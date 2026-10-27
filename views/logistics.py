"""Логистика: поставщики (OTIF, сроки), рейсы склад → магазин (пунктуальность, потери), почасовой профиль."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

BUCKETS = ["раньше<br>> 30 мин", "в окне<br>±30 мин", "+30–60<br>мин", "+1–2 ч", "+2–4 ч", "> +4 ч"]


def _wavg(df: pd.DataFrame, col: str, w: str) -> float:
    m = df[col].notna() & (df[w] > 0)
    return float(np.average(df.loc[m, col], weights=df.loc[m, w])) if m.any() else np.nan


def render(f: Q.Filters):
    ui.page_header("Логистика", "Поставки от поставщиков на склады и рейсы со складов в магазины", f)
    py_ok = common.py_available(f)
    if f.cats:
        st.caption(":material/info: Фильтр по категориям применяется к заказам поставщикам; рейсы в магазины — сборные, по ним он не действует.")

    by_dc = Q.orders_by(f, "dc_id")
    trips_dc = Q.deliveries_by(f, "dc_id")
    if by_dc.empty and trips_dc.empty:
        ui.empty_state()
        return
    by_dc_py = Q.orders_by(f.py(), "dc_id") if py_ok else pd.DataFrame()
    trips_py = Q.deliveries_by(f.py(), "dc_id") if py_ok else pd.DataFrame()

    otif = _wavg(by_dc, "otif", "orders")
    on_time = _wavg(by_dc, "on_time", "orders")
    fill = M.safe_div((by_dc["fill_rate"] * by_dc["orders"]).sum(), by_dc["orders"].sum())
    lt = _wavg(by_dc, "lt_avg", "orders")
    trips_ok = _wavg(trips_dc, "on_time", "trips")
    loss = 1 - M.safe_div(trips_dc["cost_received"].sum(), trips_dc["cost_shipped"].sum())

    def pp(cur, df, col, w):
        return (cur - _wavg(df, col, w)) if py_ok and len(df) else np.nan

    k = st.columns(3)
    ui.kpi(k[0], "OTIF поставщиков", ui.pct(otif), ui.delta_pp(pp(otif, by_dc_py, "otif", "orders")),
           help="Доля заказов, пришедших в обещанную дату или раньше и в объёме не менее 98% заказанного")
    ui.kpi(k[1], "Пунктуальность поставщиков", ui.pct(on_time), ui.delta_pp(pp(on_time, by_dc_py, "on_time", "orders")))
    ui.kpi(k[2], "Срок поставки", f"{ui.num(lt, 1)} дн.",
           ui.delta_abs(lt - _wavg(by_dc_py, "lt_avg", "orders"), " дн.") if py_ok and len(by_dc_py) else None,
           good_when_up=False, help="От даты заказа до даты прихода, среднее по заказам")
    k = st.columns(3)
    ui.kpi(k[0], "Полнота поставок", ui.pct(fill), None, help="Принято / заказано, в штуках")
    ui.kpi(k[1], "Рейсы в окно ±30 мин", ui.pct(trips_ok), ui.delta_pp(pp(trips_ok, trips_py, "on_time", "trips")),
           help="Рейсы склад → магазин, прибывшие не позже 30 минут после плана")
    ui.kpi(k[2], "Потери при доставке", ui.pct(loss, 2), None, good_when_up=False,
           help="1 − принято магазином / отгружено со склада, в себестоимости")

    st.subheader("Поставщики")
    _suppliers(f)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        _lead_time_box(f)
    with c2:
        _otif_trend(f)

    st.subheader("Рейсы склад → магазин")
    c3, c4 = st.columns(2, gap="large")
    with c3:
        _trips_trend(f)
    with c4:
        _delay_hist(f)
    c5, c6 = st.columns(2, gap="large")
    with c5:
        _by_hour(f)
    with c6:
        _dc_table(trips_dc)


def _suppliers(f: Q.Filters):
    s = Q.orders_by(f, "supplier_id")
    if s.empty:
        ui.empty_state()
        return
    sup = Q.dims()["supplier"].set_index("supplier_id")["supplier"]
    s["name"] = s["supplier_id"].map(sup)
    s["status"] = np.select([s["otif"] < 0.7, s["otif"] < 0.8], ["critical", "warning"], "good")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = go.Figure()
        labels = {"critical": "OTIF < 70%", "warning": "OTIF 70–80%", "good": "OTIF ≥ 80%"}
        for stt in ("good", "warning", "critical"):
            d = s[s["status"] == stt]
            if d.empty:
                continue
            fig.add_scatter(
                x=d["lt_avg"], y=d["otif"], mode="markers", name=labels[stt],
                marker=dict(size=np.clip(np.sqrt(d["cost_received"] / s["cost_received"].max()) * 34, 8, 34),
                            color=ui.STATUS[stt], line=dict(color="white", width=2), opacity=0.9),
                customdata=np.c_[d["name"], d["orders"], d["delay_avg"]],
                hovertemplate="%{customdata[0]}<br>OTIF: %{y:.1%}<br>Срок: %{x:.1f} дн.<br>Заказов: %{customdata[1]}"
                              "<br>Средняя задержка: %{customdata[2]:.2f} дн.<extra></extra>")
        worst = s.nsmallest(3, "otif")
        for _, r in worst.iterrows():
            fig.add_annotation(x=r["lt_avg"], y=r["otif"], text=r["name"], showarrow=False, yshift=-16,
                               font=dict(size=11, color=ui.INK2))
        fig.update_layout(title="Поставщики: срок и OTIF",
                          xaxis=dict(title="средний срок поставки, дн."), yaxis=dict(tickformat=".0%", title="OTIF"))
        ui.chart(fig, 380, key="lg_sup_scatter")
        st.caption("Размер точки — объём поставок в себестоимости.")
    with right:
        t = s.sort_values("otif")
        show = pd.DataFrame({
            "": t["status"].map({"critical": "🔴", "warning": "🟡", "good": "🟢"}),
            "Поставщик": t["name"], "Заказов": t["orders"], "OTIF": t["otif"], "Вовремя": t["on_time"],
            "Полнота": t["fill_rate"], "Срок, дн.": t["lt_avg"], "Разброс срока, дн.": t["lt_std"],
            "Поставлено, млн ₽": t["cost_received"] / 1e6,
        })
        st.dataframe(show, hide_index=True, height=380, column_config={
            "": st.column_config.TextColumn(width=36),
            "OTIF": st.column_config.ProgressColumn(format="percent", min_value=0.5, max_value=1),
            "Вовремя": st.column_config.NumberColumn(format="percent"),
            "Полнота": st.column_config.NumberColumn(format="percent"),
            "Срок, дн.": st.column_config.NumberColumn(format="%.1f"),
            "Разброс срока, дн.": st.column_config.NumberColumn(format="%.1f"),
            "Поставлено, млн ₽": st.column_config.NumberColumn(format="%.1f"),
        })
        ui.download({"Поставщики": show}, "postavshchiki.xlsx", key="dl_lg_sup")


def _lead_time_box(f: Q.Filters):
    fr = Q.lead_time_freq(f)
    if fr.empty:
        return
    dc = Q.dims()["dc"].set_index("dc_id")["name"]
    fig = go.Figure()
    for dc_id, g in fr.groupby("dc_id"):
        g = g.sort_values("lead_time_days")
        cum = g["n"].cumsum() / g["n"].sum()
        q = {p: float(g["lead_time_days"].iloc[int(np.searchsorted(cum.values, p))]) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
        fig.add_trace(go.Box(name=dc[dc_id], x=[dc[dc_id]], q1=[q[0.25]], median=[q[0.5]], q3=[q[0.75]],
                             lowerfence=[q[0.05]], upperfence=[q[0.95]],
                             mean=[float(np.average(g["lead_time_days"], weights=g["n"]))],
                             marker=dict(color=ui.SERIES[0]), fillcolor="rgba(42,120,214,0.18)",
                             line=dict(color=ui.SERIES[0], width=1.5), showlegend=False))
    fig.update_layout(title="Срок поставки по складам, дней", yaxis=dict(title="дней"),
                      xaxis=dict(tickangle=-20))
    ui.chart(fig, 340, key="lg_lt_box")
    st.caption("Ящик — от 25-го до 75-го процентиля, линия — медиана, пунктир — среднее, усы — 5–95%.")


def _otif_trend(f: Q.Filters):
    w = Q.orders_by(f, "week_start")
    if w.empty:
        return
    w = w.set_index("day").sort_index()
    days = w["orders"].resample("W-SUN").count()
    wk = pd.DataFrame({
        "otif": (w["otif"] * w["orders"]).resample("W-SUN").sum() / w["orders"].resample("W-SUN").sum(),
        "on_time": (w["on_time"] * w["orders"]).resample("W-SUN").sum() / w["orders"].resample("W-SUN").sum(),
    })[days == 7].dropna()
    fig = go.Figure()
    fig.add_scatter(x=wk.index, y=wk["on_time"], name="Вовремя", mode="lines", line=dict(color=ui.SERIES[2], width=2))
    fig.add_scatter(x=wk.index, y=wk["otif"], name="OTIF", mode="lines", line=dict(color=ui.SERIES[0], width=2.5))
    fig.update_layout(title="OTIF и пунктуальность поставщиков по неделям", yaxis=dict(tickformat=".0%"),
                      hovermode="x unified")
    ui.chart(fig, 340, key="lg_otif_trend", dates=True)


def _trips_trend(f: Q.Filters):
    t = Q.deliveries_by_dc_day(f)
    if t.empty:
        return
    dc = Q.dims()["dc"].set_index("dc_id")
    fig = go.Figure()
    for i, (dc_id, g) in enumerate(t.groupby("dc_id")):
        r = g.set_index("date").resample("W-SUN")
        n_days = r["trips"].count()
        g = r[["trips", "on_time_trips"]].sum()
        g = g[(g["trips"] > 0) & (n_days == 7)]
        fig.add_scatter(x=g.index, y=g["on_time_trips"] / g["trips"], name=dc.loc[dc_id, "name"], mode="lines",
                        line=dict(color=ui.SERIES[(dc_id - 1) % len(ui.SERIES)], width=2),
                        hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Доля рейсов в окне ±30 мин по складам", yaxis=dict(tickformat=".0%"), hovermode="x unified",
                      legend=dict(orientation="v", x=1.02, y=1, yanchor="top", xanchor="left", font=dict(size=11)),
                      margin=dict(t=48))
    ui.chart(fig, 360, key="lg_trips_trend", dates=True)


def _delay_hist(f: Q.Filters):
    b = Q.delay_buckets(f)
    if b.empty:
        return
    b = b.set_index("bucket").reindex(range(6)).fillna(0)
    share = b["trips"] / b["trips"].sum()
    colors = [ui.NEUTRAL, ui.STATUS["good"], ui.STATUS["warning"], ui.STATUS["serious"], ui.STATUS["critical"], ui.STATUS["critical"]]
    fig = go.Figure(go.Bar(x=BUCKETS, y=share.values, marker=dict(color=colors, cornerradius=4),
                           text=[ui.pct(v) for v in share.values], textposition="outside",
                           hovertemplate="%{x}: %{y:.1%} рейсов<extra></extra>"))
    fig.update_layout(title="Отклонение прибытия от плана", xaxis=dict(tickangle=0),
                      yaxis=dict(tickformat=".0%", range=[0, share.max() * 1.18]),
                      showlegend=False)
    ui.chart(fig, 360, key="lg_delay")


def _by_hour(f: Q.Filters):
    h = Q.deliveries_by(f, "planned_hour")
    if h.empty:
        return
    h = h.sort_values("planned_hour")
    fig = go.Figure(go.Bar(
        x=h["planned_hour"].astype(int).map(lambda v: f"{v:02d}:00"), y=h["trips"],
        marker=dict(color=ui.SERIES[0], cornerradius=4), customdata=h["on_time"],
        hovertemplate="%{x}: %{y:,.0f} рейсов, вовремя %{customdata:.1%}<extra></extra>"))
    fig.update_layout(title="Рейсы по плановому часу прибытия", yaxis=dict(title="рейсов"), showlegend=False)
    ui.chart(fig, 320, key="lg_hour")
    st.caption("Почасовой профиль — основа для ежечасного мониторинга рейсов: кэш данных в приложении живёт один час.")


def _dc_table(trips: pd.DataFrame):
    if trips.empty:
        return
    dc = Q.dims()["dc"].set_index("dc_id")
    t = trips.sort_values("on_time")
    show = pd.DataFrame({
        "Склад": t["dc_id"].map(dc["name"]), "Сеть": t["dc_id"].map(dc["chain_name"]), "Рейсов": t["trips"],
        "В окне": t["on_time"], "Среднее опоздание, мин": t["late_delay_avg"],
        "Потери, тыс ₽": (t["cost_shipped"] - t["cost_received"]) / 1e3,
        "Потери, %": (1 - t["cost_received"] / t["cost_shipped"]) * 100,
    })
    st.markdown("**Склады: качество доставки в магазины**")
    st.dataframe(show, hide_index=True, column_config={
        "В окне": st.column_config.ProgressColumn(format="percent", min_value=0.6, max_value=1),
        "Среднее опоздание, мин": st.column_config.NumberColumn(format="%.0f", help="Среди рейсов, опоздавших более чем на 30 минут"),
        "Потери, тыс ₽": st.column_config.NumberColumn(format="%.0f"),
        "Потери, %": st.column_config.NumberColumn(format="%.2f%%"),
    })
    ui.download({"Склады": show}, "dostavka_po_skladam.xlsx", key="dl_lg_dc")
