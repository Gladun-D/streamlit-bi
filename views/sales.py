"""Продажи: динамика, факторы (трафик и чек), категории, промо, рейтинг магазинов."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import config as C
from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def render(f: Q.Filters):
    ui.page_header("Продажи", "Динамика, факторы изменения выручки, категории и промо", f)
    py_ok = common.py_available(f)
    cur = common.load_period(f)
    if common.no_data(cur.sales):
        return
    prev = common.load_period(f.py()) if py_ok else None
    common.notes(f, py_ok)
    k, kp = cur.k, (prev.k if prev else {})

    def d(key):
        return M.change(k[key], kp.get(key, np.nan)) if py_ok else np.nan

    days = max(f.days, 1)
    cols = st.columns(5)
    ui.kpi(cols[0], "Выручка в день", ui.rub(k["revenue"] / days), ui.delta_pct(d("revenue")))
    ui.kpi(cols[1], "Валовая прибыль", ui.rub(k["gross_profit"]), ui.delta_pct(d("gross_profit")))
    ui.kpi(cols[2], "Средний чек", ui.rub(k["avg_check"], 0), ui.delta_pct(d("avg_check")))
    ui.kpi(cols[3], "Доля промо в выручке", ui.pct(k["promo_share"]),
           ui.delta_pp(k["promo_share"] - kp.get("promo_share", np.nan)) if py_ok else None, good_when_up=None)
    has_g = (not f.chains) or ("G" in f.chains)
    if has_g:
        fg = Q.resolve(Q.Filters(f.start, f.end, ("G",), f.dcs, f.regions, f.stores, f.cats))
        tg = Q.traffic_daily(fg)
        loy = M.safe_div(tg["loyalty_checks"].sum(), tg["checks"].sum()) if len(tg) else np.nan
        ui.kpi(cols[4], "Чеки с картой (сеть А)", ui.pct(loy),
               help="Только сеть с программой лояльности; у дискаунтера карта не используется",
               good_when_up=None, arrow="off")
    else:
        ui.kpi(cols[4], "Чеков в день на магазин",
               ui.num(k["checks"] / max(cur.traffic["store_days"].sum(), 1)), None)

    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        _decomposition(cur, prev, py_ok)
    with c2:
        _weekday(cur)

    st.subheader("Категории")
    c3, c4 = st.columns(2, gap="large")
    cat_cur = Q.sales_by(f, "category_id")
    cats = Q.dims()["category"].set_index("category_id")
    with c3:
        _cat_structure(cat_cur, cats)
    with c4:
        _cat_growth(f, cat_cur, cats, py_ok)

    st.subheader("Промо")
    c5, c6 = st.columns(2, gap="large")
    with c5:
        _promo_share(cat_cur, cats)
    with c6:
        _promo_uplift(f, cats)

    if has_g and cur.traffic["delivery_orders"].sum() > 0:
        st.subheader("Онлайн-доставка")
        w = M.full_weeks(cur.traffic, "delivery_orders")
        fig = go.Figure(go.Bar(x=w.index, y=w.values, marker=dict(color=ui.SERIES[0], cornerradius=4),
                               hovertemplate="Неделя с %{x|%d.%m}: %{y:,.0f} заказов<extra></extra>"))
        fig.update_layout(title="Заказов доставки в неделю", yaxis=dict(title="заказов"))
        ui.chart(fig, 260, key="sl_delivery", dates=True)

    st.subheader("Рейтинг магазинов")
    _store_rank(f, py_ok)


def _decomposition(cur: common.Period, prev: common.Period | None, py_ok: bool):
    if not py_ok or prev is None or prev.k["checks"] == 0:
        st.markdown("**Факторы изменения выручки**")
        st.caption("Нужен период с базой прошлого года.")
        return
    dec = M.revenue_decomposition(cur.k["checks"], cur.k["avg_check"], prev.k["checks"], prev.k["avg_check"])
    fig = go.Figure(go.Waterfall(
        x=["Прошлый год", "Трафик (чеки)", "Средний чек", "Текущий период"],
        measure=["absolute", "relative", "relative", "total"],
        y=[dec["base"] / 1e6, dec["traffic"] / 1e6, dec["check"] / 1e6, dec["current"] / 1e6],
        text=[ui.rub(dec["base"]), ui.rub(dec["traffic"]), ui.rub(dec["check"]), ui.rub(dec["current"])],
        textposition="outside",
        connector=dict(line=dict(color=ui.AXIS, width=1)),
        increasing=dict(marker=dict(color=ui.SERIES[0])),
        decreasing=dict(marker=dict(color=ui.SERIES[7])),
        totals=dict(marker=dict(color=ui.NEUTRAL)),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    lo = min(dec["base"], dec["current"]) / 1e6 * 0.85
    fig.update_layout(title="Факторы изменения выручки магазинов",
                      yaxis=dict(range=[lo, max(dec["base"], dec["current"]) / 1e6 * 1.08], tickformat=",.0f",
                                 title="млн ₽"), showlegend=False)
    ui.chart(fig, 340, key="sl_waterfall")
    st.caption("Эффект трафика = Δчеков × чек прошлого года; эффект чека = Δчека × чеки текущего периода. "
               "Считается по выручке магазинов целиком.")


def _weekday(cur: common.Period):
    t = cur.traffic.copy()
    t["wd"] = t["date"].dt.dayofweek
    g = t.groupby("wd").agg(rev=("revenue", "mean"), checks=("checks", "mean"))
    idx = g["rev"] / g["rev"].mean()
    fig = go.Figure(go.Bar(x=[WEEKDAYS[i] for i in g.index], y=idx.values,
                           marker=dict(color=ui.SERIES[0], cornerradius=4),
                           text=[f"{v:.2f}" for v in idx.values], textposition="outside",
                           hovertemplate="%{x}: индекс %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=ui.AXIS, width=1, dash="dot"))
    fig.update_layout(title="Профиль недели: индекс выручки к среднему дню", yaxis=dict(range=[0.8, idx.max() * 1.08]),
                      showlegend=False)
    ui.chart(fig, 340, key="sl_weekday")


def _cat_structure(cat_cur: pd.DataFrame, cats: pd.DataFrame):
    df = cat_cur.assign(name=cat_cur["category_id"].map(cats["category"]))
    df["share"] = df["revenue"] / df["revenue"].sum()
    df["margin"] = M.safe_div(df["revenue"] - df["cost"], df["revenue"])
    df = df.sort_values("share")
    fig = go.Figure(go.Bar(
        x=df["share"], y=df["name"], orientation="h", marker=dict(color=ui.SERIES[0], cornerradius=4),
        text=[ui.pct(v) for v in df["share"]], textposition="outside",
        customdata=np.c_[df["revenue"] / 1e6, df["margin"] * 100],
        hovertemplate="%{y}<br>Доля: %{x:.1%}<br>Выручка: %{customdata[0]:,.1f} млн ₽<br>Маржа: %{customdata[1]:.1f}%<extra></extra>"))
    fig.update_layout(title="Структура выручки", xaxis=dict(tickformat=".0%", range=[0, df["share"].max() * 1.25]),
                      showlegend=False)
    ui.chart(fig, 420, key="sl_cat_struct")


def _cat_growth(f: Q.Filters, cat_cur: pd.DataFrame, cats: pd.DataFrame, py_ok: bool):
    if not py_ok:
        st.markdown("**Рост категорий г/г**")
        st.caption("Нужен период с базой прошлого года.")
        return
    comp = M.comparable_stores(Q.dims()["store"], f.start)
    ids = tuple(sorted(i for i in comp if f.store_ids is None or i in set(f.store_ids)))
    if not ids:
        st.caption("Нет сопоставимых магазинов.")
        return
    cf = Q.Filters(f.start, f.end, f.chains, f.dcs, f.regions, f.stores, f.cats, ids, f.dc_ids)
    cur_c = Q.sales_by(cf, "category_id")[["category_id", "revenue"]]
    py = Q.sales_by(cf.py(), "category_id")[["category_id", "revenue"]].rename(columns={"revenue": "py"})
    df = cur_c.merge(py, on="category_id")
    df["yoy"] = df["revenue"] / df["py"] - 1
    df["name"] = df["category_id"].map(cats["category"])
    df = df.sort_values("yoy")
    colors = [ui.SERIES[0] if v >= 0 else ui.SERIES[7] for v in df["yoy"]]
    fig = go.Figure(go.Bar(x=df["yoy"], y=df["name"], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[ui.delta_pct(v, "") for v in df["yoy"]], textposition="outside",
                           hovertemplate="%{y}: %{x:+.1%} г/г<extra></extra>"))
    lim = max(abs(df["yoy"]).max() * 1.35, 0.05)
    fig.add_vline(x=0, line=dict(color=ui.AXIS, width=1))
    fig.update_layout(title="Рост категорий в сопоставимых магазинах (LFL)", xaxis=dict(tickformat="+.0%", range=[-lim if df["yoy"].min() < 0 else -lim / 6, lim]),
                      showlegend=False)
    ui.chart(fig, 420, key="sl_cat_growth")


def _promo_share(cat_cur: pd.DataFrame, cats: pd.DataFrame):
    df = cat_cur.assign(name=cat_cur["category_id"].map(cats["category"]))
    df["promo"] = M.safe_div(df["promo_revenue"], df["revenue"])
    df = df.sort_values("promo")
    fig = go.Figure(go.Bar(x=df["promo"], y=df["name"], orientation="h", marker=dict(color=ui.SERIES[0], cornerradius=4),
                           text=[ui.pct(v) for v in df["promo"]], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%} выручки по промо<extra></extra>"))
    fig.update_layout(title="Доля промо в выручке категории", xaxis=dict(tickformat=".0%", range=[0, df["promo"].max() * 1.3]),
                      showlegend=False)
    ui.chart(fig, 400, key="sl_promo_share")


def _promo_uplift(f: Q.Filters, cats: pd.DataFrame):
    u = Q.promo_uplift(f)
    if u.empty:
        ui.empty_state()
        return
    p = u.pivot_table(index="category_id", columns="promo", values=["demand", "sku_days"], aggfunc="sum")
    try:
        base = p[("demand", 0)] / p[("sku_days", 0)]
        promo = p[("demand", 1)] / p[("sku_days", 1)]
    except KeyError:
        st.caption("В выбранном периоде нет акций.")
        return
    up = (promo / base).dropna().rename("uplift").reset_index()
    up["name"] = up["category_id"].map(cats["category"])
    up = up.sort_values("uplift")
    fig = go.Figure(go.Bar(x=up["uplift"], y=up["name"], orientation="h", marker=dict(color=ui.SERIES[2], cornerradius=4),
                           text=[f"×{v:.1f}" for v in up["uplift"]], textposition="outside",
                           hovertemplate="%{y}: продажи в акции ×%{x:.2f} к обычному дню<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=ui.AXIS, width=1, dash="dot"))
    fig.update_layout(title="Эффект акций: продажи SKU в день акции к обычному дню",
                      xaxis=dict(range=[0, up["uplift"].max() * 1.2]), showlegend=False)
    ui.chart(fig, 400, key="sl_promo_uplift")
    st.caption("По отгрузкам со складов; это валовой эффект — без учёта каннибализации и последующего провала спроса.")


def _store_rank(f: Q.Filters, py_ok: bool):
    df = common.store_table(f)
    if df.empty:
        ui.empty_state()
        return
    df["lfl"] = np.where(df["comparable"], df["yoy"], np.nan)
    sort_by = st.segmented_control("Сортировка", ["Выручка", "LFL", "Выполнение плана", "Дефицит"], default="Выручка",
                                   key="sl_sort", label_visibility="collapsed")
    col = {"Выручка": ("revenue", False), "LFL": ("lfl", True), "Выполнение плана": ("plan_pct", True),
           "Дефицит": ("oos", False)}[sort_by or "Выручка"]
    df = df.sort_values(col[0], ascending=col[1], na_position="last")
    show = pd.DataFrame({
        "Магазин": df["store_name"], "Сеть": df["chain_name"], "Регион": df["region"], "РЦ": df["dc_name"],
        "Выручка, тыс ₽": df["revenue"] / 1e3, "Г/г": df["yoy"], "LFL": df["lfl"],
        "План": df["plan_pct"], "Ср. чек, ₽": df["avg_check"], "Чеков в день": df["checks_day"],
        "Маржа": df["margin"], "OOS": df["oos"], "Открыт": df["open_date"].dt.date,
    })
    st.dataframe(show, hide_index=True, height=420, column_config={
        "Выручка, тыс ₽": st.column_config.NumberColumn(format="%.0f"),
        "Г/г": st.column_config.NumberColumn(format="percent", help="Для новых магазинов не показательно"),
        "LFL": st.column_config.NumberColumn(format="percent", help="Только сопоставимые магазины"),
        "План": st.column_config.ProgressColumn(format="percent", min_value=0.8, max_value=1.1),
        "Ср. чек, ₽": st.column_config.NumberColumn(format="%.0f"),
        "Чеков в день": st.column_config.NumberColumn(format="%.0f"),
        "Маржа": st.column_config.NumberColumn(format="percent"),
        "OOS": st.column_config.NumberColumn(format="percent"),
        "Открыт": st.column_config.DateColumn(format="DD.MM.YYYY"),
    })
    ui.download({"Магазины": show}, "reyting_magazinov.xlsx", key="dl_sl_stores")
