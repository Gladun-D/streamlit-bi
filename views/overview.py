"""Обзор сети: ключевые показатели, тренд к прошлому году и плану, сравнение сетей, РЦ и регионов."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common


def render(f: Q.Filters):
    ui.page_header("Обзор сети", "Ключевые показатели к прошлому году и плану", f)
    py_ok = common.py_available(f)
    cur = common.load_period(f)
    if common.no_data(cur.sales):
        return
    prev = common.load_period(f.py()) if py_ok else None
    common.notes(f, py_ok)

    k, kp = cur.k, (prev.k if prev else {})
    stores = common.store_table(f)
    comp = stores[stores["comparable"] & stores["revenue_py"].notna()]
    lfl = M.change(comp["revenue"].sum(), comp["revenue_py"].sum()) if py_ok and len(comp) else np.nan
    plan_pct = M.safe_div(stores.loc[stores["plan"].notna(), "store_revenue"].sum(), stores["plan"].sum())
    otif = common.otif_summary(f)
    otif_py = common.otif_summary(f.py()) if py_ok else np.nan

    def d(key):
        return M.change(k[key], kp.get(key, np.nan)) if py_ok else np.nan

    def dpp(key):
        return k[key] - kp.get(key, np.nan) if py_ok else np.nan

    spark_src = Q.sales_daily(f.with_dates(max(f.start, f.end - timedelta(days=181)), f.end))
    spark_tr = Q.traffic_daily(f.with_dates(max(f.start, f.end - timedelta(days=181)), f.end))
    wk_rev, wk_cost = M.full_weeks(spark_src, "revenue"), M.full_weeks(spark_src, "cost")
    wk_chk, wk_trev = M.full_weeks(spark_tr, "checks"), M.full_weeks(spark_tr, "revenue")
    sp_margin = ((wk_rev - wk_cost) / wk_rev * 100).round(2).tail(13).tolist()
    sp_check = (wk_trev / wk_chk).round(1).tail(13).tolist()
    r1 = st.columns(5)
    ui.kpi(r1[0], "Выручка", ui.rub(k["revenue"]), ui.delta_pct(d("revenue")),
           help="Розничная выручка за период. Сравнение — с тем же периодом прошлого года (сдвиг 364 дня). "
                "График — по неделям, до 26 недель.",
           spark=wk_rev.tail(13).round(0).tolist())
    ui.kpi(r1[1], "LFL выручки", ui.pct(lfl), None if not np.isfinite(lfl) else
           f"{int(comp['store_id'].nunique())} магазинов", good_when_up=None, arrow="off",
           help="Like-for-like: изменение выручки магазинов, открытых не позже чем за год до начала периода.",
           height=ui.KPI_H)
    ui.kpi(r1[2], "Валовая маржа", ui.pct(k["margin"]), ui.delta_pp(dpp("margin")),
           help="(Выручка − себестоимость продаж) / выручка", spark=sp_margin)
    ui.kpi(r1[3], "Средний чек", ui.rub(k["avg_check"], 0), ui.delta_pct(d("avg_check")),
           help="Выручка магазинов / число чеков (по магазину целиком)", spark=sp_check)
    ui.kpi(r1[4], "Чеки", ui.big(k["checks"]), ui.delta_pct(d("checks")),
           help="Количество покупок — показатель трафика", spark=wk_chk.tail(13).round(0).tolist())

    r2 = st.columns(5)
    ui.kpi(r2[0], "Выполнение плана", ui.pct(plan_pct),
           None if not np.isfinite(plan_pct) else ui.delta_pct(plan_pct - 1, "к плану"),
           help="Выручка магазинов / план выручки за дни периода (план задан по месяцам и магазинам)")
    ui.kpi(r2[1], "Дефицит (OOS)", ui.pct(k["oos"]), ui.delta_pp(dpp("oos")), good_when_up=False,
           help="Доля позиций с нулевым остатком на полке, взвешенная по выручке")
    ui.kpi(r2[2], "Запас в магазинах", f"{ui.num(k['cover_days'], 1)} дн.",
           ui.delta_abs(dpp("cover_days"), " дн."), good_when_up=None,
           help="Средний остаток в себестоимости / средняя дневная себестоимость продаж")
    ui.kpi(r2[3], "Списания", ui.pct(k["writeoff_rate"], 2), ui.delta_pp(dpp("writeoff_rate"), digits=2),
           good_when_up=False, help="Списания / себестоимость продаж")
    ui.kpi(r2[4], "OTIF поставщиков", ui.pct(otif), ui.delta_pp(otif - otif_py if py_ok else np.nan),
           help="Доля заказов, пришедших в срок и в полном объёме (≥ 98% заказанного)")

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        _trend(f, py_ok)
    with right:
        _chains(stores, f, py_ok)

    st.subheader("РЦ и регионы: где отклонения")
    _regions(stores, f, py_ok, k["oos"])

    st.subheader("Требуют внимания")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        _worst_stores(stores, py_ok)
    with c2:
        _categories(f, py_ok)


def _trend(f: Q.Filters, py_ok: bool):
    tf = f if f.days >= 56 else f.with_dates(f.end - timedelta(days=90), f.end)
    cur = M.full_weeks(Q.sales_daily(tf), "revenue")
    fig = go.Figure()
    title = "Выручка по неделям"
    if tf.py().start >= Q.data_bounds()[0]:
        py = M.full_weeks(Q.sales_daily(tf.py()), "revenue")
        py.index = py.index + pd.Timedelta(days=364)
        fig.add_scatter(x=py.index, y=py.values / 1e6, name="Прошлый год", mode="lines",
                        line=dict(color=ui.NEUTRAL, width=2, dash="dot"),
                        hovertemplate="Прошлый год: %{y:,.1f} млн ₽<extra></extra>")
    if not f.cat_filtered:
        plan = _weekly_plan(tf)
        if len(plan):
            fig.add_scatter(x=plan.index, y=plan.values / 1e6, name="План", mode="lines",
                            line=dict(color=ui.SERIES[2], width=2, dash="dash"),
                            hovertemplate="План: %{y:,.1f} млн ₽<extra></extra>")
    fig.add_scatter(x=cur.index, y=cur.values / 1e6, name="Текущий год", mode="lines+markers",
                    line=dict(color=ui.SERIES[0], width=2.5), marker=dict(size=6),
                    hovertemplate="Факт: %{y:,.1f} млн ₽<extra></extra>")
    fig.update_layout(title=title, hovermode="x unified", yaxis=dict(title="млн ₽", tickformat=",.0f"),
                      )
    ui.chart(fig, 360, key="ov_trend", dates=True)
    if tf is not f:
        st.caption("Период короче 8 недель — тренд показан за последние 13 недель.")


@st.cache_data(ttl=Q.TTL, show_spinner=False)
def _weekly_plan(f: Q.Filters) -> pd.Series:
    """План по неделям: месячный план распределяется по дням с учётом типового профиля дня недели."""
    rows = []
    m = pd.Timestamp(f.start).replace(day=1)
    while m <= pd.Timestamp(f.end):
        mf = f.with_dates(m.date(), min((m + pd.offsets.MonthEnd(0)).date(), Q.data_bounds()[1]))
        p = Q.plan_for_period(mf)
        total = p["plan"].sum() if len(p) else 0
        days = pd.date_range(mf.start, mf.end)
        wd = np.array([0.95, 0.95, 0.97, 0.99, 1.08, 1.11, 0.95])[days.dayofweek]
        rows.append(pd.Series(total * wd / wd.sum(), index=days))
        m = m + pd.offsets.MonthBegin(1)
    s = pd.concat(rows) if rows else pd.Series(dtype=float)
    s = s[(s.index >= pd.Timestamp(f.start)) & (s.index <= pd.Timestamp(f.end))]
    return M.full_weeks(s.rename("plan").rename_axis("date").reset_index(), "plan")


def _chains(stores: pd.DataFrame, f: Q.Filters, py_ok: bool):
    st.markdown("**Сравнение сетей**")
    g = common.group_table(stores, ["chain", "chain_name"], f).sort_values("chain")
    for _, r in g.iterrows():
        with st.container(border=True):
            a, b = st.columns([1.25, 1])
            title, _, fmt = str(r["chain_name"]).partition(" · ")
            a.markdown(f"**{title}**  \n"
                       f"<span style='color:{ui.MUTED};font-size:0.82rem'>{fmt + ' · ' if fmt else ''}"
                       f"{int(r['stores'])} маг. · {ui.pct(r['revenue'] / g['revenue'].sum(), 0)} выручки</span>",
                       unsafe_allow_html=True)
            b.markdown(f"<div style='text-align:right;font-size:1.35rem;font-weight:650'>{ui.rub(r['revenue'])}</div>",
                       unsafe_allow_html=True)
            c = st.columns(4)
            for col, (lab, val) in zip(c, [("Г/г", ui.delta_pct(r["yoy"], "") or "—"),
                                           ("LFL", ui.delta_pct(r["lfl"], "") or "—"),
                                           ("Маржа", ui.pct(r["margin"])), ("OOS", ui.pct(r["oos"]))]):
                col.markdown(f"<div style='color:{ui.MUTED};font-size:0.78rem'>{lab}</div>"
                             f"<div style='font-weight:600'>{val}</div>", unsafe_allow_html=True)
    st.caption("Г/г включает новые магазины; LFL — только сопоставимые.")


def _regions(stores: pd.DataFrame, f: Q.Filters, py_ok: bool, oos_ref: float):
    by = st.segmented_control("Разрез", ["РЦ / склад", "Регион"], default="РЦ / склад", key="ov_region_by",
                              label_visibility="collapsed")
    key = ["chain_name", "dc_name"] if by == "РЦ / склад" else ["chain_name", "region"]
    g = common.group_table(stores, key, f)
    g["status"] = g.apply(lambda r: common.status_icon(r, oos_ref), axis=1)
    g = g.sort_values("revenue", ascending=False)
    show = pd.DataFrame({
        "": g["status"],
        "Сеть": g["chain_name"],
        ("РЦ / склад" if by == "РЦ / склад" else "Регион"): g[key[1]],
        "Магазинов": g["stores"].astype(int),
        "Выручка, млн ₽": g["revenue"] / 1e6,
        "Г/г": g["yoy"],
        "LFL": g["lfl"],
        "План": g["plan_pct"],
        "Маржа": g["margin"],
        "OOS": g["oos"],
        "Списания": g["writeoff_rate"],
    })
    st.dataframe(show, hide_index=True, height=min(38 + 35 * len(show), 460), column_config={
        "": st.column_config.TextColumn(width=36, help="🔴 LFL < −3% или дефицит в 1,5 раза выше среднего; "
                                                         "🟡 LFL < 0, план < 97% или дефицит выше среднего на 20%"),
        "Выручка, млн ₽": st.column_config.NumberColumn(format="%.1f"),
        "Г/г": st.column_config.NumberColumn(format="percent"),
        "LFL": st.column_config.NumberColumn(format="percent"),
        "План": st.column_config.ProgressColumn(format="percent", min_value=0.8, max_value=1.1,
                                                help="Выполнение плана выручки"),
        "Маржа": st.column_config.NumberColumn(format="percent"),
        "OOS": st.column_config.NumberColumn(format="percent"),
        "Списания": st.column_config.NumberColumn(format="percent"),
    })
    ui.download({"Свод": show}, "svod_rc_regiony.xlsx", key="dl_ov_regions")


def _worst_stores(stores: pd.DataFrame, py_ok: bool):
    st.markdown("**Магазины с наибольшим падением LFL**")
    if not py_ok:
        st.caption("Нужен период с базой прошлого года.")
        return
    comp = stores[stores["comparable"] & stores["revenue_py"].notna()].copy()
    if comp.empty:
        ui.empty_state("Нет сопоставимых магазинов")
        return
    comp["lfl"] = comp["revenue"] / comp["revenue_py"] - 1
    worst = comp.nsmallest(10, "lfl")
    show = pd.DataFrame({
        "Магазин": worst["store_name"], "Регион": worst["region"], "LFL": worst["lfl"],
        "Чеки г/г": worst["checks"] / worst["checks_py"] - 1, "OOS": worst["oos"],
    })
    st.dataframe(show, hide_index=True, column_config={
        "LFL": st.column_config.NumberColumn(format="percent"),
        "Чеки г/г": st.column_config.NumberColumn(format="percent"),
        "OOS": st.column_config.NumberColumn(format="percent"),
    })


def _categories(f: Q.Filters, py_ok: bool):
    st.markdown("**Категории: рост дефицита и списаний**")
    cur = Q.sales_by(f, "category_id")
    cats = Q.dims()["category"].set_index("category_id")["category"]
    cur["oos"] = M.safe_div(cur["oos_w"], cur["revenue"])
    cur["wo"] = M.safe_div(cur["writeoff"], cur["cost"])
    if py_ok:
        py = Q.sales_by(f.py(), "category_id")
        py["oos_py"] = M.safe_div(py["oos_w"], py["revenue"])
        py["wo_py"] = M.safe_div(py["writeoff"], py["cost"])
        cur = cur.merge(py[["category_id", "oos_py", "wo_py"]], on="category_id", how="left")
    else:
        cur["oos_py"] = np.nan
        cur["wo_py"] = np.nan
    cur["d_oos"] = cur["oos"] - cur["oos_py"]
    cur["d_wo"] = cur["wo"] - cur["wo_py"]
    cur = cur.sort_values("d_oos", ascending=False, na_position="last")
    show = pd.DataFrame({
        "Категория": cur["category_id"].map(cats), "OOS": cur["oos"], "Δ OOS, п.п.": cur["d_oos"] * 100,
        "Списания": cur["wo"], "Δ списаний, п.п.": cur["d_wo"] * 100,
    })
    st.dataframe(show, hide_index=True, height=38 + 35 * len(show), column_config={
        "OOS": st.column_config.NumberColumn(format="percent"),
        "Δ OOS, п.п.": st.column_config.NumberColumn(format="%+.2f"),
        "Списания": st.column_config.NumberColumn(format="percent"),
        "Δ списаний, п.п.": st.column_config.NumberColumn(format="%+.2f"),
    })
