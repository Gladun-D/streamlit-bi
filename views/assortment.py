"""Ассортимент: ABC/XYZ-анализ SKU, кривая Парето, рекомендации по управлению позициями."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common


def render(f: Q.Filters):
    ui.page_header("Ассортимент: ABC/XYZ", "Вклад позиций в результат и стабильность спроса — по отгрузкам со складов в магазины", f)
    if f.store_filtered and (f.regions or f.stores):
        st.caption(":material/info: Анализ строится по складам: фильтры по региону и магазину сюда не применяются, "
                   "применяются сеть, склад и категория.")
    if f.days < 56:
        st.warning("Для XYZ-анализа нужен период не короче 8 недель — выберите период длиннее.", icon=":material/warning:")

    c = st.columns([1.2, 1, 1])
    basis = c[0].segmented_control("ABC по", ["Выручка", "Валовая прибыль", "Штуки"], default="Выручка", key="as_basis")
    x_thr = c[1].slider("Граница X (CV)", 0.05, 0.5, 0.25, 0.05, key="as_x",
                        help="Коэффициент вариации недельных продаж, до которого спрос считается стабильным")
    y_thr = c[2].slider("Граница Y (CV)", 0.2, 1.0, 0.5, 0.05, key="as_y")
    if y_thr <= x_thr:
        y_thr = x_thr + 0.05

    wk = Q.sku_weekly(f)
    if wk.empty:
        ui.empty_state()
        return
    wk["week_start"] = pd.to_datetime(wk["week_start"])
    # только полные недели внутри периода
    wk = wk[(wk["week_start"] >= pd.Timestamp(f.start)) & (wk["week_start"] + pd.Timedelta(days=6) <= pd.Timestamp(f.end))]
    if wk.empty:
        ui.empty_state("В периоде нет ни одной полной недели")
        return
    col = {"Выручка": "revenue", "Валовая прибыль": "gross_profit", "Штуки": "qty"}[basis or "Выручка"]
    agg = wk.groupby("sku_id").agg(revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"),
                                   qty=("qty", "sum"), promo_days=("promo_days", "sum"))
    agg["cv"] = M.weekly_cv(wk, "qty")
    agg["abc"] = M.abc_classes(agg[col])
    agg["xyz"] = M.xyz_classes(agg["cv"], x_thr, y_thr)
    agg["cls"] = agg["abc"] + agg["xyz"]
    agg["share"] = agg[col] / agg[col].sum()
    sku = Q.dims()["sku"].set_index("sku_id")
    cats = Q.dims()["category"].set_index("category_id")["category"]
    sup = Q.dims()["supplier"].set_index("supplier_id")["supplier"]
    agg = agg.join(sku[["sku", "category_id", "supplier_id", "private_label", "lifecycle"]])
    agg["category"] = agg["category_id"].map(cats)
    agg["supplier"] = agg["supplier_id"].map(sup)
    agg = agg[agg[col] > 0]

    a_share = agg.loc[agg["abc"] == "A", col].sum() / agg[col].sum()
    k = st.columns(5)
    ui.kpi(k[0], "SKU с продажами", ui.num(len(agg)), None)
    ui.kpi(k[1], "Группа A", f"{ui.num((agg['abc'] == 'A').sum())} SKU",
           f"{ui.pct(a_share, 0)} результата", good_when_up=None, arrow="off",
           help=f"{ui.pct((agg['abc'] == 'A').mean(), 0)} позиций дают {ui.pct(a_share, 0)} показателя «{basis}»")
    ui.kpi(k[2], "AX — ядро", ui.num((agg["cls"] == "AX").sum()), None, help="Высокий вклад и стабильный спрос")
    ui.kpi(k[3], "AZ — риск", ui.num((agg["cls"] == "AZ").sum()), None,
           help="Высокий вклад, нестабильный спрос — главный источник дефицита и излишков")
    ui.kpi(k[4], "CZ — на вывод", ui.num((agg["cls"] == "CZ").sum()), None)

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        _matrix(agg, col)
    with right:
        _pareto(agg, col, basis or "Выручка")

    st.subheader("Позиции и рекомендации")
    fc = st.columns([2, 1])
    sel = fc[0].pills("Группы", [a + x for a in "ABC" for x in "XYZ"], selection_mode="multi", key="as_cls",
                      help="Пусто — все группы")
    q = fc[1].text_input("Поиск SKU", placeholder="артикул или категория", key="as_q")
    t = agg.reset_index()
    if sel:
        t = t[t["cls"].isin(sel)]
    if q:
        ql = q.lower()
        t = t[t["sku"].str.lower().str.contains(ql, regex=False) | t["category"].str.lower().str.contains(ql, regex=False)]
    t = t.sort_values(col, ascending=False)
    show = pd.DataFrame({
        "Группа": t["cls"], "SKU": t["sku"], "Категория": t["category"], "Поставщик": t["supplier"],
        "Выручка, млн ₽": t["revenue"] / 1e6, "Вал. прибыль, млн ₽": t["gross_profit"] / 1e6, "Штук": t["qty"],
        "Доля": t["share"], "CV": t["cv"], "СТМ": t["private_label"],
        "Рекомендация": t["cls"].map(M.ABC_XYZ_ACTIONS),
    })
    st.dataframe(show, hide_index=True, height=440, column_config={
        "Выручка, млн ₽": st.column_config.NumberColumn(format="%.1f"),
        "Вал. прибыль, млн ₽": st.column_config.NumberColumn(format="%.1f"),
        "Штук": st.column_config.NumberColumn(format="localized"),
        "Доля": st.column_config.NumberColumn(format="percent"),
        "CV": st.column_config.NumberColumn(format="%.2f"),
        "СТМ": st.column_config.CheckboxColumn(),
        "Рекомендация": st.column_config.TextColumn(width="large"),
    })
    ui.download({"ABC-XYZ": show}, "abc_xyz.xlsx", key="dl_as")


def _matrix(agg: pd.DataFrame, col: str):
    cnt = agg.pivot_table(index="abc", columns="xyz", values=col, aggfunc="size").reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
    val = agg.pivot_table(index="abc", columns="xyz", values=col, aggfunc="sum").reindex(index=list("ABC"), columns=list("XYZ")).fillna(0)
    share = val / val.values.sum()
    text = [[f"<b>{a}{x}</b><br>{int(cnt.loc[a, x])} SKU<br>{share.loc[a, x]:.0%}" for x in "XYZ"] for a in "ABC"]
    fig = go.Figure(go.Heatmap(
        z=share.values, x=["X<br>стабильный", "Y<br>колеблющийся", "Z<br>нерегулярный"],
        y=["A · 80%", "B · 15%", "C · 5%"], text=text, texttemplate="%{text}",
        textfont=dict(size=13), colorscale=[[0, ui.SEQ[0]], [0.5, ui.SEQ[3]], [1, ui.SEQ[6]]],
        showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y} × %{x}<br>Доля: %{z:.1%}<extra></extra>"))
    fig.update_layout(title="ABC × XYZ: SKU и доля результата", yaxis=dict(autorange="reversed"),
                      xaxis=dict(side="bottom", tickangle=0))
    ui.chart(fig, 380, key="as_matrix")


def _pareto(agg: pd.DataFrame, col: str, label: str):
    s = agg[col].sort_values(ascending=False)
    cum = s.cumsum() / s.sum()
    x = np.arange(1, len(s) + 1) / len(s)
    fig = go.Figure()
    fig.add_scatter(x=x, y=cum.values, mode="lines", line=dict(color=ui.SERIES[0], width=2.5), name="Накопленная доля",
                    hovertemplate="%{x:.0%} позиций → %{y:.0%} результата<extra></extra>")
    for thr, name in ((0.8, "A"), (0.95, "B")):
        n = int((cum.values < thr).sum()) + 1
        xi = n / len(s)
        fig.add_shape(type="line", x0=0, x1=xi, y0=thr, y1=thr, line=dict(color=ui.AXIS, width=1, dash="dot"))
        fig.add_shape(type="line", x0=xi, x1=xi, y0=0, y1=thr, line=dict(color=ui.AXIS, width=1, dash="dot"))
        fig.add_annotation(x=xi, y=thr, text=f"{name}: {xi:.0%} SKU → {thr:.0%}", showarrow=False,
                           xanchor="left", yanchor="top", xshift=6, font=dict(color=ui.INK2, size=12))
    fig.update_layout(title=f"Кривая Парето: {label.lower()}", xaxis=dict(tickformat=".0%", title="доля SKU"),
                      yaxis=dict(tickformat=".0%", range=[0, 1.02]), showlegend=False)
    ui.chart(fig, 380, key="as_pareto")
