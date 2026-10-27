"""Запасы и пополнение: покрытие, дефицит, излишки, неликвид, точки заказа и рекомендуемый заказ."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

STATUS_COLOR = {"Нет в наличии": ui.STATUS["critical"], "Ниже точки заказа": ui.STATUS["serious"],
                "Излишек": ui.STATUS["warning"], "Неликвид": "#52514e", "Норма": ui.STATUS["good"]}


def render(f: Q.Filters):
    ui.page_header("Запасы и пополнение", "Склады — по SKU на конец периода; магазины — по категориям", f)
    c = st.columns([1, 1, 2])
    sl = c[0].selectbox("Уровень сервиса", [0.90, 0.95, 0.97, 0.98, 0.99], index=1, format_func=lambda v: f"{v:.0%}",
                        key="sk_sl", help="Вероятность не уйти в дефицит за срок поставки — определяет страховой запас")
    dead = c[1].selectbox("Неликвид, дней без продаж", [30, 60, 90], index=0, key="sk_dead")

    pos = Q.stock_position(f)
    if pos.empty:
        ui.empty_state()
        return
    d = Q.dims()
    rt = M.reorder_table(pos, d["sku"], f.end, service_level=sl, dead_days=dead)
    rt = rt[(rt["stock_qty"] > 0) | (rt["d_avg"] > 0)]
    dc_stock = Q.dc_stock_daily(f)
    sales = Q.sales_daily(f)
    k_sales = M.sales_kpis(sales, Q.traffic_daily(f)) if not sales.empty else {}

    last28 = dc_stock.tail(28)
    dc_cover = M.safe_div(dc_stock["stock_cost"].iloc[-1] if len(dc_stock) else np.nan, last28["cogs"].mean())
    n = len(rt)
    oos_n = int((rt["status"] == "Нет в наличии").sum())
    below_n = int((rt["status"] == "Ниже точки заказа").sum())
    dead_v = rt.loc[rt["status"] == "Неликвид", "stock_value"].sum()
    k = st.columns(3)
    ui.kpi(k[0], "Запас на складах", ui.rub(rt["stock_value"].sum()), None,
           help="Остаток на конец периода в себестоимости")
    ui.kpi(k[1], "Покрытие складов", f"{ui.num(dc_cover, 1)} дн.", None, good_when_up=None,
           help="Остаток на конец периода / средняя дневная отгрузка за 28 дней (в себестоимости)")
    ui.kpi(k[2], "Запас в магазинах", f"{ui.num(k_sales.get('cover_days', np.nan), 1)} дн.", None, good_when_up=None,
           help="Средний остаток в себестоимости / средняя дневная себестоимость продаж за период")
    k = st.columns(3)
    ui.kpi(k[0], "Нет в наличии на складе", ui.num(oos_n), f"{ui.pct(oos_n / max(n, 1))} позиций склад × SKU",
           good_when_up=None, arrow="off")
    ui.kpi(k[1], "Ниже точки заказа", ui.num(below_n), f"{ui.pct(below_n / max(n, 1))} позиций — нужен заказ",
           good_when_up=None, arrow="off")
    ui.kpi(k[2], "Неликвид и излишек", ui.rub(dead_v + rt["excess_value"].sum()),
           f"неликвид {ui.rub(dead_v)}", good_when_up=None, arrow="off",
           help="Неликвид — остаток без продаж дольше выбранного срока; излишек — запас сверх максимума "
                "(1,5 × (точка заказа + спрос за период пересмотра) + страховой запас)")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        _status_bar(rt)
    with right:
        _dc_trend(dc_stock)

    st.subheader("Магазины: запас по регионам и категориям")
    _store_heatmap(f)

    st.subheader("Что заказать и что разобрать")
    _action_table(rt, d)


def _status_bar(rt: pd.DataFrame):
    g = rt.groupby("status").agg(n=("sku_id", "size"), value=("stock_value", "sum")).reindex(M.STOCK_STATUS_ORDER).fillna(0)
    labels = [f"{M.STOCK_STATUS_ICON[s]} {s}" for s in g.index]
    fig = go.Figure(go.Bar(
        x=g["n"], y=labels, orientation="h", marker=dict(color=[STATUS_COLOR[s] for s in g.index], cornerradius=4),
        text=[f"{int(v)} · {ui.rub(val)}" for v, val in zip(g["n"], g["value"])], textposition="outside",
        hovertemplate="%{y}: %{x} позиций<extra></extra>"))
    fig.update_layout(title="Позиции склад × SKU по статусу", yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, g["n"].max() * 1.7]), showlegend=False)
    ui.chart(fig, 320, key="sk_status")


def _dc_trend(dc_stock: pd.DataFrame):
    if dc_stock.empty:
        return
    s = dc_stock.set_index("date")["stock_cost"].resample("W-SUN").mean()
    cov = (dc_stock.set_index("date")["stock_cost"] / dc_stock.set_index("date")["cogs"].rolling(28, min_periods=7).mean()
           ).resample("W-SUN").mean()
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s.values / 1e6, mode="lines", line=dict(color=ui.SERIES[0], width=2.5), name="Запас",
                    customdata=cov.reindex(s.index).values,
                    hovertemplate="Неделя по %{x|%d.%m}: %{y:,.0f} млн ₽ · покрытие %{customdata:.1f} дн.<extra></extra>",
                    fill="tozeroy", fillcolor="rgba(42,120,214,0.08)")
    fig.update_layout(title="Запас на складах (средний за неделю, в себестоимости)", yaxis=dict(tickformat=",.0f", title="млн ₽"),
                      showlegend=False)
    ui.chart(fig, 320, key="sk_trend", dates=True)


def _store_heatmap(f: Q.Filters):
    sc = Q.sales_by_store_cat(f)
    if sc.empty:
        ui.empty_state()
        return
    d = Q.dims()
    stores = d["store"].set_index("store_id")
    cats = d["category"].set_index("category_id")
    sc["region"] = sc["store_id"].map(stores["region"])
    sc["chain"] = sc["store_id"].map(stores["chain_name"])
    metric = st.segmented_control("Показатель", ["Покрытие к нормативу", "Дефицит (OOS)"], default="Покрытие к нормативу",
                                  key="sk_heat_metric", label_visibility="collapsed")
    g = sc.groupby(["region", "category_id"]).agg(stock=("stock_cost", "sum"), cost=("cost", "sum"),
                                                  oos_w=("oos_w", "sum"), rev=("revenue", "sum")).reset_index()
    g["cover"] = M.safe_div(g["stock"], g["cost"])
    g["norm"] = g["category_id"].map(cats["norm_days"])
    g["ratio"] = g["cover"] / g["norm"]
    g["oos"] = M.safe_div(g["oos_w"], g["rev"])
    val = "ratio" if metric != "Дефицит (OOS)" else "oos"
    piv = g.pivot(index="region", columns="category_id", values=val)
    piv.columns = [cats.loc[c, "category"] for c in piv.columns]
    order = g.groupby("region")["rev"].sum().sort_values(ascending=False).index
    piv = piv.reindex(order)
    if val == "ratio":
        # расходящаяся шкала: 1 = норматив
        zmax = max(float(np.nanmax(np.abs(piv.values - 1))), 0.2)
        cs = [[0, ui.DIV_NEG], [0.5, ui.DIV_MID], [1, ui.DIV_POS]]
        z, zmin, zmax_ = piv.values, 1 - zmax, 1 + zmax
        tmpl = "%{y} · %{x}<br>Запас к нормативу: %{z:.2f}<extra></extra>"
        txt = np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v:.2f}")(piv.values)
    else:
        cs = [[0, ui.SEQ[0]], [1, ui.SEQ[6]]]
        z, zmin, zmax_ = piv.values, 0, float(np.nanmax(piv.values))
        tmpl = "%{y} · %{x}<br>Дефицит: %{z:.1%}<extra></extra>"
        txt = np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v * 100:.1f}")(piv.values)
    fig = go.Figure(go.Heatmap(z=z, x=list(piv.columns), y=list(piv.index), colorscale=cs, zmin=zmin, zmax=zmax_,
                               text=txt, texttemplate="%{text}", textfont=dict(size=10), xgap=2, ygap=2,
                               hovertemplate=tmpl, colorbar=dict(thickness=10, len=0.6)))
    fig.update_layout(xaxis=dict(side="top", tickangle=-30), yaxis=dict(autorange="reversed"), margin=dict(t=90))
    ui.chart(fig, max(320, 26 * len(piv) + 120), key="sk_heat")
    if val == "ratio":
        st.caption("Покрытие в днях / норматив категории: 1 — норма, меньше 1 (красный) — риск дефицита, "
                   "больше 1 (синий) — избыточный запас.")
    else:
        st.caption("Доля позиций без остатка на полке, %; взвешено по выручке.")


def _action_table(rt: pd.DataFrame, d: dict):
    sel = st.pills("Статус", M.STOCK_STATUS_ORDER, selection_mode="multi",
                   default=["Нет в наличии", "Ниже точки заказа"], key="sk_status_sel",
                   format_func=lambda s: f"{M.STOCK_STATUS_ICON[s]} {s}")
    t = rt[rt["status"].isin(sel)] if sel else rt
    t = t.copy()
    t["order_value"] = t["order_qty"] * t["unit_cost"]
    order = {s: i for i, s in enumerate(M.STOCK_STATUS_ORDER)}
    t = t.sort_values(["status", "order_value", "stock_value"], key=lambda c: c.map(order) if c.name == "status" else -c)
    dc = d["dc"].set_index("dc_id")["name"]
    cats = d["category"].set_index("category_id")["category"]
    sup = d["supplier"].set_index("supplier_id")["supplier"]
    show = pd.DataFrame({
        "Статус": t["status"].map(lambda s: f"{M.STOCK_STATUS_ICON[s]} {s}"),
        "SKU": t["sku"], "Категория": t["category_id"].map(cats), "Склад": t["dc_id"].map(dc),
        "Поставщик": t["supplier_id"].map(sup),
        "Остаток, шт": t["stock_qty"], "В пути, шт": t["in_transit"], "Спрос в день, шт": t["d_avg"],
        "Покрытие, дн.": t["cover_days"], "Срок поставки, дн.": t["lt_avg"],
        "Страховой запас": t["safety_stock"], "Точка заказа": t["rop"],
        "Заказать, шт": t["order_qty"], "Сумма заказа, ₽": t["order_value"],
        "Остаток, ₽": t["stock_value"], "Дней без продаж": t["days_since_sale"],
    })
    st.caption(f"Позиций: {len(show)} · рекомендуемый заказ на {ui.rub(t['order_value'].sum())}")
    st.dataframe(show, hide_index=True, height=440, column_config={
        c: st.column_config.NumberColumn(format="%.0f") for c in
        ["Остаток, шт", "В пути, шт", "Страховой запас", "Точка заказа", "Заказать, шт", "Сумма заказа, ₽", "Остаток, ₽",
         "Дней без продаж"]
    } | {
        "Спрос в день, шт": st.column_config.NumberColumn(format="%.1f"),
        "Покрытие, дн.": st.column_config.NumberColumn(format="%.1f"),
        "Срок поставки, дн.": st.column_config.NumberColumn(format="%.1f"),
    })
    ui.download({"Пополнение": show}, "popolnenie.xlsx", key="dl_sk")
    with st.expander("Как считается"):
        st.markdown(
            "- **Спрос в день** и его разброс — по последним 28 дням до конца периода.\n"
            "- **Страховой запас** = z × √(срок поставки × σ²спроса + спрос² × σ²срока), z — по уровню сервиса.\n"
            "- **Точка заказа** = спрос × срок поставки + страховой запас.\n"
            "- **Заказать** = точка заказа + спрос × 7 дней (период пересмотра) − остаток − в пути, "
            "с округлением вверх до упаковки.\n"
            "- **Неликвид** — есть остаток, но не было отгрузок дольше выбранного срока.")
