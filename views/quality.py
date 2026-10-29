"""Качество и потери: брак при приёмке, прогноз потерь, риск по срокам годности на РЦ, декларации и сертификаты."""

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

PAGE = "quality"


def render(f: Q.Filters):
    ui.page_header("Качество и потери", "Брак при приёмке, сроки годности на РЦ, декларации и прогноз потерь", f, PAGE)
    ui.register_widgets(PAGE, ["Брак по категориям", "Брак по поставщикам", "Риск по срокам годности", "Декларации"])
    d = Q.dims()
    cat = Q.defects_by(f, "category_id")
    if cat.empty:
        ui.empty_state()
        return
    rate = cat["qty_defect"].sum() / (cat["qty_received"].sum() + cat["qty_defect"].sum())
    cost_def = cat["cost_defect"].sum()
    py_ok = common.py_available(f)
    rate_py = np.nan
    if py_ok:
        cp = Q.defects_by(f.py(), "category_id")
        if len(cp):
            rate_py = cp["qty_defect"].sum() / (cp["qty_received"].sum() + cp["qty_defect"].sum())
    sales = Q.sales_by(f, "category_id")
    wo = sales["writeoff"].sum()
    # прогноз потерь на 4 недели: темп последних 4 недель × сезонный коэффициент прошлого года
    e = f.end
    last4 = f.with_dates(e - timedelta(days=27), e)
    l4 = Q.defects_by(last4, "category_id")["cost_defect"].sum() + Q.sales_by(last4, "category_id")["writeoff"].sum()
    season = np.nan
    if f.with_dates(e - timedelta(days=27), e).py().start >= Q.data_bounds()[0]:
        nxt_py = f.with_dates(e + timedelta(days=1), e + timedelta(days=28)).py()
        cur_py = last4.py()
        a = Q.defects_by(nxt_py, "category_id")["cost_defect"].sum() + Q.sales_by(nxt_py, "category_id")["writeoff"].sum()
        b = Q.defects_by(cur_py, "category_id")["cost_defect"].sum() + Q.sales_by(cur_py, "category_id")["writeoff"].sum()
        season = a / b if b else np.nan
    fc_loss = l4 * (season if np.isfinite(season) else 1.0)

    k = st.columns(4)
    ui.kpi(k[0], "Брак при приёмке", ui.pct(rate, 2), ui.delta_pp(rate - rate_py, digits=2), good_when_up=False, height=150,
           help="Забраковано / (принято + забраковано), в штуках, по заказам поставщикам")
    ui.kpi(k[1], "Стоимость брака", ui.rub(cost_def), None, good_when_up=None, height=150, help="В себестоимости закупки")
    ui.kpi(k[2], "Списания в магазинах", ui.rub(wo), None, good_when_up=None, height=150)
    ui.kpi(k[3], "Прогноз потерь, 4 нед.", ui.rub(fc_loss), None, good_when_up=None, height=150,
           insight=(f"сезонность прошлого года: ×{season:.2f}".replace(".", ",") if np.isfinite(season) else
                    "по темпу последних 4 недель"),
           help="Брак + списания: темп последних 4 недель × отношение следующих 4 недель к текущим в прошлом году")

    c1, c2 = st.columns(2, gap="large")
    names = d["category"].set_index("category_id")["category"]
    with c1:
        cat["rate"] = cat["qty_defect"] / (cat["qty_received"] + cat["qty_defect"])
        cat["name"] = cat["key"].map(names)
        cc = cat.sort_values("rate")
        fig = go.Figure(go.Bar(y=cc["name"], x=cc["rate"] * 100, orientation="h",
                               marker=dict(color=ui.SERIES[1], cornerradius=3),
                               customdata=cc["cost_defect"] / 1e6,
                               hovertemplate="%{y}: %{x:.2f}%<br>%{customdata:,.1f} млн ₽<extra></extra>"))
        fig.update_layout(title="Брак при приёмке по категориям, %", xaxis=dict(title="%"))
        ui.chart(fig, 380, key="q_cat", export_name="Брак по категориям")
    with c2:
        sup = Q.defects_by(f, "supplier_id")
        sup = sup[sup["orders"] >= 20]
        sup["rate"] = sup["qty_defect"] / (sup["qty_received"] + sup["qty_defect"])
        sup["name"] = sup["key"].map(d["supplier"].set_index("supplier_id")["supplier"])
        top = sup.nlargest(10, "rate").sort_values("rate")
        fig = go.Figure(go.Bar(y=top["name"], x=top["rate"] * 100, orientation="h",
                               marker=dict(color=[ui.STATUS["critical"] if r > 0.02 else ui.STATUS["warning"]
                                                  if r > 0.01 else ui.SERIES[0] for r in top["rate"]], cornerradius=3),
                               hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        fig.update_layout(title="Поставщики с наибольшим браком, %", xaxis=dict(title="%"))
        ui.chart(fig, 380, key="q_sup", export_name="Брак по поставщикам")

    daily = Q.defects_by(f, "receipt_date")
    if len(daily) > 14:
        w = common.resample(daily.rename(columns={"key": "date"}), ["qty_defect", "qty_received"],
                            "Неделя" if f.days >= 56 else "День")
        r = w["qty_defect"] / (w["qty_defect"] + w["qty_received"])
        fig = go.Figure(go.Scatter(x=r.index, y=r * 100, mode="lines+markers", line=dict(color=ui.SERIES[1], width=2),
                                   hovertemplate="%{y:.2f}%<extra></extra>"))
        fig.update_layout(title="Доля брака при приёмке в динамике", yaxis=dict(title="%"))
        ui.chart(fig, 280, key="q_trend", dates=True)

    st.subheader("Риск истечения срока годности на РЦ")
    pos = Q.stock_position(f)
    if len(pos):
        sku = d["sku"].set_index("sku_id")
        p = pos.copy()
        p["shelf"] = p["sku_id"].map(sku["shelf_life_days"])
        p["unit_cost"] = p["sku_id"].map(sku["unit_cost"])
        p["cover"] = M.safe_div(p["stock_qty"], p["d_avg"])
        share = st.slider("Риск, если запаса хватит дольше чем на … срока годности", 0.3, 0.9, 0.5, 0.05,
                          key="q_share", format="%.2f")
        p["risk_days"] = p["cover"] - p["shelf"] * share
        risk = p[(p["shelf"] <= 30) & (p["stock_qty"] > 0) & ((p["risk_days"] > 0) | p["d_avg"].fillna(0).eq(0))].copy()
        risk["value_at_risk"] = np.where(risk["d_avg"].fillna(0) > 0,
                                         (risk["stock_qty"] - risk["d_avg"] * risk["shelf"] * share).clip(lower=0),
                                         risk["stock_qty"]) * risk["unit_cost"]
        risk = risk.sort_values("value_at_risk", ascending=False)
        ui.insight(f"Скоропортящихся позиций в зоне риска: <b>{len(risk)}</b>, товар на <b>"
                   f"{ui.rub(risk['value_at_risk'].sum())}</b> может не успеть продаться до окончания срока годности.",
                   "warn" if len(risk) else "good")
        dc = d["dc"].set_index("dc_id")["name"]
        show = pd.DataFrame({"SKU": risk["sku_id"].map(sku["sku"]), "РЦ": risk["dc_id"].map(dc),
                             "Срок годности, дн.": risk["shelf"], "Остаток, шт": risk["stock_qty"],
                             "Спрос в день, шт": risk["d_avg"], "Хватит на, дн.": risk["cover"],
                             "Под риском, ₽": risk["value_at_risk"].round(0).astype("int64")}).head(50)
        st.dataframe(show, hide_index=True, column_config={
            "Остаток, шт": st.column_config.NumberColumn(format="%.0f"),
            "Спрос в день, шт": st.column_config.NumberColumn(format="%.1f"),
            "Хватит на, дн.": st.column_config.NumberColumn(format="%.1f"),
            "Под риском, ₽": st.column_config.NumberColumn(format="localized")})
        ui.collect_table("Риск по срокам годности", show)

    st.subheader("Декларации и сертификаты соответствия")
    days = st.slider("Показывать истекающие в ближайшие, дней", 7, 120, 30, key="q_cert_days")
    c = Q.certificates(f)
    c = c[c["lifecycle"] != "delisted"].copy()
    c["days"] = (c["valid_to"] - pd.Timestamp(f.end)).dt.days
    exp = c[c["days"] < 0]
    soon = c[(c["days"] >= 0) & (c["days"] <= days)]
    k = st.columns(3)
    ui.kpi(k[0], "Истекли", ui.num(len(exp)), None, good_when_up=None, help="Товар в матрице, документ просрочен")
    ui.kpi(k[1], f"Истекают за {days} дн.", ui.num(len(soon)), None, good_when_up=None)
    ui.kpi(k[2], "Документов в контроле", ui.num(len(c)), None, good_when_up=None)
    lst = pd.concat([exp, soon]).sort_values("days")
    if len(lst):
        show = pd.DataFrame({"": np.where(lst["days"] < 0, "🔴", np.where(lst["days"] <= 7, "🟠", "🟡")),
                             "SKU": lst["sku"], "Поставщик": lst["supplier_id"].map(
                                 d["supplier"].set_index("supplier_id")["supplier"]),
                             "Документ": lst["doc_type"], "Номер (условный)": lst["doc_number"],
                             "Действует до": lst["valid_to"].dt.date, "Дней": lst["days"]})
        st.dataframe(show, hide_index=True, column_config={"": st.column_config.TextColumn(width=36)})
        ui.collect_table("Декларации", show)
        st.caption("Эти же события формируют уведомления (правило «Декларация или сертификат истекает»).")
    ui.export_panel(PAGE)
