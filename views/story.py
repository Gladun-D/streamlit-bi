"""Data story: общая картина → диагностика → прогноз и сценарии → выводы.

Тексты собираются из расчётов автоматически: при смене фильтров меняются и выводы.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import config as C
from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common


def render(f: Q.Filters):
    ui.page_header("Data story", "От общей картины к причинам, прогнозу и решениям", f, "story")
    py_ok = common.py_available(f)
    cur = common.load_period(f)
    if common.no_data(cur.sales):
        return
    prev = common.load_period(f.py()) if py_ok else None
    common.notes(f, py_ok)
    stores = common.store_table(f)
    findings: list[tuple[str, str]] = []

    # 1. Общая картина
    st.markdown('<div class="bi-step">Шаг 1 · Общая картина</div>', unsafe_allow_html=True)
    st.subheader("Что произошло")
    k = cur.k
    comp = stores[stores["comparable"] & stores["revenue_py"].notna()]
    lfl = M.change(comp["revenue"].sum(), comp["revenue_py"].sum()) if py_ok and len(comp) else np.nan
    plan_pct = M.safe_div(stores.loc[stores["plan"].notna(), "store_revenue"].sum(), stores["plan"].sum())
    if py_ok and prev is not None:
        yoy = M.change(k["revenue"], prev.k["revenue"])
        new_stores = stores[~stores["comparable"]]
        new_contrib = new_stores["revenue"].sum() - new_stores["revenue_py"].fillna(0).sum()
        text = (f"Выручка за период — <b>{ui.rub(k['revenue'])}</b>, {ui.delta_pct(yoy)}. "
                f"Сопоставимые магазины дали <b>{ui.pct(lfl)}</b> (LFL), остальное — новые магазины "
                f"(+{ui.rub(new_contrib)}). План выполнен на <b>{ui.pct(plan_pct)}</b>.")
        ui.insight(text, "good" if lfl >= 0 and plan_pct >= 0.98 else "warn")
    else:
        ui.insight(f"Выручка за период — <b>{ui.rub(k['revenue'])}</b>, план выполнен на <b>{ui.pct(plan_pct)}</b>. "
                   "Для сравнения с прошлым годом выберите период позже 31.12.2025.")

    if py_ok and prev is not None:
        dec = M.revenue_decomposition(cur.k["checks"], cur.k["avg_check"], prev.k["checks"], prev.k["avg_check"])
        c1, c2 = st.columns([1, 1.3], gap="large")
        with c1:
            drv = "средний чек" if abs(dec["check"]) > abs(dec["traffic"]) else "трафик"
            st.markdown(
                f"Изменение выручки магазинов раскладывается на трафик ({ui.rub(dec['traffic'])}) "
                f"и средний чек ({ui.rub(dec['check'])}). Основной драйвер — **{drv}**. "
                f"Средний чек {ui.delta_pct(M.change(cur.k['avg_check'], prev.k['avg_check']))}: "
                f"при заложенной инфляции цен около {ui.pct(C.PRICE_INFLATION, 0)} это означает, что "
                + ("корзина в штуках сжимается." if M.change(cur.k['avg_check'], prev.k['avg_check']) < C.PRICE_INFLATION - 0.01
                   else "корзина в натуральном выражении стабильна."))
        with c2:
            _contrib_chart(stores, f)

    # 2. Диагностика
    st.markdown('<div class="bi-step">Шаг 2 · Диагностика</div>', unsafe_allow_html=True)
    st.subheader("Почему")
    if py_ok:
        reg = common.group_table(stores, ["chain_name", "region"], f)
        reg = reg[reg["stores"] >= 5]
        if len(reg):
            w = reg.nsmallest(1, "lfl").iloc[0]
            b = reg.nlargest(1, "lfl").iloc[0]
            mean_lfl = lfl
            if w["lfl"] < mean_lfl - 0.03:
                findings.append(("warn", f"<b>{w['region']}</b> ({w['chain_name']}): LFL {ui.pct(w['lfl'])} против "
                                         f"{ui.pct(mean_lfl)} в среднем — проверить конкурентное окружение и трафик."))
            ui.insight(f"Слабее всего регион <b>{w['region']}</b> ({w['chain_name']}): LFL {ui.pct(w['lfl'])}, "
                       f"сильнее всего — <b>{b['region']}</b>: {ui.pct(b['lfl'])}.", "warn" if w["lfl"] < 0 else "info")
            _region_traffic(stores, w["region"], w["chain_name"])

    dc_otif = Q.orders_by(f, "dc_id")
    trips = Q.deliveries_by(f, "dc_id")
    dcs = Q.dims()["dc"].set_index("dc_id")
    if len(dc_otif):
        dc_otif["name"] = dc_otif["dc_id"].map(dcs["name"])
        avg = np.average(dc_otif["otif"], weights=dc_otif["orders"])
        w = dc_otif.nsmallest(1, "otif").iloc[0]
        st_dc = stores.groupby("dc_name").apply(lambda g: M.safe_div(g["oos_w"].sum(), g["revenue"].sum()),
                                                include_groups=False)
        oos_w = st_dc.get(w["name"], np.nan)
        msg = (f"Слабое звено поставок — <b>{w['name']}</b>: OTIF поставщиков {ui.pct(w['otif'])} при среднем "
               f"{ui.pct(avg)}; дефицит на полке в его магазинах {ui.pct(oos_w)} против {ui.pct(k['oos'])} по выборке.")
        if len(trips):
            tw = trips.set_index("dc_id").loc[w["dc_id"]] if w["dc_id"] in trips["dc_id"].values else None
            if tw is not None:
                msg += f" Рейсов в окно ±30 мин: {ui.pct(tw['on_time'])}."
        kind = "warn" if w["otif"] < avg - 0.02 else "info"
        ui.insight(msg, kind)
        if kind == "warn":
            findings.append(("warn", f"<b>{w['name']}</b>: OTIF {ui.pct(w['otif'])} — разобрать с поставщиками и "
                                     "транспортом; поднять страховой запас на время исправления."))

    cat = Q.sales_by(f, "category_id")
    cats = Q.dims()["category"].set_index("category_id")
    cat["wo"] = M.safe_div(cat["writeoff"], cat["cost"])
    cat["wo_rub"] = cat["writeoff"]
    top_wo = cat.nlargest(2, "wo_rub")
    names = ", ".join(f"<b>{cats.loc[c, 'category']}</b> ({ui.rub(v)}, {ui.pct(r, 1)})"
                      for c, v, r in zip(top_wo["category_id"], top_wo["wo_rub"], top_wo["wo"]))
    ui.insight(f"Больше всего списаний в {names}. Это категории с коротким сроком годности — "
               "рычаги: точность заказа, частота поставок, уценка до окончания срока.", "info")
    findings.append(("info", f"Списания: фокус на {names} — пересмотреть нормативы запаса и частоту поставок."))

    # 3. Прогноз и сценарии
    st.markdown('<div class="bi-step">Шаг 3 · Прогноз и сценарии</div>', unsafe_allow_html=True)
    st.subheader("Что будет")
    _forecast(f)
    _scenarios(f, cur, stores)

    # 4. Выводы
    st.markdown('<div class="bi-step">Шаг 4 · Выводы</div>', unsafe_allow_html=True)
    st.subheader("Что делать")
    pos = Q.stock_position(f)
    if len(pos):
        rt = M.reorder_table(pos, Q.dims()["sku"], f.end)
        dead = rt.loc[rt["status"] == "Неликвид", "stock_value"].sum()
        below = int(rt["status"].isin(["Нет в наличии", "Ниже точки заказа"]).sum())
        findings.append(("warn" if below else "info",
                         f"Склады: {below} позиций склад × SKU без остатка или ниже точки заказа — "
                         f"рекомендуемый заказ на странице «Запасы и пополнение»."))
        if dead > 0:
            findings.append(("info", f"Неликвид на складах — {ui.rub(dead)}: распродать или вернуть поставщикам, "
                                     "вывести позиции из матрицы."))
    if np.isfinite(plan_pct) and plan_pct < 0.98:
        findings.append(("warn", f"План выполнен на {ui.pct(plan_pct)} — проверить магазины с наибольшим отставанием "
                                 "(страница «Продажи», сортировка по выполнению плана)."))
    for kind, text in findings:
        ui.insight(text, kind)
    st.caption("Выводы сформированы правилами по расчётным показателям. Данные синтетические: закономерности "
               "заложены в генератор намеренно (см. «О данных и методика»).")
    ui.export_panel("story")


def _contrib_chart(stores: pd.DataFrame, f: Q.Filters):
    g = common.group_table(stores, ["chain_name", "dc_name"], f)
    g["delta"] = g["revenue"] - g["revenue_py"]
    g = g.dropna(subset=["delta"]).sort_values("delta")
    colors = [ui.SERIES[0] if v >= 0 else ui.SERIES[7] for v in g["delta"]]
    fig = go.Figure(go.Bar(x=g["delta"] / 1e6, y=g["dc_name"], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[ui.rub(v) for v in g["delta"]], textposition="outside",
                           hovertemplate="%{y}: %{x:,.0f} млн ₽<extra></extra>"))
    lim = abs(g["delta"]).max() / 1e6 * 1.45
    fig.update_layout(title="Вклад РЦ в изменение выручки г/г",
                      xaxis=dict(range=[min(-lim * 0.1, g["delta"].min() / 1e6 * 1.45), lim], tickformat=",.0f",
                                 title="млн ₽"), showlegend=False)
    ui.chart(fig, 300, key="st_contrib")


def _region_traffic(stores: pd.DataFrame, region: str, chain_name: str):
    f: Q.Filters = st.session_state["F"]
    ids = stores[(stores["region"] == region) & (stores["chain_name"] == chain_name) & stores["comparable"]]["store_id"]
    if ids.empty:
        return
    rf = Q.Filters(f.start, f.end, store_ids=tuple(int(i) for i in ids), dc_ids=None)
    start = max(Q.data_bounds()[0] + timedelta(days=364), f.end - timedelta(days=364))
    rf = rf.with_dates(start, f.end)
    cur = M.full_weeks(Q.traffic_daily(rf), "checks")
    py = M.full_weeks(Q.traffic_daily(rf.py()), "checks")
    py.index = py.index + pd.Timedelta(days=364)
    idx = (cur / py.reindex(cur.index) - 1).dropna()
    colors = [ui.SERIES[0] if v >= 0 else ui.SERIES[7] for v in idx]
    fig = go.Figure(go.Bar(x=idx.index, y=idx.values, marker=dict(color=colors, cornerradius=3),
                           hovertemplate="Неделя с %{x|%d.%m.%Y}: %{y:+.1%} чеков г/г<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=ui.AXIS, width=1))
    fig.update_layout(title=f"{region}: чеки сопоставимых магазинов к прошлому году по неделям",
                      yaxis=dict(tickformat="+.0%"), showlegend=False)
    ui.chart(fig, 280, key="st_region_traffic", dates=True)


def _forecast(f: Q.Filters):
    hist_f = f.with_dates(Q.data_bounds()[0], f.end)
    w = M.full_weeks(Q.sales_daily(hist_f), "revenue")
    fc = M.seasonal_forecast(w, horizon=8)
    if fc.empty:
        st.caption("Для прогноза нужно не меньше 62 недель истории.")
        return
    tail = w.tail(26) / 1e6
    fc_m = fc.assign(forecast=fc["forecast"] / 1e6, low=fc["low"] / 1e6, high=fc["high"] / 1e6)
    fig = go.Figure()
    fig.add_scatter(x=list(fc_m["week"]) + list(fc_m["week"][::-1]), y=list(fc_m["high"]) + list(fc_m["low"][::-1]),
                    fill="toself", fillcolor="rgba(42,120,214,0.12)", mode="lines", line=dict(width=0),
                    name="Интервал 80%", hoverinfo="skip")
    # прогноз начинается от последней фактической точки — линия без разрыва
    fx = [tail.index[-1]] + list(fc_m["week"])
    fy = [tail.iloc[-1]] + list(fc_m["forecast"])
    fig.add_scatter(x=tail.index, y=tail.values, name="Факт", mode="lines", line=dict(color=ui.SERIES[0], width=2.5),
                    hovertemplate="%{y:,.0f} млн ₽<extra>Факт</extra>")
    fig.add_scatter(x=fx, y=fy, name="Прогноз", mode="lines+markers",
                    line=dict(color=ui.SERIES[0], width=2, dash="dash"), marker=dict(size=6),
                    hovertemplate="%{y:,.0f} млн ₽<extra>Прогноз</extra>")
    fig.update_layout(title="Выручка по неделям: прогноз на 8 недель", yaxis=dict(tickformat=",.0f", title="млн ₽"),
                      hovermode="x unified")
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        ui.chart(fig, 340, key="st_fc", dates=True)
    with c2:
        total = fc["forecast"].sum()
        same_ly = w.reindex(fc["week"] - pd.Timedelta(days=364)).sum()
        st.metric("Прогноз выручки на 8 недель", ui.rub(total),
                  ui.delta_pct(M.change(total, same_ly)) if same_ly > 0 else None, border=True)
        st.caption(f"Метод: сезонный наивный — та же неделя прошлого года × темп последних 8 недель к прошлому году. "
                   f"Ошибка метода на последних 26 неделях (MAPE): {ui.pct(fc.attrs.get('mape', np.nan))}. "
                   "Это ориентир, а не план: акции и открытия магазинов он не учитывает.")


def _scenarios(f: Q.Filters, cur: common.Period, stores: pd.DataFrame):
    st.markdown("**Сценарии «что если»**")
    k = cur.k
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**Снизить дефицит на полке**")
            red = st.slider("Снижение OOS, п.п.", 0.0, 3.0, 1.0, 0.25, key="sc_oos")
            subst = st.slider("Доля спроса, уходящего к конкуренту при дефиците", 0.2, 0.8, 0.5, 0.05, key="sc_subst",
                              help="Остальное покупатель заменяет другим товаром в этом же магазине (ДОПУЩЕНИЕ)")
            extra = k["revenue"] * red / 100 * subst
            st.metric("Доп. выручка за период", ui.rub(extra), f"+{ui.rub(extra * k['margin'])} валовой прибыли",
                      delta_color="normal", border=False)
    with c2:
        with st.container(border=True):
            st.markdown("**Изменить срок поставки**")
            dlt = st.slider("Изменение срока, дней", -3, 3, -1, 1, key="sc_lt")
            pos = Q.stock_position(f)
            if len(pos):
                rt0 = M.reorder_table(pos, Q.dims()["sku"], f.end)
                pos2 = pos.copy()
                pos2["lt_avg"] = (pos2["lt_avg"] + dlt).clip(lower=1)
                rt1 = M.reorder_table(pos2, Q.dims()["sku"], f.end)
                # целевой средний запас ≈ страховой + половина цикла пополнения; меняется на d·ΔLT·0 + ΔSS
                d_ss = ((rt1["safety_stock"] - rt0["safety_stock"]) * rt0["unit_cost"]).sum()
                d_pipe = ((rt1["lt_avg"] - rt0["lt_avg"]) * rt0["d_avg"] * rt0["unit_cost"]).sum()
                st.metric("Страховой запас, изменение", ui.rub(d_ss))
                st.caption(f"Плюс товары в пути: {ui.rub(d_pipe)}. Оборотный капитал меняется на {ui.rub(d_ss + d_pipe)}.")
    with c3:
        with st.container(border=True):
            st.markdown("**Изменить долю промо**")
            dp = st.slider("Изменение доли промо, п.п.", -10, 10, 5, 1, key="sc_promo")
            disc = st.slider("Средняя скидка", 0.05, 0.3, 0.15, 0.01, key="sc_disc", format="%.2f")
            incr = st.slider("Доля прироста, которая не каннибализирует", 0.1, 0.6, 0.3, 0.05, key="sc_incr",
                             help="Часть промо-продаж, которая действительно добавляется к выручке (ДОПУЩЕНИЕ)")
            moved = k["revenue"] * dp / 100
            d_rev = moved * incr
            d_gp = d_rev * k["margin"] - moved * disc
            st.metric("Выручка", ui.rub(d_rev), ui.signed(("+" if d_gp >= 0 else "") + f"{ui.rub(d_gp)} валовой прибыли"),
                      delta_color="normal")
    st.caption("Сценарии — упрощённые модели с явными допущениями (ползунки). Они показывают порядок эффекта, "
               "а не заменяют расчёт по методологии компании.")
