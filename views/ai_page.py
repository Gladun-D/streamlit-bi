"""AI-помощник: аномалии и их факторы, магазины-выбросы, негативные тренды, топ-5 факторов, сценарии «что если»."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import ai
from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

PAGE = "ai"


def render(f: Q.Filters):
    ui.page_header("AI-помощник", "Поиск закономерностей, аномалий и причин отклонений", f, PAGE)
    ui.register_widgets(PAGE, ["Аномалии", "Магазины-выбросы", "Негативные тренды", "Топ-5 факторов", "Что если"])
    st.caption(":material/info: Методы — статистические (робастные отклонения, регрессия трендов, разложение "
               "отклонений). Генеративная модель в прототипе не подключена; в целевой системе к этим расчётам "
               "добавляется текстовый помощник.")
    common.notes(f, common.py_available(f))
    t1, t2, t3, t4, t5 = st.tabs(["Аномалии", "Магазины-выбросы", "Негативные тренды", "Топ-5 факторов",
                                  "Что если"])
    with t1:
        _anomalies(f)
    with t2:
        _outliers(f)
    with t3:
        _trends(f)
    with t4:
        _factors(f)
    with t5:
        _whatif(f)
    ui.export_panel(PAGE)


@st.cache_data(ttl=Q.TTL, show_spinner=False)
def _daily(f: Q.Filters) -> pd.DataFrame:
    s = Q.sales_daily(f)
    t = Q.traffic_daily(f)[["date", "checks", "revenue"]].rename(columns={"revenue": "store_revenue"})
    return s.merge(t, on="date")


def _anomalies(f: Q.Filters):
    d0 = Q.data_bounds()[0]
    ff = f.with_dates(max(d0, min(f.start, f.end - timedelta(days=364))), f.end)
    warm = f.with_dates(max(d0, ff.start - timedelta(days=150)), f.end)   # разгон базы и z-оценки
    c = st.columns([1, 1.4, 1.6])
    z = c[0].slider("Чувствительность (порог z)", 2.0, 6.0, 3.0, 0.5, key="ai_z",
                    help="Чем меньше порог, тем больше дней считается аномальными")
    win = c[1].segmented_control("Окно поиска", ["12 месяцев", "Период фильтра"], default="12 месяцев",
                                 key="ai_win", help="Аномалии редки: по умолчанию ищем за последний год, "
                                                    "чтобы попали праздники и сезонные события")
    show_from = pd.Timestamp(ff.start if win != "Период фильтра" else f.start)
    series, an = ai.anomalies(_daily(warm), z_thr=z)
    series = series[series["date"] >= show_from]
    an = an[an["date"] >= show_from] if len(an) else an
    fig = go.Figure()
    fig.add_scatter(x=series["date"], y=series["base"] / 1e6, name="Обычный уровень (тот же день недели)",
                    mode="lines", line=dict(color=ui.NEUTRAL, width=2, dash="dot"))
    fig.add_scatter(x=series["date"], y=series["revenue"] / 1e6, name="РТО", mode="lines",
                    line=dict(color=ui.SERIES[0], width=2), hovertemplate="%{y:,.1f} млн ₽<extra>РТО</extra>")
    if len(an):
        up = an[an["deviation"] > 0]
        dn = an[an["deviation"] <= 0]
        fig.add_scatter(x=up["date"], y=up["revenue"] / 1e6, mode="markers", name="Всплеск",
                        marker=dict(size=11, color=ui.STATUS["good"], line=dict(color="white", width=2)),
                        customdata=np.c_[up["deviation"] * 100, up["main_factor"]],
                        hovertemplate="%{x|%d.%m.%Y}: %{customdata[0]:+.1f}%<br>%{customdata[1]}<extra>Всплеск</extra>")
        fig.add_scatter(x=dn["date"], y=dn["revenue"] / 1e6, mode="markers", name="Провал",
                        marker=dict(size=11, color=ui.STATUS["critical"], line=dict(color="white", width=2)),
                        customdata=np.c_[dn["deviation"] * 100, dn["main_factor"]],
                        hovertemplate="%{x|%d.%m.%Y}: %{customdata[0]:+.1f}%<br>%{customdata[1]}<extra>Провал</extra>")
    fig.update_layout(title="РТО по дням: аномалии отмечены точками", yaxis=dict(title="млн ₽"), hovermode="x unified")
    ui.chart(fig, 380, key="ai_an_chart", dates=True, export_name="Аномалии")
    if an.empty:
        ui.insight("Аномалий в выбранном окне не найдено при текущей чувствительности: попробуйте окно «12 месяцев» "
                   "или порог пониже.", "good")
    else:
        show = pd.DataFrame({
            "Дата": an["date"].dt.date, "Отклонение": an["deviation"], "z": an["z"], "Главный фактор": an["main_factor"],
            "Трафик": an["traffic"], "Средний чек": an["check"], "Доля промо, п.п.": an["promo_pp"] * 100,
            "OSA, п.п.": an["osa_pp"] * 100}).sort_values("Дата", ascending=False)
        st.dataframe(show, hide_index=True, column_config={
            "Дата": st.column_config.DateColumn(format="DD.MM.YYYY"),
            "Отклонение": st.column_config.NumberColumn(format="percent"),
            "z": st.column_config.NumberColumn(format="%.1f"),
            "Трафик": st.column_config.NumberColumn(format="percent", help="Чеки к обычному уровню"),
            "Средний чек": st.column_config.NumberColumn(format="percent"),
            "Доля промо, п.п.": st.column_config.NumberColumn(format="%+.1f"),
            "OSA, п.п.": st.column_config.NumberColumn(format="%+.2f")})
        ui.collect_table("Аномалии", show)
        last = an.sort_values("date").iloc[-1]
        ui.insight(f"Последняя аномалия — <b>{last['date']:%d.%m.%Y}</b>: РТО {ui.delta_pct(last['deviation'], '')} к обычному "
                   f"уровню. Главный фактор — <b>{last['main_factor']}</b> (трафик {ui.delta_pct(last['traffic'], '')}, "
                   f"чек {ui.delta_pct(last['check'], '')}).", "warn" if last["deviation"] < 0 else "info")
    links = ai.factor_links(series)
    if len(links):
        st.markdown("**Какие факторы связаны с отклонениями РТО** (корреляция по всем дням)")
        st.dataframe(links, hide_index=True, column_config={
            "Корреляция с отклонением РТО": st.column_config.ProgressColumn(format="%.2f", min_value=-1, max_value=1)})


def _outliers(f: Q.Filters):
    if not common.py_available(f):
        st.caption("Нужен период с базой прошлого года.")
        return
    stores = common.store_table(f)
    z = st.slider("Порог отклонения от магазинов своего дивизиона (z)", 1.5, 4.0, 2.5, 0.5, key="ai_out_z")
    out = ai.store_outliers(stores, z)
    low = out[out["z"] < 0] if len(out) else out
    if low.empty:
        ui.insight("Магазинов с аномально низким РТО относительно своего дивизиона не найдено.", "good")
    else:
        ui.insight(f"Найдено <b>{len(low)}</b> магазинов с аномально низким LFL относительно магазинов своего "
                   f"дивизиона. Сильнее всего отстаёт <b>{low.iloc[0]['store_name']}</b>: LFL {low.iloc[0]['lfl']:+.1%} "
                   f"при медиане дивизиона {low.iloc[0]['peer_median']:+.1%}.", "warn")
    if len(out):
        show = pd.DataFrame({
            "Магазин": out["store_name"], "Дивизион": out["division"], "УГМ": out["ugm"], "LFL": out["lfl"],
            "Медиана дивизиона": out["peer_median"], "z": out["z"], "Направление": out["direction"],
            "Чеки г/г": out["checks"] / out["checks_py"] - 1, "OSA": out["osa"]})
        st.dataframe(ui.styled(show, {"LFL": "lfl", "OSA": "osa"}), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ["LFL", "Медиана дивизиона", "Чеки г/г", "OSA"]}
            | {"z": st.column_config.NumberColumn(format="%.1f")})
        ui.collect_table("Магазины-выбросы", show)
        st.caption("Если у магазина падают и чеки, и РТО при нормальной OSA — вероятна внешняя причина (конкурент, "
                   "ремонт дороги); если падает OSA — проблема пополнения.")


def _trends(f: Q.Filters):
    if not common.py_available(f.with_dates(f.end - timedelta(days=90), f.end)):
        st.caption("Нужна история прошлого года.")
        return
    c = st.columns([1.2, 1, 2])
    ent = c[0].segmented_control("Объект", ["Магазин", "УГМ", "Дивизион", "Категория"], default="УГМ",
                                 key="ai_tr_ent") or "УГМ"
    weeks = c[1].slider("Недель", 8, 26, 13, key="ai_tr_w")
    end = f.end
    start = end - timedelta(days=7 * weeks - 1)
    ff = f.with_dates(start, end)
    if ent == "Категория":
        cur, py = Q.cat_daily(ff), Q.cat_daily(ff.py())
        names = Q.dims()["category"].set_index("category_id")["category"]
        cur["key"], py["key"] = cur["category_id"].map(names), py["category_id"].map(names)
    else:
        cur, py = Q.store_daily(ff), Q.store_daily(ff.py())
        col = {"Магазин": "store_name", "УГМ": "ugm", "Дивизион": "division"}[ent]
        stv = Q.dims()["store"].set_index("store_id")
        comp = set(M.comparable_stores(Q.dims()["store"], start))
        cur, py = cur[cur["store_id"].isin(comp)], py[py["store_id"].isin(comp)]
        cur["key"], py["key"] = cur["store_id"].map(stv[col]), py["store_id"].map(stv[col])
    py["date"] = py["date"] + pd.Timedelta(days=364)
    for d in (cur, py):
        d["week"] = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    w = cur.groupby(["key", "week"])["revenue"].sum().to_frame().join(
        py.groupby(["key", "week"])["revenue"].sum().rename("revenue_py")).reset_index()
    res = ai.negative_trends(w, "key", weeks)
    if res.empty:
        ui.insight(f"Устойчивых негативных трендов по объекту «{ent}» за {weeks} недель не найдено.", "good")
        return
    ui.insight(f"Найдено <b>{len(res)}</b> объектов с устойчиво снижающимся РТО к прошлому году. "
               f"Сильнее всего — <b>{res.iloc[0]['key']}</b>: индекс к ПГ снизился с {res.iloc[0]['index_start']:.2f} "
               f"до {res.iloc[0]['index_end']:.2f}.", "warn")
    show = pd.DataFrame({ent: res["key"], "Индекс к ПГ по неделям": res["spark"],
                         "Индекс в начале": res["index_start"], "Индекс в конце": res["index_end"],
                         "Наклон, п.п. в неделю": res["slope_pp_week"] * 100})
    st.dataframe(show, hide_index=True, column_config={
        "Индекс к ПГ по неделям": st.column_config.LineChartColumn(width="medium"),
        "Индекс в начале": st.column_config.NumberColumn(format="%.3f"),
        "Индекс в конце": st.column_config.NumberColumn(format="%.3f"),
        "Наклон, п.п. в неделю": st.column_config.NumberColumn(format="%.2f")})
    ui.collect_table("Негативные тренды", show.drop(columns=["Индекс к ПГ по неделям"]))
    st.caption("Индекс = РТО недели / РТО той же недели прошлого года (только сопоставимые магазины). Тренд "
               "негативный, если наклон регрессии отрицательный и последние 4 недели ниже первых 4.")


def _factors(f: Q.Filters):
    if not common.py_available(f):
        st.caption("Нужен период с базой прошлого года.")
        return
    stores = common.store_table(f)
    cat_c, cat_p = Q.sales_by(f, "category_id"), Q.sales_by(f.py(), "category_id")
    names = Q.dims()["category"].set_index("category_id")["category"]
    fac = ai.top_factors(stores, cat_c, cat_p, names)
    total = stores["revenue"].sum() - stores["revenue_py"].fillna(0).sum()
    drivers = fac[fac["Группа"] == "Драйверы"]
    detail = fac[fac["Группа"] != "Драйверы"]
    neg = detail.sort_values("Эффект, ₽").head(5)
    pos = detail.sort_values("Эффект, ₽", ascending=False).head(5)
    sign = "выросли" if total >= 0 else "снизились"
    reasons = ", ".join(f"{r['Фактор']} ({ui.rub(r['Эффект, ₽'])})" for _, r in neg.iterrows() if r["Эффект, ₽"] < 0)
    ui.insight(f"РТО {sign} на <b>{ui.rub(total)}</b> к прошлому году. "
               + (f"Сдерживающие факторы: {reasons}." if reasons else "Отрицательных факторов не найдено."),
               "warn" if total < 0 else "info")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dd = drivers.copy()
        fig = go.Figure(go.Waterfall(
            x=list(dd["Фактор"]) + ["Итого изменение"], measure=["relative"] * len(dd) + ["total"],
            y=list(dd["Эффект, ₽"] / 1e6) + [0], text=[ui.rub(v) for v in dd["Эффект, ₽"]] + [ui.rub(total)],
            textposition="outside", connector=dict(line=dict(color=ui.AXIS, width=1)),
            increasing=dict(marker=dict(color=ui.SERIES[0])), decreasing=dict(marker=dict(color=ui.SERIES[7])),
            totals=dict(marker=dict(color=ui.NEUTRAL))))
        fig.update_layout(title="Драйверы изменения РТО г/г", yaxis=dict(title="млн ₽"), showlegend=False)
        ui.chart(fig, 380, key="ai_fac_wf", export_name="Драйверы")
        st.caption("OSA — оценка: изменение доступности × РТО × доля спроса, не перешедшего на замену (допущение).")
    with c2:
        top = pd.concat([neg, pos]).drop_duplicates("Фактор").sort_values("Эффект, ₽")
        colors = [ui.SERIES[7] if v < 0 else ui.SERIES[0] for v in top["Эффект, ₽"]]
        fig = go.Figure(go.Bar(y=top["Фактор"], x=top["Эффект, ₽"] / 1e6, orientation="h",
                               marker=dict(color=colors, cornerradius=3), customdata=top["Комментарий"],
                               hovertemplate="%{y}<br>%{x:,.1f} млн ₽<br>%{customdata}<extra></extra>"))
        fig.update_layout(title="Топ-5 факторов снижения и роста (регионы и категории)", xaxis=dict(title="млн ₽"))
        ui.chart(fig, 380, key="ai_fac_top", export_name="Топ-5 факторов")
    show = fac.sort_values("Эффект, ₽").rename(columns={"Эффект, ₽": "Эффект, млн ₽"})
    show["Эффект, млн ₽"] = show["Эффект, млн ₽"] / 1e6
    with st.expander("Все факторы"):
        st.dataframe(show, hide_index=True, column_config={"Эффект, млн ₽": st.column_config.NumberColumn(format="%+.1f")})
    ui.collect_table("Факторы", show)


def _whatif(f: Q.Filters):
    st.markdown("**Изменение цены категории**")
    cats = Q.dims()["category"].set_index("category_id")
    c = st.columns([1.5, 1, 1, 1])
    cat = c[0].selectbox("Категория", cats.index.tolist(), format_func=lambda i: cats.loc[i, "category"], key="wi_cat",
                         index=11 if 12 in cats.index else 0)
    change = c[1].slider("Изменение цены, %", -30, 30, -15, 1, key="wi_price") / 100
    el = c[2].number_input("Эластичность спроса", -4.0, 0.0, float(ai.ELASTICITY.get(int(cat), -1.0)), 0.1,
                           key=f"wi_el_{cat}", help="На сколько % меняется спрос при росте цены на 1% (допущение)")
    weeks = c[3].slider("Горизонт, недель", 1, 13, 4, key="wi_w")
    base = Q.sales_by(replace(f, cats=(int(cat),) if not f.cats or int(cat) in f.cats else (-1,)).with_dates(
        f.end - timedelta(days=27), f.end), "category_id")
    if base.empty:
        ui.empty_state("Категория недоступна при текущих фильтрах или правах")
        return
    rev4, cost4 = float(base["revenue"].sum()), float(base["cost"].sum())
    k = weeks / 4
    res = ai.price_scenario(rev4 * k, cost4 * k, change, el)
    cols = st.columns(4)
    ui.kpi(cols[0], "Продажи в штуках", ui.delta_pct(res["volume"], "") or "—", None, good_when_up=None)
    ui.kpi(cols[1], f"РТО за {weeks} нед.", ui.rub(res["revenue"]), ui.signed(ui.delta_abs(res["d_revenue"] / 1e6, " млн ₽", "")))
    ui.kpi(cols[2], "Валовая прибыль", ui.rub(res["gross"]), ui.signed(ui.delta_abs(res["d_gross"] / 1e6, " млн ₽", "")))
    ui.kpi(cols[3], "Маржа", ui.pct(res["margin_new"]), ui.delta_pp(res["margin_new"] - res["margin_old"], ""))
    verdict = "увеличит" if res["d_gross"] > 0 else "уменьшит"
    ui.insight(f"Изменение цены «{cats.loc[cat, 'category']}» на {change:+.0%} при эластичности {el:.1f} "
               f"{verdict} валовую прибыль на {ui.rub(abs(res['d_gross']))} за {weeks} нед.",
               "good" if res["d_gross"] > 0 else "warn")
    xs = np.arange(-30, 31, 1) / 100
    gp = [ai.price_scenario(rev4 * k, cost4 * k, x, el)["d_gross"] / 1e6 for x in xs]
    fig = go.Figure(go.Scatter(x=xs * 100, y=gp, mode="lines", line=dict(color=ui.SERIES[0], width=2),
                               hovertemplate="Цена %{x:+.0f}%: %{y:+,.1f} млн ₽<extra></extra>"))
    fig.add_vline(x=change * 100, line=dict(color=ui.SERIES[1], dash="dash"))
    fig.add_hline(y=0, line=dict(color=ui.AXIS))
    fig.update_layout(title="Изменение валовой прибыли в зависимости от изменения цены",
                      xaxis=dict(title="изменение цены, %"), yaxis=dict(title="млн ₽"))
    ui.chart(fig, 320, key="wi_curve", export_name="Что если: цена")
    st.caption("База — РТО и себестоимость категории за последние 4 недели выборки. Модель постоянной эластичности; "
               "каннибализация и реакция конкурентов не учитываются.")

    st.markdown("**Другие сценарии**")
    k = common.load_period(f).k
    c1, c2 = st.columns(2, gap="large")
    with c1.container(border=True):
        st.markdown("Снизить дефицит на полке")
        red = st.slider("Рост OSA, п.п.", 0.0, 3.0, 1.0, 0.25, key="wi_osa")
        subst = st.slider("Доля спроса, уходящего при дефиците", 0.2, 0.8, 0.5, 0.05, key="wi_subst")
        extra = k["revenue"] * red / 100 * subst
        st.metric("Доп. РТО за период", ui.rub(extra), f"+{ui.rub(extra * k['margin'])} валовой прибыли")
    with c2.container(border=True):
        st.markdown("Изменить долю промо")
        dp = st.slider("Изменение доли промо, п.п.", -10, 10, 5, 1, key="wi_promo")
        disc = st.slider("Средняя скидка", 0.05, 0.3, 0.15, 0.01, key="wi_disc")
        incr = st.slider("Доля прироста без каннибализации", 0.1, 0.6, 0.3, 0.05, key="wi_incr")
        moved = k["revenue"] * dp / 100
        d_gp = moved * incr * k["margin"] - moved * disc
        st.metric("Изменение РТО", ui.rub(moved * incr), ui.signed(("+" if d_gp >= 0 else "") + f"{ui.rub(d_gp)} валовой прибыли"))
