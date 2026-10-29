"""Обзор сети: KPI с прогнозом, план-факт-прогноз, комбинированный график, карта, отклонения по дивизионам."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bi import config as C
from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

PAGE = "overview"
KPI_ROW = 150


def render(f: Q.Filters):
    ui.page_header("Обзор сети", "Ключевые показатели к прошлому году, плану и прогнозу", f, PAGE)
    ui.register_widgets(PAGE, ["Тренд РТО", "РТО и маржа", "Карта", "План-факт по дивизионам", "Таблица дивизионов"])
    py_ok = common.py_available(f)
    cur = common.load_period(f)
    if common.no_data(cur.sales):
        return
    prev = common.load_period(f.py()) if py_ok else None
    common.notes(f, py_ok)
    mobile = st.session_state.get("mobile", False)

    k, kp = cur.k, (prev.k if prev else {})
    stores = common.store_table(f)
    comp = stores[stores["comparable"] & stores["revenue_py"].notna()]
    lfl = M.change(comp["revenue"].sum(), comp["revenue_py"].sum()) if py_ok and len(comp) else np.nan
    plan_pct = M.safe_div(stores.loc[stores["plan"].notna(), "store_revenue"].sum(), stores["plan"].sum())
    otif = common.otif_summary(f)
    otif_py = common.otif_summary(f.py()) if py_ok else np.nan
    fc_m, plan_m, share_m = common.month_forecast(f)

    def d(key):
        return M.change(k[key], kp.get(key, np.nan)) if py_ok else np.nan

    def dpp(key):
        return k[key] - kp.get(key, np.nan) if py_ok else np.nan

    osa = 1 - k["oos"]
    osa_py = 1 - kp["oos"] if py_ok else np.nan
    cols = st.columns(2 if mobile else 5)
    ui.kpi(cols[0], "РТО", ui.rub(k["revenue"]), ui.delta_pct(d("revenue")),
           help="Розничный товарооборот. Сравнение — с тем же периодом прошлого года (сдвиг 364 дня). "
                "Прогноз месяца: факт с начала месяца + темп последних 7 дней.", height=KPI_ROW,
           insight=f"прогноз месяца: {ui.pct(share_m)} плана" if np.isfinite(share_m) else None)
    ui.kpi(cols[1], "LFL РТО", ui.pct(lfl), None if not np.isfinite(lfl) else f"{len(comp)} магазинов",
           good_when_up=None, arrow="off", height=KPI_ROW,
           help="Like-for-like: рост РТО магазинов, открытых не позже чем за год до начала периода.")
    if not mobile:
        ui.kpi(cols[2], "Валовая маржа", ui.pct(k["margin"]), ui.delta_pp(dpp("margin")), height=KPI_ROW)
        ui.kpi(cols[3], "Средний чек", ui.rub(k["avg_check"], 0), ui.delta_pct(d("avg_check")), height=KPI_ROW)
        ui.kpi(cols[4], "Чеки", ui.big(k["checks"]), ui.delta_pct(d("checks")), height=KPI_ROW)
    cols = st.columns(2 if mobile else 5)
    ui.kpi(cols[0], "Выполнение плана", ui.pct(plan_pct),
           None if not np.isfinite(plan_pct) else ui.delta_pct(plan_pct - 1, "к плану"),
           help="РТО магазинов / план за дни периода")
    ui.kpi(cols[1], "OSA (доступность)", ui.pct(osa), ui.delta_pp(osa - osa_py),
           help="1 − доля позиций без остатка на полке, взвешенная по РТО")
    if not mobile:
        ui.kpi(cols[2], "Запас в магазинах", f"{ui.num(k['cover_days'], 1)} дн.",
               ui.delta_abs(dpp("cover_days"), " дн."), good_when_up=None)
        ui.kpi(cols[3], "Списания", ui.pct(k["writeoff_rate"], 2), ui.delta_pp(dpp("writeoff_rate"), digits=2),
               good_when_up=False)
        ui.kpi(cols[4], "OTIF поставщиков", ui.pct(otif), ui.delta_pp(otif - otif_py if py_ok else np.nan))

    st.write("")
    _trend(f, py_ok, mobile)
    if mobile:
        _division_bullets(stores, f)
        return

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        _combo(f)
    with right:
        _map(stores)

    st.subheader("Дивизионы: план, факт и прошлый год")
    _division_bullets(stores, f)
    _division_table(stores, f)

    st.subheader("Требуют внимания")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        _worst_stores(stores, py_ok)
    with c2:
        _categories(f, py_ok)
    ui.export_panel(PAGE)


# ---------------------------------------------------------------------------
def _trend(f: Q.Filters, py_ok: bool, mobile: bool):
    st.subheader("Динамика")
    c = st.columns([1, 1, 1, 1.4]) if not mobile else st.columns(2)
    gran = c[0].selectbox("Гранулярность", ["День", "Неделя", "Месяц", "Квартал"], index=1, key="ov_gran")
    metric = c[1].selectbox("Показатель", ["РТО", "Чеки", "Средний чек", "Валовая маржа"], key="ov_metric")
    if mobile:
        kind, extras = "Линия", ["Прошлый год", "План"]
    else:
        kind = c[2].segmented_control("Вид", ["Линия", "Столбцы"], default="Линия", key="ov_kind")
        extras = st.pills("Добавить на график", ["Прошлый год", "План", "Прогноз", "Скользящее среднее",
                                                "Линия тренда", "Индекс к началу"], selection_mode="multi",
                          default=["Прошлый год", "План", "Прогноз"], key="ov_extras")
    extras = extras or []
    tf = f if f.days >= 28 or gran == "День" else f.with_dates(f.end - timedelta(days=90), f.end)

    def series(ff: Q.Filters) -> pd.Series:
        s = Q.sales_daily(ff)[["date", "revenue", "cost"]]
        t = Q.traffic_daily(ff)[["date", "checks", "revenue"]].rename(columns={"revenue": "store_revenue"})
        x = common.resample(s.merge(t, on="date", how="left"), ["revenue", "cost", "checks", "store_revenue"], gran)
        if metric == "РТО":
            return x["revenue"]
        if metric == "Чеки":
            return x["checks"]
        if metric == "Средний чек":
            return x["store_revenue"] / x["checks"]
        return (x["revenue"] - x["cost"]) / x["revenue"]

    y = series(tf)
    if y.empty:
        ui.empty_state()
        return
    scale, unit = {"РТО": (1e6, "млн ₽"), "Чеки": (1e3, "тыс. чеков"), "Средний чек": (1, "₽"),
                   "Валовая маржа": (0.01, "%")}[metric]
    index_mode = "Индекс к началу" in extras
    y_plot, unit_plot = (y / y.iloc[0] * 100, "индекс, начало = 100") if index_mode else (y / scale, unit)
    fig = go.Figure()
    hov = "%{y:,.1f} " + unit_plot
    if "Прошлый год" in extras and tf.py().start >= Q.data_bounds()[0]:
        p = series(tf.py())
        p.index = p.index + pd.Timedelta(days=364)
        if gran in ("Месяц", "Квартал"):
            p.index = p.index.to_period("M" if gran == "Месяц" else "Q").to_timestamp()
        pp = p / y.iloc[0] * 100 if index_mode else p / scale
        fig.add_scatter(x=pp.index, y=pp.values, name="Прошлый год", mode="lines",
                        line=dict(color=ui.NEUTRAL, width=2, dash="dot"), hovertemplate=hov + "<extra>ПГ</extra>")
    if "План" in extras and metric == "РТО" and not f.cat_filtered and not index_mode:
        pl = Q.plan_monthly(tf)
        if len(pl):
            days = []
            for _, r in pl.groupby("month", as_index=False)["plan"].sum().iterrows():
                m0 = r["month"]
                dd = pd.date_range(max(m0, pd.Timestamp(tf.start)),
                                   min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(tf.end)))
                if len(dd):
                    days.append(pd.Series(r["plan"] / len(dd), index=dd))
            if days:
                pdays = pd.concat(days).rename("plan").rename_axis("date").reset_index()
                pr = common.resample(pdays, ["plan"], gran)["plan"]
                fig.add_scatter(x=pr.index, y=pr.values / scale, name="План", mode="lines",
                                line=dict(color=ui.SERIES[2], width=2, dash="dash"),
                                hovertemplate=hov + "<extra>План</extra>")
    if kind == "Столбцы":
        fig.add_bar(x=y_plot.index, y=y_plot.values, name="Факт", marker=dict(color=ui.SERIES[0], cornerradius=3),
                    hovertemplate=hov + "<extra>Факт</extra>")
    else:
        fig.add_scatter(x=y_plot.index, y=y_plot.values, name="Факт",
                        mode="lines+markers" if gran != "День" else "lines",
                        line=dict(color=ui.SERIES[0], width=2.5), marker=dict(size=6),
                        hovertemplate=hov + "<extra>Факт</extra>")
    if "Скользящее среднее" in extras and len(y_plot) > 4:
        w = {"День": 7, "Неделя": 4, "Месяц": 3, "Квартал": 2}[gran]
        ma = y_plot.rolling(w).mean()
        fig.add_scatter(x=ma.index, y=ma.values, name=f"Скользящее среднее ({w})", mode="lines",
                        line=dict(color=ui.SERIES[6], width=2), hovertemplate=hov + "<extra>Среднее</extra>")
    if "Линия тренда" in extras and len(y_plot) > 3:
        t = np.arange(len(y_plot))
        a, b = np.polyfit(t, y_plot.values.astype(float), 1)
        fig.add_scatter(x=y_plot.index, y=a * t + b, name="Тренд", mode="lines",
                        line=dict(color=ui.SERIES[3], width=1.5, dash="dash"), hoverinfo="skip")
    if "Прогноз" in extras and metric in ("РТО", "Чеки") and gran in ("День", "Неделя") and not index_mode:
        hist = Q.sales_daily(f.with_dates(Q.data_bounds()[0], f.end)) if metric == "РТО" else \
            Q.traffic_daily(f.with_dates(Q.data_bounds()[0], f.end))
        col = "revenue" if metric == "РТО" else "checks"
        yd = hist.set_index("date")[col]
        if len(yd) > 90:
            fc = common.cached_forecast(yd, "ets", 28, "D").set_index("date")
            if gran == "Неделя":
                # неделя, где есть и факт, и прогноз, собирается из обоих — линия без разрыва
                fact = yd[yd.index > fc.index[0] - pd.Timedelta(days=7)]
                comb = pd.concat([pd.DataFrame({"forecast": fact, "low": fact, "high": fact}), fc])
                cnt = comb["forecast"].resample("W-SUN").count()
                fc = comb.resample("W-SUN").sum()[cnt == 7]
                fc.index = fc.index - pd.Timedelta(days=6)
                fc = fc[fc.index + pd.Timedelta(days=6) >= yd.index[-1]]
                if len(y_plot) and len(fc):
                    last = y_plot.index[-1]
                    fc = pd.concat([pd.DataFrame({"forecast": [y.iloc[-1]], "low": [y.iloc[-1]], "high": [y.iloc[-1]]},
                                                 index=[last]), fc[fc.index > last]])
            if len(fc):
                fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]),
                                y=list(fc["high"] / scale) + list(fc["low"][::-1] / scale), fill="toself",
                                fillcolor="rgba(42,120,214,0.12)", mode="lines", line=dict(width=0),
                                name="Интервал 80%", hoverinfo="skip")
                fig.add_scatter(x=fc.index, y=fc["forecast"] / scale, name="Прогноз (ETS)", mode="lines+markers",
                                line=dict(color=ui.SERIES[0], width=2, dash="dash"), marker=dict(size=5),
                                hovertemplate=hov + "<extra>Прогноз</extra>")
    fig.update_layout(title=f"{metric} · {gran.lower()}", hovermode="x unified",
                      yaxis=dict(title=unit_plot, tickformat=",.0f" if metric != "Валовая маржа" else ",.1f"))
    ui.chart(fig, 380, key="ov_trend", dates=True, export_name="Тренд")
    if tf is not f:
        st.caption("Период короче 4 недель — тренд показан за последние 13 недель.")
    if "Прогноз" in extras and gran == "Неделя" and metric in ("РТО", "Чеки"):
        st.caption("Прогноз строится по дням моделью ETS и суммируется по неделям; интервал недели — сумма дневных "
                   "границ (оценка сверху). Модели, горизонт и точность — на странице «Прогноз».")


def _combo(f: Q.Filters):
    s = Q.sales_daily(f)
    w = common.resample(s, ["revenue", "cost"], "Неделя" if f.days >= 56 else "День")
    if w.empty:
        return
    margin = (w["revenue"] - w["cost"]) / w["revenue"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    fig.add_bar(x=w.index, y=w["revenue"] / 1e6, name="РТО, млн ₽", marker=dict(color=ui.SERIES[0], cornerradius=3),
                hovertemplate="%{y:,.0f} млн ₽<extra>РТО</extra>", row=1, col=1)
    fig.add_scatter(x=w.index, y=margin * 100, name="Валовая маржа, %", mode="lines+markers",
                    line=dict(color=ui.SERIES[1], width=2), marker=dict(size=5),
                    hovertemplate="%{y:.2f}%<extra>Маржа</extra>", row=2, col=1)
    fig.update_yaxes(title_text="млн ₽", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(title="РТО и валовая маржа на одной диаграмме", hovermode="x unified")
    ui.chart(fig, 400, key="ov_combo", dates=True, export_name="РТО и маржа")


def _map(stores: pd.DataFrame):
    g = stores.groupby(["city", "chain_name"], as_index=False).agg(
        revenue=("revenue", "sum"), n=("store_id", "nunique"))
    comp = stores[stores["comparable"]].groupby(["city", "chain_name"]).agg(r=("revenue", "sum"), p=("revenue_py", "sum"))
    lfl = (comp["r"] / comp["p"] - 1).replace([np.inf, -np.inf], np.nan)
    g["lfl"] = [lfl.get((c, ch), np.nan) for c, ch in zip(g["city"], g["chain_name"])]
    g["lat"] = g["city"].map(lambda c: C.CITY_COORDS.get(c, (np.nan, np.nan))[0])
    g["lon"] = g["city"].map(lambda c: C.CITY_COORDS.get(c, (np.nan, np.nan))[1])
    g = g.dropna(subset=["lat"])
    if g.empty:
        return
    lim = max(0.05, float(np.nanmax(np.abs(g["lfl"]))) if g["lfl"].notna().any() else 0.05)
    fig = go.Figure(go.Scattergeo(
        lat=g["lat"], lon=g["lon"], text=g["city"], mode="markers",
        marker=dict(size=np.clip(np.sqrt(g["revenue"] / g["revenue"].max()) * 42, 7, 42), color=g["lfl"],
                    colorscale=[[0, ui.DIV_NEG], [0.5, "#d9d8d2"], [1, ui.DIV_POS]], cmin=-lim, cmax=lim,
                    line=dict(color="white", width=1), opacity=0.9,
                    colorbar=dict(title="LFL", tickformat="+.0%", thickness=10, len=0.6)),
        customdata=np.c_[g["chain_name"], g["n"], g["revenue"] / 1e6, g["lfl"] * 100],
        hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>Магазинов: %{customdata[1]}<br>"
                      "РТО: %{customdata[2]:,.0f} млн ₽<br>LFL: %{customdata[3]:+.1f}%<extra></extra>"))
    fig.update_geos(projection_type="conic conformal", lataxis_range=[43, 60.5], lonaxis_range=[32, 60],
                    showcountries=True, showsubunits=True, resolution=50)
    fig.update_layout(title="Карта сети: размер — РТО, цвет — LFL", margin=dict(l=0, r=0, t=48, b=0))
    ui.chart(fig, 400, key="ov_map")


def _division_bullets(stores: pd.DataFrame, f: Q.Filters):
    g = common.group_table(stores, ["chain_name", "division"], f).sort_values("revenue")
    if g.empty:
        return
    g["label"] = g["division"] + " · " + g["chain_name"].str.split(" · ").str[0]
    colors = [ui.STATUS[ui.status(v, "plan")] if np.isfinite(v) else ui.NEUTRAL for v in g["plan_pct"]]
    fig = go.Figure()
    fig.add_bar(y=g["label"], x=g["store_revenue"] / 1e6, orientation="h", name="Факт",
                marker=dict(color=colors, cornerradius=3),
                customdata=np.c_[g["plan_pct"] * 100, g["yoy"] * 100, g["lfl"] * 100],
                hovertemplate="%{y}<br>Факт: %{x:,.0f} млн ₽<br>План: %{customdata[0]:.1f}%<br>"
                              "Г/г: %{customdata[1]:+.1f}% · LFL: %{customdata[2]:+.1f}%<extra></extra>")
    fig.add_scatter(y=g["label"], x=g["plan"] / 1e6, mode="markers", name="План",
                    marker=dict(symbol="line-ns", size=22,
                                line=dict(width=3, color="#f0efec" if ui.is_dark() else ui.INK2)),
                    hovertemplate="План: %{x:,.0f} млн ₽<extra></extra>")
    if common.py_available(f):
        py_rev = stores.groupby(["chain_name", "division"])["revenue_py"].sum()
        g["py"] = [py_rev.get((c, d), np.nan) for c, d in zip(g["chain_name"], g["division"])]
        fig.add_scatter(y=g["label"], x=g["py"] / 1e6, mode="markers", name="Прошлый год",
                        marker=dict(symbol="diamond", size=9, color=ui.NEUTRAL),
                        hovertemplate="Прошлый год: %{x:,.0f} млн ₽<extra></extra>")
    fig.update_layout(title="Факт (столбец), план (черта), прошлый год (ромб)", xaxis=dict(title="млн ₽"))
    ui.chart(fig, max(300, 34 * len(g) + 110), key="ov_bullets", export_name="План-факт по дивизионам")
    st.caption("Цвет столбца — выполнение плана по порогам («Пороги подсветки» в боковой панели): "
               "🟢 норма · 🟡 допуск · 🔴 отставание.")


def _division_table(stores: pd.DataFrame, f: Q.Filters):
    by = st.segmented_control("Разрез", ["Дивизион", "Федеральный округ", "РЦ", "Регион"], default="Дивизион",
                              key="ov_region_by", label_visibility="collapsed") or "Дивизион"
    col = {"Дивизион": "division", "Федеральный округ": "fed_district", "РЦ": "dc_name", "Регион": "region"}[by]
    g = common.group_table(stores, ["chain_name", col], f)
    g["status"] = g.apply(common.status_icon, axis=1)
    g = g.sort_values("revenue", ascending=False).reset_index(drop=True)
    show = pd.DataFrame({
        "": g["status"], "Сеть": g["chain_name"], by: g[col], "Магазинов": g["stores"].astype(int),
        "РТО, млн ₽": g["revenue"] / 1e6, "Г/г": g["yoy"], "LFL": g["lfl"], "План": g["plan_pct"],
        "OSA": g["osa"], "Маржа": g["margin"], "Списания": g["writeoff_rate"],
    })
    sty = ui.styled(show, {"LFL": "lfl", "План": "plan", "OSA": "osa"}, top_col="LFL", top_n=3)
    st.dataframe(sty, hide_index=True, height=min(38 + 35 * len(show), 460), column_config={
        "": st.column_config.TextColumn(width=36, help="Худший статус из LFL, плана и OSA по настраиваемым порогам"),
        "РТО, млн ₽": st.column_config.NumberColumn(format="%.1f"),
        "Г/г": st.column_config.NumberColumn(format="percent"),
        "LFL": st.column_config.NumberColumn(format="percent"),
        "План": st.column_config.NumberColumn(format="percent"),
        "OSA": st.column_config.NumberColumn(format="percent"),
        "Маржа": st.column_config.NumberColumn(format="percent"),
        "Списания": st.column_config.NumberColumn(format="percent"),
    })
    st.caption("Заливка — по порогам подсветки; зелёная черта и жирный шрифт — 3 лучших по LFL, красная черта — 3 худших.")
    ui.download({"Свод": show}, "svod.xlsx", key="dl_ov_regions")


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
    show = pd.DataFrame({"Магазин": worst["store_name"], "УГМ": worst["ugm"], "LFL": worst["lfl"],
                         "Чеки г/г": worst["checks"] / worst["checks_py"] - 1, "OSA": worst["osa"]})
    st.dataframe(ui.styled(show, {"LFL": "lfl", "OSA": "osa"}), hide_index=True, column_config={
        "LFL": st.column_config.NumberColumn(format="percent"),
        "Чеки г/г": st.column_config.NumberColumn(format="percent"),
        "OSA": st.column_config.NumberColumn(format="percent")})
    ui.collect_table("Магазины с падением LFL", show)


def _categories(f: Q.Filters, py_ok: bool):
    st.markdown("**Категории: доступность и списания**")
    cur = Q.sales_by(f, "category_id")
    cats = Q.dims()["category"].set_index("category_id")["category"]
    cur["osa"] = 1 - M.safe_div(cur["oos_w"], cur["revenue"])
    cur["wo"] = M.safe_div(cur["writeoff"], cur["cost"])
    if py_ok:
        py = Q.sales_by(f.py(), "category_id")
        py["osa_py"] = 1 - M.safe_div(py["oos_w"], py["revenue"])
        py["wo_py"] = M.safe_div(py["writeoff"], py["cost"])
        cur = cur.merge(py[["category_id", "osa_py", "wo_py"]], on="category_id", how="left")
    else:
        cur["osa_py"], cur["wo_py"] = np.nan, np.nan
    cur = cur.sort_values("osa")
    show = pd.DataFrame({"Категория": cur["category_id"].map(cats), "OSA": cur["osa"],
                         "Δ OSA, п.п.": (cur["osa"] - cur["osa_py"]) * 100, "Списания": cur["wo"],
                         "Δ списаний, п.п.": (cur["wo"] - cur["wo_py"]) * 100})
    st.dataframe(ui.styled(show, {"OSA": "osa"}), hide_index=True, column_config={
        "OSA": st.column_config.NumberColumn(format="percent"),
        "Δ OSA, п.п.": st.column_config.NumberColumn(format="%+.2f"),
        "Списания": st.column_config.NumberColumn(format="percent"),
        "Δ списаний, п.п.": st.column_config.NumberColumn(format="%+.2f")})
    ui.collect_table("Категории", show)
