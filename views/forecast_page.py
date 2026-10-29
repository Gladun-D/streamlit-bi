"""Прогноз: ETS / SARIMA / сезонный наивный, доверительные интервалы, история прогнозов и точность."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import forecast as FC
from bi import queries as Q
from bi import ui
from views import common

PAGE = "forecast"
SERIES = {
    "РТО магазинов": ("sales", "revenue", "₽"),
    "Чеки": ("traffic", "checks", "чеков"),
    "Средний чек": ("traffic", "avg_check", "₽"),
    "Отгрузки РЦ в магазины (себестоимость)": ("dc", "cogs", "₽"),
    "Остатки на РЦ (себестоимость)": ("dc", "stock_cost", "₽"),
}


def _series(f: Q.Filters, name: str) -> pd.Series:
    src, col, _ = SERIES[name]
    full = f.with_dates(Q.data_bounds()[0], f.end)
    if src == "sales":
        d = Q.sales_daily(full)
    elif src == "traffic":
        d = Q.traffic_daily(full)
        d["avg_check"] = d["revenue"] / d["checks"]
    else:
        d = Q.dc_stock_daily(full)
    return d.set_index("date")[col].astype(float)


def render(f: Q.Filters):
    ui.page_header("Прогноз", "Модели ETS и SARIMA с доверительными интервалами и историей точности", f, PAGE)
    ui.register_widgets(PAGE, ["Прогноз", "История прогнозов", "Точность по горизонтам"])
    c = st.columns([1.5, 1.15, 1.6, 1.1, 0.8])
    name = c[0].selectbox("Что прогнозируем", list(SERIES), key="fc_series",
                          help="Ряд строится по текущим фильтрам: сеть, регион, магазин, категория, поставщик, SKU")
    gran = c[1].segmented_control("Шаг", ["День", "Неделя"], default="День", key="fc_gran") or "День"
    models = list(FC.MODELS) if gran == "Неделя" else ["ets", "sarima"]
    model = c[2].selectbox("Модель", models, format_func=lambda m: FC.MODELS[m], key=f"fc_model_{gran}")
    horizon = c[3].slider("Горизонт", 7 if gran == "День" else 4, 90 if gran == "День" else 26,
                          28 if gran == "День" else 8, key=f"fc_h_{gran}",
                          help="Дней или недель вперёд от последней даты данных")
    level = c[4].selectbox("Интервал", [0.8, 0.95], format_func=lambda v: f"{v:.0%}", key="fc_level")
    use_f = st.checkbox("Учитывать календарные факторы (праздники)", value=True, key="fc_factors",
                        disabled=model != "sarima", help="Используются в модели SARIMA как внешние регрессоры")

    y = _series(f, name)
    y = y[y.index <= pd.Timestamp(f.end)]
    if gran == "Неделя":
        y = common.resample(y.rename("v").reset_index().rename(columns={"index": "date"}), ["v"], "Неделя")["v"]
    if len(y) < (90 if gran == "День" else 64) or y.sum() == 0:
        ui.empty_state("Недостаточно истории для прогноза по выбранным фильтрам")
        return
    freq = "D" if gran == "День" else "W"
    try:
        fc = common.cached_forecast(y, model, int(horizon), freq, float(level), bool(use_f))
    except Exception as e:
        st.error(f"Модель не обучилась: {e}")
        return
    _, _, unit = SERIES[name]
    scale, ut = (1e6, "млн ₽") if unit == "₽" and name != "Средний чек" else ((1, "₽") if name == "Средний чек"
                                                                              else (1e3, "тыс. чеков"))
    total_fc = fc["forecast"].sum()
    py_same = y.reindex(fc["date"] - pd.Timedelta(days=364)) if gran == "День" else \
        y.reindex(fc["date"] - pd.Timedelta(days=364))
    py_total = py_same.sum() if py_same.notna().all() else np.nan
    bt = common.cached_backtest(y, model, int(min(horizon, 28 if gran == "День" else 8)), freq,
                                6 if model != "sarima" else 4, bool(use_f))
    acc = FC.accuracy_by_horizon(bt, freq)
    k = st.columns(4)
    if name == "Средний чек":
        ui.kpi(k[0], "Прогноз: средний за горизонт", ui.rub(fc["forecast"].mean(), 0), None, good_when_up=None)
    else:
        ui.kpi(k[0], f"Прогноз на {horizon} {'дн.' if gran == 'День' else 'нед.'}",
               ui.rub(total_fc) if unit == "₽" else ui.big(total_fc),
               ui.delta_pct(total_fc / py_total - 1, "к тем же датам ПГ") if np.isfinite(py_total) else None)
    ui.kpi(k[1], "Нижняя граница", ui.rub(fc["low"].sum()) if unit == "₽" and name != "Средний чек"
           else ui.big(fc["low"].sum()), None, good_when_up=None, help=f"Сумма нижних границ интервала {level:.0%}")
    ui.kpi(k[2], "Верхняя граница", ui.rub(fc["high"].sum()) if unit == "₽" and name != "Средний чек"
           else ui.big(fc["high"].sum()), None, good_when_up=None)
    mape = float(bt["ape"].mean()) if len(bt) else np.nan
    ui.kpi(k[3], "Ошибка модели (MAPE)", ui.pct(mape, 2), None, good_when_up=None,
           help="Средняя абсолютная ошибка прогнозов «из прошлого» на истории (бэктест)")

    hist = y.tail(180 if gran == "День" else 60)
    fig = go.Figure()
    fig.add_scatter(x=list(fc["date"]) + list(fc["date"][::-1]), y=list(fc["high"] / scale) + list(fc["low"][::-1] / scale),
                    fill="toself", fillcolor="rgba(42,120,214,0.14)", mode="lines", line=dict(width=0),
                    name=f"Интервал {level:.0%}", hoverinfo="skip")
    fig.add_scatter(x=hist.index, y=hist.values / scale, name="Факт", mode="lines", line=dict(color=ui.SERIES[0], width=2),
                    hovertemplate="%{y:,.1f} " + ut + "<extra>Факт</extra>")
    fig.add_scatter(x=[hist.index[-1]] + list(fc["date"]), y=[hist.iloc[-1] / scale] + list(fc["forecast"] / scale),
                    name="Прогноз", mode="lines", line=dict(color=ui.SERIES[0], width=2, dash="dash"),
                    hovertemplate="%{y:,.1f} " + ut + "<extra>Прогноз</extra>")
    fig.update_layout(title=f"{name}: факт и прогноз · {FC.MODELS[model]}", hovermode="x unified", yaxis=dict(title=ut))
    ui.chart(fig, 420, key="fc_chart", dates=True, export_name="Прогноз")
    ui.collect_table("Прогноз", fc.assign(date=fc["date"].dt.date).rename(columns={
        "date": "Дата", "forecast": "Прогноз", "low": "Нижняя граница", "high": "Верхняя граница"}))

    st.subheader("История прогнозов и точность")
    if bt.empty:
        st.caption("Недостаточно истории для оценки точности.")
    else:
        c1, c2 = st.columns([1.4, 1], gap="large")
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=y.tail(120 if gran == "День" else 40).index, y=y.tail(120 if gran == "День" else 40) / scale,
                            name="Факт", mode="lines", line=dict(color=ui.NEUTRAL, width=2))
            for i, (orig, g) in enumerate(bt.groupby("origin")):
                fig.add_scatter(x=g["date"], y=g["forecast"] / scale, mode="lines",
                                name=f"Прогноз от {pd.Timestamp(orig):%d.%m}",
                                line=dict(width=1.5, color=(ui.SERIES_DARK if ui.is_dark() else ui.SERIES)[i % 8]))
            fig.update_layout(title="Что модель прогнозировала раньше и что произошло", yaxis=dict(title=ut),
                              hovermode="x unified")
            ui.chart(fig, 340, key="fc_bt", dates=True, export_name="История прогнозов")
        with c2:
            show = acc.rename(columns={"bucket": "Горизонт", "points": "Точек", "mape": "MAPE", "bias": "Смещение"})
            st.markdown("**Точность по горизонтам**")
            st.dataframe(show, hide_index=True, column_config={
                "MAPE": st.column_config.NumberColumn(format="percent"),
                "Смещение": st.column_config.NumberColumn(format="percent", help="+ — прогноз выше факта")})
            ui.collect_table("Точность по горизонтам", show)
            if st.toggle("Сравнить все модели", key="fc_cmp"):
                rows = []
                for m in models:
                    b = common.cached_backtest(y, m, int(min(horizon, 28 if gran == "День" else 8)), freq,
                                               6 if m != "sarima" else 4, bool(use_f))
                    rows.append({"Модель": FC.MODELS[m], "MAPE": float(b["ape"].mean()) if len(b) else np.nan})
                st.dataframe(pd.DataFrame(rows).sort_values("MAPE"), hide_index=True,
                             column_config={"MAPE": st.column_config.NumberColumn(format="percent")})
        st.caption("История прогнозов восстанавливается бэктестом: модель обучается на данных до каждой «точки "
                   "прогноза» и сравнивается с фактом. В целевой системе прогнозы сохраняются при каждом "
                   "обновлении. На синтетических данных итог по сети очень гладкий — точность здесь оптимистична; "
                   "на уровне магазина или SKU ошибка выше.")
    st.caption("Модель переобучается автоматически при обновлении данных (кэш — 1 час). Настройки модели: шаг, "
               "горизонт, уровень интервала, календарные факторы.")
    ui.export_panel(PAGE)
