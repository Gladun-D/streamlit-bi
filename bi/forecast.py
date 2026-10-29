"""Прогнозирование временных рядов: ETS (Хольт-Уинтерс), SARIMA, сезонный наивный метод.

Модели обучаются на логарифме ряда (мультипликативная сезонность, неотрицательный прогноз).
Календарные факторы (предновогодние дни, праздники) подаются в SARIMA как внешние регрессоры.
История точности — скользящий бэктест: прогнозы «из прошлого» сравниваются с фактом.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bi import metrics as M

MODELS = {
    "ets": "ETS (Хольт-Уинтерс, затухающий тренд)",
    "sarima": "SARIMA с календарными факторами",
    "naive": "Сезонный наивный (неделя прошлого года × темп)",
}


def holidays_frame(idx: pd.DatetimeIndex, weekly: bool = False) -> pd.DataFrame:
    """Календарные факторы. Для недельного ряда — доля дней недели, попадающих в окно праздника."""
    def daily(ix: pd.DatetimeIndex) -> pd.DataFrame:
        m, d = ix.month, ix.day
        return pd.DataFrame({
            "pre_new_year": ((m == 12) & (d >= 28)).astype(float),
            "new_year": ((m == 1) & (d <= 7)).astype(float),
            "march8": ((m == 3) & (d >= 6) & (d <= 8)).astype(float),
            "feb23": ((m == 2) & (d >= 21) & (d <= 23)).astype(float),
        }, index=ix)
    if not weekly:
        return daily(idx)
    days = pd.date_range(idx.min(), idx.max() + pd.Timedelta(days=6))
    h = daily(days)
    return h.resample("W-SUN", label="left", closed="left").mean().reindex(idx, method="nearest").set_axis(idx)


def _fit_predict(y: pd.Series, model: str, horizon: int, freq: str, level: float, use_factors: bool):
    """Возвращает (mean, lo, hi) на horizon шагов вперёд."""
    alpha = 1 - level
    idx_f = pd.date_range(y.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
    if model == "naive":
        if freq != "7D":
            raise ValueError("Сезонный наивный метод — только для недельного ряда")
        fc = M.seasonal_forecast(y, horizon=horizon)
        if fc.empty:
            raise ValueError("Недостаточно истории: нужно не меньше 62 недель")
        z = 1.2816 if level <= 0.8 else 1.96
        width = (fc["high"] - fc["forecast"]) / 1.2816 * z
        return (pd.Series(fc["forecast"].values, idx_f), pd.Series((fc["forecast"] - width).values, idx_f),
                pd.Series((fc["forecast"] + width).values, idx_f))
    ly = np.log(y.clip(lower=1.0))
    weekly = freq == "7D"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "ets":
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel
            seasonal = None if weekly else "add"
            m = ETSModel(ly.reset_index(drop=True), error="add", trend="add", damped_trend=True,
                         seasonal=seasonal, seasonal_periods=None if weekly else 7)
            r = m.fit(disp=False, maxiter=200)
            pred = r.get_prediction(start=len(ly), end=len(ly) + horizon - 1)
            sf = pred.summary_frame(alpha=alpha)
            mean, lo, hi = sf["mean"].values, sf["pi_lower"].values, sf["pi_upper"].values
        elif model == "sarima":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            exog = holidays_frame(y.index, weekly) if use_factors else None
            exog_f = holidays_frame(idx_f, weekly) if use_factors else None
            if exog is not None:
                keep = [c for c in exog.columns if exog[c].std() > 0]
                exog, exog_f = (exog[keep], exog_f[keep]) if keep else (None, None)
            order, sorder = ((1, 1, 1), (0, 0, 0, 0)) if weekly else ((1, 1, 1), (1, 0, 1, 7))
            m = SARIMAX(ly.values, exog=None if exog is None else exog.values, order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False)
            r = m.fit(disp=False, maxiter=60)
            pred = r.get_forecast(steps=horizon, exog=None if exog_f is None else exog_f.values)
            mean = pred.predicted_mean
            ci = pred.conf_int(alpha=alpha)
            lo, hi = ci[:, 0], ci[:, 1]
        else:
            raise ValueError(model)
    return pd.Series(np.exp(mean), idx_f), pd.Series(np.exp(lo), idx_f), pd.Series(np.exp(hi), idx_f)


def forecast(y: pd.Series, model: str, horizon: int, freq: str = "D", level: float = 0.8,
             use_factors: bool = True) -> pd.DataFrame:
    y = y.astype(float).dropna()
    mean, lo, hi = _fit_predict(y, model, horizon, "7D" if freq == "W" else "D", level, use_factors)
    return pd.DataFrame({"date": mean.index, "forecast": mean.values, "low": lo.values, "high": hi.values})


def backtest(y: pd.Series, model: str, horizon: int, freq: str = "D", n_origins: int = 6,
             use_factors: bool = True) -> pd.DataFrame:
    """История прогнозов: для каждой «точки прогноза» в прошлом — прогноз и факт по горизонтам."""
    y = y.astype(float).dropna()
    step = 7 if freq == "D" else 1
    f = "7D" if freq == "W" else "D"
    rows = []
    for k in range(n_origins, 0, -1):
        cut = len(y) - horizon - (k - 1) * step
        if cut < (70 if freq == "D" else 64):
            continue
        hist = y.iloc[:cut]
        try:
            mean, _, _ = _fit_predict(hist, model, horizon, f, 0.8, use_factors)
        except Exception:
            continue
        actual = y.reindex(mean.index)
        for h, (d, fc) in enumerate(mean.items(), start=1):
            if pd.notna(actual.get(d)):
                rows.append({"origin": hist.index[-1], "horizon": h, "date": d, "forecast": fc,
                             "actual": float(actual[d])})
    df = pd.DataFrame(rows)
    if len(df):
        df["ape"] = (df["forecast"] / df["actual"] - 1).abs()
    return df


def accuracy_by_horizon(bt: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    if freq == "D":
        bins = [0, 7, 14, 28, 1000]
        labels = ["1–7 дней", "8–14 дней", "15–28 дней", "29+ дней"]
    else:
        bins = [0, 1, 4, 8, 1000]
        labels = ["1 неделя", "2–4 недели", "5–8 недель", "9+ недель"]
    b = bt.assign(bucket=pd.cut(bt["horizon"], bins=bins, labels=labels))
    out = b.groupby("bucket", observed=True).agg(points=("ape", "size"), mape=("ape", "mean"),
                                                 bias=("forecast", "sum"), actual=("actual", "sum")).reset_index()
    out["bias"] = out["bias"] / out["actual"] - 1
    return out.drop(columns="actual")
