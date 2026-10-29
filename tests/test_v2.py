"""Модули версии 2: формулы показателей, роли и доступ, прогноз, аномалии, выгрузка."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from bi import access as A
from bi import ai, export
from bi import forecast as FC
from bi import kpi as K


# --- Формулы показателей ----------------------------------------------------------------------------------------
def test_kpi_formula_evaluates_over_sums():
    df = pd.DataFrame({"revenue": [100.0, 0.0], "cost": [75.0, 0.0]})   # поля витрины
    v = K.evaluate("(РТО - Себестоимость) / РТО", df)
    assert v.iloc[0] == pytest.approx(0.25)
    assert np.isnan(v.iloc[1])          # деление на ноль даёт пусто, а не ошибку


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "РТО ** 2", "open('x')", "РТО.real",
                                 "[РТО]", "Неизвестное / РТО", "", "1 + 2", "lambda: 1"])
def test_kpi_formula_rejects_unsafe_or_invalid(bad):
    with pytest.raises(K.FormulaError):
        K.parse(bad)


def test_kpi_library_has_unique_codes_and_valid_builder_formulas():
    lib = K.library()
    assert lib["code"].is_unique
    for _, r in lib[lib["builder"].astype(bool)].iterrows():
        K.parse(r["formula"])           # все формулы, доступные в конструкторе, разбираются


# --- Роли и доступ -----------------------------------------------------------------------------------------------
def _stores():
    return pd.DataFrame({
        "store_id": [1, 2, 3, 4], "chain": ["G", "P", "P", "P"],
        "fed_district": ["Приволжский ФО", "Центральный ФО", "Приволжский ФО", "Приволжский ФО"],
        "division": ["Ульяновск", "Центр", "Средняя Волга", "Средняя Волга"],
        "ugm": ["У-1", "Ц-1", "СВ-1", "СВ-2"], "dc_id": [1, 3, 2, 2]})


def test_role_scope_restricts_stores():
    st_ = _stores()
    mask = lambda key: A.scope_store_mask(A.role_scope(A.ROLES[key], st_), st_)  # noqa: E731
    assert mask("network").all()
    assert st_[mask("okrug")]["store_id"].tolist() == [2]
    assert st_[mask("division")]["store_id"].tolist() == [3, 4]
    assert st_[mask("ugm")]["store_id"].tolist() == [3]
    assert st_[mask("logist")]["store_id"].tolist() == [2]


def test_temporary_grant_extends_ugm_scope_only_while_valid():
    st_ = _stores()
    g = pd.DataFrame({"role_key": ["ugm"], "scope_type": ["ugm"], "scope_value": ["СВ-2"],
                      "valid_from": ["2026-10-01"], "valid_to": ["2026-10-31"]})
    inside = A.role_scope(A.ROLES["ugm"], st_, g, pd.Timestamp("2026-10-15").date())
    after = A.role_scope(A.ROLES["ugm"], st_, g, pd.Timestamp("2026-11-01").date())
    assert set(inside.ugms) == {"СВ-1", "СВ-2"}
    assert after.ugms == ("СВ-1",)


def test_only_admin_has_admin_page():
    assert [k for k, r in A.ROLES.items() if "admin" in r.pages] == ["admin"]


# --- Прогноз и аномалии ------------------------------------------------------------------------------------------
def _series(days=400, seed=1):
    idx = pd.date_range("2025-01-01", periods=days, freq="D")
    rng = np.random.default_rng(seed)
    week = np.array([0.93, 0.93, 0.96, 0.99, 1.1, 1.16, 0.94])[idx.dayofweek]
    return pd.Series(1000 * week * (1 + 0.0005 * np.arange(days)) * rng.normal(1, 0.01, days), index=idx)


@pytest.mark.parametrize("model,freq,horizon", [("ets", "D", 28), ("sarima", "D", 28), ("ets", "W", 8),
                                                ("naive", "W", 8)])
def test_forecast_shape_and_interval(model, freq, horizon):
    """Сезонный наивный метод доступен только для недельного ряда (нужно 62+ недели истории)."""
    y = _series(days=520)
    if freq == "W":
        y = y.resample("W-SUN").sum().iloc[1:-1]
    fc = FC.forecast(y, model, horizon=horizon, freq=freq)
    assert len(fc) == horizon
    assert (fc["low"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["high"]).all()
    assert fc["forecast"].between(y.tail(horizon).min() * 0.7, y.tail(horizon).max() * 1.3).all()


def test_backtest_accuracy_reasonable_on_regular_series():
    bt = FC.backtest(_series(), "ets", horizon=14, n_origins=4)
    acc = FC.accuracy_by_horizon(bt)
    assert len(acc) and (acc["mape"] < 0.1).all()


def test_anomaly_found_on_injected_spike():
    y = _series()
    y.iloc[350] *= 1.4
    daily = pd.DataFrame({"date": y.index, "revenue": y.values, "checks": y.values / 500,
                          "store_revenue": y.values, "promo_revenue": y.values * 0.25, "oos_w": y.values * 0.05})
    _, an = ai.anomalies(daily, z_thr=3.0)
    assert y.index[350] in set(an["date"])


# --- Выгрузка ----------------------------------------------------------------------------------------------------
def test_export_all_formats_with_mixed_charts():
    d = pd.date_range("2026-01-01", periods=20)
    combo = make_subplots(rows=2, cols=1, shared_xaxes=True)
    combo.add_bar(x=d, y=np.arange(20) * 100.0, name="РТО", row=1, col=1)
    combo.add_scatter(x=d, y=np.linspace(24, 25, 20), name="Маржа", row=2, col=1)
    bullet = go.Figure()
    bullet.add_bar(y=["Дивизион 1", "Дивизион 2"], x=[1.0, 2.0], orientation="h", name="Факт")
    bullet.add_scatter(y=["Дивизион 1", "Дивизион 2"], x=[1.5, 2.5], mode="markers", name="План")
    mixed = go.Figure()
    mixed.add_scatter(x=[str(x.date()) for x in d], y=np.arange(20.0))
    mixed.add_scatter(x=list(d), y=np.arange(20.0)[::-1])
    exp = {"title": "Тест", "kpis": [("РТО", "1 ₽", "+1%")],
           "tables": {"Таблица": pd.DataFrame({"a": [1.5, 2.0], "b": ["x", "y"]})},
           "figures": {"combo": combo, "bullet": bullet, "mixed": mixed,
                       "waterfall": go.Figure(go.Waterfall(x=["a", "b"], y=[1, 2]))}}
    sizes = {}
    for fmt in ["Excel", "CSV", "XML", "PDF", "PowerPoint"]:
        data, name, mime = export.build(fmt, exp)
        sizes[fmt] = len(data)
        assert name.startswith("test_") and mime
    assert sizes["PDF"] > 10_000 and sizes["PowerPoint"] > 20_000
