"""Тесты формул на эталонных примерах, посчитанных вручную."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bi import metrics as M
from bi.db import expand_params


def test_abc_classes_boundaries():
    # 50 + 30 + 10 + 6 + 4 = 100: A — до 80% (50, 30), B — до 95% (10, 6), C — остальное
    s = pd.Series([4, 50, 10, 30, 6], index=list("edbac"))
    cls = M.abc_classes(s)
    assert cls.to_dict() == {"e": "C", "d": "A", "b": "B", "a": "A", "c": "B"}


def test_xyz_classes():
    cv = pd.Series([0.1, 0.3, 0.9, np.nan])
    assert list(M.xyz_classes(cv, 0.25, 0.5)) == ["X", "Y", "Z", "Z"]


def test_weekly_cv_counts_zero_weeks():
    wk = pd.DataFrame({"sku_id": [1, 1, 1, 2, 2, 2, 2],
                       "week_start": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19",
                                                     "2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26"]),
                       "qty": [10, 10, 10, 10, 0, 10, 0]})
    cv = M.weekly_cv(wk, min_weeks=4)
    # SKU 1 в неделю 26.01 не продавался -> нули учитываются: [10,10,10,0], среднее 7,5, σ 4,33
    assert cv[1] == pytest.approx(np.std([10, 10, 10, 0]) / 7.5)
    assert cv[2] == pytest.approx(1.0)


def test_lfl_only_comparable_stores():
    cur = pd.DataFrame({"store_id": [1, 2, 3], "revenue": [110, 220, 50]})
    py = pd.DataFrame({"store_id": [1, 2], "revenue": [100, 200]})
    assert M.lfl(cur, py, comparable={1, 2, 3}) == pytest.approx(0.10)


def test_comparable_cutoff():
    stores = pd.DataFrame({"store_id": [1, 2], "open_date": pd.to_datetime(["2025-01-01", "2025-06-01"])})
    assert M.comparable_stores(stores, date(2026, 1, 1)) == {1}


def test_revenue_decomposition_adds_up():
    d = M.revenue_decomposition(cur_checks=110, cur_check=550, py_checks=100, py_check=500)
    assert d["base"] + d["traffic"] + d["check"] == pytest.approx(d["current"])
    assert d["traffic"] == pytest.approx(10 * 500)
    assert d["check"] == pytest.approx(50 * 110)


def test_reorder_point_and_order():
    pos = pd.DataFrame({"dc_id": [1], "sku_id": [1], "stock_qty": [20.0], "d_avg": [10.0], "d_std": [3.0],
                        "shipped_window": [280.0], "oos_days": [0], "last_sale": pd.to_datetime(["2026-10-26"]),
                        "lt_avg": [4.0], "lt_std": [1.0], "in_transit": [0.0]})
    sku = pd.DataFrame({"sku_id": [1], "sku": ["x"], "category_id": [1], "supplier_id": [1], "unit_cost": [50.0],
                        "case_qty": [12], "lifecycle": ["active"]})
    r = M.reorder_table(pos, sku, date(2026, 10, 26), service_level=0.95).iloc[0]
    ss = 1.6449 * np.sqrt(4 * 3 ** 2 + 10 ** 2 * 1 ** 2)
    assert r["safety_stock"] == pytest.approx(ss, rel=1e-4)
    assert r["rop"] == pytest.approx(40 + ss, rel=1e-4)
    need = 40 + ss + 70 - 20
    assert r["order_qty"] == np.ceil(need / 12) * 12
    assert r["status"] == "Ниже точки заказа"


def test_dead_stock_status():
    pos = pd.DataFrame({"dc_id": [1], "sku_id": [1], "stock_qty": [100.0], "d_avg": [0.0], "d_std": [0.0],
                        "shipped_window": [0.0], "oos_days": [0], "last_sale": pd.to_datetime(["2026-08-01"]),
                        "lt_avg": [5.0], "lt_std": [0.0], "in_transit": [0.0]})
    sku = pd.DataFrame({"sku_id": [1], "sku": ["x"], "category_id": [1], "supplier_id": [1], "unit_cost": [10.0],
                        "case_qty": [6], "lifecycle": ["delisted"]})
    r = M.reorder_table(pos, sku, date(2026, 10, 26), dead_days=30).iloc[0]
    assert r["status"] == "Неликвид" and r["order_qty"] == 0


def test_seasonal_forecast_reproduces_pure_seasonality():
    idx = pd.date_range("2024-01-01", periods=110, freq="7D")
    season = 100 + 10 * np.sin(np.arange(110) * 2 * np.pi / 52)
    y = pd.Series(season * 1.05 ** (np.arange(110) // 52), index=idx)
    fc = M.seasonal_forecast(y.iloc[:104], horizon=4)
    assert len(fc) == 4
    assert np.allclose(fc["forecast"].values, y.iloc[104:108].values, rtol=0.02)


def test_full_weeks_drops_partial():
    d = pd.DataFrame({"date": pd.date_range("2026-01-05", "2026-01-20"), "v": 1.0})   # пн 05.01 … вт 20.01
    w = M.full_weeks(d, "v")
    assert list(w.values) == [7.0, 7.0]


def test_expand_params_lists():
    sql, p = expand_params("SELECT * FROM t WHERE a IN :ids AND d BETWEEN :d0 AND :d1", {"ids": [1, 2], "d0": 1, "d1": 2})
    assert sql == "SELECT * FROM t WHERE a IN (:ids_0, :ids_1) AND d BETWEEN :d0 AND :d1"
    assert p == {"ids_0": 1, "ids_1": 2, "d0": 1, "d1": 2}
