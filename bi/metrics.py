"""Формулы показателей. Чистые функции без Streamlit — покрыты тестами (tests/test_metrics.py)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from bi import config as C

# Квантиль нормального распределения для уровня сервиса (без scipy)
_Z = {0.90: 1.2816, 0.95: 1.6449, 0.97: 1.8808, 0.98: 2.0537, 0.99: 2.3263}


def z_value(service_level: float) -> float:
    keys = np.array(sorted(_Z))
    k = keys[np.argmin(np.abs(keys - service_level))]
    return _Z[float(k)]


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b != 0, a / b, np.nan)
    return r.item() if r.ndim == 0 else r


def change(cur: float, base: float) -> float:
    """Относительное изменение; NaN, если базы нет."""
    if base is None or not np.isfinite(base) or base == 0:
        return float("nan")
    return cur / base - 1


# ---------------------------------------------------------------------------
# Сводные показатели по продажам
# ---------------------------------------------------------------------------
def sales_kpis(sales: pd.DataFrame, traffic: pd.DataFrame) -> dict[str, float]:
    """sales — агрегат fact_sales (одна строка или по дням), traffic — fact_store_day."""
    rev = float(sales["revenue"].sum())
    cost = float(sales["cost"].sum())
    n_days = int(sales["date"].nunique()) if "date" in sales else int(sales["n_days"].max() or 0)
    checks = float(traffic["checks"].sum())
    t_rev = float(traffic["revenue"].sum())
    return {
        "revenue": rev,
        "gross_profit": rev - cost,
        "margin": safe_div(rev - cost, rev),
        "checks": checks,
        "avg_check": safe_div(t_rev, checks),
        "promo_share": safe_div(sales["promo_revenue"].sum(), rev),
        "writeoff_rate": safe_div(sales["writeoff"].sum(), cost),
        "oos": safe_div(sales["oos_w"].sum(), rev),
        # запас в днях = средний остаток / средняя дневная себестоимость продаж
        "cover_days": safe_div(sales["stock_cost"].sum() / max(n_days, 1), cost / max(n_days, 1)),
        "stock_cost_avg": float(sales["stock_cost"].sum()) / max(n_days, 1),
        "loyalty_share": safe_div(traffic["loyalty_checks"].sum(), checks) if "loyalty_checks" in traffic else np.nan,
    }


def comparable_stores(stores: pd.DataFrame, start: date) -> set[int]:
    """Сопоставимые магазины: открыты не позже чем за 364 дня до начала периода."""
    cutoff = pd.Timestamp(start - timedelta(days=364))
    return set(stores.loc[stores["open_date"] <= cutoff, "store_id"].astype(int))


def lfl(cur_by_store: pd.DataFrame, py_by_store: pd.DataFrame, comparable: set[int],
        col: str = "revenue") -> float:
    c = cur_by_store[cur_by_store["store_id"].isin(comparable)]
    p = py_by_store[py_by_store["store_id"].isin(comparable)]
    common = set(c["store_id"]) & set(p["store_id"])
    if not common:
        return float("nan")
    return change(c.loc[c["store_id"].isin(common), col].sum(), p.loc[p["store_id"].isin(common), col].sum())


def revenue_decomposition(cur_checks: float, cur_check: float, py_checks: float, py_check: float) -> dict[str, float]:
    """Разложение изменения выручки: эффект трафика (по прошлому чеку) + эффект среднего чека (по текущему трафику)."""
    return {
        "base": py_checks * py_check,
        "traffic": (cur_checks - py_checks) * py_check,
        "check": (cur_check - py_check) * cur_checks,
        "current": cur_checks * cur_check,
    }


# ---------------------------------------------------------------------------
# ABC / XYZ
# ---------------------------------------------------------------------------
def abc_classes(values: pd.Series, a: float = 0.8, b: float = 0.95) -> pd.Series:
    """A — позиции, дающие первые 80% суммы, B — следующие 15%, C — остальное."""
    s = values.fillna(0).clip(lower=0).sort_values(ascending=False)
    total = s.sum()
    if total <= 0:
        return pd.Series("C", index=values.index)
    cum = s.cumsum() / total
    prev = cum.shift(fill_value=0)   # доля до позиции: позиция, пересекающая порог, остаётся в старшем классе
    cls = np.where(prev < a, "A", np.where(prev < b, "B", "C"))
    return pd.Series(cls, index=s.index).reindex(values.index)


def xyz_classes(cv: pd.Series, x: float = 0.25, y: float = 0.5) -> pd.Series:
    """X — стабильный спрос (CV ≤ x), Y — колеблющийся, Z — нерегулярный или нет данных."""
    return pd.Series(np.where(cv <= x, "X", np.where(cv <= y, "Y", "Z")), index=cv.index).where(cv.notna(), "Z")


def weekly_cv(weekly: pd.DataFrame, value: str = "qty", key: str = "sku_id", min_weeks: int = 4) -> pd.Series:
    """Коэффициент вариации недельных продаж; недели без продаж внутри периода учитываются как нули."""
    piv = weekly.pivot_table(index="week_start", columns=key, values=value, aggfunc="sum").fillna(0)
    if len(piv) < min_weeks:
        return pd.Series(np.nan, index=piv.columns)
    mean = piv.mean()
    return (piv.std(ddof=0) / mean).where(mean > 0)


ABC_XYZ_ACTIONS = {
    "AX": "Автозаказ, минимальный страховой запас, ежедневный контроль наличия",
    "AY": "Автозаказ с повышенным страховым запасом, контроль промо-календаря",
    "AZ": "Ручной контроль заказа, анализ причин нестабильности спроса",
    "BX": "Автозаказ, стандартный страховой запас",
    "BY": "Автозаказ, пересмотр параметров раз в месяц",
    "BZ": "Заказ под потребность, сократить страховой запас",
    "CX": "Укрупнить партии, реже заказывать",
    "CY": "Оценить целесообразность в матрице",
    "CZ": "Кандидат на вывод из матрицы или заказ под заказ",
}


# ---------------------------------------------------------------------------
# Запасы: страховой запас, точка заказа, статусы
# ---------------------------------------------------------------------------
def reorder_table(pos: pd.DataFrame, sku: pd.DataFrame, end: date, service_level: float = C.SERVICE_LEVEL,
                  review_days: int = C.REVIEW_DAYS, dead_days: int = 30, overstock_mult: float = 3.0) -> pd.DataFrame:
    """Страховой запас SS = z·√(LT·σd² + d²·σLT²); точка заказа ROP = d·LT + SS;
    рекомендуемый заказ = ROP + d·период пересмотра − (остаток + в пути), кратно упаковке."""
    z = z_value(service_level)
    df = pos.merge(sku[["sku_id", "sku", "category_id", "supplier_id", "unit_cost", "case_qty", "lifecycle"]],
                   on="sku_id", how="left")
    df["lt_avg"] = df["lt_avg"].fillna(df["lt_avg"].median() if df["lt_avg"].notna().any() else 5)
    df["lt_std"] = df["lt_std"].fillna(0)
    df["d_avg"] = df["d_avg"].fillna(0)
    df["d_std"] = df["d_std"].fillna(0)
    df["in_transit"] = df["in_transit"].fillna(0)
    df["safety_stock"] = z * np.sqrt(df["lt_avg"] * df["d_std"] ** 2 + df["d_avg"] ** 2 * df["lt_std"] ** 2)
    df["rop"] = df["d_avg"] * df["lt_avg"] + df["safety_stock"]
    df["cover_days"] = safe_div(df["stock_qty"], df["d_avg"])
    target = df["rop"] + df["d_avg"] * review_days
    raw = (target - df["stock_qty"] - df["in_transit"]).clip(lower=0)
    df["order_qty"] = np.ceil(raw / df["case_qty"].clip(lower=1)) * df["case_qty"].clip(lower=1)
    df["stock_value"] = df["stock_qty"] * df["unit_cost"]
    df["days_since_sale"] = (pd.Timestamp(end) - df["last_sale"]).dt.days
    max_norm = target * overstock_mult / 2 + df["safety_stock"]

    status = np.select(
        [
            (df["stock_qty"] > 0) & ((df["days_since_sale"].isna()) | (df["days_since_sale"] >= dead_days)),
            (df["stock_qty"] <= 0) & (df["d_avg"] > 0),
            (df["stock_qty"] + df["in_transit"] < df["rop"]) & (df["d_avg"] > 0),
            (df["stock_qty"] > max_norm) & (df["d_avg"] > 0),
        ],
        ["Неликвид", "Нет в наличии", "Ниже точки заказа", "Излишек"],
        default="Норма",
    )
    df["status"] = status
    df.loc[df["status"].isin(["Неликвид"]), "order_qty"] = 0
    df["excess_value"] = np.where(df["status"] == "Излишек",
                                  (df["stock_qty"] - max_norm).clip(lower=0) * df["unit_cost"], 0.0)
    return df


STOCK_STATUS_ORDER = ["Нет в наличии", "Ниже точки заказа", "Излишек", "Неликвид", "Норма"]
STOCK_STATUS_ICON = {"Нет в наличии": "🔴", "Ниже точки заказа": "🟠", "Излишек": "🟡", "Неликвид": "⚫", "Норма": "🟢"}


# ---------------------------------------------------------------------------
# Прогноз
# ---------------------------------------------------------------------------
def seasonal_forecast(weekly: pd.Series, horizon: int = 8, season: int = 52, recent: int = 8,
                      backtest: int = 26) -> pd.DataFrame:
    """Сезонный наивный прогноз с поправкой на тренд:
    ŷ(t+h) = y(t+h−52) × [сумма последних 8 недель / сумма тех же недель год назад].
    Интервал 80% — по ошибкам того же метода на последних 26 неделях."""
    y = weekly.astype(float).dropna()
    if len(y) < season + recent + 2:
        return pd.DataFrame()

    def predict(hist: pd.Series, h: int) -> float:
        r = hist.iloc[-recent:].sum() / hist.iloc[-recent - season:-season].sum()
        return hist.iloc[len(hist) - season + h - 1] * r

    errs = []
    for k in range(backtest, 0, -1):
        hist = y.iloc[:-k]
        if len(hist) < season + recent + 1:
            continue
        errs.append(y.iloc[-k] / predict(hist, 1) - 1)
    sd = float(np.std(errs)) if errs else 0.05
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=7), periods=horizon, freq="7D")
    vals = [predict(y, h) for h in range(1, horizon + 1)]
    out = pd.DataFrame({"week": idx, "forecast": vals})
    widen = np.sqrt(np.arange(1, horizon + 1))
    out["low"] = out["forecast"] * (1 - 1.2816 * sd * widen)
    out["high"] = out["forecast"] * (1 + 1.2816 * sd * widen)
    out.attrs["mape"] = float(np.mean(np.abs(errs))) if errs else np.nan
    return out


def full_weeks(daily: pd.DataFrame, value: str) -> pd.Series:
    """Сумма по неделям (пн–вс); неполные крайние недели отбрасываются."""
    s = daily.set_index("date")[value].resample("W-SUN").agg(["sum", "count"])
    s = s[s["count"] == 7]["sum"]
    s.index = s.index - pd.Timedelta(days=6)   # подпись — понедельник
    return s
