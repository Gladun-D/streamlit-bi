"""Общие расчёты для страниц: текущий период против прошлого года, разрезы по магазинам."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui


@dataclass
class Period:
    f: Q.Filters
    sales: pd.DataFrame       # по дням
    traffic: pd.DataFrame     # по дням
    k: dict


def load_period(f: Q.Filters) -> Period:
    s = Q.sales_daily(f)
    t = Q.traffic_daily(f)
    return Period(f, s, t, M.sales_kpis(s, t))


def py_available(f: Q.Filters) -> bool:
    d0, _ = Q.data_bounds()
    return f.py().start >= d0


def store_table(f: Q.Filters) -> pd.DataFrame:
    """Показатели по магазинам за период и год назад + план."""
    d = Q.dims()
    cur = Q.sales_by(f, "store_id")
    tr = Q.traffic_by_store(f)
    plan = Q.plan_for_period(f)
    df = d["store"][["store_id", "store_code", "store_name", "chain", "chain_name", "fed_district", "division", "ugm",
                     "region", "city", "dc_id", "format", "open_date"]].merge(cur, on="store_id", how="inner")
    df = df.merge(tr.rename(columns={"revenue": "store_revenue"}), on="store_id", how="left")
    df = df.merge(plan, on="store_id", how="left")
    if py_available(f):
        py = Q.sales_by(f.py(), "store_id")[["store_id", "revenue"]].rename(columns={"revenue": "revenue_py"})
        trp = Q.traffic_by_store(f.py())[["store_id", "checks"]].rename(columns={"checks": "checks_py"})
        df = df.merge(py, on="store_id", how="left").merge(trp, on="store_id", how="left")
    else:
        df["revenue_py"] = np.nan
        df["checks_py"] = np.nan
    comp = M.comparable_stores(d["store"], f.start)
    df["comparable"] = df["store_id"].isin(comp)
    df["yoy"] = M.safe_div(df["revenue"], df["revenue_py"]) - 1
    df["margin"] = M.safe_div(df["revenue"] - df["cost"], df["revenue"])
    df["oos"] = M.safe_div(df["oos_w"], df["revenue"])
    df["osa"] = 1 - df["oos"]
    df["writeoff_rate"] = M.safe_div(df["writeoff"], df["cost"])
    df["avg_check"] = M.safe_div(df["store_revenue"], df["checks"])
    df["checks_day"] = M.safe_div(df["checks"], df["days_open"])
    df["plan_pct"] = M.safe_div(df["store_revenue"], df["plan"])
    df["cover_days"] = M.safe_div(df["stock_cost"], df["cost"])
    dc_name = d["dc"].set_index("dc_id")["name"]
    df["dc_name"] = df["dc_id"].map(dc_name)
    return df


def group_table(st_df: pd.DataFrame, by: list[str], f: Q.Filters) -> pd.DataFrame:
    """Свод по региону/РЦ/сети: LFL считается только по сопоставимым магазинам."""
    def agg(g: pd.DataFrame) -> pd.Series:
        comp = g[g["comparable"] & g["revenue_py"].notna()]
        return pd.Series({
            "stores": g["store_id"].nunique(),
            "revenue": g["revenue"].sum(),
            "revenue_py": g["revenue_py"].sum(min_count=1),
            "lfl": M.change(comp["revenue"].sum(), comp["revenue_py"].sum()) if len(comp) else np.nan,
            "margin": M.safe_div(g["revenue"].sum() - g["cost"].sum(), g["revenue"].sum()),
            "avg_check": M.safe_div(g["store_revenue"].sum(), g["checks"].sum()),
            "oos": M.safe_div(g["oos_w"].sum(), g["revenue"].sum()),
            "writeoff_rate": M.safe_div(g["writeoff"].sum(), g["cost"].sum()),
            "plan_pct": M.safe_div(g.loc[g["plan"].notna(), "store_revenue"].sum(), g["plan"].sum()),
            "osa": 1 - M.safe_div(g["oos_w"].sum(), g["revenue"].sum()),
            "checks": g["checks"].sum(),
            "store_revenue": g["store_revenue"].sum(),
            "plan": g["plan"].sum(min_count=1),
            "cost": g["cost"].sum(),
            "promo_revenue": g["promo_revenue"].sum(),
            "promo_cost": g["promo_cost"].sum() if "promo_cost" in g else np.nan,
        })
    out = st_df.groupby(by, dropna=False).apply(agg, include_groups=False).reset_index()
    out["yoy"] = M.safe_div(out["revenue"], out["revenue_py"]) - 1
    return out


HIER = [("chain_name", "Сеть"), ("fed_district", "Федеральный округ"), ("division", "Дивизион"),
        ("ugm", "УГМ"), ("store_name", "Магазин")]
_RANK = {"critical": 3, "warning": 2, "good": 1, "none": 0}


def status_of(row: pd.Series) -> str:
    """Худший из статусов по LFL, плану и OSA (пороги настраиваются в боковой панели)."""
    st_ = [ui.status(row.get("lfl", np.nan), "lfl"), ui.status(row.get("plan_pct", np.nan), "plan"),
           ui.status(row.get("osa", np.nan), "osa")]
    return max(st_, key=lambda s: _RANK[s])


def status_icon(row: pd.Series, oos_ref: float | None = None) -> str:
    return ui.STATUS_ICON[status_of(row)]


def notes(f: Q.Filters, py_ok: bool):
    msgs = []
    if f.cat_filtered:
        msgs.append("Выбраны категории: чеки, средний чек и план считаются по магазину целиком.")
    if f.sku_filtered:
        msgs.append("Поставщик и товар применяются к данным РЦ, закупкам и качеству; продажи магазинов "
                    "в прототипе детализированы до категории.")
    if f.condition:
        msgs.append(f"Условие по магазинам: {Q.CONDITIONS.get(f.condition, f.condition).lower()} "
                    f"— в выборке {len(f.store_ids or [])} магазинов.")
    if not py_ok:
        msgs.append("Для сравнения с прошлым годом нужен период, начинающийся не раньше 31.12.2025 "
                    "(данные начинаются 01.01.2025).")
    for m in msgs:
        st.caption(f":material/info: {m}")


def weekly_spark(daily: pd.DataFrame, value: str, weeks: int = 13) -> list[float] | None:
    s = M.full_weeks(daily, value) if len(daily) >= 7 else pd.Series(dtype=float)
    return s.tail(weeks).round(0).tolist() if len(s) > 2 else None


def otif_summary(f: Q.Filters) -> float:
    o = Q.orders_by(f, "dc_id")
    return float(np.average(o["otif"], weights=o["orders"])) if len(o) and o["orders"].sum() else np.nan


def no_data(df: pd.DataFrame) -> bool:
    if df is None or df.empty or float(df.get("revenue", pd.Series([1])).sum()) == 0:
        ui.empty_state()
        return True
    return False


@st.cache_data(ttl=Q.TTL, show_spinner="Строю прогноз…")
def cached_forecast(y: pd.Series, model: str, horizon: int, freq: str = "D", level: float = 0.8,
                    use_factors: bool = True) -> pd.DataFrame:
    from bi import forecast as FC
    return FC.forecast(y, model, horizon, freq, level, use_factors)


@st.cache_data(ttl=Q.TTL, show_spinner="Считаю историю точности…")
def cached_backtest(y: pd.Series, model: str, horizon: int, freq: str = "D", n_origins: int = 6,
                    use_factors: bool = True) -> pd.DataFrame:
    from bi import forecast as FC
    return FC.backtest(y, model, horizon, freq, n_origins, use_factors)


def month_forecast(f: Q.Filters) -> tuple[float, float, float]:
    """Прогноз выполнения плана текущего месяца: (прогноз РТО, план месяца, доля) — темп последних 7 дней."""
    from datetime import timedelta
    end = f.end
    m0 = end.replace(day=1)
    mend = (pd.Timestamp(m0) + pd.offsets.MonthEnd(0)).date()
    mtd = Q.traffic_daily(f.with_dates(m0, end))["revenue"].sum()
    s7 = max(m0, end - timedelta(days=6))
    r7 = Q.traffic_daily(f.with_dates(s7, end))["revenue"].sum() / ((end - s7).days + 1)
    plan = Q.plan_for_period(f.with_dates(m0, mend))["plan"].sum()
    fc = mtd + r7 * (mend - end).days
    return fc, plan, (fc / plan if plan else np.nan)


def resample(daily: pd.DataFrame, cols: list[str], gran: str) -> pd.DataFrame:
    """Агрегирование дневных данных: День / Неделя / Месяц / Квартал / Год (суммы; неполные крайние недели — убираются)."""
    d = daily.set_index("date")[cols]
    if gran == "День":
        return d
    if gran == "Неделя":
        cnt = d.iloc[:, 0].resample("W-SUN").count()
        out = d.resample("W-SUN").sum()[cnt == 7]
        out.index = out.index - pd.Timedelta(days=6)
        return out
    rule = {"Месяц": "MS", "Квартал": "QS", "Год": "YS"}[gran]
    return d.resample(rule).sum()
