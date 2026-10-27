"""Запросы к источнику данных. Все функции кэшируются по значениям фильтров."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from bi import config as C
from bi.db import make_backend

TTL = 3600  # данные логистики в целевой системе обновляются ежечасно


@st.cache_resource(show_spinner="Готовлю данные: первый запуск занимает до минуты…")
def backend():
    return make_backend()


def q(sql: str, **params) -> pd.DataFrame:
    return backend().query(sql, params)


# ---------------------------------------------------------------------------
# Справочники
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def dims() -> dict[str, pd.DataFrame]:
    out = {
        "store": q("SELECT * FROM dim_store"),
        "dc": q("SELECT * FROM dim_dc"),
        "category": q("SELECT * FROM dim_category"),
        "supplier": q("SELECT * FROM dim_supplier"),
        "sku": q("SELECT * FROM dim_sku"),
    }
    for col in ("open_date",):
        out["store"][col] = pd.to_datetime(out["store"][col])
    names = C.chain_names()
    out["store"]["chain_name"] = out["store"]["chain"].map(names)
    out["dc"]["chain_name"] = out["dc"]["chain"].map(names)
    for frame in out.values():
        for c in frame.columns:
            if str(frame[c].dtype) in ("string", "str") or frame[c].dtype == object:
                frame[c] = frame[c].astype(object)
    return out


@st.cache_data(ttl=TTL, show_spinner=False)
def data_bounds() -> tuple[date, date]:
    r = q("SELECT MIN(date) AS d0, MAX(date) AS d1 FROM fact_store_day")
    return pd.Timestamp(r["d0"].iloc[0]).date(), pd.Timestamp(r["d1"].iloc[0]).date()


# ---------------------------------------------------------------------------
# Фильтры
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Filters:
    start: date
    end: date
    chains: tuple[str, ...] = ()
    dcs: tuple[int, ...] = ()
    regions: tuple[str, ...] = ()
    stores: tuple[int, ...] = ()
    cats: tuple[int, ...] = ()
    # вычисляемые
    store_ids: tuple[int, ...] | None = field(default=None)
    dc_ids: tuple[int, ...] | None = field(default=None)

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def py(self) -> "Filters":
        """Тот же период год назад (сдвиг 364 дня — совпадают дни недели)."""
        return Filters(self.start - timedelta(days=364), self.end - timedelta(days=364), self.chains,
                       self.dcs, self.regions, self.stores, self.cats, self.store_ids, self.dc_ids)

    def with_dates(self, start: date, end: date) -> "Filters":
        return Filters(start, end, self.chains, self.dcs, self.regions, self.stores, self.cats,
                       self.store_ids, self.dc_ids)

    @property
    def store_filtered(self) -> bool:
        return self.store_ids is not None

    @property
    def cat_filtered(self) -> bool:
        return bool(self.cats)


def resolve(f: Filters) -> Filters:
    d = dims()
    st_ = d["store"]
    m = pd.Series(True, index=st_.index)
    if f.chains:
        m &= st_["chain"].isin(f.chains)
    if f.dcs:
        m &= st_["dc_id"].isin(f.dcs)
    if f.regions:
        m &= st_["region"].isin(f.regions)
    if f.stores:
        m &= st_["store_id"].isin(f.stores)
    store_ids = None if m.all() else tuple(int(x) for x in st_.loc[m, "store_id"])
    dcs = d["dc"]
    md = pd.Series(True, index=dcs.index)
    if f.chains:
        md &= dcs["chain"].isin(f.chains)
    if f.dcs:
        md &= dcs["dc_id"].isin(f.dcs)
    dc_ids = None if md.all() else tuple(int(x) for x in dcs.loc[md, "dc_id"])
    return Filters(f.start, f.end, f.chains, f.dcs, f.regions, f.stores, f.cats, store_ids, dc_ids)


def _where(f: Filters, *, stores=True, cats=True, dcs=False, date_col="date") -> tuple[str, dict]:
    parts = [f"{date_col} BETWEEN :d0 AND :d1"]
    p: dict = {"d0": f.start, "d1": f.end}
    if stores and f.store_ids is not None:
        if not f.store_ids:
            parts.append("1 = 0")
        else:
            parts.append("store_id IN :sids")
            p["sids"] = list(f.store_ids)
    if cats and f.cats:
        parts.append("category_id IN :cats")
        p["cats"] = list(f.cats)
    if dcs and f.dc_ids is not None:
        if not f.dc_ids:
            parts.append("1 = 0")
        else:
            parts.append("dc_id IN :dcids")
            p["dcids"] = list(f.dc_ids)
    return " AND ".join(parts), p


SALES_MEASURES = """
    SUM(revenue) AS revenue, SUM(cost) AS cost, SUM(qty) AS qty, SUM(promo_revenue) AS promo_revenue,
    SUM(writeoff) AS writeoff, SUM(stock_cost) AS stock_cost, SUM(oos_share * revenue) AS oos_w,
    COUNT(DISTINCT date) AS n_days
"""


# ---------------------------------------------------------------------------
# Продажи
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def sales_daily(f: Filters) -> pd.DataFrame:
    w, p = _where(f)
    df = q(f"SELECT date, {SALES_MEASURES} FROM fact_sales WHERE {w} GROUP BY date ORDER BY date", **p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def sales_daily_by_cat(f: Filters) -> pd.DataFrame:
    w, p = _where(f)
    df = q(f"""SELECT date, category_id, SUM(revenue) AS revenue, SUM(promo_revenue) AS promo_revenue
               FROM fact_sales WHERE {w} GROUP BY date, category_id""", **p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def sales_by(f: Filters, dim: str) -> pd.DataFrame:
    assert dim in ("store_id", "category_id")
    w, p = _where(f)
    return q(f"SELECT {dim}, {SALES_MEASURES} FROM fact_sales WHERE {w} GROUP BY {dim}", **p)


@st.cache_data(ttl=TTL, show_spinner=False)
def sales_by_store_cat(f: Filters) -> pd.DataFrame:
    w, p = _where(f)
    return q(f"""SELECT store_id, category_id, SUM(revenue) AS revenue, SUM(cost) AS cost,
                 SUM(stock_cost) AS stock_cost, SUM(oos_share * revenue) AS oos_w, COUNT(DISTINCT date) AS n_days
                 FROM fact_sales WHERE {w} GROUP BY store_id, category_id""", **p)


@st.cache_data(ttl=TTL, show_spinner=False)
def traffic_daily(f: Filters) -> pd.DataFrame:
    w, p = _where(f, cats=False)
    df = q(f"""SELECT date, SUM(checks) AS checks, SUM(loyalty_checks) AS loyalty_checks,
               SUM(delivery_orders) AS delivery_orders, SUM(revenue) AS revenue, COUNT(*) AS store_days
               FROM fact_store_day WHERE {w} GROUP BY date ORDER BY date""", **p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def traffic_by_store(f: Filters) -> pd.DataFrame:
    w, p = _where(f, cats=False)
    return q(f"""SELECT store_id, SUM(checks) AS checks, SUM(revenue) AS revenue, COUNT(*) AS days_open
                 FROM fact_store_day WHERE {w} GROUP BY store_id""", **p)


@st.cache_data(ttl=TTL, show_spinner=False)
def plan_for_period(f: Filters) -> pd.DataFrame:
    """План по магазинам за период, пропорционально числу дней периода в каждом месяце."""
    m0 = f.start.replace(day=1)
    w, p = _where(f.with_dates(m0, f.end), cats=False, date_col="month")
    df = q(f"SELECT month, store_id, revenue_plan FROM fact_plan WHERE {w}", **p)
    if df.empty:
        return pd.DataFrame(columns=["store_id", "plan"])
    df["month"] = pd.to_datetime(df["month"])
    mend = df["month"] + pd.offsets.MonthEnd(0)
    s = df["month"].clip(lower=pd.Timestamp(f.start))
    e = mend.clip(upper=pd.Timestamp(f.end))
    frac = ((e - s).dt.days + 1).clip(lower=0) / df["month"].dt.days_in_month
    df["plan"] = df["revenue_plan"] * frac
    return df.groupby("store_id", as_index=False)["plan"].sum()


# ---------------------------------------------------------------------------
# Склады и SKU
# ---------------------------------------------------------------------------
def _sku_where(f: Filters, alias: str = "f") -> tuple[str, dict]:
    parts = [f"{alias}.date BETWEEN :d0 AND :d1"]
    p: dict = {"d0": f.start, "d1": f.end}
    if f.dc_ids is not None:
        parts.append(f"{alias}.dc_id IN :dcids" if f.dc_ids else "1 = 0")
        if f.dc_ids:
            p["dcids"] = list(f.dc_ids)
    if f.cats:
        parts.append("s.category_id IN :cats")
        p["cats"] = list(f.cats)
    return " AND ".join(parts), p


@st.cache_data(ttl=TTL, show_spinner=False)
def sku_weekly(f: Filters) -> pd.DataFrame:
    w, p = _sku_where(f)
    return q(f"""SELECT f.sku_id, f.week_start, SUM(f.shipped_qty) AS qty, SUM(f.shipped_qty * s.price) AS revenue,
                 SUM(f.shipped_qty * (s.price - s.unit_cost)) AS gross_profit, SUM(f.demand_qty) AS demand,
                 SUM(f.promo) AS promo_days
                 FROM fact_sku_dc f JOIN dim_sku s ON s.sku_id = f.sku_id
                 WHERE {w} GROUP BY f.sku_id, f.week_start""", **p)


@st.cache_data(ttl=TTL, show_spinner=False)
def promo_uplift(f: Filters) -> pd.DataFrame:
    """Средние продажи SKU в дни акции и без неё — по категориям."""
    w, p = _sku_where(f)
    return q(f"""SELECT s.category_id, f.promo, SUM(f.demand_qty) AS demand, COUNT(*) AS sku_days
                 FROM fact_sku_dc f JOIN dim_sku s ON s.sku_id = f.sku_id
                 WHERE {w} GROUP BY s.category_id, f.promo""", **p)


@st.cache_data(ttl=TTL, show_spinner=False)
def dc_stock_daily(f: Filters) -> pd.DataFrame:
    w, p = _sku_where(f)
    df = q(f"""SELECT f.date, SUM(f.stock_qty * s.unit_cost) AS stock_cost, SUM(f.shipped_qty * s.unit_cost) AS cogs,
               SUM(f.demand_qty * s.unit_cost) AS demand_cost
               FROM fact_sku_dc f JOIN dim_sku s ON s.sku_id = f.sku_id
               WHERE {w} GROUP BY f.date ORDER BY f.date""", **p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def stock_position(f: Filters, window_days: int = 28) -> pd.DataFrame:
    """Срез запасов на конец периода по SKU × склад с параметрами для точки заказа."""
    end = f.end
    w0 = end - timedelta(days=window_days - 1)
    parts, p = [], {"w0": w0, "end": end}
    if f.dc_ids is not None:
        parts.append("f.dc_id IN :dcids" if f.dc_ids else "1 = 0")
        if f.dc_ids:
            p["dcids"] = list(f.dc_ids)
    if f.cats:
        parts.append("s.category_id IN :cats")
        p["cats"] = list(f.cats)
    extra = (" AND " + " AND ".join(parts)) if parts else ""
    stats = q(f"""SELECT f.dc_id, f.sku_id,
                    SUM(CASE WHEN f.date = :end THEN f.stock_qty ELSE 0 END) AS stock_qty,
                    AVG(CASE WHEN f.promo = 0 THEN f.demand_qty END) AS d_avg,
                    STDDEV_POP(CASE WHEN f.promo = 0 THEN f.demand_qty END) AS d_std,
                    SUM(f.shipped_qty) AS shipped_window,
                    SUM(CASE WHEN f.shipped_qty < f.demand_qty THEN 1 ELSE 0 END) AS oos_days
                  FROM fact_sku_dc f JOIN dim_sku s ON s.sku_id = f.sku_id
                  WHERE f.date BETWEEN :w0 AND :end{extra}
                  GROUP BY f.dc_id, f.sku_id""", **p)
    last = q(f"""SELECT f.dc_id, f.sku_id, MAX(f.date) AS last_sale
                 FROM fact_sku_dc f JOIN dim_sku s ON s.sku_id = f.sku_id
                 WHERE f.shipped_qty > 0 AND f.date <= :end{extra.replace(':w0', ':end')}
                 GROUP BY f.dc_id, f.sku_id""", **{k: v for k, v in p.items() if k != "w0"})
    p2 = {k: v for k, v in p.items() if k != "w0"}
    p2["lt0"] = end - timedelta(days=180)
    extra_o = extra.replace("f.dc_id", "o.dc_id")
    lt = q(f"""SELECT o.dc_id, o.sku_id, AVG(o.lead_time_days) AS lt_avg, STDDEV_POP(o.lead_time_days) AS lt_std,
                 SUM(CASE WHEN o.order_date <= :end AND (o.receipt_date IS NULL OR o.receipt_date > :end)
                          THEN o.qty_ordered ELSE 0 END) AS in_transit
               FROM fact_supplier_order o JOIN dim_sku s ON s.sku_id = o.sku_id
               WHERE o.order_date BETWEEN :lt0 AND :end{extra_o}
               GROUP BY o.dc_id, o.sku_id""", **p2)
    df = stats.merge(last, on=["dc_id", "sku_id"], how="left").merge(lt, on=["dc_id", "sku_id"], how="left")
    df["last_sale"] = pd.to_datetime(df["last_sale"])
    return df


# ---------------------------------------------------------------------------
# Логистика
# ---------------------------------------------------------------------------
def _order_where(f: Filters) -> tuple[str, dict]:
    parts = ["o.receipt_date BETWEEN :d0 AND :d1"]
    p: dict = {"d0": f.start, "d1": f.end}
    if f.dc_ids is not None:
        parts.append("o.dc_id IN :dcids" if f.dc_ids else "1 = 0")
        if f.dc_ids:
            p["dcids"] = list(f.dc_ids)
    if f.cats:
        parts.append("s.category_id IN :cats")
        p["cats"] = list(f.cats)
    return " AND ".join(parts), p


ORDER_MEASURES = """
    COUNT(*) AS orders, AVG(o.on_time) AS on_time, AVG(o.in_full) AS in_full, AVG(o.otif) AS otif,
    AVG(o.lead_time_days) AS lt_avg, STDDEV_POP(o.lead_time_days) AS lt_std, AVG(o.delay_days) AS delay_avg,
    SUM(o.qty_received) / SUM(o.qty_ordered) AS fill_rate, SUM(o.cost_received) AS cost_received
"""


@st.cache_data(ttl=TTL, show_spinner=False)
def orders_by(f: Filters, dim: str) -> pd.DataFrame:
    assert dim in ("supplier_id", "dc_id", "week_start", "category_id")
    w, p = _order_where(f)
    if dim == "week_start":
        sel = "o.receipt_date AS day"
        grp = "o.receipt_date"
    elif dim == "category_id":
        sel, grp = "s.category_id", "s.category_id"
    else:
        sel, grp = f"o.{dim}", f"o.{dim}"
    df = q(f"""SELECT {sel}, {ORDER_MEASURES}
               FROM fact_supplier_order o JOIN dim_sku s ON s.sku_id = o.sku_id
               WHERE {w} GROUP BY {grp}""", **p)
    if dim == "week_start":
        df["day"] = pd.to_datetime(df["day"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def lead_time_freq(f: Filters) -> pd.DataFrame:
    w, p = _order_where(f)
    return q(f"""SELECT o.dc_id, o.lead_time_days, COUNT(*) AS n
                 FROM fact_supplier_order o JOIN dim_sku s ON s.sku_id = o.sku_id
                 WHERE {w} GROUP BY o.dc_id, o.lead_time_days""", **p)


DELIVERY_MEASURES = """
    COUNT(*) AS trips, AVG(on_time) AS on_time, AVG(delay_min) AS delay_avg,
    SUM(CASE WHEN delay_min > 30 THEN delay_min ELSE 0 END) / NULLIF(SUM(CASE WHEN delay_min > 30 THEN 1 ELSE 0 END), 0) AS late_delay_avg,
    SUM(cost_shipped) AS cost_shipped, SUM(cost_received) AS cost_received
"""


@st.cache_data(ttl=TTL, show_spinner=False)
def deliveries_by(f: Filters, dim: str) -> pd.DataFrame:
    assert dim in ("dc_id", "store_id", "planned_hour", "date")
    w, p = _where(f, cats=False, dcs=True)
    df = q(f"SELECT {dim}, {DELIVERY_MEASURES} FROM fact_delivery WHERE {w} GROUP BY {dim}", **p)
    if dim == "date":
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def deliveries_by_dc_day(f: Filters) -> pd.DataFrame:
    w, p = _where(f, cats=False, dcs=True)
    df = q(f"""SELECT date, dc_id, COUNT(*) AS trips, SUM(on_time) AS on_time_trips
               FROM fact_delivery WHERE {w} GROUP BY date, dc_id""", **p)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=TTL, show_spinner=False)
def delay_buckets(f: Filters) -> pd.DataFrame:
    w, p = _where(f, cats=False, dcs=True)
    return q(f"""SELECT CASE WHEN delay_min < -30 THEN 0
                             WHEN delay_min <= 30 THEN 1
                             WHEN delay_min <= 60 THEN 2
                             WHEN delay_min <= 120 THEN 3
                             WHEN delay_min <= 240 THEN 4
                             ELSE 5 END AS bucket, COUNT(*) AS trips
                 FROM fact_delivery WHERE {w} GROUP BY 1""", **p)


# ---------------------------------------------------------------------------
# Качество данных
# ---------------------------------------------------------------------------
@st.cache_data(ttl=TTL, show_spinner=False)
def data_quality() -> pd.DataFrame:
    checks = [
        ("fact_sales", "Строк", "SELECT COUNT(*) AS v FROM fact_sales"),
        ("fact_sales", "Отрицательная выручка или себестоимость", "SELECT COUNT(*) AS v FROM fact_sales WHERE revenue < 0 OR cost < 0"),
        ("fact_sales", "Себестоимость выше выручки", "SELECT COUNT(*) AS v FROM fact_sales WHERE cost > revenue * 1.2 AND revenue > 0"),
        ("fact_sales", "Пустые ключи", "SELECT COUNT(*) AS v FROM fact_sales WHERE store_id IS NULL OR category_id IS NULL OR date IS NULL"),
        ("fact_sales", "Дубли ключа день × магазин × категория",
         "SELECT COUNT(*) AS v FROM (SELECT date, store_id, category_id FROM fact_sales GROUP BY date, store_id, category_id HAVING COUNT(*) > 1) t"),
        ("fact_store_day", "Строк", "SELECT COUNT(*) AS v FROM fact_store_day"),
        ("fact_store_day", "Выручка без чеков", "SELECT COUNT(*) AS v FROM fact_store_day WHERE checks = 0 AND revenue > 0"),
        ("fact_sku_dc", "Строк", "SELECT COUNT(*) AS v FROM fact_sku_dc"),
        ("fact_sku_dc", "Отрицательный остаток", "SELECT COUNT(*) AS v FROM fact_sku_dc WHERE stock_qty < 0"),
        ("fact_sku_dc", "Отгрузка больше спроса", "SELECT COUNT(*) AS v FROM fact_sku_dc WHERE shipped_qty > demand_qty + 0.5"),
        ("fact_supplier_order", "Строк", "SELECT COUNT(*) AS v FROM fact_supplier_order"),
        ("fact_supplier_order", "Заказы в пути (нет даты прихода)", "SELECT COUNT(*) AS v FROM fact_supplier_order WHERE receipt_date IS NULL"),
        ("fact_supplier_order", "Приход раньше заказа", "SELECT COUNT(*) AS v FROM fact_supplier_order WHERE receipt_date < order_date"),
        ("fact_delivery", "Строк", "SELECT COUNT(*) AS v FROM fact_delivery"),
        ("fact_delivery", "Принято больше отгруженного", "SELECT COUNT(*) AS v FROM fact_delivery WHERE cost_received > cost_shipped"),
        ("dim_store", "Магазины без РЦ", "SELECT COUNT(*) AS v FROM dim_store WHERE dc_id IS NULL"),
    ]
    rows = []
    for table, check, sql in checks:
        v = int(q(sql)["v"].iloc[0])
        rows.append({"Таблица": table, "Проверка": check, "Значение": v,
                     "Статус": "—" if check == "Строк" else ("✅ OK" if v == 0 else "⚠️ Проверить")})
    return pd.DataFrame(rows)
