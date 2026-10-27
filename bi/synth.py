"""Генератор синтетических данных товародвижения двух продуктовых сетей.

Структура сети (число магазинов, РЦ, регионы, рост) — по открытым данным, см. bi/config.py.
Все операционные показатели — синтетические, генерируются с фиксированным seed,
поэтому при каждом запуске получаются одни и те же цифры.

Таблицы (parquet в каталоге данных):
  dim_dc, dim_store, dim_category, dim_supplier, dim_sku
  fact_store_day        день × магазин: чеки, чеки с картой, заказы доставки, выручка
  fact_sales            день × магазин × категория: выручка, себестоимость, штуки, промо,
                        списания, остаток (в себестоимости), доля SKU в дефиците
  fact_plan             месяц × магазин: план выручки
  fact_sku_dc           день × склад × SKU: спрос, отгрузка в магазины, приход, остаток, промо
  fact_supplier_order   заказ поставщику: даты заказа/обещания/прихода, заказано/принято
  fact_delivery         рейс склад → магазин: плановое и фактическое время, отгружено/принято

Заложенные в данные сценарии (чтобы аналитике было что «найти»; описаны в разделе «О данных»):
  1. С апреля 2026 в Самарской обл. трафик сети Б снижается (условный конкурент).
  2. Летом 2026 у РЦ Краснодар падает пунктуальность поставщиков и рейсов -> растёт дефицит.
  3. В августе растут списания овощей и фруктов (жара).
  4. ~5% SKU выводятся из матрицы, но остатки на складах остаются -> неликвид.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bi import config as C

TABLES = [
    "dim_dc", "dim_store", "dim_category", "dim_supplier", "dim_sku",
    "fact_store_day", "fact_sales", "fact_plan", "fact_sku_dc",
    "fact_supplier_order", "fact_delivery",
]
VERSION = "3"  # меняйте при изменении логики генерации, чтобы кэш пересобрался


@dataclass
class Ctx:
    rng: np.random.Generator
    dates: pd.DatetimeIndex
    stores: pd.DataFrame
    cats: pd.DataFrame
    open_mask: np.ndarray  # [D, S] магазин открыт


# ---------------------------------------------------------------------------
# Календарные факторы
# ---------------------------------------------------------------------------
WEEKDAY_TRAFFIC = np.array([0.95, 0.95, 0.97, 0.99, 1.08, 1.11, 0.95])
WEEKDAY_CHECK = np.array([0.98, 0.98, 0.99, 1.00, 1.02, 1.05, 0.99])


def _holiday_traffic(d: pd.Timestamp) -> float:
    m, day = d.month, d.day
    if m == 12 and day >= 28:
        return 1.15 + 0.08 * (day - 28)          # 28.12 — 1.15 … 31.12 — 1.39
    if m == 1 and day == 1:
        return 0.55
    if m == 1 and day <= 7:
        return 0.88
    if m == 3 and day == 7:
        return 1.22
    if m == 2 and day == 22:
        return 1.08
    return 1.0


def _holiday_check(d: pd.Timestamp) -> float:
    if d.month == 12 and d.day >= 20:
        return 1.12
    if d.month == 3 and d.day in (6, 7):
        return 1.06
    return 1.0


HOLIDAY_CAT_BOOST = {8: 1.8, 6: 1.6, 3: 1.3, 1: 1.2, 7: 1.3}  # 28–31 декабря, по категориям


def _cat_season(cats: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """[D, C] множитель сезонности категории."""
    phase = (dates.dayofyear.values[:, None] / 365.25 * 12.0) - (cats["peak_month"].values[None, :] - 0.5)
    season = 1 + cats["season_amp"].values[None, :] * np.cos(2 * np.pi * phase / 12.0)
    dec = (dates.month == 12) & (dates.day >= 28)
    for cid, boost in HOLIDAY_CAT_BOOST.items():
        j = cats.index[cats["category_id"] == cid][0]
        season[dec, j] *= boost
    return season


# ---------------------------------------------------------------------------
# Справочники
# ---------------------------------------------------------------------------
def _dims(rng: np.random.Generator) -> tuple[pd.DataFrame, ...]:
    dc = pd.DataFrame(C.DCS)

    cats = pd.DataFrame(C.CATEGORIES, columns=[
        "category_id", "category", "share_G", "share_P", "margin", "writeoff",
        "season_amp", "peak_month", "promo_share", "norm_days", "skus", "fresh",
    ])

    rows = []
    end = pd.Timestamp(C.DATA_END)
    start = pd.Timestamp(C.DATA_START)
    sid = 0
    for chain, prm in C.CHAINS.items():
        n_old, n_new = prm["stores_start"], prm["stores_end"] - prm["stores_start"]
        if chain == "G":
            shares = np.array([c[2] for c in C.G_CITIES])
            geo_idx = rng.choice(len(C.G_CITIES), size=n_old + n_new, p=shares / shares.sum())
            geo = [(C.G_CITIES[i][0], C.G_CITIES[i][1], C.G_CITIES[i][3]) for i in geo_idx]
            first_open = pd.Timestamp("2002-06-01")
        else:
            w = np.array([r[2] for r in C.P_REGIONS], dtype=float)
            geo_idx = rng.choice(len(C.P_REGIONS), size=n_old + n_new, p=w / w.sum())
            geo = [(C.P_REGIONS[i][0], C.P_REGIONS[i][1], C.P_REGIONS[i][3]) for i in geo_idx]
            first_open = pd.Timestamp("2015-02-01")
        old_open = first_open + pd.to_timedelta(
            np.sort(rng.uniform(0, (start - first_open).days - 30, n_old)).astype(int), unit="D")
        new_open = start + pd.to_timedelta(
            np.sort(rng.uniform(20, (end - start).days - 10, n_new)).astype(int), unit="D")
        opens = list(old_open) + list(new_open)
        prefix = "А" if chain == "G" else "Б"
        for k, ((region, city, dc_id), od) in enumerate(zip(geo, opens), start=1):
            sid += 1
            rows.append({
                "store_id": sid,
                "store_code": f"{prefix}-{k:03d}",
                "chain": chain,
                "format": prm["format"],
                "region": region,
                "city": city,
                "dc_id": dc_id,
                "open_date": od.normalize(),
                "size_factor": float(np.exp(rng.normal(0, 0.25))),
                "growth": float(rng.normal(0, 0.06)),
                "price_factor": float(np.exp(rng.normal(0, 0.04))),
                "slot_hour": int(rng.integers(5, 15)),
            })
    stores = pd.DataFrame(rows)
    stores["store_name"] = stores["store_code"] + " · " + stores["city"]
    g_ids = stores.index[stores["chain"] == "G"]
    gulliver_delivery = rng.choice(g_ids, size=C.CHAINS["G"]["delivery_stores"], replace=False)
    stores["has_delivery"] = False
    stores.loc[gulliver_delivery, "has_delivery"] = True
    # гастромаркет — один магазин «Гулливера» (ФАКТ: один гастромаркет)
    gastro = stores.index[(stores["chain"] == "G") & (stores["city"] == "Ульяновск")][0]
    stores.loc[gastro, "format"] = "Гастромаркет"

    sup = []
    for i in range(1, C.N_SUPPLIERS + 1):
        sup.append({
            "supplier_id": i,
            "supplier": f"Поставщик {i:02d}",
            "on_time": float(np.clip(rng.beta(12, 1.6), 0.6, 0.995)),
            "fill_rate": float(np.clip(rng.beta(30, 1.5), 0.8, 1.0)),
        })
    sup = pd.DataFrame(sup)
    # Три заметно ненадёжных поставщика — чтобы рейтинг был содержательным
    sup.loc[[4, 17, 33], "on_time"] = [0.62, 0.68, 0.71]
    sup.loc[[4, 17, 33], "fill_rate"] = [0.86, 0.9, 0.88]

    base_price = {1: 120, 2: 95, 3: 330, 4: 55, 5: 110, 6: 140, 7: 95, 8: 520, 9: 260, 10: 210, 11: 230, 12: 380}
    sku_rows = []
    skid = 0
    sup_by_cat = {c: rng.choice(sup["supplier_id"].values, size=6, replace=False) for c in cats["category_id"]}
    for _, c in cats.iterrows():
        for k in range(int(c["skus"])):
            skid += 1
            cv_class = rng.choice(["X", "Y", "Z"], p=[0.4, 0.35, 0.25])
            cv = {"X": rng.uniform(0.08, 0.2), "Y": rng.uniform(0.25, 0.45), "Z": rng.uniform(0.55, 1.0)}[cv_class]
            fresh = bool(c["fresh"])
            sku_rows.append({
                "sku_id": skid,
                "sku": f"{c['category'].split(',')[0].split(' и ')[0]} · арт. {skid:04d}",
                "category_id": int(c["category_id"]),
                "supplier_id": int(rng.choice(sup_by_cat[c["category_id"]])),
                "price": float(round(base_price[int(c["category_id"])] * np.exp(rng.normal(0, 0.35)), 2)),
                "weight": float(np.exp(rng.normal(0, 1.0))),
                "cv": float(cv),
                "lt_mean": float(rng.uniform(1, 2.5) if fresh else rng.uniform(4, 11)),
                "case_qty": int(rng.choice([6, 8, 10, 12, 20, 24])),
                "private_label": bool(rng.random() < 0.18),
            })
    sku = pd.DataFrame(sku_rows)
    sku["unit_cost"] = (sku["price"] * (1 - sku["category_id"].map(cats.set_index("category_id")["margin"])
                                        * np.exp(rng.normal(0, 0.08, len(sku))))).round(2)
    # Матрицы сетей
    sku["in_G"] = True
    p_mask = rng.random(len(sku)) < C.CHAINS["P"]["assortment_share"]
    ready = sku["category_id"] == 10
    p_mask[ready.values] = rng.random(int(ready.sum())) < 0.25
    sku["in_P"] = p_mask
    # Жизненный цикл: выводимые и новые позиции
    n = len(sku)
    life = rng.choice(["active", "delisted", "new"], size=n, p=[0.9, 0.05, 0.05])
    span = (pd.Timestamp(C.DATA_END) - pd.Timestamp(C.DATA_START)).days
    sku["lifecycle"] = life
    sku["delist_day"] = np.where(life == "delisted", rng.integers(int(span * 0.35), int(span * 0.85), n), -1)
    sku["launch_day"] = np.where(life == "new", rng.integers(int(span * 0.2), int(span * 0.8), n), -1)
    return dc, cats, stores, sup, sku


# ---------------------------------------------------------------------------
# Магазины: трафик и чек
# ---------------------------------------------------------------------------
def _store_day(ctx: Ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает expected_checks [D,S] (без шума), checks [D,S], avg_check [D,S]."""
    rng, dates, st = ctx.rng, ctx.dates, ctx.stores
    D, S = len(dates), len(st)
    t_years = ((dates - pd.Timestamp(C.DATA_START)).days.values / 365.25)[:, None]
    base_checks = st["chain"].map({k: v["checks_per_day"] for k, v in C.CHAINS.items()}).values[None, :]
    base_check = st["chain"].map({k: v["avg_check"] for k, v in C.CHAINS.items()}).values[None, :]

    wd = dates.dayofweek.values
    hol_t = np.array([_holiday_traffic(d) for d in dates])[:, None]
    hol_c = np.array([_holiday_check(d) for d in dates])[:, None]
    season = (1 + 0.035 * np.cos(2 * np.pi * (dates.dayofyear.values - 200) / 365.25))[:, None]

    growth = (1 + C.TRAFFIC_TREND + st["growth"].values[None, :]) ** t_years
    age = (dates.values[:, None] - st["open_date"].values[None, :]).astype("timedelta64[D]").astype(int)
    ramp = np.clip(0.55 + 0.45 * age / C.NEW_STORE_RAMP_DAYS, 0, 1)

    # Сценарий 1: условный конкурент в Самарской обл. для сети Б с апреля 2026
    comp = np.ones((D, S))
    samara = ((st["region"] == "Самарская обл.") & (st["chain"] == "P")).values
    after = np.asarray(dates >= pd.Timestamp("2026-04-01"))
    ramp_c = np.clip((dates - pd.Timestamp("2026-04-01")).days.values / 60, 0, 1)
    comp[np.ix_(after, samara)] = (1 - 0.11 * ramp_c[after])[:, None]

    expected = (base_checks * st["size_factor"].values[None, :] * WEEKDAY_TRAFFIC[wd][:, None]
                * hol_t * season * growth * ramp * comp)
    expected = np.where(ctx.open_mask, expected, 0.0)
    checks = np.round(expected * np.exp(rng.normal(0, 0.055, (D, S)))).astype(np.float32)
    checks = np.where(ctx.open_mask, checks, 0)

    avg_check = (base_check * st["price_factor"].values[None, :] * (1 + C.PRICE_INFLATION) ** t_years
                 * WEEKDAY_CHECK[wd][:, None] * hol_c * np.exp(rng.normal(0, 0.025, (D, S))))
    return expected.astype(np.float32), checks, avg_check.astype(np.float32)


# ---------------------------------------------------------------------------
# Склады: посуточная симуляция запасов по SKU
# ---------------------------------------------------------------------------
def _simulate_dc(ctx: Ctx, dc: pd.DataFrame, sku: pd.DataFrame, sup: pd.DataFrame,
                 exp_rev_dc_cat: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """exp_rev_dc_cat [D, n_dc, C] — ожидаемая выручка категории в магазинах склада.

    Возвращает fact_sku_dc, fact_supplier_order, lost_share [D, n_dc, C] (доля спроса, не обеспеченная складом).
    """
    rng, dates, cats = ctx.rng, ctx.dates, ctx.cats
    D = len(dates)
    cat_pos = {cid: j for j, cid in enumerate(cats["category_id"])}
    dc_pos = {d: i for i, d in enumerate(dc["dc_id"])}

    # Ряды склад × SKU по матрицам сетей
    pairs = []
    for _, d in dc.iterrows():
        col = "in_G" if d["chain"] == "G" else "in_P"
        ids = sku.index[sku[col]].values
        pairs.append(pd.DataFrame({"dc_id": d["dc_id"], "sku_idx": ids}))
    pairs = pd.concat(pairs, ignore_index=True)
    s = sku.loc[pairs["sku_idx"]].reset_index(drop=True)
    N = len(pairs)
    j_cat = s["category_id"].map(cat_pos).values
    i_dc = pairs["dc_id"].map(dc_pos).values

    # Доля SKU внутри категории склада
    w = s["weight"].values
    grp = i_dc * 100 + j_cat
    w_sum = pd.Series(w).groupby(grp).transform("sum").values
    sku_share = w / w_sum
    price = s["price"].values
    margin = cats["margin"].values[j_cat]
    # ожидаемый дневной спрос в штуках [D, N]
    base_qty = exp_rev_dc_cat[:, i_dc, j_cat] * sku_share[None, :] / price[None, :]

    # Жизненный цикл
    dayidx = np.arange(D)[:, None]
    alive = np.ones((D, N), dtype=bool)
    dl = s["delist_day"].values
    ln = s["launch_day"].values
    alive &= ~((dl[None, :] >= 0) & (dayidx >= dl[None, :]))
    alive &= ~((ln[None, :] >= 0) & (dayidx < ln[None, :]))

    # Промо: недели акций по SKU
    chain_of_dc = dc.set_index("dc_id")["chain"]
    chain = pairs["dc_id"].map(chain_of_dc).to_numpy(dtype=object)
    uplift = np.where(chain == "G", C.CHAINS["G"]["promo_uplift"], C.CHAINS["P"]["promo_uplift"])
    uplift = uplift * np.exp(rng.normal(0, 0.2, N))
    week = ((dates - dates[0]).days.values // 7)
    n_weeks = week.max() + 1
    promo_rate = cats["promo_share"].values[j_cat] / 2.2   # вероятность акции в неделю
    # акция одновременно на сети: один и тот же SKU в одну неделю по всем складам сети
    key = pd.Series(chain).astype(str) + "_" + s["sku_id"].astype(str)
    uniq, inv = np.unique(key.values, return_inverse=True)
    promo_weeks_u = rng.random((n_weeks, len(uniq))) < np.repeat(promo_rate, 1)[np.unique(inv, return_index=True)[1]][None, :]
    promo = promo_weeks_u[week][:, inv] & alive  # [D, N]

    noise = np.exp(rng.normal(0, s["cv"].values[None, :], (D, N)) - s["cv"].values[None, :] ** 2 / 2)
    demand = base_qty * noise * np.where(promo, uplift[None, :], 1.0) * alive
    # спрос в промо уже содержит сезонность через exp_rev_dc_cat; cat_season учтён там
    demand = demand.astype(np.float32)

    # Параметры пополнения
    sup_idx = s["supplier_id"].values - 1
    on_time = sup["on_time"].values[sup_idx]
    fill = sup["fill_rate"].values[sup_idx]
    lt_mean = s["lt_mean"].values
    krasnodar = (pairs["dc_id"].values == 5)
    d_plan = np.maximum(base_qty[:28].mean(axis=0), 0.05)
    sigma = d_plan * s["cv"].values
    z = 1.65
    sigma_lt = 1.2  # разброс срока поставки, дни (планировщик закладывает в страховой запас)
    case = s["case_qty"].values

    # Промо известно заранее: планировщик учитывает ~70% эффекта акций в горизонте LT + пересмотр
    horizon = np.round(lt_mean).astype(int) + C.REVIEW_DAYS
    pc = np.vstack([np.zeros((1, N)), np.cumsum(promo, axis=0)])
    rows_ = np.arange(D)[:, None]
    hi = np.minimum(rows_ + 1 + horizon[None, :], D)
    lo = np.minimum(rows_ + 1, D)
    cols_ = np.arange(N)[None, :]
    promo_ahead = (pc[hi, cols_] - pc[lo, cols_]).astype(np.float32)   # дней акций впереди
    promo_extra = (uplift - 1) * 0.7

    def _rop(dp, sg):
        return dp * lt_mean + z * np.sqrt(lt_mean * sg ** 2 + dp ** 2 * sigma_lt ** 2)

    stock = np.zeros((D, N), dtype=np.float32)
    shipped = np.zeros((D, N), dtype=np.float32)
    received = np.zeros((D, N), dtype=np.float32)
    arrivals = np.zeros((D + 40, N), dtype=np.float32)
    on_order = np.zeros(N, dtype=np.float32)
    orders = []

    rop = _rop(d_plan, sigma)
    cur = (rop + d_plan * C.REVIEW_DAYS / 2).astype(np.float32)
    summer26 = (dates >= pd.Timestamp("2026-06-01")) & (dates <= pd.Timestamp("2026-09-15"))
    for t in range(D):
        # пересчёт плана раз в 4 недели по фактическим отгрузкам (планировщик не знает про промо)
        if t >= 28 and t % 28 == 0:
            recent = np.where(promo[t - 28:t], np.nan, demand[t - 28:t])
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d_new = np.nanmean(recent, axis=0)
                s_new = np.nanstd(recent, axis=0)
            d_plan = np.maximum(np.nan_to_num(d_new, nan=d_plan), 0.02)
            sigma = np.maximum(np.nan_to_num(s_new, nan=sigma), d_plan * 0.1)
            rop = _rop(d_plan, sigma)
        inc = arrivals[t]
        cur = cur + inc
        on_order = on_order - inc
        received[t] = inc
        sh = np.minimum(demand[t], cur)
        cur = cur - sh
        shipped[t] = sh
        stock[t] = cur

        position = cur + on_order
        rop_t = rop + d_plan * promo_extra * promo_ahead[t]
        need = alive[min(t + 7, D - 1)] & (position < rop_t)
        if need.any():
            idx = np.nonzero(need)[0]
            target = rop_t[idx] + d_plan[idx] * C.REVIEW_DAYS
            qty = np.ceil(np.maximum(target - position[idx], 0) / case[idx]) * case[idx]
            idx, qty = idx[qty > 0], qty[qty > 0]
            if len(idx):
                ot = on_time[idx].copy()
                if summer26[t]:
                    ot = np.where(krasnodar[idx], ot - 0.22, ot)
                lt_prom = np.maximum(1, np.round(lt_mean[idx] * np.exp(rng.normal(0, 0.15, len(idx))))).astype(int)
                late = rng.random(len(idx)) > ot
                delay = np.where(late, rng.integers(1, 6, len(idx)), 0)
                lt_fact = lt_prom + delay
                fr = np.clip(rng.normal(fill[idx], 0.04), 0.5, 1.0)
                fr = np.where(rng.random(len(idx)) < 0.9, np.maximum(fr, 0.995), fr)
                q_recv = np.floor(qty * fr)
                arr_day = t + lt_fact
                ok = arr_day < D + 40
                np.add.at(arrivals, (arr_day[ok], idx[ok]), q_recv[ok])
                on_order[idx[ok]] += q_recv[ok]
                orders.append(pd.DataFrame({
                    "series": idx, "order_day": t, "promised_day": t + lt_prom,
                    "receipt_day": np.where(arr_day < D, arr_day, -1),
                    "qty_ordered": qty, "qty_received": np.where(arr_day < D, q_recv, np.nan),
                }))

    # lost share по складу и категории, взвешенный по спросу
    lost = (demand - shipped)
    lost_dc_cat = np.zeros((D, len(dc), len(cats)), dtype=np.float32)
    dem_dc_cat = np.zeros_like(lost_dc_cat)
    for i in range(len(dc)):
        for j in range(len(cats)):
            m = (i_dc == i) & (j_cat == j)
            if m.any():
                lost_dc_cat[:, i, j] = lost[:, m].sum(axis=1)
                dem_dc_cat[:, i, j] = demand[:, m].sum(axis=1)
    lost_share = np.divide(lost_dc_cat, dem_dc_cat, out=np.zeros_like(lost_dc_cat), where=dem_dc_cat > 0)

    date_vals = dates.values.astype("datetime64[D]")
    fact = pd.DataFrame({
        "date": np.repeat(date_vals, N),
        "dc_id": np.tile(pairs["dc_id"].values, D).astype(np.int16),
        "sku_id": np.tile(s["sku_id"].values, D).astype(np.int16),
        "demand_qty": demand.ravel().round(1),
        "shipped_qty": shipped.ravel().round(1),
        "received_qty": received.ravel(),
        "stock_qty": stock.ravel().round(1),
        "promo": promo.ravel().astype(np.int8),
    })
    fact["week_start"] = fact["date"] - pd.to_timedelta(pd.DatetimeIndex(fact["date"]).dayofweek, unit="D")
    # пустые ряды (SKU ещё не введён и ничего нет) не храним
    fact = fact[(fact["demand_qty"] > 0) | (fact["stock_qty"] > 0) | (fact["received_qty"] > 0)]

    o = pd.concat(orders, ignore_index=True)
    o["dc_id"] = pairs["dc_id"].values[o["series"]]
    o["sku_id"] = s["sku_id"].values[o["series"]]
    o["supplier_id"] = s["supplier_id"].values[o["series"]]
    o["order_date"] = date_vals[o["order_day"].values]
    o["promised_date"] = pd.to_datetime(C.DATA_START) + pd.to_timedelta(o["promised_day"], unit="D")
    o["receipt_date"] = pd.NaT
    got = o["receipt_day"] >= 0
    o.loc[got, "receipt_date"] = pd.to_datetime(C.DATA_START) + pd.to_timedelta(o.loc[got, "receipt_day"], unit="D")
    o["unit_cost"] = s["unit_cost"].values[o["series"]]
    o.insert(0, "order_id", np.arange(1, len(o) + 1))
    o = o[["order_id", "supplier_id", "dc_id", "sku_id", "order_date", "promised_date", "receipt_date",
           "qty_ordered", "qty_received", "unit_cost"]]
    o["order_date"] = pd.to_datetime(o["order_date"])
    # Производные поля, чтобы SQL был одинаковым в DuckDB и MySQL
    o["lead_time_days"] = (o["receipt_date"] - o["order_date"]).dt.days
    o["delay_days"] = (o["receipt_date"] - o["promised_date"]).dt.days.clip(lower=0)
    o["on_time"] = (o["receipt_date"] <= o["promised_date"]).astype("Int8").where(o["receipt_date"].notna())
    o["in_full"] = (o["qty_received"] >= o["qty_ordered"] * 0.98).astype("Int8").where(o["receipt_date"].notna())
    o["otif"] = (o["on_time"] * o["in_full"]).astype("Int8")
    o["cost_received"] = (o["qty_received"] * o["unit_cost"]).round(0)
    return fact, o, lost_share


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------
def generate(out_dir: Path, progress=None) -> dict[str, int]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    def say(msg: str, frac: float):
        if progress:
            progress(msg, frac)

    rng = np.random.default_rng(C.SEED)
    dc, cats, stores, sup, sku = _dims(rng)
    dates = pd.date_range(C.DATA_START, C.DATA_END, freq="D")
    D, S, NC = len(dates), len(stores), len(cats)
    open_mask = dates.values[:, None] >= stores["open_date"].values[None, :]
    ctx = Ctx(rng, dates, stores, cats, open_mask)

    say("Трафик и средний чек магазинов", 0.05)
    expected_checks, checks, avg_check = _store_day(ctx)
    revenue_sd = checks * avg_check

    # Доли категорий [S, C] с индивидуальностью магазина
    share_chain = np.where(stores["chain"].to_numpy(dtype=object)[:, None] == "G",
                           cats["share_G"].values[None, :], cats["share_P"].values[None, :])
    store_cat = share_chain * np.exp(rng.normal(0, 0.12, (S, NC)))
    season = _cat_season(cats, dates)  # [D, C]

    # Ожидаемая выручка категорий по складам — для калибровки спроса на складах
    exp_rev_sd = expected_checks * avg_check
    dc_pos = {d: i for i, d in enumerate(dc["dc_id"])}
    st_dc = stores["dc_id"].map(dc_pos).values
    shares_n = store_cat / store_cat.sum(axis=1, keepdims=True)
    exp_rev_dc_cat = np.zeros((D, len(dc), NC), dtype=np.float64)
    for k in range(len(dc)):
        m = st_dc == k
        if m.any():
            exp_rev_dc_cat[:, k, :] = exp_rev_sd[:, m] @ shares_n[m]
    exp_rev_dc_cat *= (season / season.mean(axis=0, keepdims=True))[:, None, :]

    say("Симуляция запасов на складах по SKU", 0.15)
    fact_sku_dc, orders, lost_share = _simulate_dc(ctx, dc, sku, sup, exp_rev_dc_cat.astype(np.float32))
    fact_sku_dc.to_parquet(out_dir / "fact_sku_dc.parquet", index=False)
    orders.to_parquet(out_dir / "fact_supplier_order.parquet", index=False)

    say("Продажи, остатки и списания по категориям", 0.55)
    margin = cats["margin"].values
    writeoff = cats["writeoff"].values
    promo_share = cats["promo_share"].values
    disc = np.where(stores["chain"].to_numpy(dtype=object) == "G", C.CHAINS["G"]["promo_discount"], C.CHAINS["P"]["promo_discount"])
    norm = cats["norm_days"].values.astype(float)
    store_norm = norm[None, :] * np.exp(rng.normal(0, 0.18, (S, NC)))
    base_oos = np.where(cats["fresh"].to_numpy(dtype=bool), 0.035, 0.02)
    unit_price = np.array([120, 95, 330, 55, 110, 140, 95, 520, 260, 210, 230, 380], dtype=float)
    store_margin = np.exp(rng.normal(0, 0.04, S))
    week = ((dates - dates[0]).days.values // 7)
    promo_int = np.exp(rng.normal(0, 0.3, (week.max() + 1, NC)))
    t_years = ((dates - dates[0]).days.values / 365.25)

    aug = np.asarray(dates.month == 8)
    veg = 0  # индекс «Овощи и фрукты»

    writer = None
    cov_state = np.log(np.ones((S, NC)))
    cogs_hist = None
    chunk = 31
    for a in range(0, D, chunk):
        b = min(a + chunk, D)
        n = b - a
        w = store_cat[None, :, :] * season[a:b, None, :] * np.exp(rng.normal(0, 0.05, (n, S, NC)))
        w /= w.sum(axis=2, keepdims=True)
        rev = revenue_sd[a:b, :, None] * w
        p_share = np.clip(promo_share[None, None, :] * promo_int[week[a:b]][:, None, :]
                          * np.exp(rng.normal(0, 0.1, (n, S, NC))), 0, 0.7)
        promo_rev = rev * p_share
        m_eff = margin[None, None, :] * store_margin[None, :, None]
        cost = (rev - promo_rev) * (1 - m_eff) + promo_rev * (1 - (m_eff - disc[None, :, None]) / (1 - disc[None, :, None]))
        qty = rev / (unit_price[None, None, :] * (1 + C.PRICE_INFLATION) ** t_years[a:b, None, None])
        wo_rate = writeoff[None, None, :] * np.exp(rng.normal(0, 0.25, (n, S, NC)))
        wo_rate[aug[a:b], :, veg] *= 1.7   # сценарий 3
        wo = rev * wo_rate * (1 - margin[None, None, :])

        # Остатки: покрытие (в днях) — процесс AR(1) вокруг норматива магазина
        cov = np.empty((n, S, NC), dtype=np.float32)
        for k in range(n):
            cov_state = 0.9 * cov_state + rng.normal(0, 0.06, (S, NC))
            cov[k] = store_norm * np.exp(cov_state)
        cogs_ma = cost if cogs_hist is None else np.concatenate([cogs_hist, cost], axis=0)
        roll = pd.DataFrame(cogs_ma.reshape(len(cogs_ma), -1)).rolling(7, min_periods=1).mean().values
        roll = roll[-n:].reshape(n, S, NC)
        stock_cost = roll * cov
        cogs_hist = cost[-6:]

        dc_lost = lost_share[a:b][:, st_dc, :]   # [n, S, C]
        oos = np.clip(base_oos[None, None, :] * (store_norm[None] / cov) ** 1.2 + 0.5 * dc_lost
                      + rng.normal(0, 0.004, (n, S, NC)), 0, 0.6)

        open_ = open_mask[a:b][:, :, None] & np.ones((1, 1, NC), dtype=bool)
        df = pd.DataFrame({
            "date": np.repeat(dates.values[a:b].astype("datetime64[D]"), S * NC),
            "store_id": np.tile(np.repeat(stores["store_id"].values, NC), n).astype(np.int16),
            "category_id": np.tile(cats["category_id"].values, n * S).astype(np.int8),
            "revenue": rev.ravel().round(0),
            "cost": cost.ravel().round(0),
            "qty": qty.ravel().round(0),
            "promo_revenue": promo_rev.ravel().round(0),
            "writeoff": wo.ravel().round(0),
            "stock_cost": stock_cost.ravel().round(0),
            "oos_share": oos.ravel().astype(np.float32).round(4),
        })
        df = df[open_.ravel()]
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / "fact_sales.parquet", table.schema, compression="zstd")
        writer.write_table(table)
        say("Продажи, остатки и списания по категориям", 0.55 + 0.3 * b / D)
    writer.close()

    say("Чеки, план и доставка в магазины", 0.88)
    loyalty = stores["chain"].map({k: v["loyalty_share"] for k, v in C.CHAINS.items()}).values
    loy = np.round(checks * np.clip(loyalty[None, :] + rng.normal(0, 0.02, checks.shape), 0, 1))
    deliv_orders = np.zeros_like(checks)
    hd = stores["has_delivery"].values
    if hd.any():
        per_store = C.CHAINS["G"]["delivery_orders_day"] / hd.sum()
        growth_d = (1.4) ** t_years  # ФАКТ: рост ~40% в год
        lam = per_store * growth_d[:, None] * WEEKDAY_TRAFFIC[dates.dayofweek.values][:, None] / growth_d[-1]
        deliv_orders[:, hd] = rng.poisson(np.broadcast_to(lam, (D, int(hd.sum()))))
    deliv_orders = np.where(open_mask, deliv_orders, 0)
    sd = pd.DataFrame({
        "date": np.repeat(dates.values.astype("datetime64[D]"), S),
        "store_id": np.tile(stores["store_id"].values, D).astype(np.int16),
        "checks": checks.ravel().astype(np.int32),
        "loyalty_checks": loy.ravel().astype(np.int32),
        "delivery_orders": deliv_orders.ravel().astype(np.int32),
        "revenue": revenue_sd.ravel().round(0),
    })
    sd = sd[open_mask.ravel()]
    sd.to_parquet(out_dir / "fact_store_day.parquet", index=False)

    # План: ожидаемая выручка по модели × бюджетный «оптимизм»
    exp_df = pd.DataFrame(exp_rev_sd, index=dates, columns=stores["store_id"].values)
    month_plan = exp_df.resample("MS").sum()
    covered = exp_df.iloc[:, 0].resample("MS").count()
    month_plan = month_plan.mul(month_plan.index.days_in_month / covered, axis=0)  # неполный месяц -> полный план
    optimism = np.exp(rng.normal(0.025, 0.03, month_plan.shape))
    month_plan = month_plan * optimism
    plan = month_plan.stack().rename("revenue_plan").reset_index()
    plan.columns = ["month", "store_id", "revenue_plan"]
    plan = plan[plan["revenue_plan"] > 0]
    plan["revenue_plan"] = plan["revenue_plan"].round(0)
    plan.to_parquet(out_dir / "fact_plan.parquet", index=False)

    # Рейсы склад → магазин
    prob = np.where(stores["chain"].to_numpy(dtype=object) == "G", 0.9, 0.82)
    has = (rng.random((D, S)) < prob[None, :]) & open_mask
    di, si = np.nonzero(has)
    n_del = len(di)
    dc_ids = stores["dc_id"].values[si]
    plan_ts = (dates.values[di].astype("datetime64[m]")
               + (stores["slot_hour"].values[si] * 60 + rng.integers(-20, 21, n_del)).astype("timedelta64[m]"))
    mix = rng.random(n_del)
    delay = np.where(mix < 0.87, rng.normal(0, 12, n_del),
             np.where(mix < 0.97, rng.uniform(30, 120, n_del), rng.uniform(120, 360, n_del)))
    kr_summer = (dc_ids == 5) & (dates.values[di] >= np.datetime64("2026-06-01")) & (dates.values[di] <= np.datetime64("2026-09-15"))
    delay = np.where(kr_summer & (rng.random(n_del) < 0.25), delay + rng.uniform(60, 240, n_del), delay)
    act_ts = plan_ts + np.round(delay).astype("timedelta64[m]")
    daily_cost = (revenue_sd * 0.74)[di, si]
    shipped_cost = daily_cost / prob[si] * np.exp(rng.normal(0, 0.15, n_del))
    loss = np.where(rng.random(n_del) < 0.03, rng.uniform(0.01, 0.06, n_del), rng.uniform(0, 0.004, n_del))
    delv = pd.DataFrame({
        "delivery_id": np.arange(1, n_del + 1, dtype=np.int32),
        "date": dates.values[di].astype("datetime64[D]"),
        "dc_id": dc_ids.astype(np.int16),
        "store_id": stores["store_id"].values[si].astype(np.int16),
        "planned_ts": plan_ts.astype("datetime64[s]"),
        "actual_ts": act_ts.astype("datetime64[s]"),
        "cost_shipped": shipped_cost.round(0),
        "cost_received": (shipped_cost * (1 - loss)).round(0),
    })
    delv["delay_min"] = np.round(delay).astype(np.int32)
    delv["on_time"] = (delv["delay_min"] <= 30).astype(np.int8)
    delv["planned_hour"] = pd.DatetimeIndex(delv["planned_ts"]).hour.astype(np.int8)
    delv.to_parquet(out_dir / "fact_delivery.parquet", index=False)

    say("Справочники", 0.97)
    dc.to_parquet(out_dir / "dim_dc.parquet", index=False)
    cats_out = cats.drop(columns=["share_G", "share_P", "season_amp", "peak_month", "skus"])
    cats_out.to_parquet(out_dir / "dim_category.parquet", index=False)
    stores_out = stores.drop(columns=["size_factor", "growth", "price_factor", "slot_hour"])
    stores_out.to_parquet(out_dir / "dim_store.parquet", index=False)
    sup[["supplier_id", "supplier"]].to_parquet(out_dir / "dim_supplier.parquet", index=False)
    sku_out = sku[["sku_id", "sku", "category_id", "supplier_id", "price", "unit_cost", "case_qty",
                   "private_label", "lifecycle", "in_G", "in_P"]]
    sku_out.to_parquet(out_dir / "dim_sku.parquet", index=False)

    (out_dir / "VERSION").write_text(f"{VERSION}:{C.SEED}:{C.DATA_START}:{C.DATA_END}")
    say("Готово", 1.0)
    counts = {t: pq.ParquetFile(out_dir / f"{t}.parquet").metadata.num_rows for t in TABLES}
    counts["_seconds"] = round(time.time() - t0, 1)
    return counts


def is_ready(out_dir: Path) -> bool:
    v = out_dir / "VERSION"
    return v.exists() and v.read_text() == f"{VERSION}:{C.SEED}:{C.DATA_START}:{C.DATA_END}" and all(
        (out_dir / f"{t}.parquet").exists() for t in TABLES)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "generated"
    print(generate(target, progress=lambda m, f: print(f"{f:4.0%} {m}", flush=True)))
