"""Уведомления: правила, проверка текущих значений и прогноза, история срабатываний.

Каналы доставки в прототипе — только интерфейс BI. E-mail, Telegram, MAX, NextCloud Talk и задачи
в Битрикс24 подключаются в целевой системе через API (в правиле хранится выбранный канал).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from bi import ai
from bi import queries as Q

KINDS = {
    "osa_store": ("OSA магазина ниже порога, последние 7 дней", 0.92, "%"),
    "plan_forecast": ("Прогноз выполнения плана месяца ниже порога (дивизион)", 0.97, "%"),
    "sales_anomaly": ("Аномальное отклонение РТО от обычного уровня, последние 7 дней", 3.0, "z"),
    "stock_excess": ("Запас категории в регионе выше норматива в N раз, последние 7 дней", 1.3, "×"),
    "otif_supplier": ("OTIF поставщика ниже порога, 28 дней", 0.70, "%"),
    "defect_supplier": ("Брак при приёмке у поставщика выше порога, 28 дней", 0.02, "%"),
    "cert_expiry": ("Декларация или сертификат истекает в течение N дней", 30, "дн."),
}
CHANNELS = ["Интерфейс BI", "E-mail (целевая система)", "Telegram (целевая система)", "MAX (целевая система)",
            "Задача в Битрикс24 (целевая система)"]


def fmt_value(v: float, unit: str) -> str:
    if unit == "%":
        return f"{v * 100:.1f}%".replace(".", ",")
    if unit == "×":
        return f"×{v:.2f}".replace(".", ",")
    if unit == "z":
        return f"z = {v:+.1f}".replace(".", ",")
    return f"{v:.0f} {unit}"


def _sev(value: float, thr: float, below: bool) -> str:
    gap = (thr - value) if below else (value - thr)
    rel = gap / abs(thr) if thr else gap
    return "критично" if rel > 0.1 else "внимание"


def evaluate(kind: str, threshold: float, f: Q.Filters) -> pd.DataFrame:
    """Срабатывания правила на конец выбранного периода."""
    end = f.end
    d = Q.dims()
    stores = d["store"].set_index("store_id")
    rows = []
    if kind == "osa_store":
        s = Q.sales_by(f.with_dates(end - timedelta(days=6), end), "store_id")
        s["osa"] = 1 - s["oos_w"] / s["revenue"]
        for _, r in s[s["osa"] < threshold].iterrows():
            st_ = stores.loc[int(r["store_id"])]
            rows.append((st_["store_name"], f"{st_['division']} · {st_['ugm']}", r["osa"], _sev(r["osa"], threshold, True)))
    elif kind == "plan_forecast":
        m0 = end.replace(day=1)
        days_in = pd.Timestamp(end).days_in_month
        fm = f.with_dates(m0, end)
        tr = Q.traffic_by_store(fm)[["store_id", "revenue"]]
        last7 = Q.traffic_by_store(f.with_dates(max(m0, end - timedelta(days=6)), end))[["store_id", "revenue"]]
        n7 = (end - max(m0, end - timedelta(days=6))).days + 1
        plan = Q.plan_monthly(f.with_dates(m0, (pd.Timestamp(m0) + pd.offsets.MonthEnd(0)).date()))
        x = tr.merge(last7.rename(columns={"revenue": "r7"}), on="store_id", how="left") \
              .merge(plan.groupby("store_id", as_index=False)["plan"].sum(), on="store_id", how="inner")
        x["division"] = x["store_id"].map(stores["division"])
        remaining = days_in - end.day
        x["fc"] = x["revenue"] + x["r7"].fillna(0) / n7 * remaining
        g = x.groupby("division")[["fc", "plan"]].sum()
        g["v"] = g["fc"] / g["plan"]
        for dv, r in g[g["v"] < threshold].iterrows():
            rows.append((f"Дивизион «{dv}»", f"прогноз на {pd.Timestamp(end).strftime('%m.%Y')}", r["v"],
                         _sev(r["v"], threshold, True)))
    elif kind == "sales_anomaly":
        s = Q.sales_daily(f.with_dates(max(Q.data_bounds()[0], end - timedelta(days=140)), end))
        t = Q.traffic_daily(f.with_dates(max(Q.data_bounds()[0], end - timedelta(days=140)), end))
        if len(s) > 60:
            daily = s.merge(t[["date", "checks", "revenue"]].rename(columns={"revenue": "store_revenue"}), on="date")
            _, an = ai.anomalies(daily, z_thr=threshold)
            if len(an):
                an = an[an["date"] >= pd.Timestamp(end - timedelta(days=6))]
                for _, r in an.iterrows():
                    rows.append((f"РТО за {r['date']:%d.%m.%Y}", f"{r['deviation']:+.1%} к обычному уровню; "
                                 f"фактор: {r['main_factor']}", r["z"], "критично" if abs(r["z"]) > threshold * 1.5
                                 else "внимание"))
    elif kind == "stock_excess":
        sc = Q.stock_by_attr_daily(f.with_dates(end - timedelta(days=6), end), "region")
        g = sc.groupby(["attr", "category_id"])[["stock_cost", "cost"]].sum().reset_index()
        g["cover"] = g["stock_cost"] / g["cost"]
        g["norm"] = g["category_id"].map(d["category"].set_index("category_id")["norm_days"])
        g["v"] = g["cover"] / g["norm"]
        cats = d["category"].set_index("category_id")["category"]
        for _, r in g[g["v"] > threshold].iterrows():
            rows.append((f"{cats[r['category_id']]} · {r['attr']}", f"запас {r['cover']:.1f} дн. при нормативе "
                         f"{r['norm']:.0f}", r["v"], _sev(r["v"], threshold, False)))
    elif kind in ("otif_supplier", "defect_supplier"):
        o = Q.orders_by(f.with_dates(end - timedelta(days=27), end), "supplier_id")
        sup = d["supplier"].set_index("supplier_id")["supplier"]
        if kind == "otif_supplier":
            for _, r in o[o["otif"] < threshold].iterrows():
                rows.append((sup[int(r["supplier_id"])], f"{int(r['orders'])} заказов", r["otif"],
                             _sev(r["otif"], threshold, True)))
        else:
            o["v"] = o["qty_defect"] / (o["qty_received"] + o["qty_defect"])
            for _, r in o[o["v"] > threshold].iterrows():
                rows.append((sup[int(r["supplier_id"])], f"{int(r['orders'])} заказов", r["v"],
                             _sev(r["v"], threshold, False)))
    elif kind == "cert_expiry":
        c = Q.certificates(f)
        c = c[c["lifecycle"] != "delisted"]
        days = (c["valid_to"] - pd.Timestamp(end)).dt.days
        c = c.assign(days=days)
        c = c[c["days"] <= threshold]
        for _, r in c.sort_values("days").iterrows():
            state_ = "истёк" if r["days"] < 0 else f"истекает через {int(r['days'])} дн."
            rows.append((r["sku"], f"{r['doc_type']} {r['doc_number']}: {state_}", float(r["days"]),
                         "критично" if r["days"] < 7 else "внимание"))
    return pd.DataFrame(rows, columns=["Объект", "Детали", "value", "Важность"])


@st.cache_data(ttl=Q.TTL, show_spinner=False)
def evaluate_all(f: Q.Filters, rules: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in rules[rules["active"] == 1].iterrows():
        try:
            ev = evaluate(r["kind"], float(r["threshold"]), f)
        except Exception as e:   # правило не должно ронять страницу
            ev = pd.DataFrame([("Ошибка проверки правила", str(e)[:120], np.nan, "внимание")],
                              columns=["Объект", "Детали", "value", "Важность"])
        if len(ev):
            unit = KINDS[r["kind"]][2]
            ev["Правило"] = r["name"]
            ev["kind"] = r["kind"]
            ev["Значение"] = ev["value"].map(lambda v: fmt_value(v, unit) if pd.notna(v) else "—")
            ev["Порог"] = fmt_value(float(r["threshold"]), unit)
            ev["Канал"] = r["channel"]
            out.append(ev)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["Объект", "Детали", "value", "Важность", "Правило", "kind", "Значение", "Порог", "Канал"])


@st.cache_data(ttl=Q.TTL, show_spinner=False)
def history(f: Q.Filters, rules: pd.DataFrame, weeks: int = 26) -> pd.DataFrame:
    """Срабатывания по неделям за прошлые периоды (для анализа частоты и причин)."""
    end = f.end
    start = end - timedelta(days=7 * weeks - 1)
    d = Q.dims()
    stores = d["store"].set_index("store_id")
    rows = []
    active = rules[rules["active"] == 1]
    for _, r in active.iterrows():
        kind, thr = r["kind"], float(r["threshold"])
        if kind == "osa_store":
            s = Q.sales_daily_by_store(f.with_dates(start, end))
            s["week"] = s["date"] - pd.to_timedelta(s["date"].dt.dayofweek, unit="D")
            g = s.groupby(["week", "store_id"])[["revenue", "oos_w"]].sum().reset_index()
            g["v"] = 1 - g["oos_w"] / g["revenue"]
            hit = g[g["v"] < thr]
            for _, h in hit.iterrows():
                rows.append((h["week"], r["name"], kind, stores.loc[int(h["store_id"]), "store_name"],
                             stores.loc[int(h["store_id"]), "division"], h["v"]))
        elif kind == "stock_excess":
            sc = Q.stock_by_attr_daily(f.with_dates(start, end), "region")
            sc["week"] = sc["date"] - pd.to_timedelta(sc["date"].dt.dayofweek, unit="D")
            g = sc.groupby(["week", "attr", "category_id"])[["stock_cost", "cost"]].sum().reset_index()
            g["v"] = g["stock_cost"] / g["cost"] / g["category_id"].map(d["category"].set_index("category_id")["norm_days"])
            cats = d["category"].set_index("category_id")["category"]
            for _, h in g[g["v"] > thr].iterrows():
                rows.append((h["week"], r["name"], kind, f"{cats[h['category_id']]} · {h['attr']}", h["attr"], h["v"]))
        elif kind in ("otif_supplier", "defect_supplier"):
            o = Q.orders_daily_by_supplier(f.with_dates(start, end))
            o["week"] = o["date"] - pd.to_timedelta(o["date"].dt.dayofweek, unit="D")
            g = o.groupby(["week", "supplier_id"])[["orders", "otif_n", "qty_received", "qty_defect"]].sum().reset_index()
            g = g[g["orders"] >= 20]
            sup = d["supplier"].set_index("supplier_id")["supplier"]
            if kind == "otif_supplier":
                g["v"] = g["otif_n"] / g["orders"]
                hit = g[g["v"] < thr]
            else:
                g["v"] = g["qty_defect"] / (g["qty_received"] + g["qty_defect"])
                hit = g[g["v"] > thr]
            for _, h in hit.iterrows():
                rows.append((h["week"], r["name"], kind, sup[int(h["supplier_id"])], "Поставщики", h["v"]))
    df = pd.DataFrame(rows, columns=["week", "Правило", "kind", "Объект", "Группа", "value"])
    if df.empty:
        return df
    # результат: закрыто, если на следующей неделе правило по объекту не сработало
    df = df.sort_values("week")
    nxt = set(zip(df["Объект"], df["week"] - pd.Timedelta(days=7)))
    last_week = df["week"].max()
    df["Результат"] = [("активно" if w == last_week else "повторилось" if (o, w) in nxt else "закрыто")
                       for o, w in zip(df["Объект"], df["week"])]
    return df
