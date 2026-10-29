"""Аналитические алгоритмы «AI-помощника» (статистика, без генеративной модели).

- anomalies:        резкие всплески и падения дневного ряда относительно того же дня недели
                    (робастное отклонение) + факторы: трафик, чек, промо, OSA, календарь;
- factor_links:     связь отклонений РТО с отклонениями факторов (корреляция по всем дням);
- store_outliers:   магазины с аномально низким (высоким) LFL / OSA относительно своего дивизиона;
- negative_trends:  объекты с устойчиво снижающимся индексом РТО к прошлому году;
- top_factors:      разложение изменения РТО к прошлому году на факторы (топ-5);
- price_scenario:   «что если» изменить цену категории (ценовая эластичность — допущение).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOLIDAYS = {(12, 28): "предновогодние дни", (12, 29): "предновогодние дни", (12, 30): "предновогодние дни",
            (12, 31): "31 декабря", (1, 1): "1 января", (3, 7): "канун 8 Марта", (3, 8): "8 Марта",
            (2, 22): "канун 23 февраля"}


def _robust_z(x: pd.Series, window: int = 91) -> pd.Series:
    med = x.rolling(window, min_periods=28).median()
    mad = (x - med).abs().rolling(window, min_periods=28).median() * 1.4826
    return (x - med) / mad.replace(0, np.nan)


def anomalies(daily: pd.DataFrame, z_thr: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """daily: date, revenue, checks, store_revenue, promo_revenue, oos_w (суммы за день по выборке).
    Возвращает (ряд с базой и z-оценкой, список аномалий с факторами)."""
    d = daily.sort_values("date").set_index("date").copy()
    d["avg_check"] = d["store_revenue"] / d["checks"]
    d["promo"] = d["promo_revenue"] / d["revenue"]
    d["osa"] = 1 - d["oos_w"] / d["revenue"]
    d["wd"] = d.index.dayofweek
    base_cols = {"revenue": "base", "checks": "base_checks", "avg_check": "base_check", "promo": "base_promo",
                 "osa": "base_osa"}
    for col, b in base_cols.items():
        # база: медиана того же дня недели за 8 предыдущих недель
        d[b] = d.groupby("wd")[col].transform(lambda s: s.shift(1).rolling(8, min_periods=4).median())
    d["resid"] = np.log(d["revenue"] / d["base"])
    d["z"] = _robust_z(d["resid"])
    out = d[d["z"].abs() >= z_thr].copy()
    rows = []
    for day, r in out.iterrows():
        f_traffic = r["checks"] / r["base_checks"] - 1
        f_check = r["avg_check"] / r["base_check"] - 1
        f_promo = r["promo"] - r["base_promo"]
        f_osa = r["osa"] - r["base_osa"]
        hol = HOLIDAYS.get((day.month, day.day), "")
        cands = [("трафик (чеки)", f_traffic, abs(f_traffic)), ("средний чек", f_check, abs(f_check)),
                 ("доля промо", f_promo, abs(f_promo) * 2), ("доступность на полке (OSA)", f_osa, abs(f_osa) * 4)]
        main = max(cands, key=lambda c: c[2])
        rows.append({"date": day, "revenue": r["revenue"], "base": r["base"], "deviation": r["revenue"] / r["base"] - 1,
                     "z": r["z"], "traffic": f_traffic, "check": f_check, "promo_pp": f_promo, "osa_pp": f_osa,
                     "holiday": hol, "main_factor": f"календарь: {hol}" if hol else main[0]})
    return d.reset_index(), pd.DataFrame(rows)


def factor_links(series: pd.DataFrame) -> pd.DataFrame:
    """Корреляция отклонений РТО от базы с отклонениями факторов по всем дням ряда."""
    s = series.dropna(subset=["resid", "base_checks", "base_check", "base_promo", "base_osa"]).copy()
    if len(s) < 30:
        return pd.DataFrame()
    facts = {
        "Трафик (чеки)": np.log(s["checks"] / s["base_checks"]),
        "Средний чек": np.log(s["avg_check"] / s["base_check"]),
        "Доля промо": s["promo"] - s["base_promo"],
        "OSA": s["osa"] - s["base_osa"],
    }
    rows = []
    for name, v in facts.items():
        c = np.corrcoef(s["resid"], v)[0, 1]
        rows.append({"Фактор": name, "Корреляция с отклонением РТО": c,
                     "Сила связи": "сильная" if abs(c) >= 0.6 else "умеренная" if abs(c) >= 0.3 else "слабая"})
    return pd.DataFrame(rows).sort_values("Корреляция с отклонением РТО", key=np.abs, ascending=False)


def store_outliers(st_df: pd.DataFrame, z_thr: float = 2.5) -> pd.DataFrame:
    """Магазины с аномальным LFL относительно магазинов того же дивизиона (робастная z-оценка)."""
    x = st_df[st_df["comparable"] & st_df["revenue_py"].gt(0)].copy()
    if len(x) < 8:
        return pd.DataFrame()
    x["lfl"] = x["revenue"] / x["revenue_py"] - 1
    g = x.groupby(["chain", "division"])["lfl"]
    x["peer_median"] = g.transform("median")
    x["peer_mad"] = g.transform(lambda s: (s - s.median()).abs().median() * 1.4826)
    x["z"] = (x["lfl"] - x["peer_median"]) / x["peer_mad"].replace(0, np.nan)
    x = x[x["z"].abs() >= z_thr].copy()
    x["direction"] = np.where(x["z"] < 0, "ниже", "выше")
    return x.sort_values("z")


def negative_trends(weekly: pd.DataFrame, key: str, weeks: int = 13, slope_thr: float = -0.004) -> pd.DataFrame:
    """weekly: key, week, revenue, revenue_py. Индекс к прошлому году; наклон линейной регрессии за weeks недель."""
    rows = []
    for k, g in weekly.groupby(key):
        g = g.sort_values("week").tail(weeks)
        g = g[g["revenue_py"] > 0]
        if len(g) < max(6, weeks // 2):
            continue
        idx = (g["revenue"] / g["revenue_py"]).values
        t = np.arange(len(idx))
        slope = np.polyfit(t, idx, 1)[0]
        first, last = idx[:4].mean(), idx[-4:].mean()
        if slope <= slope_thr and last < first:
            rows.append({key: k, "slope_pp_week": slope, "index_start": first, "index_end": last,
                         "drop_pp": last - first, "spark": list(np.round(idx, 3))})
    return pd.DataFrame(rows).sort_values("slope_pp_week") if rows else pd.DataFrame()


def top_factors(st_df: pd.DataFrame, cat_cur: pd.DataFrame, cat_py: pd.DataFrame, cat_names: pd.Series,
                osa_substitution: float = 0.5) -> pd.DataFrame:
    """Разложение изменения РТО к прошлому году. Эффекты в ₽; отрицательные — факторы снижения."""
    rows = []
    s = st_df.copy()
    comp = s[s["comparable"] & s["revenue_py"].notna()]
    new = s[~s["comparable"]]
    rows.append(("Новые магазины", new["revenue"].sum() - new["revenue_py"].fillna(0).sum(),
                 f"{len(new)} магазинов открыты менее года назад"))
    # трафик и чек сопоставимых магазинов
    c_chk, p_chk = comp["checks"].sum(), comp["checks_py"].sum()
    if p_chk > 0 and c_chk > 0:
        c_avg = comp["store_revenue"].sum() / c_chk
        p_avg = comp["revenue_py"].sum() / p_chk
        rows.append(("Трафик сопоставимых магазинов", (c_chk - p_chk) * p_avg, f"чеки {c_chk / p_chk - 1:+.1%}"))
        rows.append(("Средний чек сопоставимых магазинов", (c_avg - p_avg) * c_chk, f"чек {c_avg / p_avg - 1:+.1%}"))
    # дивизионы (LFL)
    for (ch, dv), g in comp.groupby(["chain_name", "division"]):
        eff = g["revenue"].sum() - g["revenue_py"].sum()
        lfl = g["revenue"].sum() / g["revenue_py"].sum() - 1
        rows.append((f"Дивизион «{dv}» ({str(ch).split(' · ')[0]})", eff, f"LFL {lfl:+.1%}"))
    # категории
    cc = cat_cur.merge(cat_py[["category_id", "revenue"]].rename(columns={"revenue": "py"}), on="category_id")
    for _, r in cc.iterrows():
        rows.append((f"Категория «{cat_names.get(r['category_id'], r['category_id'])}»", r["revenue"] - r["py"],
                     f"{r['revenue'] / r['py'] - 1:+.1%} г/г"))
    # доступность на полке (OSA): упущенные продажи
    if {"oos_w"}.issubset(cat_cur.columns) and "oos_w" in cat_py.columns:
        osa_c = 1 - cat_cur["oos_w"].sum() / cat_cur["revenue"].sum()
        osa_p = 1 - cat_py["oos_w"].sum() / cat_py["revenue"].sum()
        eff = (osa_c - osa_p) * cat_cur["revenue"].sum() * osa_substitution
        rows.append(("Доступность на полке (OSA)", eff, f"OSA {osa_c:.1%} против {osa_p:.1%}; "
                                                       f"доля потерянного спроса — допущение {osa_substitution:.0%}"))
    df = pd.DataFrame(rows, columns=["Фактор", "Эффект, ₽", "Комментарий"])
    df["Группа"] = np.where(df["Фактор"].str.startswith("Дивизион"), "Регионы",
                   np.where(df["Фактор"].str.startswith("Категория"), "Категории", "Драйверы"))
    return df


def price_scenario(base_rev: float, base_cost: float, price_change: float, elasticity: float) -> dict:
    """Изменение цены на price_change (доля) при эластичности спроса elasticity."""
    vol = (1 + price_change) ** elasticity - 1
    rev = base_rev * (1 + price_change) * (1 + vol)
    cost = base_cost * (1 + vol)
    return {"volume": vol, "revenue": rev, "d_revenue": rev - base_rev,
            "gross": rev - cost, "d_gross": (rev - cost) - (base_rev - base_cost),
            "margin_new": (rev - cost) / rev if rev else np.nan, "margin_old": (base_rev - base_cost) / base_rev}


ELASTICITY = {1: -1.0, 2: -0.7, 3: -0.9, 4: -0.4, 5: -1.1, 6: -1.4, 7: -1.3, 8: -1.6, 9: -1.2, 10: -0.8,
              11: -1.2, 12: -1.8}   # ДОПУЩЕНИЕ: типовые значения ценовой эластичности по категориям
