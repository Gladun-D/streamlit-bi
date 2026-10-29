"""Сравнение периодов: два периода на одном экране (разделённый экран), показатели и отклонения по разрезу."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

PAGE = "compare"
DIMS = {"Дивизион": "division", "Федеральный округ": "fed_district", "РЦ": "dc_name", "Регион": "region",
        "Категория": "category"}


def _period_picker(col, key: str, default: tuple[date, date], f: Q.Filters) -> tuple[date, date]:
    d0, d1 = Q.data_bounds()
    opts = ["Как в фильтре", "Год назад", "Предыдущий", "Свои даты"]
    mode = col.segmented_control("Период", opts, default=opts[0] if key == "A" else opts[1], key=f"cmp_mode_{key}",
                                 help="«Год назад» — те же дни недели год назад (−364 дня); «Предыдущий» — период "
                                      "той же длины непосредственно перед выбранным") \
        or opts[0]
    if mode == "Как в фильтре":
        return f.start, f.end
    if mode == "Год назад":
        s, e = f.start - timedelta(days=364), f.end - timedelta(days=364)
        return max(s, d0), max(e, d0)
    if mode == "Предыдущий":
        e = f.start - timedelta(days=1)
        return max(e - timedelta(days=f.days - 1), d0), max(e, d0)
    rng = col.date_input("Даты", value=default, min_value=d0, max_value=d1, format="DD.MM.YYYY", key=f"cmp_rng_{key}")
    return tuple(rng) if isinstance(rng, (tuple, list)) and len(rng) == 2 else default


def render(f: Q.Filters):
    ui.page_header("Сравнение периодов", "Два периода рядом: показатели, структура и отклонения", None, PAGE)
    ui.register_widgets(PAGE, ["Показатели A и B", "Разрез A против B", "Тренды A и B"])
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown("#### Период A")
        a0, a1 = _period_picker(ca, "A", (f.start, f.end), f)
        st.caption(f"{a0:%d.%m.%Y} — {a1:%d.%m.%Y} · {(a1 - a0).days + 1} дн.")
    with cb:
        st.markdown("#### Период B")
        b0, b1 = _period_picker(cb, "B", (f.start - timedelta(days=364), f.end - timedelta(days=364)), f)
        st.caption(f"{b0:%d.%m.%Y} — {b1:%d.%m.%Y} · {(b1 - b0).days + 1} дн.")
    fa, fb = f.with_dates(a0, a1), f.with_dates(b0, b1)
    pa, pb = common.load_period(fa), common.load_period(fb)
    if pa.sales.empty or pb.sales.empty:
        ui.empty_state("В одном из периодов нет данных")
        return
    days_a, days_b = (a1 - a0).days + 1, (b1 - b0).days + 1
    norm = st.toggle("Сравнивать РТО и чеки в среднем за день", value=days_a != days_b, key="cmp_norm",
                     help="Полезно, если периоды разной длины")
    ka, kb = pa.k, pb.k
    rows = [
        ("РТО", ka["revenue"] / (days_a if norm else 1), kb["revenue"] / (days_b if norm else 1), "rub"),
        ("Чеки", ka["checks"] / (days_a if norm else 1), kb["checks"] / (days_b if norm else 1), "num"),
        ("Средний чек", ka["avg_check"], kb["avg_check"], "rub0"),
        ("Валовая маржа", ka["margin"], kb["margin"], "pct"),
        ("OSA", 1 - ka["oos"], 1 - kb["oos"], "pct"),
        ("Доля промо", ka["promo_share"], kb["promo_share"], "pct"),
        ("Списания", ka["writeoff_rate"], kb["writeoff_rate"], "pct"),
        ("Запас в магазинах, дн.", ka["cover_days"], kb["cover_days"], "days"),
    ]
    fmt = {"rub": ui.rub, "rub0": lambda v: ui.rub(v, 0), "num": ui.big, "pct": ui.pct,
           "days": lambda v: f"{ui.num(v, 1)} дн."}
    cols = st.columns(4)
    for i, (name, va, vb, kind) in enumerate(rows):
        delta = (ui.delta_pp(va - vb, "к B") if kind == "pct" else ui.delta_pct(M.change(va, vb), "к B"))
        good = None if name in ("Доля промо", "Запас в магазинах, дн.") else (name != "Списания")
        ui.kpi(cols[i % 4], name + (" в день" if norm and name in ("РТО", "Чеки") else ""), fmt[kind](va), delta,
               good_when_up=good, help=f"Период B: {fmt[kind](vb)}")
    table = pd.DataFrame([(n, fmt[k](a), fmt[k](b), (ui.delta_pp(a - b, "") if k == "pct" else ui.delta_pct(M.change(a, b), "")))
                          for n, a, b, k in rows], columns=["Показатель", "Период A", "Период B", "A к B"])
    ui.collect_table("Показатели A и B", table)

    st.subheader("Разрез: A против B")
    dim = st.segmented_control("Разрез", list(DIMS), default="Дивизион", key="cmp_dim") or "Дивизион"
    ga, gb = _by_dim(fa, DIMS[dim]), _by_dim(fb, DIMS[dim])
    m = ga.merge(gb, on="key", how="outer", suffixes=("_a", "_b")).fillna({"revenue_a": 0, "revenue_b": 0})
    if norm:
        m["revenue_a"] /= days_a
        m["revenue_b"] /= days_b
    m["delta"] = m["revenue_a"] / m["revenue_b"] - 1
    m = m.sort_values("revenue_a", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=m["key"], x=m["revenue_b"] / 1e6, orientation="h", name="Период B",
                marker=dict(color=ui.NEUTRAL, cornerradius=3), hovertemplate="B: %{x:,.1f} млн ₽<extra></extra>")
    fig.add_bar(y=m["key"], x=m["revenue_a"] / 1e6, orientation="h", name="Период A",
                marker=dict(color=ui.SERIES[0], cornerradius=3), hovertemplate="A: %{x:,.1f} млн ₽<extra></extra>")
    fig.update_layout(title=f"РТО{' в день' if norm else ''} по разрезу «{dim}»", barmode="group",
                      xaxis=dict(title="млн ₽"))
    ui.chart(fig, max(320, 30 * len(m) + 120), key="cmp_bars", export_name="Разрез A против B")
    show = pd.DataFrame({dim: m["key"], "РТО A, млн ₽": m["revenue_a"] / 1e6, "РТО B, млн ₽": m["revenue_b"] / 1e6,
                         "A к B": m["delta"], "Маржа A": m["margin_a"], "Маржа B": m["margin_b"],
                         "OSA A": m["osa_a"], "OSA B": m["osa_b"]}).sort_values("A к B")
    mln = lambda v: f"{v:,.1f}".replace(",", " ").replace(".", ",")  # noqa: E731
    st.dataframe(ui.styled(show, {"OSA A": "osa", "OSA B": "osa"}, top_col="A к B", top_n=3)
                 .format({"РТО A, млн ₽": mln, "РТО B, млн ₽": mln}), hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                ["A к B", "Маржа A", "Маржа B", "OSA A", "OSA B"]}
                 )
    ui.download({"Сравнение": show, "Показатели": table}, "sravnenie.xlsx", key="dl_cmp")

    st.subheader("Тренды периодов рядом")
    c1, c2 = st.columns(2, gap="large")
    for col, ff, name, color in ((c1, fa, "A", ui.SERIES[0]), (c2, fb, "B", ui.NEUTRAL)):
        with col:
            s = Q.sales_daily(ff)
            fig = go.Figure(go.Scatter(x=s["date"], y=s["revenue"] / 1e6, mode="lines", line=dict(color=color, width=2),
                                       hovertemplate="%{y:,.1f} млн ₽<extra></extra>"))
            fig.update_layout(title=f"Период {name}: РТО по дням", yaxis=dict(title="млн ₽"))
            ui.chart(fig, 300, key=f"cmp_trend_{name}", dates=True)
    ui.export_panel(PAGE)


def _by_dim(f: Q.Filters, col: str) -> pd.DataFrame:
    if col == "category":
        c = Q.sales_by(f, "category_id")
        c["key"] = c["category_id"].map(Q.dims()["category"].set_index("category_id")["category"])
    else:
        c = Q.sales_by(f, "store_id")
        stv = Q.dims()["store"].set_index("store_id")
        c["key"] = c["store_id"].map(stv[col])
        c = c.groupby("key", as_index=False)[["revenue", "cost", "oos_w"]].sum()
    c["margin"] = M.safe_div(c["revenue"] - c["cost"], c["revenue"])
    c["osa"] = 1 - M.safe_div(c["oos_w"], c["revenue"])
    return c[["key", "revenue", "margin", "osa"]]
