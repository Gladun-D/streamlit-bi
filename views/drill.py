"""Детализация по иерархии: сеть → округ → дивизион → УГМ → магазин → категория → SKU.

Клик или выделение на диаграмме — кросс-фильтр для тренда и таблицы; «Детализировать» — уровень ниже;
хлебные крошки — быстрый возврат на любой уровень.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import metrics as M
from bi import queries as Q
from bi import ui
from views import common

PAGE = "drill"
LEVELS = [("chain_name", "Сеть"), ("fed_district", "Федеральный округ"), ("division", "Дивизион"), ("ugm", "УГМ"),
          ("store_name", "Магазин"), ("category", "Категория"), ("sku", "SKU (отгрузки РЦ)")]
METRICS = {
    "РТО": ("revenue", "₽", None), "LFL": ("lfl", "%", "lfl"), "Выполнение плана": ("plan_pct", "%", "plan"),
    "OSA": ("osa", "%", "osa"), "Валовая маржа": ("margin", "%", None), "Средний чек": ("avg_check", "₽", None),
    "Г/г": ("yoy", "%", None),
}


def render(f: Q.Filters):
    ui.page_header("Детализация по иерархии", "Провал по показателю до магазина, категории и SKU", f, PAGE)
    ui.register_widgets(PAGE, ["Диаграмма уровня", "Тренд выбранных", "Таблица уровня"])
    py_ok = common.py_available(f)
    common.notes(f, py_ok)
    path: list[tuple[int, str]] = st.session_state.setdefault("drill_path", [])
    level = len(path)

    # Хлебные крошки
    crumbs = st.container(horizontal=True, gap="small", vertical_alignment="center")
    with crumbs:
        if st.button("Вся выборка", icon=":material/home:", type="tertiary" if level else "secondary", key="cr_all"):
            st.session_state["drill_path"] = []
            st.rerun()
        for i, (lv, val) in enumerate(path):
            st.markdown("›")
            if st.button(f"{LEVELS[lv][1]}: {val}", key=f"cr_{i}", type="tertiary" if i < level - 1 else "secondary"):
                st.session_state["drill_path"] = path[: i + 1]
                st.rerun()

    data, child_col = _level_data(f, path, py_ok)
    if data is None or data.empty:
        ui.empty_state("На этом уровне нет данных для выбранных фильтров")
        return
    label = LEVELS[level][1]

    c = st.columns([1.3, 1, 2])
    avail = [m for m in METRICS if METRICS[m][0] in data.columns and data[METRICS[m][0]].notna().any()]
    metric = c[0].selectbox("Показатель", avail, key="dr_metric") or avail[0]
    if metric not in avail:
        metric = avail[0]
    mode = c[1].segmented_control("Режим", ["Диаграмма", "Таблица"], default="Диаграмма", key="dr_mode") or "Диаграмма"
    col, unit, kind = METRICS[metric]
    data = data.sort_values(col, ascending=metric in ("LFL", "Выполнение плана", "OSA", "Г/г"), na_position="last")

    selected: list[str] = []
    if mode == "Диаграмма":
        show = data.head(40)
        colors = ([ui.STATUS[ui.status(v, kind)] if np.isfinite(v) else ui.NEUTRAL for v in show[col]] if kind
                  else ui.SERIES[0])
        x = show[col] / 1e6 if unit == "₽" and col == "revenue" else show[col] * (100 if unit == "%" else 1)
        xt = "млн ₽" if col == "revenue" else ("%" if unit == "%" else "₽")
        hover = _hover(show)
        fig = go.Figure(go.Bar(y=show[child_col], x=x, orientation="h", marker=dict(color=colors, cornerradius=3),
                               customdata=hover, hovertemplate="%{customdata}<extra></extra>"))
        fig.update_layout(title=f"{label}: {metric}" + (" (первые 40)" if len(data) > 40 else ""),
                          yaxis=dict(autorange="reversed", title=None), xaxis=dict(title=xt), dragmode="zoom",
                          clickmode="event+select")  # при dragmode=select клик по столбцу не выделяет его
        ev = ui.chart(fig, max(280, 26 * len(show) + 110), key=f"dr_chart_{level}", on_select="rerun",
                      selection_mode=("points", "box"), export_name=f"{label}: {metric}")
        try:
            pts = ev.selection.points if ev is not None else []
            selected = list(dict.fromkeys(p["y"] for p in pts if "y" in p))
            if not selected and ev is not None and ev.selection.box:
                # рамка: берём все строки, попавшие по оси категорий (Plotly выделяет столбец только по его концу)
                cats = show[child_col].tolist()
                for bx in ev.selection.box:
                    lo, hi = sorted(bx.get("y", [0, -1])[:2])
                    selected += [c_ for i, c_ in enumerate(cats) if lo <= i <= hi]
                selected = list(dict.fromkeys(selected))
        except Exception:
            selected = []
        st.caption("Клик по столбцу или выделение рамкой (кнопка «Box select» над графиком) — кросс-фильтр тренда и "
                   "таблицы ниже. Shift + клик — несколько элементов. "
                   "Цвет — статус по порогам подсветки.")
    else:
        selected = st.multiselect(f"Выбрать {label.lower()} для кросс-фильтра", data[child_col].tolist(),
                                  key=f"dr_sel_{level}", placeholder="Все элементы уровня")

    # Детализация на уровень ниже
    if level < len(LEVELS) - 1:
        sel_key, last_key = f"dr_go_sel_{level}", f"_dr_last_sel_{level}"
        options = data[child_col].tolist()
        # клик по столбцу подставляет элемент в поле «Детализировать» (только при смене выбора на графике)
        if selected != st.session_state.get(last_key):
            st.session_state[last_key] = selected
            if len(selected) == 1 and selected[0] in options:
                st.session_state[sel_key] = selected[0]
        if st.session_state.get(sel_key) not in options:
            st.session_state.pop(sel_key, None)
        dc1, dc2 = st.columns([2, 1], vertical_alignment="bottom")
        target = dc1.selectbox(f"Детализировать: {label.lower()}", options, index=None,
                               placeholder="Выберите элемент или кликните на столбец", key=sel_key)
        if dc2.button(f"Перейти на уровень «{LEVELS[level + 1][1]}»", icon=":material/subdirectory_arrow_right:",
                      disabled=target is None, key=f"dr_go_{level}", width="stretch"):
            st.session_state["drill_path"] = path + [(level, target)]
            ui.audit("Детализация", f"{label}: {target}")
            st.rerun()

    _trend(f, path, data, child_col, selected)

    tbl = data[data[child_col].isin(selected)] if selected else data
    show = _table(tbl, child_col, label)
    st.markdown(f"**{label}: показатели**" + (f" · выбрано {len(selected)}" if selected else ""))
    status_cols = {k: v for k, v in {"LFL": "lfl", "План": "plan", "OSA": "osa"}.items() if k in show.columns}
    top = "LFL" if "LFL" in show.columns and show["LFL"].notna().any() else ("РТО, млн ₽" if "РТО, млн ₽" in show else None)
    st.dataframe(ui.styled(show, status_cols, top_col=top), hide_index=True, height=min(38 + 35 * len(show), 520),
                 column_config=_colcfg(show))
    ui.download({f"{label}": show}, "detalizaciya.xlsx", key="dl_drill")
    if level == 6:
        st.caption("SKU-уровень показан по отгрузкам РЦ, который снабжает магазин, — продажи по SKU в магазинах "
                   "в прототипе не детализированы.")
    ui.export_panel(PAGE)


# ---------------------------------------------------------------------------
def _level_data(f: Q.Filters, path, py_ok) -> tuple[pd.DataFrame | None, str]:
    level = len(path)
    stores = common.store_table(f)
    for lv, val in path:
        if lv <= 4:
            stores = stores[stores[LEVELS[lv][0]] == val]
    if level <= 4:
        col = LEVELS[level][0]
        g = common.group_table(stores, [col], f)
        g = g.rename(columns={col: "name"})
        return g, "name"
    store = stores.iloc[0] if len(stores) else None
    if store is None:
        return None, "name"
    fs = replace(f, store_ids=(int(store["store_id"]),))
    cats = Q.dims()["category"].set_index("category_id")
    if level == 5:
        cur = Q.sales_by_store_cat(fs)
        cur = cur.groupby("category_id", as_index=False)[["revenue", "cost", "stock_cost", "oos_w"]].sum()
        if py_ok:
            py = Q.sales_by_store_cat(fs.py()).groupby("category_id", as_index=False)["revenue"].sum() \
                .rename(columns={"revenue": "revenue_py"})
            cur = cur.merge(py, on="category_id", how="left")
            cur["yoy"] = cur["revenue"] / cur["revenue_py"] - 1
            cur["lfl"] = cur["yoy"] if bool(store["comparable"]) else np.nan
        cur["name"] = cur["category_id"].map(cats["category"])
        cur["margin"] = M.safe_div(cur["revenue"] - cur["cost"], cur["revenue"])
        cur["osa"] = 1 - M.safe_div(cur["oos_w"], cur["revenue"])
        cur["cover_days"] = M.safe_div(cur["stock_cost"], cur["cost"])
        return cur, "name"
    cat_name = path[-1][1]
    cat_id = int(cats.index[cats["category"] == cat_name][0])
    fsku = replace(f, dc_ids=(int(store["dc_id"]),), cats=(cat_id,), store_ids=None)
    wk = Q.sku_weekly(fsku)
    if wk.empty:
        return None, "name"
    sku = Q.dims()["sku"].set_index("sku_id")
    g = wk.groupby("sku_id", as_index=False)[["revenue", "gross_profit", "qty", "demand"]].sum()
    g["name"] = g["sku_id"].map(sku["sku"])
    g["margin"] = M.safe_div(g["gross_profit"], g["revenue"])
    g["osa"] = M.safe_div(g["qty"], g["demand"])
    return g, "name"


def _hover(d: pd.DataFrame) -> list[str]:
    out = []
    for _, r in d.iterrows():
        parts = [f"<b>{r['name']}</b>", f"РТО: {ui.rub(r.get('revenue', np.nan))}"]
        for k, lab in (("lfl", "LFL"), ("plan_pct", "План"), ("osa", "OSA"), ("margin", "Маржа"), ("yoy", "Г/г")):
            v = r.get(k, np.nan)
            if v is not None and pd.notna(v):
                parts.append(f"{lab}: {ui.pct(v)}")
        if pd.notna(r.get("stores", np.nan)):
            parts.append(f"Магазинов: {int(r['stores'])}")
        out.append("<br>".join(parts))
    return out


def _trend(f: Q.Filters, path, data: pd.DataFrame, child_col: str, selected: list[str]):
    level = len(path)
    names = selected or data.sort_values("revenue", ascending=False)[child_col].head(6).tolist()
    if level <= 4:
        st_ = common.store_table(f)
        for lv, val in path:
            st_ = st_[st_[LEVELS[lv][0]] == val]
        key_col = LEVELS[level][0]
        mp = st_.set_index("store_id")[key_col]
        sd = Q.store_daily(replace(f, store_ids=tuple(int(i) for i in st_["store_id"])))
        sd["name"] = sd["store_id"].map(mp)
        sd = sd[sd["name"].isin(names)]
        piv = sd.pivot_table(index="date", columns="name", values="revenue", aggfunc="sum")
    elif level == 5:
        store_id = int(common.store_table(f).query("store_name == @path[-1][1]")["store_id"].iloc[0])
        cd = Q.cat_daily(replace(f, store_ids=(store_id,)))
        cats = Q.dims()["category"].set_index("category_id")["category"]
        cd["name"] = cd["category_id"].map(cats)
        piv = cd[cd["name"].isin(names)].pivot_table(index="date", columns="name", values="revenue", aggfunc="sum")
    else:
        return
    if piv.empty:
        return
    gran = "Неделя" if f.days >= 56 else "День"
    w = common.resample(piv.reset_index(), list(piv.columns), gran)
    fig = go.Figure()
    for i, n in enumerate(names[:8]):
        if n in w:
            fig.add_scatter(x=w.index, y=w[n] / 1e6, name=str(n)[:40], mode="lines",
                            line=dict(width=2, color=(ui.SERIES_DARK if ui.is_dark() else ui.SERIES)[i % 8]),
                            hovertemplate="%{y:,.1f} млн ₽<extra>" + str(n)[:40] + "</extra>")
    title = "Тренд выбранных элементов" if selected else "Тренд: крупнейшие элементы уровня"
    fig.update_layout(title=f"{title} · РТО по {('неделям' if gran == 'Неделя' else 'дням')}", hovermode="x unified",
                      yaxis=dict(title="млн ₽"))
    ui.chart(fig, 320, key=f"dr_trend_{level}", dates=True, export_name="Тренд выбранных")


def _table(d: pd.DataFrame, child_col: str, label: str) -> pd.DataFrame:
    cols = {child_col: label}
    out = pd.DataFrame({label: d[child_col].values})
    if "status" not in d and {"lfl", "plan_pct", "osa"} & set(d.columns):
        out.insert(0, "", [common.status_icon(r) for _, r in d.iterrows()])
    for src, name, scale in (("stores", "Магазинов", 1), ("revenue", "РТО, млн ₽", 1e-6), ("yoy", "Г/г", 1),
                             ("lfl", "LFL", 1), ("plan_pct", "План", 1), ("osa", "OSA", 1), ("margin", "Маржа", 1),
                             ("avg_check", "Средний чек, ₽", 1), ("writeoff_rate", "Списания", 1),
                             ("cover_days", "Запас, дн.", 1), ("qty", "Штук", 1)):
        if src in d.columns and d[src].notna().any():
            out[name] = (d[src] * scale).values
    del cols
    return out


def _colcfg(show: pd.DataFrame) -> dict:
    cfg = {"": st.column_config.TextColumn(width=36)}
    for c in show.columns:
        if c in ("Г/г", "LFL", "План", "OSA", "Маржа", "Списания"):
            cfg[c] = st.column_config.NumberColumn(format="percent")
        elif c in ("РТО, млн ₽",):
            cfg[c] = st.column_config.NumberColumn(format="%.2f")
        elif c in ("Средний чек, ₽", "Штук", "Магазинов"):
            cfg[c] = st.column_config.NumberColumn(format="%.0f")
        elif c == "Запас, дн.":
            cfg[c] = st.column_config.NumberColumn(format="%.1f")
    return cfg
