"""Конструктор отчётов (self-service): сводные таблицы по проверенной витрине продаж.

Строки и столбцы — измерения оргструктуры, товара и времени; показатели — из библиотеки KPI
и вычисляемые поля пользователя. Все показатели считаются формулой над суммами, поэтому подытоги
и итоги корректны для долей и средних. Данные ограничены правами пользователя (RLS) и глобальными фильтрами.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bi import kpi as K
from bi import queries as Q
from bi import state, ui

PAGE = "builder"
DIMS = {
    "Сеть": ("store", "chain_name"), "Федеральный округ": ("store", "fed_district"),
    "Дивизион": ("store", "division"), "УГМ": ("store", "ugm"), "Регион": ("store", "region"),
    "Город": ("store", "city"), "РЦ": ("store", "dc_name"), "Формат": ("store", "format"),
    "Магазин": ("store", "store_name"), "Категория": ("cat", "category"), "Склад (зона)": ("cat", "zone"),
    "Год": ("time", "year"), "Квартал": ("time", "quarter"), "Месяц": ("time", "month"),
}
PALETTES = {
    "Корпоративная": ui.SERIES,
    "Синяя шкала": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#86b6ef"],
    "Тёплая": ["#eb6834", "#eda100", "#e34948", "#e87ba4", "#d55181", "#c98500", "#ec835a", "#fab219"],
    "Монохром": ["#52514e", "#898781", "#b9b7ae", "#383835", "#c3c2b7", "#6f6e69", "#a3a199", "#2c2c2a"],
}
CHARTS = ["Без графика", "Столбцы", "Горизонтальные столбцы", "Линия", "Два показателя (столбцы + линия)",
          "Тепловая карта"]
KEYS = ["bld_rows", "bld_col", "bld_meas", "bld_sort", "bld_desc", "bld_topn", "bld_chart", "bld_palette",
        "bld_sub", "bld_hl"]
DEFAULT = {"bld_rows": ["Дивизион"], "bld_col": None, "bld_meas": ["РТО", "Валовая маржа, %", "OSA, %"],
           "bld_sort": "РТО", "bld_desc": True, "bld_topn": 0, "bld_chart": "Горизонтальные столбцы",
           "bld_palette": "Корпоративная", "bld_sub": True, "bld_hl": True}
STATUS_BY_CODE = {"plan_pct": "plan", "osa": "osa", "yoy": "lfl"}


def _cfg() -> dict:
    return {k: st.session_state.get(k, DEFAULT[k]) for k in KEYS}


def _set_cfg(cfg: dict):
    for k in KEYS:
        if k in cfg:
            st.session_state[k] = cfg[k]


def _undo():
    pos = st.session_state.get("bld_pos", 0)
    if pos > 0:
        st.session_state["bld_pos"] = pos - 1
        _set_cfg(st.session_state["bld_hist"][pos - 1])


def _redo():
    pos, hist = st.session_state.get("bld_pos", 0), st.session_state.get("bld_hist", [])
    if pos < len(hist) - 1:
        st.session_state["bld_pos"] = pos + 1
        _set_cfg(hist[pos + 1])


def render(f: Q.Filters):
    role = ui.role()
    ui.page_header("Конструктор отчётов", "Сводные таблицы и графики из витрины продаж — в рамках ваших прав", f, PAGE)
    ui.register_widgets(PAGE, ["Сводная таблица", "График конструктора"])
    for k, v in DEFAULT.items():
        st.session_state.setdefault(k, v)

    lib = K.library(state.custom_kpis())
    lib = lib[lib["builder"].astype(bool)]
    adhoc = st.session_state.setdefault("bld_adhoc", {})        # вычисляемые поля сессии: имя -> (формула, ед.)
    measures = {r["name"]: (r["formula"], r["unit"], r["code"]) for _, r in lib.iterrows()}
    measures.update({n: (fml, unit, "adhoc") for n, (fml, unit) in adhoc.items()})
    st.session_state["bld_meas"] = [m for m in st.session_state["bld_meas"] if m in measures]

    # --- панель: отмена/повтор, сохранённые отчёты
    top = st.container(horizontal=True, gap="small", vertical_alignment="bottom")
    with top:
        st.button("Отменить", icon=":material/undo:", on_click=_undo, key="bld_undo",
                  disabled=st.session_state.get("bld_pos", 0) == 0)
        st.button("Повторить", icon=":material/redo:", on_click=_redo, key="bld_redo",
                  disabled=st.session_state.get("bld_pos", 0) >= len(st.session_state.get("bld_hist", [])) - 1)
        reps = state.reports_for(role.key)
        if len(reps):
            st.selectbox("Сохранённые отчёты", reps["id"].tolist(), key="bld_load_pick",
                         format_func=lambda i: reps.set_index("id").loc[i, "name"]
                         + (" · общий" if reps.set_index("id").loc[i, "shared"] else " · мой"))

            def _load():
                row = reps.set_index("id").loc[st.session_state["bld_load_pick"]]
                p = json.loads(row["payload"])
                st.session_state["bld_adhoc"].update({k: tuple(v) for k, v in p.get("adhoc", {}).items()})
                _set_cfg(p["cfg"])
                ui.audit("Открыт сохранённый отчёт", row["name"])
            st.button("Открыть", icon=":material/folder_open:", on_click=_load, key="bld_load")
        with st.popover("Сохранить / опубликовать", icon=":material/save:", disabled=not role.can_build):
            name = st.text_input("Название отчёта", max_chars=80, key="bld_save_name")
            shared = st.toggle("Опубликовать для коллег", value=True, key="bld_save_shared",
                               help="Коллеги увидят отчёт в списке, а данные — только в рамках своих прав")
            if st.button("Сохранить", icon=":material/check:", key="bld_save_btn") and name.strip():
                state.save_report(name, role.key, shared, {"cfg": _cfg(), "adhoc": adhoc})
                ui.audit("Сохранён отчёт", name, "опубликован" if shared else "личный")
                st.success("Отчёт сохранён")

    # --- настройки
    c1, c2, c3 = st.columns([1.4, 1, 1.6])
    rows = c1.multiselect("Строки (порядок = уровни группировки)", list(DIMS), key="bld_rows",
                          placeholder="Например: Сеть, Дивизион, УГМ")
    col_dim = c2.selectbox("Столбцы (необязательно)", [d for d in DIMS if d not in rows], key="bld_col",
                           index=None, placeholder="Без столбцов")
    meas = c3.multiselect("Показатели", list(measures), key="bld_meas", max_selections=6,
                         placeholder="Выберите показатели")
    c4, c5, c6, c7, c8 = st.columns([1.2, 0.8, 0.9, 1.4, 1.1])
    sort_opts = meas or ["РТО"]
    if st.session_state.get("bld_sort") not in sort_opts:
        st.session_state["bld_sort"] = sort_opts[0]
    sort_by = c4.selectbox("Сортировать по", sort_opts, key="bld_sort")
    desc = c5.toggle("Убывание", key="bld_desc")
    topn = c6.number_input("Топ N строк (0 — все)", 0, 1000, key="bld_topn")
    chart_t = c7.selectbox("Визуализация", CHARTS, key="bld_chart")
    palette = c8.selectbox("Цвета", list(PALETTES), key="bld_palette")
    o1, o2, o3 = st.columns(3)
    subtotals = o1.toggle("Подытоги по уровням", key="bld_sub")
    highlight = o2.toggle("Подсветка (пороги, лучшие/худшие)", key="bld_hl")
    mobile_prev = o3.toggle("Предпросмотр на телефоне", key="bld_mobile")

    # история для отмены/повтора
    cfg = _cfg()
    hist = st.session_state.setdefault("bld_hist", [])
    pos = st.session_state.get("bld_pos", -1)
    if not hist or cfg != hist[pos]:
        del hist[pos + 1:]
        hist.append(cfg)
        st.session_state["bld_pos"] = len(hist) - 1
        del hist[:-30]
        st.session_state["bld_pos"] = len(hist) - 1

    with st.expander("Вычисляемое поле", icon=":material/function:"):
        st.caption("Формула над полями витрины: " + ", ".join(f"`{k}`" for k in K.BASE_FIELDS)
                   + ". Пример: `(РТО - Себестоимость) / РТО`. Разрешены + − × / и скобки.")
        e1, e2, e3 = st.columns([1.2, 2, 0.7])
        fname = e1.text_input("Название", max_chars=60, key="bld_f_name")
        fexpr = e2.text_input("Формула", max_chars=300, key="bld_f_expr")
        funit = e3.selectbox("Единица", ["₽", "%", "дн.", "шт"], key="bld_f_unit")
        b1, b2 = st.columns(2)
        if b1.button("Добавить в отчёт", icon=":material/add:", key="bld_f_add") and fname.strip():
            try:
                K.parse(fexpr)
                adhoc[fname.strip()] = (fexpr, funit)
                st.session_state["bld_meas"] = (st.session_state["bld_meas"] + [fname.strip()])[:6]
                ui.audit("Вычисляемое поле", fname, fexpr)
                st.rerun()
            except K.FormulaError as e:
                st.error(str(e))
        if role.is_admin and b2.button("Сохранить в библиотеку показателей", icon=":material/library_add:",
                                       key="bld_f_lib") and fname.strip():
            try:
                K.parse(fexpr)
                state.upsert_kpi(K.slug(fname), fname, fexpr, funit, ["Продажи"], "Создан в конструкторе",
                                 role.title, "up", role.title)
                ui.audit("Показатель добавлен в библиотеку", fname, fexpr)
                st.success("Показатель добавлен в библиотеку")
            except K.FormulaError as e:
                st.error(str(e))
        if not role.is_admin:
            st.caption("Сохранять показатели в общую библиотеку может только администратор BI.")

    if not rows or not meas:
        st.info("Выберите хотя бы одно измерение в строках и один показатель.", icon=":material/info:")
        return

    try:
        base = _data(f, tuple(rows + ([col_dim] if col_dim else [])), tuple(meas), measures)
    except K.FormulaError as e:
        st.error(str(e))
        return
    if base.empty:
        ui.empty_state()
        return
    table, flat = _pivot(base, rows, col_dim, meas, measures, subtotals)
    if sort_by in flat.columns:
        flat = flat.sort_values(sort_by, ascending=not desc, na_position="last")
    if topn:
        flat = flat.head(int(topn))
    ctx = (base, list(meas), measures)
    if col_dim is None:
        table = _order_with_subtotals(table, rows, flat, subtotals, ctx)
    else:
        order = flat.set_index(rows).index
        table = table.set_index(rows).reindex(order).reset_index()
    ui.collect_table("Отчёт", table)

    def show_all():
        if chart_t != "Без графика":
            _chart(flat, rows, col_dim, meas, measures, chart_t, palette, base)
        status_cols = {}
        if highlight:
            for m in meas:
                code = measures[m][2]
                if code in STATUS_BY_CODE:
                    status_cols.update({c: STATUS_BY_CODE[code] for c in table.columns if str(c).startswith(m)})
        sty = ui.styled(table, status_cols, top_col=(meas[0] if highlight and meas[0] in table.columns
                                                     and col_dim is None else None))
        st.dataframe(sty, hide_index=True, height=min(40 + 35 * len(table), 640), column_config=_colcfg(table, measures))

    if mobile_prev:
        st.caption(":material/smartphone: Предпросмотр вёрстки на экране 390 px")
        with st.container(width=390, border=True):
            show_all()
    else:
        show_all()
    st.caption(f"Строк: {len(table)}. Ограничений на число строк и столбцов нет; полный результат — в выгрузке. "
               "Чеки, средний чек и план доступны без разреза по категориям; «прошлый год» — сдвиг 364 дня.")
    ui.download({"Отчёт": table}, "konstruktor.xlsx", key="dl_bld")
    ui.export_panel(PAGE)


# ---------------------------------------------------------------------------
def _data(f: Q.Filters, dims: tuple[str, ...], meas: tuple[str, ...], measures: dict) -> pd.DataFrame:
    need = set()
    for m in meas:
        need |= K.fields_needed(measures[m][0])
    kinds = {DIMS[d][0] for d in dims}
    time_keys = []
    td = [d for d in dims if DIMS[d][0] == "time"]
    if td:
        time_keys = ["year"] + (["quarter"] if "Квартал" in td else []) + (["month"] if "Месяц" in td else [])
    keys = (["store_id"] if "store" in kinds else []) + (["category_id"] if "cat" in kinds else []) + time_keys
    base = Q.sales_grouped(f, tuple(keys))
    store_fields = {"checks", "store_revenue", "plan"} & need
    if store_fields and "cat" not in kinds:
        sk = [k for k in keys if k != "category_id"]
        sd = Q.store_day_grouped(f, tuple(sk))
        base = base.merge(sd, on=sk, how="left") if sk else base.assign(**{c: sd[c].iloc[0] for c in sd.columns})
        if "plan" in need:
            pl = Q.plan_monthly(f)
            pl["year"], pl["quarter"], pl["month_n"] = pl["month"].dt.year, pl["month"].dt.quarter, pl["month"].dt.month
            pl = pl.drop(columns="month").rename(columns={"month_n": "month"})
            pk = [k for k in sk]
            pg = pl.groupby(pk, as_index=False)["plan"].sum() if pk else pd.DataFrame({"plan": [pl["plan"].sum()]})
            base = base.merge(pg, on=pk, how="left") if pk else base.assign(plan=pg["plan"].iloc[0])
    if "revenue_py" in need:
        py = Q.sales_grouped(f.py(), tuple(keys))[keys + ["revenue"]].rename(columns={"revenue": "revenue_py"})
        if "year" in py:
            py["year"] = py["year"] + 1
        base = base.merge(py, on=keys, how="left") if keys else base.assign(revenue_py=py["revenue_py"].iloc[0])
    d = Q.dims()
    if "store_id" in base:
        stv = d["store"].set_index("store_id")
        for dim in dims:
            if DIMS[dim][0] == "store":
                base[dim] = base["store_id"].map(stv[DIMS[dim][1]])
    if "category_id" in base:
        cv = d["category"].set_index("category_id")
        for dim in dims:
            if DIMS[dim][0] == "cat":
                base[dim] = base["category_id"].map(cv[DIMS[dim][1]])
    for dim in dims:
        if DIMS[dim][0] == "time":
            col = DIMS[dim][1]
            if dim == "Год":
                base[dim] = base["year"].astype(str)
            elif dim == "Квартал":
                base[dim] = base["year"].astype(str) + " Q" + base["quarter"].astype(str)
            else:
                base[dim] = base["year"].astype(str) + "-" + base["month"].astype(int).map("{:02d}".format)
            del col
    return base


SUM_COLS = ["revenue", "cost", "qty", "promo_revenue", "promo_cost", "writeoff", "stock_cost", "oos_w", "checks",
            "store_revenue", "plan", "revenue_py"]


def _agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = [c for c in SUM_COLS if c in df.columns]
    if by:
        return df.groupby(by, dropna=False, as_index=False)[cols].sum(min_count=1)
    return df[cols].sum(min_count=1).to_frame().T


def _apply(df: pd.DataFrame, meas: list[str], measures: dict) -> pd.DataFrame:
    out = df.copy()
    for m in meas:
        out[m] = K.evaluate(measures[m][0], out)
    return out


def _pivot(base, rows, col_dim, meas, measures, subtotals):
    flat = _apply(_agg(base, rows + ([col_dim] if col_dim else [])), meas, measures)
    if col_dim:
        wide = flat.pivot_table(index=rows, columns=col_dim, values=meas, aggfunc="first")
        wide.columns = [f"{m} · {c}" for m, c in wide.columns]
        tot = _apply(_agg(base, rows), meas, measures).set_index(rows)[meas]
        tot.columns = [f"{m} · Итого" for m in meas]
        table = wide.join(tot).reset_index()
        flat_rows = _apply(_agg(base, rows), meas, measures)
        return table, flat_rows
    return flat[rows + meas], flat


def _order_with_subtotals(table: pd.DataFrame, rows: list[str], flat: pd.DataFrame, subtotals: bool,
                          ctx) -> pd.DataFrame:
    """Сохраняет порядок сортировки и вставляет подытоги по первому уровню и общий итог."""
    ordered = flat[rows + [c for c in table.columns if c not in rows]].reset_index(drop=True)
    if not subtotals or len(rows) < 2:
        return _with_total(ordered, rows, ctx)
    out = []
    first = rows[0]
    for val in ordered[first].drop_duplicates():
        part = ordered[ordered[first] == val]
        out.append(part)
        out.append(_subtotal_row(rows, val, ctx))
    res = pd.concat(out, ignore_index=True)
    return _with_total(res, rows, ctx)


def _subtotal_row(rows: list[str], val, ctx) -> pd.DataFrame:
    """Подытог пересчитывается формулой по суммам исходных полей — доли и средние корректны."""
    base, meas, measures = ctx
    sub = base[base[rows[0]] == val]
    r = _apply(_agg(sub, []), meas, measures)
    row = {rows[0]: f"Итого: {val}"}
    for rr in rows[1:]:
        row[rr] = ""
    for m in meas:
        row[m] = r[m].iloc[0]
    return pd.DataFrame([row])


def _with_total(df: pd.DataFrame, rows: list[str], ctx) -> pd.DataFrame:
    base, meas, measures = ctx
    r = _apply(_agg(base, []), meas, measures)
    row = {rows[0]: "ИТОГО"}
    for rr in rows[1:]:
        row[rr] = ""
    for m in meas:
        if m in df.columns:
            row[m] = r[m].iloc[0]
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def _chart(flat, rows, col_dim, meas, measures, chart_t, palette, base):
    pal = PALETTES[palette]
    label = flat[rows].astype(str).agg(" · ".join, axis=1) if len(rows) > 1 else flat[rows[0]].astype(str)
    m0 = meas[0]
    unit0 = measures[m0][1]
    sc = 100 if unit0 == "%" else (1e-6 if unit0 == "₽" else 1)
    ut = {"%": "%", "₽": "млн ₽"}.get(unit0, unit0)
    if chart_t == "Тепловая карта":
        if not col_dim:
            st.caption("Для тепловой карты выберите измерение в «Столбцы».")
            return
        flat2 = _apply(_agg(base, rows + [col_dim]), [m0], measures)
        flat2["label"] = flat2[rows].astype(str).agg(" · ".join, axis=1)
        piv = flat2.pivot_table(index="label", columns=col_dim, values=m0, aggfunc="first")
        fig = go.Figure(go.Heatmap(z=piv.values * sc, x=[str(c) for c in piv.columns], y=list(piv.index),
                                   colorscale=[[0, ui.SEQ[0]], [1, ui.SEQ[6]]], xgap=2, ygap=2,
                                   hovertemplate="%{y} · %{x}: %{z:,.1f} " + ut + "<extra></extra>",
                                   colorbar=dict(thickness=10)))
        fig.update_layout(title=f"{m0}: {' · '.join(rows)} × {col_dim}", yaxis=dict(autorange="reversed"))
        ui.chart(fig, max(320, 24 * len(piv) + 120), key="bld_chart_heat", export_name="График конструктора")
        return
    flat = flat.head(60)
    label = label.loc[flat.index]
    if chart_t == "Два показателя (столбцы + линия)" and len(meas) >= 2:
        m1 = meas[1]
        u1 = measures[m1][1]
        s1 = 100 if u1 == "%" else (1e-6 if u1 == "₽" else 1)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
        fig.add_bar(x=label, y=flat[m0] * sc, name=m0, marker=dict(color=pal[0], cornerradius=3), row=1, col=1)
        fig.add_scatter(x=label, y=flat[m1] * s1, name=m1, mode="lines+markers", line=dict(color=pal[1], width=2),
                        row=2, col=1)
        fig.update_layout(title=f"{m0} и {m1}")
        ui.chart(fig, 460, key="bld_chart_combo", export_name="График конструктора")
        return
    fig = go.Figure()
    horiz = chart_t == "Горизонтальные столбцы"
    for i, m in enumerate(meas[:3] if chart_t != "Линия" else meas[:4]):
        u = measures[m][1]
        if u != unit0:
            continue
        s = 100 if u == "%" else (1e-6 if u == "₽" else 1)
        if chart_t == "Линия":
            fig.add_scatter(x=label, y=flat[m] * s, name=m, mode="lines+markers", line=dict(color=pal[i % 8], width=2))
        elif horiz:
            fig.add_bar(y=label, x=flat[m] * s, name=m, orientation="h", marker=dict(color=pal[i % 8], cornerradius=3))
        else:
            fig.add_bar(x=label, y=flat[m] * s, name=m, marker=dict(color=pal[i % 8], cornerradius=3))
    fig.update_layout(title=" · ".join(rows) + f": {', '.join(m for m in meas if measures[m][1] == unit0)}",
                      barmode="group")
    if horiz:
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(title=ut))
    else:
        fig.update_layout(yaxis=dict(title=ut))
    ui.chart(fig, max(340, 22 * len(flat) + 120) if horiz else 380, key="bld_chart_main",
             export_name="График конструктора")
    if any(measures[m][1] != unit0 for m in meas[:3]):
        st.caption("На одном графике — показатели одной единицы измерения; для двух разных используйте "
                   "«Два показателя (столбцы + линия)».")


def _colcfg(table: pd.DataFrame, measures: dict) -> dict:
    cfg = {}
    for c in table.columns:
        name = str(c).split(" · ")[0]
        if name in measures:
            unit = measures[name][1]
            cfg[c] = st.column_config.NumberColumn(format="percent" if unit == "%" else
                                                   ("localized" if unit in ("₽", "шт") else "%.1f"))
    return cfg
