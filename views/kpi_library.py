"""Библиотека показателей: формула, источник, ответственный, частота, направления; поиск, фильтры, экспорт;
добавление, изменение и архивирование пользовательских показателей администратором."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bi import kpi as K
from bi import queries as Q
from bi import state, ui

PAGE = "kpi"


def render(f: Q.Filters):
    role = ui.role()
    ui.page_header("Библиотека показателей", "Глоссарий KPI: формулы, источники, ответственные, частота обновления",
                   None, PAGE)
    custom = state.custom_kpis()
    lib = K.library(custom, include_archived=True)
    c = st.columns([2, 2, 1.2])
    q = c[0].text_input("Поиск", placeholder="название, формула, описание", key="kpi_q")
    dirs = c[1].multiselect("Направления", K.DIRECTIONS, placeholder="Все направления", key="kpi_dirs")
    kind = c[2].segmented_control("Показать", ["Действующие", "Архив"], default="Действующие", key="kpi_kind")
    v = lib[lib["archived"] == (1 if kind == "Архив" else 0)]
    if q:
        ql = q.lower()
        v = v[v[["name", "formula", "description"]].astype(str).apply(lambda s: s.str.lower().str.contains(ql, regex=False))
              .any(axis=1)]
    if dirs:
        v = v[v["directions"].astype(str).apply(lambda s: any(d in s.split(",") for d in dirs))]
    show = pd.DataFrame({"Показатель": v["name"], "Формула": v["formula"], "Ед.": v["unit"],
                         "Направления": v["directions"].astype(str).str.replace(",", ", "),
                         "Источник": v["source"], "Ответственный": v["owner"], "Обновление": v["freq"],
                         "Тип": v["kind"], "В конструкторе": v["builder"].astype(bool), "Описание": v["description"]})
    st.dataframe(show, hide_index=True, height=min(40 + 35 * len(show), 600), column_config={
        "Формула": st.column_config.TextColumn(width="medium"), "Описание": st.column_config.TextColumn(width="large"),
        "В конструкторе": st.column_config.CheckboxColumn()})
    st.caption(f"Показателей: {len(show)}. Один показатель может относиться к нескольким направлениям. Формулы "
               "считаются над суммами, поэтому показатели корректно агрегируются на любом уровне.")
    ui.download({"Библиотека показателей": show}, "biblioteka_kpi.xlsx", key="dl_kpi")
    ui.collect_table("Библиотека показателей", show)

    st.subheader("Карточка показателя")
    pick = st.selectbox("Показатель", v["code"].tolist(), format_func=lambda c_: v.set_index("code").loc[c_, "name"],
                        key="kpi_pick") if len(v) else None
    if pick:
        r = v.set_index("code").loc[pick]
        c1, c2 = st.columns([1.4, 1], gap="large")
        with c1:
            st.markdown(f"**{r['name']}** · {r['kind'].lower()}")
            st.code(r["formula"], language=None)
            st.markdown(f"{r['description']}\n\n- Источник: {r['source']}\n- Ответственный: {r['owner']}\n"
                        f"- Частота обновления: {r['freq']}\n- Направления: {str(r['directions']).replace(',', ', ')}")
        with c2:
            if bool(r["builder"]):
                try:
                    from views.builder import _data
                    measures = {r["name"]: (r["formula"], r["unit"], pick)}
                    base = _data(f, (), (r["name"],), measures)
                    from views.builder import _agg
                    val = K.evaluate(r["formula"], _agg(base, [])).iloc[0]
                    st.metric("Значение на текущих фильтрах", K.fmt_value(val, r["unit"]), border=True,
                              help=f"{f.start:%d.%m.%Y} — {f.end:%d.%m.%Y}, в рамках ваших прав")
                except Exception as e:
                    st.caption(f"Не удалось рассчитать: {e}")
            else:
                st.caption("Показатель считается отдельным алгоритмом и выводится на тематических страницах.")

    st.subheader("Управление показателями")
    if not role.is_admin:
        st.caption(":material/lock: Добавлять, изменять и архивировать показатели может только администратор BI. "
                   "Остальные пользователи не могут переопределить утверждённые формулы.")
        ui.export_panel(PAGE)
        return
    cus = custom.copy()
    mode = st.segmented_control("Действие", ["Новый показатель", "Изменить", "Архив / восстановление"],
                                default="Новый показатель", key="kpi_mode")
    if mode == "Новый показатель" or (mode == "Изменить" and len(cus)):
        cur = None
        if mode == "Изменить":
            code = st.selectbox("Пользовательский показатель", cus["code"].tolist(),
                                format_func=lambda c_: cus.set_index("code").loc[c_, "name"], key="kpi_edit_pick")
            cur = cus.set_index("code").loc[code]
        with st.form("kpi_form", border=True):
            c1, c2, c3 = st.columns([1.5, 2, 0.8])
            name = c1.text_input("Название", value="" if cur is None else cur["name"], max_chars=80)
            formula = c2.text_input("Формула", value="" if cur is None else cur["formula"], max_chars=300,
                                    help="Поля: " + ", ".join(K.BASE_FIELDS))
            unit = c3.selectbox("Единица", ["%", "₽", "дн.", "шт"],
                                index=0 if cur is None else ["%", "₽", "дн.", "шт"].index(cur["unit"]))
            c4, c5, c6 = st.columns([2, 1.5, 1])
            dirs_ = c4.multiselect("Направления", K.DIRECTIONS, placeholder="Выберите направления",
                                   default=[] if cur is None else [d for d in str(cur["directions"]).split(",") if d])
            owner = c5.text_input("Ответственный", value="" if cur is None else cur["owner"], max_chars=80)
            good = c6.selectbox("Хорошо, когда", ["up", "down", "none"],
                                format_func={"up": "растёт", "down": "снижается", "none": "нейтрально"}.get,
                                index=0 if cur is None else ["up", "down", "none"].index(cur["good"]))
            desc = st.text_area("Описание", value="" if cur is None else cur["description"], max_chars=500, height=70)
            if st.form_submit_button("Проверить и сохранить", icon=":material/check:"):
                try:
                    K.parse(formula)
                    code = K.slug(name) if cur is None else cur.name
                    state.upsert_kpi(code, name, formula, unit, dirs_ or ["Продажи"], desc, owner, good, role.title)
                    ui.audit("Изменён показатель" if cur is not None else "Добавлен показатель", name, formula)
                    st.success("Сохранено. Значения за всю историю пересчитаны по новой формуле автоматически.")
                    st.rerun()
                except K.FormulaError as e:
                    st.error(str(e))
    elif mode == "Изменить":
        st.caption("Пользовательских показателей пока нет.")
    else:
        if not len(cus):
            st.caption("Пользовательских показателей пока нет.")
        else:
            code = st.selectbox("Показатель", cus["code"].tolist(),
                                format_func=lambda c_: cus.set_index("code").loc[c_, "name"]
                                + (" · в архиве" if cus.set_index("code").loc[c_, "archived"] else ""), key="kpi_arch_pick")
            arch = bool(cus.set_index("code").loc[code, "archived"])
            if st.button("Восстановить" if arch else "В архив", key="kpi_arch_btn"):
                state.archive_kpi(code, not arch)
                ui.audit("Архив показателя", code, "восстановлен" if arch else "в архив")
                st.rerun()
    st.caption("Стандартные показатели утверждены и не редактируются в интерфейсе; изменения формул фиксируются "
               "в журнале аудита.")
    ui.export_panel(PAGE)
