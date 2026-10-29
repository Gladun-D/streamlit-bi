"""Центр уведомлений: правила, активные срабатывания (по факту и прогнозу), история и анализ причин."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import alerts as AL
from bi import queries as Q
from bi import state, ui

PAGE = "alerts"
SEV_ICON = {"критично": "🔴", "внимание": "🟡"}


def render(f: Q.Filters):
    role = ui.role()
    ui.page_header("Уведомления", "Отклонения KPI, прогнозные риски, аномалии, запасы, поставщики и качество", f, PAGE)
    ui.register_widgets(PAGE, ["Активные уведомления", "История срабатываний", "Правила"])
    rules = state.alert_rules()
    al = AL.evaluate_all(f, rules)
    crit = int((al["Важность"] == "критично").sum())
    warn = int((al["Важность"] == "внимание").sum())
    k = st.columns(4)
    ui.kpi(k[0], "Активных уведомлений", ui.num(len(al)), None, good_when_up=None)
    ui.kpi(k[1], "Критичных", ui.num(crit), None, good_when_up=None)
    ui.kpi(k[2], "Требуют внимания", ui.num(warn), None, good_when_up=None)
    ui.kpi(k[3], "Правил включено", f"{int(rules['active'].sum())} из {len(rules)}", None, good_when_up=None)

    t1, t2, t3 = st.tabs(["Активные", "История и причины", "Правила"])
    with t1:
        if al.empty:
            ui.insight("Активных уведомлений нет.", "good")
        else:
            rule_pick = st.pills("Правило", sorted(al["Правило"].unique()), selection_mode="multi", key="al_rule")
            v = al[al["Правило"].isin(rule_pick)] if rule_pick else al
            v = v.sort_values(["Важность", "Правило"], ascending=[True, True])
            show = pd.DataFrame({"": v["Важность"].map(SEV_ICON), "Правило": v["Правило"], "Объект": v["Объект"],
                                 "Значение": v["Значение"], "Порог": v["Порог"], "Детали": v["Детали"]})
            st.dataframe(show, hide_index=True, height=min(40 + 35 * len(show), 560),
                         column_config={"": st.column_config.TextColumn(width=36),
                                        "Правило": st.column_config.TextColumn(width="medium"),
                                        "Детали": st.column_config.TextColumn(width="medium")})
            ui.collect_table("Активные уведомления", show)
            st.caption("Канал доставки в прототипе — интерфейс BI. Правило «Прогноз выполнения плана месяца» срабатывает по прогнозу (факт с начала месяца + "
                       "темп последних 7 дней), а не по факту. Значения считаются на конец выбранного периода "
                       "и в рамках ваших прав.")
    with t2:
        hist = AL.history(f, rules)
        if hist.empty:
            st.caption("Нет срабатываний за последние 26 недель.")
        else:
            wk = hist.groupby(["week", "Правило"]).size().unstack(fill_value=0)
            fig = go.Figure()
            for i, c in enumerate(wk.columns):
                fig.add_bar(x=wk.index, y=wk[c], name=c[:45], marker=dict(color=ui.SERIES[i % 8]))
            fig.update_layout(title="Срабатывания по неделям", barmode="stack", yaxis=dict(title="срабатываний"))
            ui.chart(fig, 340, key="al_hist", dates=True, export_name="История срабатываний")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Чаще всего повторяются**")
                rep = hist.groupby(["Правило", "Объект"]).agg(Срабатываний=("week", "size"),
                                                               Последнее=("week", "max")).reset_index()
                rep = rep.sort_values("Срабатываний", ascending=False).head(15)
                rep["Последнее"] = rep["Последнее"].dt.strftime("%d.%m.%Y")
                st.dataframe(rep, hide_index=True)
                ui.collect_table("Повторяющиеся срабатывания", rep)
            with c2:
                st.markdown("**Где возникают и чем заканчиваются**")
                grp = hist.groupby("Группа").size().sort_values(ascending=False).head(10).rename("Срабатываний")
                res = hist["Результат"].value_counts().rename("Срабатываний")
                st.dataframe(grp.reset_index(), hide_index=True)
                st.dataframe(res.rename_axis("Результат").reset_index(), hide_index=True)
            st.caption("История восстанавливается пересчётом правил по прошлым неделям; «закрыто» — на следующей "
                       "неделе правило по объекту не сработало, «повторилось» — сработало снова.")
    with t3:
        can_edit = role.is_admin or role.key == "analyst"
        show = rules.assign(**{"Тип": rules["kind"].map(lambda k_: AL.KINDS.get(k_, (k_,))[0])})
        st.dataframe(pd.DataFrame({"Вкл.": show["active"].astype(bool), "Правило": show["name"],
                                   "Порог": [AL.fmt_value(t, AL.KINDS[k_][2]) for t, k_ in zip(show["threshold"], show["kind"])],
                                   "Канал": show["channel"], "Автор": show["created_by"]}), hide_index=True)
        if can_edit:
            with st.form("al_new", border=True):
                st.markdown("**Новое правило**")
                c1, c2, c3 = st.columns([2, 1, 1.4])
                kind = c1.selectbox("Тип", list(AL.KINDS), format_func=lambda k_: AL.KINDS[k_][0])
                unit = AL.KINDS[kind][2]
                thr = c2.number_input("Порог" + (" (доля, напр. 0,92)" if unit == "%" else ""),
                                      value=float(AL.KINDS[kind][1]), step=0.01 if unit == "%" else 1.0)
                ch = c3.selectbox("Канал", AL.CHANNELS, help="В прототипе работает только «Интерфейс BI»")
                name = st.text_input("Название", value=AL.KINDS[kind][0], max_chars=80)
                if st.form_submit_button("Добавить правило", icon=":material/add_alert:"):
                    state.add_rule(name, kind, thr, "сеть", ch, role.title)
                    ui.audit("Добавлено правило уведомлений", name)
                    st.rerun()
            rid = st.selectbox("Включить / выключить правило", rules["id"].tolist(),
                               format_func=lambda i: rules.set_index("id").loc[i, "name"], key="al_toggle_pick")
            cur = bool(rules.set_index("id").loc[rid, "active"])
            if st.button("Выключить" if cur else "Включить", key="al_toggle"):
                state.set_rule_active(int(rid), not cur)
                ui.audit("Изменено правило уведомлений", rules.set_index("id").loc[rid, "name"],
                         "выключено" if cur else "включено")
                st.rerun()
        else:
            st.caption("Правила настраивают аналитик и администратор BI.")
        st.caption("Каналы доставки: в прототипе — интерфейс BI; e-mail, Telegram, MAX, NextCloud Talk и задачи "
                   "в Битрикс24 подключаются в целевой системе через API по тем же правилам.")
    ui.export_panel(PAGE)
