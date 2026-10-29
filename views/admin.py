"""Администрирование: роли и доступ, временные доступы, журнал прав, аудит, мониторинг, данные, подключения."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bi import access as A
from bi import queries as Q
from bi import state, ui

PAGE = "admin"


def render(f: Q.Filters):
    role = ui.role()
    ui.page_header("Администрирование", "Доступ, аудит, мониторинг использования и свежесть данных", None, PAGE)
    if not role.is_admin:
        st.warning("Раздел доступен только администратору BI.", icon=":material/lock:")
        return
    t = st.tabs(["Роли и доступ", "Временный доступ", "Журнал прав", "Аудит действий", "Мониторинг", "Данные",
                 "Подключения"])
    stores = Q.dims()["store"]
    with t[0]:
        rows = []
        for r in A.ROLES.values():
            sc = A.role_scope(r, stores, state.grants(), Q.data_bounds()[1])
            rows.append({"Роль": r.title, "Данные (RLS)": sc.label, "Экспорт": r.can_export,
                         "Конструктор": r.can_build, "Администрирование": r.is_admin,
                         **{A.PAGE_TITLES[p]: (p in r.pages) for p in A.ALL_PAGES if p not in ("home", "about")}})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, column_config={c: st.column_config.CheckboxColumn(width="small")
                                                         for c in df.columns if df[c].dtype == bool})
        st.caption("Модель: роль = набор страниц (page-level security) + ограничение строк данных (row-level "
                   "security) + права на экспорт и публикацию. Все отчёты, в том числе пользовательские из "
                   "конструктора, наследуют ограничения роли. В целевой системе роли назначаются через SSO / "
                   "Active Directory по должности и подразделению; здесь — выбором демо-пользователя.")
        ui.collect_table("Роли и доступ", df)
    with t[1]:
        g = state.grants()
        st.dataframe(g.rename(columns={"role_key": "Роль", "scope_type": "Тип", "scope_value": "Объект",
                                       "valid_from": "С", "valid_to": "По", "reason": "Причина",
                                       "created_by": "Выдал", "created_at": "Когда"}).drop(columns=["id"]),
                     hide_index=True)
        ugms = sorted(stores["ugm"].unique())
        with st.form("grant_form", border=True):
            st.markdown("**Выдать временный доступ** (например, на время замещения)")
            c1, c2, c3, c4 = st.columns([1.2, 1.4, 1, 1])
            rk = c1.selectbox("Роль", ["ugm"], format_func=lambda k: A.ROLES[k].title)
            val = c2.selectbox("УГМ", ugms)
            d1 = Q.data_bounds()[1]
            vf = c3.date_input("С", d1, format="DD.MM.YYYY")
            vt = c4.date_input("По", d1 + timedelta(days=14), format="DD.MM.YYYY")
            reason = st.text_input("Причина", "Замещение", max_chars=200)
            if st.form_submit_button("Выдать", icon=":material/key:"):
                state.add_grant(rk, "ugm", val, str(vf), str(vt), reason, role.title)
                ui.audit("Выдан временный доступ", val, f"{vf} — {vt}")
                st.rerun()
        if len(g):
            gid = st.selectbox("Отозвать доступ", g["id"].tolist(),
                               format_func=lambda i: f"{g.set_index('id').loc[i, 'scope_value']} до "
                                                     f"{g.set_index('id').loc[i, 'valid_to']}", key="grant_rev")
            if st.button("Отозвать", key="grant_rev_btn"):
                state.revoke_grant(int(gid), role.title)
                ui.audit("Отозван временный доступ", str(gid))
                st.rerun()
        st.caption("Временный доступ автоматически перестаёт действовать после даты «По»: попробуйте демо-роль УГМ — "
                   "в её данных видна вторая группа магазинов.")
    with t[2]:
        j = state.rights_journal()
        st.dataframe(j.rename(columns={"ts": "Когда", "actor": "Кто", "action": "Действие", "details": "Детали"})
                     .drop(columns=["id"]), hide_index=True)
        st.caption("Все изменения прав и временных доступов фиксируются автоматически.")
    with t[3]:
        log = state.audit_log(30)
        c1, c2, c3 = st.columns(3)
        acts = c1.multiselect("Действие", sorted(log["action"].unique()), key="aud_act", placeholder="Все действия")
        rls = c2.multiselect("Роль", sorted(log["role"].unique()), key="aud_role", placeholder="Все роли",
                             format_func=lambda k: A.ROLES[k].title if k in A.ROLES else k)
        demo = c3.toggle("Показывать демонстрационную историю", value=True, key="aud_demo")
        v = log
        if acts:
            v = v[v["action"].isin(acts)]
        if rls:
            v = v[v["role"].isin(rls)]
        if not demo:
            v = v[v["demo"] == 0]
        show = pd.DataFrame({"Когда": v["ts"], "Сессия": v["session"], "Роль": v["role"].map(
            lambda k: A.ROLES[k].title if k in A.ROLES else k), "Действие": v["action"], "Объект": v["object"],
            "Детали": v["details"]})
        st.dataframe(show, hide_index=True, height=480)
        st.caption("Фиксируются вход, просмотр отчётов, экспорт, сохранение фильтров и отчётов, комментарии, изменения "
                   "показателей и правил. Персональные данные не пишутся: в прототипе — только демо-роль и "
                   "обезличенный номер сессии. Демонстрационная история помечена отдельно.")
        ui.collect_table("Аудит", show)
    with t[4]:
        log = state.audit_log(14)
        if log.empty:
            st.caption("Нет событий.")
        else:
            log["day"] = pd.to_datetime(log["ts"]).dt.normalize()
            k = st.columns(4)
            ui.kpi(k[0], "Сессий за 14 дней", ui.num(log["session"].nunique()), None, good_when_up=None)
            ui.kpi(k[1], "Просмотров отчётов", ui.num(int((log["action"] == "Просмотр отчёта").sum())), None,
                   good_when_up=None)
            ui.kpi(k[2], "Выгрузок", ui.num(int((log["action"] == "Экспорт").sum())), None, good_when_up=None)
            ui.kpi(k[3], "Ошибок загрузки", ui.num(int((log["action"] == "Ошибка загрузки отчёта").sum())), None,
                   good_when_up=None)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                pv = log[log["action"] == "Просмотр отчёта"]["object"].value_counts().sort_values()
                fig = go.Figure(go.Bar(y=pv.index, x=pv.values, orientation="h",
                                       marker=dict(color=ui.SERIES[0], cornerradius=3)))
                fig.update_layout(title="Популярность отчётов (просмотры)")
                ui.chart(fig, 380, key="mon_pop", export_name="Популярность отчётов")
            with c2:
                dd = log.groupby(["day", "action"]).size().unstack(fill_value=0)
                fig = go.Figure()
                for i, c in enumerate(dd.columns):
                    fig.add_bar(x=dd.index, y=dd[c], name=c, marker=dict(color=ui.SERIES[i % 8]))
                fig.update_layout(title="Активность по дням", barmode="stack")
                ui.chart(fig, 380, key="mon_days", dates=True, export_name="Активность по дням")
            byrole = log.groupby("role").agg(Сессий=("session", "nunique"), Событий=("id", "size")).reset_index()
            byrole["role"] = byrole["role"].map(lambda k_: A.ROLES[k_].title if k_ in A.ROLES else k_)
            st.dataframe(byrole.rename(columns={"role": "Роль"}).sort_values("Событий", ascending=False),
                         hide_index=True)
            err = log[log["action"] == "Ошибка загрузки отчёта"]
            if len(err):
                st.markdown("**Ошибки при загрузке отчётов**")
                st.dataframe(err[["ts", "object", "details"]].rename(columns={"ts": "Когда", "object": "Отчёт",
                                                                              "details": "Ошибка"}), hide_index=True)
    with t[5]:
        m = Q.meta()
        st.markdown("**Свежесть данных** (время последней загрузки из источников)")
        st.dataframe(pd.DataFrame({"Витрина": m["table_name"], "Описание": m["description"],
                                   "Источник": m["source_system"], "Регламент": m["refresh"],
                                   "Загружено": m["loaded_at"].dt.strftime("%d.%m.%Y %H:%M"),
                                   "Данные по": pd.to_datetime(m["data_until"]).dt.strftime("%d.%m.%Y")}),
                     hide_index=True)
        st.markdown("**История изменений оргструктуры**")
        h = Q.store_history()
        names = stores.set_index("store_id")
        st.dataframe(pd.DataFrame({"Дата": h["changed_on"].dt.date, "Магазин": h["store_id"].map(names["store_name"]),
                                   "Атрибут": h["attribute"], "Было": h["old_value"], "Стало": h["new_value"]})
                     .sort_values("Дата", ascending=False), hide_index=True, height=320)
        st.caption("Отчёты строятся по текущей оргструктуре: при переименовании магазина или переводе в другую УГМ "
                   "или дивизион история продаж автоматически отображается в новой структуре, итоги прошлых "
                   "периодов сохраняются.")
        st.markdown("**Проверки качества данных**")
        st.dataframe(Q.data_quality(), hide_index=True)
    with t[6]:
        st.markdown(f"Сейчас подключено: **{Q.backend().name}**")
        st.markdown(
            "| Источник | Как подключается | Статус в прототипе |\n|---|---|---|\n"
            "| DuckDB + Parquet | встроенный, по умолчанию | ✅ работает |\n"
            "| MySQL / MariaDB | `DATA_BACKEND=mysql`, `MYSQL_URL` | ✅ проверено на MariaDB 10.11 |\n"
            "| PostgreSQL, ClickHouse, MS SQL | `DATA_BACKEND=sql`, `SQL_URL` (SQLAlchemy) | ⚠️ подключается, SQL нужно "
            "проверить под диалект |\n"
            "| Excel / CSV | загрузка списков в фильтры, выгрузка отчётов | ✅ работает |\n"
            "| Уведомления: e-mail, Telegram, MAX, Битрикс24 | через API в целевой системе | ⏳ не подключено |\n")
        st.caption("Расчёты выполняются на стороне сервера и базы данных: скорость не зависит от компьютера "
                   "пользователя. Кэш отчётов — 1 час, оперативных данных (рейсы) — 15 минут.")
    ui.export_panel(PAGE)
