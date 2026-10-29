"""Главная: единый портал — направления, избранное, последние отчёты, уведомления, свежесть данных."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from bi import access as A
from bi import alerts as AL
from bi import config as C
from bi import metrics as M
from bi import queries as Q
from bi import state, ui
from views import common

DIRECTIONS = [
    ("Продажи", ":material/point_of_sale:", ["overview", "drill", "sales", "compare"],
     "РТО, LFL, план, промо, детализация до магазина"),
    ("Ассортимент", ":material/category:", ["assortment", "builder"], "ABC/XYZ, Парето, конструктор отчётов"),
    ("Запасы", ":material/inventory_2:", ["stock"], "Покрытие, дефицит, излишки, точка заказа"),
    ("Логистика", ":material/local_shipping:", ["logistics"], "OTIF поставщиков, рейсы РЦ → магазин"),
    ("Качество и потери", ":material/verified:", ["quality"], "Брак, сроки годности, декларации"),
    ("Прогноз и AI", ":material/neurology:", ["forecast", "ai", "story"], "Прогнозы, аномалии, факторы, сценарии"),
]


def _link(key: str, label: str | None = None, icon: str | None = None):
    pages = st.session_state.get("page_obj", {})
    if key in pages:
        st.page_link(pages[key], label=label or A.PAGE_TITLES[key], icon=icon)


def render(f: Q.Filters):
    role = ui.role()
    ui.page_header("Главная", f"{role.title} · {st.session_state.get('SCOPE').label}", f, "home")
    allowed = st.session_state.get("allowed_pages", [])

    cur = common.load_period(f)
    if not cur.sales.empty and cur.sales["revenue"].sum() > 0:
        k = cur.k
        fc, plan, share = common.month_forecast(f)
        c = st.columns(4)
        ui.kpi(c[0], "РТО за период", ui.rub(k["revenue"]), None, good_when_up=None, height=130,
               insight=f"прогноз: {ui.pct(share)} плана" if np.isfinite(share) else None,
               help="Прогноз выполнения плана текущего месяца: факт с начала месяца + темп последних 7 дней")
        ui.kpi(c[1], "Валовая маржа", ui.pct(k["margin"]), None, height=130)
        ui.kpi(c[2], "OSA", ui.pct(1 - k["oos"]), None, height=130)
        ui.kpi(c[3], "Средний чек", ui.rub(k["avg_check"], 0), None, height=130)

    st.subheader("Направления")
    tiles = DIRECTIONS + [("HR", ":material/groups:", [], "Текучесть, эффективность персонала")]
    for row in (tiles[:4], tiles[4:]):
        cols = st.columns(4)
        for col, (title, icon, pages, desc) in zip(cols, row):
            with col.container(border=True, height="stretch"):
                st.markdown(f"**{title}**")
                st.caption(desc)
                vis = [p for p in pages if p in allowed]
                for p in vis[:3]:
                    _link(p)
                if title == "HR":
                    st.caption(":material/database_off: Нет данных в прототипе")
                elif not vis and st.session_state.get("mobile"):
                    st.caption(":material/desktop_windows: Доступно в полной версии")
                elif not vis:
                    st.caption(":material/lock: Нет доступа для роли")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Избранное")
        favs = [p for p in st.session_state.get("favorites", role.default_favorites) if p in allowed]
        if favs:
            for p in favs:
                _link(p, icon=":material/star:")
        else:
            st.caption("Закрепите отчёты кнопкой «В избранное» в шапке страницы.")
    with right:
        st.subheader("Недавние")
        rec = [p for p in st.session_state.get("recent", []) if p in allowed and p != "home"]
        if rec:
            for p in rec[:5]:
                _link(p, icon=":material/history:")
        else:
            st.caption("Здесь появятся отчёты, которые вы открывали в этой сессии.")

    if "alerts" in allowed:
        st.subheader("Уведомления")
        rules = state.alert_rules()
        al = AL.evaluate_all(f, rules)
        crit = int((al["Важность"] == "критично").sum())
        warn = int((al["Важность"] == "внимание").sum())
        c1, c2, c3 = st.columns([1, 1, 2])
        ui.kpi(c1, "Критичных", ui.num(crit), None, good_when_up=None)
        ui.kpi(c2, "Требуют внимания", ui.num(warn), None, good_when_up=None)
        with c3:
            if len(al):
                top = al.sort_values("Важность").groupby("Правило").size().sort_values(ascending=False).head(4)
                for rule, n in top.items():
                    st.markdown(f"- {rule}: **{n}**")
            _link("alerts", "Открыть центр уведомлений", ":material/arrow_forward:")

    with st.expander("Свежесть данных и источники", icon=":material/update:"):
        m = Q.meta()
        show = pd.DataFrame({"Витрина": m["description"], "Система-источник": m["source_system"],
                             "Обновление": m["refresh"], "Загружено": m["loaded_at"].dt.strftime("%d.%m.%Y %H:%M"),
                             "Данные по": pd.to_datetime(m["data_until"]).dt.strftime("%d.%m.%Y")})
        st.dataframe(show, hide_index=True)

    with st.expander(f"Что нового в версии {C.APP_VERSION}", icon=":material/new_releases:"):
        st.markdown(
            "- **Главная** с направлениями, избранным и недавними отчётами; уведомления и свежесть данных.\n"
            "- **Демо-роли**: директор сети, округа, дивизиона, УГМ, категорийный менеджер, логист, аналитик, "
            "администратор — свои страницы, свои данные, права на экспорт.\n"
            "- **Иерархия** «Сеть → Округ → Дивизион → УГМ → Магазин → Категория → SKU» с хлебными крошками, "
            "кросс-фильтром и детализацией по клику.\n"
            "- **Фильтры**: округ, дивизион, УГМ, РЦ, склад, поставщик, товар; загрузка списка магазинов из файла; "
            "условия («магазины с падением РТО»); сохранённые и общие наборы; ссылка с фильтрами.\n"
            "- **Конструктор отчётов**: сводные таблицы с подытогами, вычисляемые поля, подсветка, тип графика, "
            "сохранение и публикация, отмена и повтор, предпросмотр на телефоне.\n"
            "- **Прогноз** ETS и SARIMA с доверительными интервалами и историей точности; **AI-помощник**: "
            "аномалии с факторами, негативные тренды, топ-5 факторов, «что если» по цене.\n"
            "- **Уведомления**: правила, срабатывания по факту и прогнозу, история и частота причин.\n"
            "- **Качество и потери**: брак при приёмке, риск по срокам годности на РЦ, сроки деклараций.\n"
            "- **Библиотека показателей** с формулами, ответственными и пользовательскими KPI.\n"
            "- **Экспорт** в Excel, CSV, XML, PDF и PowerPoint; **тёмная тема** (меню ⋮ справа вверху); "
            "**мобильный режим**; комментарии к отчётам; аудит и мониторинг использования.\n"
            "- **Данные**: 1 РЦ у сети А и 6 РЦ у сети Б, склады по температурным зонам, история изменений "
            "оргструктуры.")
