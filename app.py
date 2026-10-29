"""Товародвижение · BI — точка входа Streamlit.

Запуск: streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

st.set_page_config(page_title="Товародвижение · BI", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="auto")

from bi import config as C  # noqa: E402
from bi import queries as Q  # noqa: E402
from bi import ui  # noqa: E402
from bi.db import setting  # noqa: E402
from views import about, assortment, logistics, overview, sales, stock, story  # noqa: E402

ui.inject_css()

try:
    Q.backend()
    D = Q.dims()
    D0, D1 = Q.data_bounds()
except Exception as exc:  # понятное сообщение вместо трассировки
    st.error("Не удалось подключиться к источнику данных.", icon=":material/error:")
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------------------------
# Навигация
# ---------------------------------------------------------------------------
def _page(module):
    def run():
        module.render(st.session_state["F"])
    run.__name__ = f"page_{module.__name__.split('.')[-1]}"
    return run


PAGES = {
    "Управление": [
        st.Page(_page(overview), title="Обзор сети", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(_page(story), title="Data story", icon=":material/auto_stories:", url_path="story"),
    ],
    "Анализ": [
        st.Page(_page(sales), title="Продажи", icon=":material/point_of_sale:", url_path="sales"),
        st.Page(_page(assortment), title="Ассортимент ABC/XYZ", icon=":material/category:", url_path="assortment"),
        st.Page(_page(stock), title="Запасы и пополнение", icon=":material/inventory_2:", url_path="stock"),
        st.Page(_page(logistics), title="Логистика", icon=":material/local_shipping:", url_path="logistics"),
    ],
    "Справка": [
        st.Page(_page(about), title="О данных и методика", icon=":material/info:", url_path="about"),
    ],
}
nav = st.navigation(PAGES, position="sidebar", expanded=True)


# ---------------------------------------------------------------------------
# Фильтры (общие для всех страниц, сохраняются при переходах)
# ---------------------------------------------------------------------------
PRESETS = {
    "Последние 4 недели": 28,
    "Последние 13 недель": 91,
    "Последние 26 недель": 182,
    "С начала года": "ytd",
    "Последние 52 недели": 364,
    "Весь период": "all",
    "Произвольный": "custom",
}
KEYS = ["f_preset", "f_range", "f_chains", "f_dcs", "f_regions", "f_stores", "f_cats"]


def _reset():
    for k in KEYS:
        st.session_state.pop(k, None)


def _keep_valid(key: str, options):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in set(options)]


names = C.chain_names()
stores, dcs, cats = D["store"], D["dc"], D["category"]

with st.sidebar:
    st.markdown('<div class="bi-brand">Товародвижение · BI</div>'
                '<div class="bi-sub">Продажи, запасы и логистика продуктовой розницы</div>', unsafe_allow_html=True)
    st.link_button("Открыть версию 2", setting("V2_URL", "https://retail-bi-v2.streamlit.app/"),
                   icon=":material/upgrade:", width="stretch", help="Прототип по функциональным требованиям: роли и доступ, конструктор "
                                         "отчётов, прогноз, AI-помощник, уведомления")
    st.caption(":material/history: Вы в версии 1")
    st.markdown('<span class="bi-badge">Синтетические данные</span>'
                f'<span class="bi-badge blue">по {D1:%d.%m.%Y}</span>', unsafe_allow_html=True)

    st.subheader("Фильтры", divider="gray")
    preset = st.selectbox("Период", list(PRESETS), index=1, key="f_preset")
    spec = PRESETS[preset]
    if spec == "custom":
        rng = st.date_input("Даты", value=(D1 - timedelta(days=90), D1), min_value=D0, max_value=D1,
                            format="DD.MM.YYYY", key="f_range")
        start, end = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (D1 - timedelta(days=90), D1))
    elif spec == "ytd":
        start, end = date(D1.year, 1, 1), D1
    elif spec == "all":
        start, end = D0, D1
    else:
        start, end = D1 - timedelta(days=spec - 1), D1
    start = max(start, D0)

    chains = st.pills("Сеть", ["G", "P"], selection_mode="multi", format_func=lambda c: names[c],
                      key="f_chains", help="Пусто — обе сети")
    chains = tuple(chains or ())

    dc_opts = dcs[dcs["chain"].isin(chains)] if chains else dcs
    _keep_valid("f_dcs", dc_opts["dc_id"].tolist())
    sel_dcs = st.multiselect("Склад / РЦ", dc_opts["dc_id"].tolist(),
                             format_func=lambda i: dcs.set_index("dc_id").loc[i, "name"],
                             placeholder="Все склады", key="f_dcs")

    st_opts = stores
    if chains:
        st_opts = st_opts[st_opts["chain"].isin(chains)]
    if sel_dcs:
        st_opts = st_opts[st_opts["dc_id"].isin(sel_dcs)]
    region_opts = sorted(st_opts["region"].unique())
    _keep_valid("f_regions", region_opts)
    sel_regions = st.multiselect("Регион", region_opts, placeholder="Все регионы", key="f_regions")
    if sel_regions:
        st_opts = st_opts[st_opts["region"].isin(sel_regions)]
    _keep_valid("f_stores", st_opts["store_id"].tolist())
    name_by_id = stores.set_index("store_id")["store_name"]
    sel_stores = st.multiselect("Магазин", st_opts["store_id"].tolist(), format_func=lambda i: name_by_id[i],
                                placeholder=f"Все магазины ({len(st_opts)})", key="f_stores")

    cat_name = cats.set_index("category_id")["category"]
    sel_cats = st.multiselect("Категория", cats["category_id"].tolist(), format_func=lambda i: cat_name[i],
                              placeholder="Все категории", key="f_cats")

    st.button("Сбросить фильтры", icon=":material/restart_alt:", on_click=_reset, width="stretch")
    st.caption(f"Источник: {Q.backend().name}")

F = Q.resolve(Q.Filters(start=start, end=end, chains=chains, dcs=tuple(sel_dcs), regions=tuple(sel_regions),
                        stores=tuple(sel_stores), cats=tuple(sel_cats)))
st.session_state["F"] = F

nav.run()
