"""Смоук-тест: каждая страница отрисовывается без ошибок при разных фильтрах и ролях.

Запуск: pytest -q  (первый запуск сгенерирует данные, ~15 секунд; весь набор — 10–15 минут)
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["home", "overview", "story", "alerts_page", "drill", "compare", "sales", "assortment", "stock", "logistics",
         "quality", "forecast_page", "ai_page", "builder", "kpi_library", "admin", "about"]
COMBOS = {
    "13 недель": "D1 - timedelta(days=90), D1",
    "сеть Б: округ, категории, условие": "date(2026, 1, 1), D1, chains=('P',), okrugs=('Центральный ФО',), "
                                         "cats=(2, 3), condition='lfl_neg'",
    "без базы прошлого года": "date(2025, 2, 1), date(2025, 4, 30)",
}
EXTRA_COMBOS = {
    "один магазин": "D1 - timedelta(days=27), D1, stores=(5,)",
    "поставщики и зона склада": "D1 - timedelta(days=90), D1, suppliers=(5, 10), zones=('Охлаждённый (фреш)',)",
}
TEMPLATE = """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from datetime import date, timedelta
from bi import ui, queries as Q, access as A, state
from bi.seed import seed
from views import {page}
ui.inject_css()
Q.backend(); state.init(seed)
D0, D1 = Q.data_bounds()
st.session_state["role_key"] = {role!r}
R = A.ROLES[{role!r}]
S = A.role_scope(R, Q.dims()["store"], state.grants(), D1)
st.session_state["SCOPE"] = S
st.session_state["allowed_pages"] = list(R.pages)
F = Q.resolve(Q.Filters({flt}), scope=S, thresholds=ui.thresholds())
st.session_state["F"] = F
{page}.render(F)
"""


def _run(page: str, role: str, flt: str) -> AppTest:
    at = AppTest.from_string(TEMPLATE.format(root=str(ROOT), page=page, role=role, flt=flt), default_timeout=400)
    at.run()
    return at


@pytest.mark.parametrize("combo", list(COMBOS))
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_admin(page, combo):
    at = _run(page, "admin", COMBOS[combo])
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("combo", list(EXTRA_COMBOS))
@pytest.mark.parametrize("page", ["overview", "drill", "stock", "logistics", "quality", "builder"])
def test_page_renders_narrow_filters(page, combo):
    at = _run(page, "analyst", EXTRA_COMBOS[combo])
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("role", ["network", "okrug", "division", "ugm", "category", "logist", "analyst"])
@pytest.mark.parametrize("page", ["home", "overview", "drill", "alerts_page"])
def test_roles_render(page, role):
    at = _run(page, role, COMBOS["13 недель"])
    assert not at.exception, [e.value for e in at.exception]


def test_app_entrypoint():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=400)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Главная"
    assert len(at.metric) >= 4
