"""Смоук-тест: каждая страница отрисовывается без ошибок при разных фильтрах.

Запуск: pytest -q  (первый запуск сгенерирует данные, ~15 секунд)
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "story", "sales", "assortment", "stock", "logistics", "about"]
COMBOS = {
    "13 недель": "Q.Filters(D1 - timedelta(days=90), D1)",
    "сеть А": "Q.Filters(D1 - timedelta(days=90), D1, chains=('G',))",
    "регион + категории": "Q.Filters(date(2026, 1, 1), D1, chains=('P',), regions=('Самарская обл.',), cats=(1, 2))",
    "без базы прошлого года": "Q.Filters(date(2025, 2, 1), date(2025, 4, 30))",
    "один магазин": "Q.Filters(D1 - timedelta(days=27), D1, stores=(5,))",
}
TEMPLATE = """
import sys
sys.path.insert(0, {root!r})
import streamlit as st
from datetime import date, timedelta
from bi import queries as Q, ui
from views import {page}
ui.inject_css()
D0, D1 = Q.data_bounds()
F = Q.resolve({flt})
st.session_state["F"] = F
{page}.render(F)
"""


@pytest.mark.parametrize("combo", list(COMBOS))
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, combo):
    at = AppTest.from_string(TEMPLATE.format(root=str(ROOT), page=page, flt=COMBOS[combo]), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_entrypoint():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Обзор сети"
    assert len(at.metric) >= 10
