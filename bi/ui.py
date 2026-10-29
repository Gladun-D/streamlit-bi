"""Оформление: форматирование чисел, KPI-карточки, единый стиль графиков (светлая и тёмная тема),
шапка страницы (избранное, ссылка, комментарии, описание отчёта), подсветка таблиц, выгрузка."""

from __future__ import annotations

import io
import itertools
import math
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Палитра (проверенный набор: категориальные слоты в фиксированном порядке, статусные — отдельно)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "🟢", "warning": "🟡", "critical": "🔴", "none": "⚪"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#b9b7ae"
CHAIN_COLOR = {"G": SERIES[0], "P": SERIES[1]}

_FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def _template(dark: bool) -> go.layout.Template:
    ink, ink2, muted = ("#f0efec", "#c3c2b7", "#9a988f") if dark else (INK, INK2, MUTED)
    grid, axis = ("#2c2c2a", "#383835") if dark else (GRID, AXIS)
    return go.layout.Template(layout=dict(
        font=dict(family=_FONT, size=13, color=ink2),
        title=dict(font=dict(size=15, color=ink), x=0, xanchor="left", xref="paper", y=0.985, yanchor="top",
                   yref="container"),
        colorway=SERIES_DARK if dark else SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a1a19" if dark else "#ffffff",
        margin=dict(l=8, r=12, t=72, b=8),
        xaxis=dict(showgrid=False, linecolor=axis, tickcolor=axis, ticks="outside", ticklen=4, automargin=True,
                   tickfont=dict(color=muted), title=dict(font=dict(color=muted))),
        yaxis=dict(gridcolor=grid, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=muted),
                   automargin=True, title=dict(font=dict(color=muted))),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None,
                    font=dict(color=ink2)),
        hoverlabel=dict(bgcolor="#262624" if dark else "white", bordercolor=grid, font=dict(color=ink, size=12)),
        hovermode="closest",
        separators=", ",
        bargap=0.25,
        geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#262624" if dark else "#f1f0ec",
                 countrycolor=axis, coastlinecolor=axis, showcountries=True, showland=True,
                 showlakes=False, subunitcolor=grid),
    ))


pio.templates["bi"] = _template(False)
pio.templates["bi_dark"] = _template(True)
pio.templates.default = "bi"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "locale": "ru",
    "topojsonURL": "/app/static/topojson/",   # карта мира из репозитория — без обращения к внешнему CDN
}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def chart(fig: go.Figure, height: int = 340, key: str | None = None, dates: bool = False, on_select=None,
          selection_mode=("points", "box"), export_name: str | None = None):
    """Рисует график в едином стиле. on_select='rerun' — включает выбор элементов (кросс-фильтр, детализация)."""
    dark = is_dark()
    fig.update_layout(template="bi_dark" if dark else "bi", height=height,
                      plot_bgcolor="#1a1a19" if dark else "#ffffff", paper_bgcolor="rgba(0,0,0,0)")
    if dates:
        fig.update_xaxes(tickformat="%d.%m.%y", hoverformat="%d.%m.%Y")
    if not export_name:   # по умолчанию в выгрузку попадает каждый график страницы под своим заголовком
        try:
            export_name = fig.layout.title.text or key
        except Exception:
            export_name = key
    if export_name:
        collect_figure(export_name, fig)
    if on_select:
        return st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key, on_select=on_select,
                               selection_mode=selection_mode)
    return st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


# ---------------------------------------------------------------------------
# Форматирование (русская локаль: пробел — разделитель разрядов, запятая — десятичных)
# ---------------------------------------------------------------------------
def _num(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s.replace("-", "−")


def rub(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    a = abs(x)
    if a >= 1e9:
        return f"{_num(x / 1e9, digits)} млрд ₽"
    if a >= 1e6:
        return f"{_num(x / 1e6, digits)} млн ₽"
    if a >= 1e3:
        return f"{_num(x / 1e3, digits)} тыс. ₽"
    return f"{_num(x, 0)} ₽"


def num(x: float, digits: int = 0) -> str:
    return _num(x, digits)


def big(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    a = abs(x)
    if a >= 1e6:
        return f"{_num(x / 1e6, digits)} млн"
    if a >= 1e4:
        return f"{_num(x / 1e3, digits)} тыс."
    return _num(x, 0)


def pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{_num(x * 100, digits)}%"


def delta_pct(x: float, suffix: str = "г/г", digits: int = 1) -> str | None:
    if x is None or not np.isfinite(x):
        return None
    sign = "+" if x > 0 else ""
    return f"{sign}{_num(x * 100, digits)}% {suffix}".strip()


def delta_pp(x: float, suffix: str = "г/г", digits: int = 1) -> str | None:
    if x is None or not np.isfinite(x):
        return None
    sign = "+" if x > 0 else ""
    return f"{sign}{_num(x * 100, digits)} п.п. {suffix}".strip()


def delta_abs(x: float, unit: str = "", suffix: str = "г/г", digits: int = 1) -> str | None:
    if x is None or not np.isfinite(x):
        return None
    sign = "+" if x > 0 else ""
    return f"{sign}{_num(x, digits)}{unit} {suffix}".strip()


# ---------------------------------------------------------------------------
# Сессия: роль, права, пороги подсветки
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "plan_green": 1.00, "plan_yellow": 0.97,      # выполнение плана: ≥ зелёный, ≥ жёлтый, иначе красный
    "lfl_yellow": 0.0, "lfl_red": -0.03,          # LFL: ≥ 0 зелёный, ≥ −3% жёлтый, иначе красный
    "osa_green": 0.95, "osa_red": 0.92,           # OSA: ≥ 95% зелёный, ≥ 92% жёлтый, иначе красный
    "top_n": 10,
}


def thresholds() -> dict:
    t = dict(DEFAULT_THRESHOLDS)
    t.update(st.session_state.get("thresholds", {}))
    return t


def status(value: float, kind: str) -> str:
    """good / warning / critical / none по настраиваемым порогам."""
    if value is None or not np.isfinite(value):
        return "none"
    t = thresholds()
    if kind == "plan":
        return "good" if value >= t["plan_green"] else "warning" if value >= t["plan_yellow"] else "critical"
    if kind == "lfl":
        return "good" if value >= t["lfl_yellow"] else "warning" if value >= t["lfl_red"] else "critical"
    if kind == "osa":
        return "good" if value >= t["osa_green"] else "warning" if value >= t["osa_red"] else "critical"
    return "none"


def role():
    from bi.access import ROLES, DEFAULT_ROLE
    return ROLES[st.session_state.get("role_key", DEFAULT_ROLE)]


def user_label() -> str:
    return f"Демо: {role().title}"


def can_export() -> bool:
    return role().can_export


def audit(action: str, obj: str = "", details: str = "") -> None:
    from bi import state
    state.audit(st.session_state.get("sid", "—"), role().key, action, obj, details)


# ---------------------------------------------------------------------------
# Стили
# ---------------------------------------------------------------------------
CSS_COMMON = """
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }
h1 { font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
h2, h3 { font-weight: 620 !important; letter-spacing: -0.005em; }
[data-testid="stMetric"] { border-radius: 12px; }
[class*="st-key-bikpi_"] [data-testid="stMetric"] { background: transparent !important; }
[data-testid="stMetricLabel"] p { font-size: 0.86rem; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 650; }
.bi-caption { opacity: .75; font-size: 0.82rem; margin: -0.4rem 0 0.4rem 0; }
.bi-brand { font-weight: 700; font-size: 1.05rem; margin: 0 0 0.1rem 0; }
.bi-sub { opacity: .7; font-size: 0.78rem; margin-bottom: 0.4rem; }
.bi-badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.74rem; margin: 0 6px 4px 0; }
.bi-step { color:#2a78d6; font-weight:650; font-size:0.8rem; letter-spacing:0.04em; text-transform:uppercase; }
.bi-crumbs { font-size: 0.85rem; opacity: .8; }
.bi-tile { border-radius: 12px; padding: 0.8rem 1rem; height: 100%; }
.bi-tile h4 { margin: 0 0 0.2rem 0; font-size: 1rem; }
.bi-tile p { margin: 0; font-size: 0.82rem; opacity: .75; }
</style>
"""
CSS_LIGHT = """
<style>
:root { --bi-ink:#0b0b0b; --bi-ink2:#52514e; --bi-line:#e1e0d9; }
[data-testid="stMetric"] { background: #ffffff; border-color: var(--bi-line) !important; }
[class*="st-key-bikpi_"] { background: #ffffff; }
[data-testid="stMetricValue"] { color: var(--bi-ink); }
[data-testid="stSidebar"] { border-right: 1px solid var(--bi-line); }
.bi-badge { background:#fff4e5; color:#8a5300; border:1px solid #f5d9a8; }
.bi-badge.blue { background:#eef5fd; color:#1c5cab; border-color:#cde2fb; }
.bi-badge.green { background:#eef8ee; color:#006300; border-color:#c7e8c7; }
.bi-insight { border-left: 3px solid #2a78d6; background:#f6f9fd; padding: 0.7rem 0.9rem; border-radius: 6px;
              margin: 0.3rem 0 0.6rem 0; color: var(--bi-ink); }
.bi-insight.warn { border-left-color: #ec835a; background:#fdf6f2; }
.bi-insight.good { border-left-color: #0ca30c; background:#f3faf3; }
.bi-tile { background:#ffffff; border:1px solid var(--bi-line); }
</style>
"""
CSS_DARK = """
<style>
:root { --bi-ink:#f0efec; --bi-ink2:#c3c2b7; --bi-line:#383835; }
[data-testid="stMetric"] { background: #1f1f1d; border-color: var(--bi-line) !important; }
[class*="st-key-bikpi_"] { background: #1f1f1d; }
[data-testid="stSidebar"] { border-right: 1px solid var(--bi-line); }
.bi-badge { background:#3a2a10; color:#f5c97a; border:1px solid #5a4318; }
.bi-badge.blue { background:#13283f; color:#9ec5f4; border-color:#1c3d63; }
.bi-badge.green { background:#12301a; color:#8fd98f; border-color:#1d4d27; }
.bi-insight { border-left: 3px solid #3987e5; background:#16212e; padding: 0.7rem 0.9rem; border-radius: 6px;
              margin: 0.3rem 0 0.6rem 0; color: var(--bi-ink); }
.bi-insight.warn { border-left-color: #ec835a; background:#2c1d16; }
.bi-insight.good { border-left-color: #0ca30c; background:#14271a; }
.bi-tile { background:#1f1f1d; border:1px solid var(--bi-line); }
</style>
"""


def inject_css():
    st.markdown(CSS_COMMON + (CSS_DARK if is_dark() else CSS_LIGHT), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Шапка страницы: избранное, ссылка, комментарии, описание
# ---------------------------------------------------------------------------
def share_url(page_key: str) -> str:
    base = ""
    try:
        base = str(st.context.url or "").split("?")[0]
    except Exception:
        pass
    params = st.session_state.get("_share_params", {})
    q = urlencode({k: v for k, v in params.items() if v not in ("", None)}, safe=",")
    return f"{base}?{q}" if q else base


def page_header(title: str, subtitle: str, f=None, page_key: str | None = None):
    st.session_state["_export"] = {"title": title, "kpis": [], "tables": {}, "figures": {}, "filters": f}
    if page_key:
        _track_view(page_key, title)
    st.title(title)
    extra = f" · {f.start:%d.%m.%Y} — {f.end:%d.%m.%Y} ({f.days} дн.)" if f is not None else ""
    st.markdown(f'<div class="bi-caption">{subtitle}{extra}</div>', unsafe_allow_html=True)
    if page_key:
        _actions(page_key, title)
    if page_key:
        from bi.docs import REPORTS
        doc = REPORTS.get(page_key)
        if doc:
            with st.expander("Об этом отчёте: назначение, источники, как работать", icon=":material/menu_book:"):
                st.markdown(doc)


def _track_view(page_key: str, title: str):
    recent = st.session_state.setdefault("recent", [])
    if not recent or recent[0] != page_key:
        if page_key in recent:
            recent.remove(page_key)
        recent.insert(0, page_key)
        del recent[8:]
    if st.session_state.get("_last_logged_page") != page_key:
        st.session_state["_last_logged_page"] = page_key
        audit("Просмотр отчёта", title)


def _actions(page_key: str, title: str):
    from bi import state
    favs = st.session_state.setdefault("favorites", list(role().default_favorites))
    row = st.container(horizontal=True, gap="small", vertical_alignment="center")
    with row:
        is_fav = page_key in favs
        if st.button("В избранном" if is_fav else "В избранное", icon=":material/star:" if is_fav else
                     ":material/star_border:", key=f"fav_{page_key}", help="Закрепить отчёт на главной странице",
                     type="tertiary"):
            if is_fav:
                favs.remove(page_key)
            else:
                favs.append(page_key)
            st.rerun()
        with st.popover("Поделиться", icon=":material/link:", type="tertiary"):
            st.caption("Ссылка открывает этот отчёт с текущими фильтрами. Данные будут показаны "
                       "в рамках прав того, кто откроет ссылку.")
            st.code(share_url(page_key), language=None, wrap_lines=True)
        n_new = 0
        try:
            cm = state.comments(page_key)
            n_new = int((cm["status"] == "новый").sum())
        except Exception:
            cm = pd.DataFrame()
        with st.popover(f"Комментарии · {n_new}" if n_new else "Комментарии", icon=":material/chat:", type="tertiary"):
            _comments_panel(page_key, cm)


def _comments_panel(page_key: str, cm: pd.DataFrame):
    from bi import state
    widgets = ["Отчёт целиком"] + list(st.session_state.get("_widgets", {}).get(page_key, []))
    with st.form(f"cm_form_{page_key}", clear_on_submit=True, border=False):
        w = st.selectbox("К чему", widgets, key=f"cm_w_{page_key}")
        txt = st.text_area("Комментарий", max_chars=500, key=f"cm_t_{page_key}", height=80)
        if st.form_submit_button("Добавить", icon=":material/add_comment:"):
            state.add_comment(page_key, w, txt, user_label())
            audit("Комментарий", page_key, w)
            st.rerun()
    if cm.empty:
        st.caption("Комментариев пока нет.")
        return
    for _, r in cm.head(20).iterrows():
        badge = "🟠 новый" if r["status"] == "новый" else "✅ решён"
        st.markdown(f"**{r['author']}** · {r['created_at'][:16]} · _{r['widget']}_ · {badge}")
        st.text(r["text"])
        if r["status"] == "новый" and st.button("Отметить решённым", key=f"cm_res_{r['id']}", type="tertiary"):
            state.set_comment_status(int(r["id"]), "решён")
            st.rerun()


def register_widgets(page_key: str, names: list[str]):
    st.session_state.setdefault("_widgets", {})[page_key] = names


def signed(delta: str | None) -> str | None:
    """Streamlit определяет направление стрелки по ведущему ASCII-минусу — заменяем типографский минус."""
    if delta is None:
        return None
    return "-" + delta[1:] if delta.startswith("−") else delta


_KPI_SEQ = itertools.count()
KPI_H = 196  # высота карточки со спарклайном — карточки без графика в том же ряду выравниваются по ней


def kpi(col, label: str, value: str, delta: str | None = None, good_when_up: bool | None = True,
        help: str | None = None, spark: list | None = None, arrow: str = "auto", height="content",
        insight: str | None = None):
    """KPI-карточка. insight — строка AI-подсказки под значением (прогноз, ожидание)."""
    color = "off" if good_when_up is None else ("normal" if good_when_up else "inverse")
    has_spark = spark is not None and len(spark) > 2
    if st.session_state.get("mobile"):
        height = "content"   # на телефоне карточки идут столбиком — выравнивать высоту не нужно
    with col:
        if insight:
            # AI-подсказка — отдельной строкой внутри карточки (класс bikpi — фон как у обычной карточки)
            with st.container(border=True, gap=None, height=height if isinstance(height, int) else "content",
                              key=f"bikpi_{next(_KPI_SEQ)}"):
                st.metric(label, value, signed(delta), delta_color=color, help=help, border=False,
                          delta_arrow=arrow, chart_data=spark if has_spark else None, chart_type="area")
                st.caption(f":material/auto_awesome: {insight}")
        else:
            st.metric(label, value, signed(delta), delta_color=color, help=help, border=True, delta_arrow=arrow,
                      chart_data=spark if has_spark else None, chart_type="area",
                      height=KPI_H if has_spark else height)
    exp = st.session_state.get("_export")
    if exp is not None:
        exp["kpis"].append((label, value, delta or ""))


def insight(text: str, kind: str = "info"):
    cls = {"info": "", "warn": " warn", "good": " good"}[kind]
    st.markdown(f'<div class="bi-insight{cls}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Подсветка таблиц (условное форматирование)
# ---------------------------------------------------------------------------
_BG = {"good": "rgba(12,163,12,0.16)", "warning": "rgba(250,178,25,0.22)", "critical": "rgba(208,59,59,0.20)",
       "none": ""}


def styled(df: pd.DataFrame, status_cols: dict[str, str] | None = None, top_col: str | None = None,
           top_n: int | None = None, top_desc: bool = True):
    """status_cols: {колонка: вид порога (plan/lfl/osa)} — заливка светофором;
    top_col: подсветить лучшие и худшие top_n строк по колонке (жирный шрифт и маркер)."""
    sty = df.style
    money = [c for c in df.columns if isinstance(c, str) and ("млн ₽" in c or "тыс ₽" in c or "тыс. ₽" in c)]
    if money:   # русский формат чисел в денежных колонках (пробел — разряды, запятая — дробь)
        sty = sty.format({c: (lambda v: "—" if pd.isna(v) else f"{v:,.1f}".replace(",", " ").replace(".", ","))
                          for c in money})
    for col, kind in (status_cols or {}).items():
        if col in df.columns:
            sty = sty.map(lambda v, k=kind: f"background-color: {_BG[status(v, k)]}" if _BG[status(v, k)] else "",
                          subset=[col])
    if top_col and top_col in df.columns and len(df) > 2:
        n = min(top_n or thresholds()["top_n"], max(1, len(df) // 3))
        s = pd.to_numeric(df[top_col], errors="coerce")
        best = set(s.nlargest(n).index if top_desc else s.nsmallest(n).index)
        worst = set(s.nsmallest(n).index if top_desc else s.nlargest(n).index)

        def row_style(r):
            if r.name in best:
                return ["border-left: 3px solid #0ca30c; font-weight: 600" if i == 0 else "font-weight: 600"
                        for i in range(len(r))]
            if r.name in worst:
                return ["border-left: 3px solid #d03b3b" if i == 0 else "" for i in range(len(r))]
            return [""] * len(r)
        sty = sty.apply(row_style, axis=1)
    return sty


# ---------------------------------------------------------------------------
# Выгрузка
# ---------------------------------------------------------------------------
def collect_table(name: str, df: pd.DataFrame):
    exp = st.session_state.get("_export")
    if exp is not None:
        exp["tables"][name] = df


def collect_figure(name: str, fig: go.Figure):
    exp = st.session_state.get("_export")
    if exp is not None:
        exp["figures"][name] = fig


def to_excel(frames: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        used = set()
        for name, df in frames.items():
            sheet = (name[:28] or "Лист").replace("/", "-").replace(":", " ")
            while sheet in used:
                sheet = sheet[:26] + f"_{len(used)}"
            used.add(sheet)
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]) and getattr(out[c].dt, "tz", None):
                    out[c] = out[c].dt.tz_localize(None)
            out.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            for i, c in enumerate(out.columns, start=1):
                width = min(max([len(str(c))] + [len(str(v)) for v in out[c].head(200)]) + 2, 48)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
    return buf.getvalue()


def download(frames: dict[str, pd.DataFrame], file_name: str, label: str = "Excel", key: str | None = None):
    for n, df in frames.items():
        collect_table(n, df)
    if not can_export():
        st.button(label, icon=":material/block:", disabled=True, key=key,
                  help="Экспорт недоступен для вашей роли")
        return
    st.download_button(label, to_excel(frames), file_name=file_name, icon=":material/download:",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=key,
                       on_click=lambda: audit("Экспорт", file_name, "xlsx"))


def export_panel(page_key: str):
    """Экспорт страницы целиком: Excel, CSV, XML, PDF, PowerPoint."""
    from bi import export
    exp = st.session_state.get("_export") or {}
    st.divider()
    if not can_export():
        st.caption(":material/block: Экспорт отчёта недоступен для вашей роли.")
        return
    with st.expander("Экспорт отчёта: Excel · CSV · XML · PDF · PowerPoint", icon=":material/ios_share:"):
        tables = exp.get("tables", {})
        figures = exp.get("figures", {})
        st.caption(f"В выгрузку попадут: {len(exp.get('kpis', []))} показателей, {len(figures)} графиков, "
                   f"{len(tables)} таблиц. Графики по отдельности сохраняются в PNG кнопкой «камера» над графиком.")
        fmt = st.segmented_control("Формат", ["Excel", "CSV", "XML", "PDF", "PowerPoint"], default="Excel",
                                   key=f"exp_fmt_{page_key}")
        if st.button("Сформировать файл", icon=":material/build:", key=f"exp_build_{page_key}"):
            with st.spinner("Формирую файл…"):
                data, fname, mime = export.build(fmt or "Excel", exp)
            st.session_state[f"exp_file_{page_key}"] = (data, fname, mime)
        built = st.session_state.get(f"exp_file_{page_key}")
        if built:
            data, fname, mime = built
            st.download_button(f"Скачать {fname}", data, file_name=fname, mime=mime, icon=":material/download:",
                               key=f"exp_dl_{page_key}", on_click=lambda: audit("Экспорт", fname, fmt or ""))


def empty_state(msg: str = "Нет данных для выбранных фильтров"):
    st.info(msg, icon=":material/filter_alt_off:")
