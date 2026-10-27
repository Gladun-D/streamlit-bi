"""Оформление: форматирование чисел, KPI-карточки, единый стиль графиков, выгрузка."""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Палитра (проверенный набор: категориальные слоты в фиксированном порядке, статусные — отдельно)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#f0efec", "#2a78d6"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#b9b7ae"
PAGE_BG = "#f9f9f7"   # = theme.backgroundColor в .streamlit/config.toml

CHAIN_COLOR = {"G": SERIES[0], "P": SERIES[1]}

pio.templates["bi"] = go.layout.Template(
    layout=dict(
        font=dict(family='system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', size=13, color=INK2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", xref="paper", y=0.985, yanchor="top",
                   yref="container"),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=8, r=12, t=72, b=8),
        xaxis=dict(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=4, automargin=True,
                   tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED), automargin=True,
                   title=dict(font=dict(color=MUTED))),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None,
                    font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)),
        hovermode="closest",
        separators=", ",
        bargap=0.25,
    )
)
pio.templates.default = "bi"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "locale": "ru",
}


def chart(fig: go.Figure, height: int = 340, key: str | None = None, dates: bool = False):
    fig.update_layout(template="bi", height=height, plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)")
    if dates:
        fig.update_xaxes(tickformat="%d.%m.%y", hoverformat="%d.%m.%Y")
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


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
# Компоненты
# ---------------------------------------------------------------------------
CSS = """
<style>
:root { --bi-ink:#0b0b0b; --bi-ink2:#52514e; --bi-muted:#898781; --bi-line:#e1e0d9; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }
h1 { font-size: 1.65rem !important; font-weight: 650 !important; letter-spacing: -0.01em; }
h2, h3 { font-weight: 620 !important; letter-spacing: -0.005em; }
[data-testid="stMetric"] { background: #ffffff; border-color: var(--bi-line) !important; border-radius: 12px; }
[data-testid="stMetricLabel"] p { color: var(--bi-ink2); font-size: 0.86rem; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 650; color: var(--bi-ink); }
[data-testid="stSidebar"] { border-right: 1px solid var(--bi-line); }
.bi-caption { color: var(--bi-muted); font-size: 0.82rem; margin: -0.4rem 0 0.8rem 0; }
.bi-brand { font-weight: 700; font-size: 1.05rem; color: var(--bi-ink); margin: 0 0 0.1rem 0; }
.bi-sub { color: var(--bi-muted); font-size: 0.78rem; margin-bottom: 0.4rem; }
.bi-badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.74rem;
            background:#fff4e5; color:#8a5300; border:1px solid #f5d9a8; margin-right:6px; }
.bi-badge.blue { background:#eef5fd; color:#1c5cab; border-color:#cde2fb; }
.bi-insight { border-left: 3px solid #2a78d6; background:#f6f9fd; padding: 0.7rem 0.9rem; border-radius: 6px;
              margin: 0.3rem 0 0.6rem 0; color: var(--bi-ink); }
.bi-insight.warn { border-left-color: #ec835a; background:#fdf6f2; }
.bi-insight.good { border-left-color: #0ca30c; background:#f3faf3; }
.bi-step { color:#2a78d6; font-weight:650; font-size:0.8rem; letter-spacing:0.04em; text-transform:uppercase; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, f=None):
    st.title(title)
    extra = ""
    if f is not None:
        extra = f" · {f.start:%d.%m.%Y} — {f.end:%d.%m.%Y} ({f.days} дн.)"
    st.markdown(f'<div class="bi-caption">{subtitle}{extra}</div>', unsafe_allow_html=True)


def signed(delta: str | None) -> str | None:
    """Streamlit определяет направление стрелки по ведущему ASCII-минусу — заменяем типографский минус."""
    if delta is None:
        return None
    return "-" + delta[1:] if delta.startswith("−") else delta


KPI_H = 196  # высота карточки со спарклайном — карточки без графика в том же ряду выравниваются по ней


def kpi(col, label: str, value: str, delta: str | None = None, good_when_up: bool | None = True,
        help: str | None = None, spark: list | None = None, arrow: str = "auto", height="content"):
    color = "off" if good_when_up is None else ("normal" if good_when_up else "inverse")
    has_spark = spark is not None and len(spark) > 2
    with col:
        st.metric(label, value, signed(delta), delta_color=color, help=help, border=True, delta_arrow=arrow,
                  chart_data=spark if has_spark else None, chart_type="area",
                  height=KPI_H if has_spark else height)


def insight(text: str, kind: str = "info"):
    cls = {"info": "", "warn": " warn", "good": " good"}[kind]
    st.markdown(f'<div class="bi-insight{cls}">{text}</div>', unsafe_allow_html=True)


def to_excel(frames: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in frames.items():
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.tz_localize(None) if getattr(out[c].dt, "tz", None) else out[c]
            out.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for i, c in enumerate(out.columns, start=1):
                width = min(max(len(str(c)), *(len(str(v)) for v in out[c].head(200))) + 2, 48)
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
    return buf.getvalue()


def download(frames: dict[str, pd.DataFrame], file_name: str, label: str = "Выгрузить в Excel", key: str | None = None):
    st.download_button(label, to_excel(frames), file_name=file_name, icon=":material/download:",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=key,
                       on_click="ignore")


def empty_state(msg: str = "Нет данных для выбранных фильтров"):
    st.info(msg, icon=":material/filter_alt_off:")
