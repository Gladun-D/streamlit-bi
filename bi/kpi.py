"""Библиотека показателей (KPI) и безопасный вычислитель формул.

Формулы пишутся по-русски над базовыми полями витрины, например:
    (РТО - Себестоимость) / РТО
Допускаются только поля из BASE_FIELDS, числа, + − × /, скобки и унарный минус.
Показатель всегда считается как формула над суммами — поэтому корректно агрегируется
на любом уровне (магазин, дивизион, сеть) и автоматически пересчитывается за всю историю
при изменении формулы.
"""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

# Базовые поля витрины: имя в формуле -> (колонка в агрегате, описание, уровень)
BASE_FIELDS: dict[str, tuple[str, str, str]] = {
    "РТО": ("revenue", "Розничный товарооборот, ₽", "категория"),
    "Себестоимость": ("cost", "Себестоимость продаж, ₽", "категория"),
    "Штуки": ("qty", "Продажи в штуках", "категория"),
    "Промо_РТО": ("promo_revenue", "РТО товаров в акции, ₽", "категория"),
    "Промо_себестоимость": ("promo_cost", "Себестоимость товаров в акции, ₽", "категория"),
    "Списания": ("writeoff", "Списания в себестоимости, ₽", "категория"),
    "Остаток_сумма": ("stock_cost", "Сумма дневных остатков в себестоимости, ₽", "категория"),
    "OOS_взвеш": ("oos_w", "Доля позиций без остатка × РТО (для OSA)", "категория"),
    "Чеки": ("checks", "Количество чеков", "магазин"),
    "РТО_магазинов": ("store_revenue", "РТО магазинов целиком (для среднего чека и плана), ₽", "магазин"),
    "План": ("plan", "План РТО, ₽", "магазин"),
    "РТО_ПГ": ("revenue_py", "РТО за тот же период прошлого года, ₽", "категория"),
}

DIRECTIONS = ["Продажи", "Финансы", "Маркетинг", "Логистика", "Закупки", "Качество"]

# Стандартная (утверждённая) библиотека. builder=True — показатель доступен в конструкторе.
STANDARD = [
    dict(code="rto", name="РТО", formula="РТО", unit="₽", directions="Продажи,Финансы", good="up", builder=True,
         description="Розничный товарооборот: продажи в розничных ценах.", source="ERP · fact_sales",
         owner="Коммерческий директор", freq="ежедневно"),
    dict(code="gross", name="Валовая прибыль", formula="РТО - Себестоимость", unit="₽", directions="Финансы,Продажи",
         good="up", builder=True, description="РТО минус себестоимость проданных товаров.", source="ERP · fact_sales",
         owner="Финансовый директор", freq="ежедневно"),
    dict(code="margin", name="Валовая маржа, %", formula="(РТО - Себестоимость) / РТО", unit="%",
         directions="Финансы,Продажи", good="up", builder=True, description="Доля валовой прибыли в РТО.",
         source="ERP · fact_sales", owner="Финансовый директор", freq="ежедневно"),
    dict(code="checks", name="Чеки", formula="Чеки", unit="шт", directions="Продажи,Маркетинг", good="up", builder=True,
         description="Количество покупок — показатель трафика. Считается по магазину целиком.",
         source="Кассовая система · fact_store_day", owner="Операционный директор", freq="ежедневно"),
    dict(code="avg_check", name="Средний чек", formula="РТО_магазинов / Чеки", unit="₽", directions="Продажи,Маркетинг",
         good="up", builder=True, description="РТО магазинов / число чеков.",
         source="Кассовая система · fact_store_day", owner="Операционный директор", freq="ежедневно"),
    dict(code="osa", name="OSA, %", formula="1 - OOS_взвеш / РТО", unit="%", directions="Логистика,Продажи", good="up",
         builder=True, description="Доступность товара на полке: 1 − доля позиций без остатка, взвешенная по РТО.",
         source="ERP · fact_sales", owner="Директор по логистике", freq="ежедневно"),
    dict(code="promo_share", name="Доля промо в РТО, %", formula="Промо_РТО / РТО", unit="%",
         directions="Маркетинг,Продажи", good="none", builder=True, description="Доля РТО товаров в акции.",
         source="ERP · fact_sales", owner="Директор по маркетингу", freq="ежедневно"),
    dict(code="promo_margin", name="Маржа в промо, %", formula="(Промо_РТО - Промо_себестоимость) / Промо_РТО",
         unit="%", directions="Маркетинг,Финансы", good="up", builder=True,
         description="Валовая маржа товаров, проданных по акции.", source="ERP · fact_sales",
         owner="Директор по маркетингу", freq="ежедневно"),
    dict(code="regular_margin", name="Маржа вне промо, %",
         formula="(РТО - Промо_РТО - (Себестоимость - Промо_себестоимость)) / (РТО - Промо_РТО)", unit="%",
         directions="Маркетинг,Финансы", good="up", builder=True, description="Валовая маржа товаров без акции.",
         source="ERP · fact_sales", owner="Директор по маркетингу", freq="ежедневно"),
    dict(code="writeoff", name="Списания, %", formula="Списания / Себестоимость", unit="%",
         directions="Логистика,Финансы,Качество", good="down", builder=True,
         description="Списания в себестоимости к себестоимости продаж.", source="ERP · fact_sales",
         owner="Директор по логистике", freq="ежедневно"),
    dict(code="cover", name="Запас в магазинах, дн.", formula="Остаток_сумма / Себестоимость", unit="дн.",
         directions="Логистика", good="none", builder=True,
         description="Средний остаток в себестоимости / средняя дневная себестоимость продаж.",
         source="ERP · fact_sales", owner="Директор по логистике", freq="ежедневно"),
    dict(code="plan_pct", name="Выполнение плана РТО, %", formula="РТО_магазинов / План", unit="%",
         directions="Продажи,Финансы", good="up", builder=True,
         description="РТО магазинов к плану за дни периода.", source="Бюджетирование · fact_plan",
         owner="Финансовый директор", freq="ежемесячно (план), ежедневно (факт)"),
    dict(code="yoy", name="Рост РТО г/г, %", formula="РТО / РТО_ПГ - 1", unit="%", directions="Продажи", good="up",
         builder=True, description="РТО к тому же периоду прошлого года (сдвиг 364 дня), включая новые магазины.",
         source="ERP · fact_sales", owner="Коммерческий директор", freq="ежедневно"),
    # Показатели, которые считаются отдельными алгоритмами (не формулой над суммами витрины продаж)
    dict(code="lfl", name="LFL РТО, %", formula="РТО_сопоставимых / РТО_сопоставимых_ПГ - 1", unit="%",
         directions="Продажи", good="up", builder=False,
         description="Рост РТО магазинов, открытых не позже чем за 364 дня до начала периода.",
         source="ERP · fact_sales", owner="Коммерческий директор", freq="ежедневно"),
    dict(code="loyalty", name="Доля чеков с картой, %", formula="Чеки_с_картой / Чеки", unit="%",
         directions="Маркетинг", good="up", builder=False, description="Только сеть с программой лояльности.",
         source="Кассовая система · fact_store_day", owner="Директор по маркетингу", freq="ежедневно"),
    dict(code="otif", name="OTIF поставщиков, %", formula="Заказы_вовремя_и_полностью / Заказы", unit="%",
         directions="Логистика,Закупки", good="up", builder=False,
         description="Доля заказов, пришедших не позже обещанной даты и в объёме ≥ 98% заказанного.",
         source="ERP / закупки · fact_supplier_order", owner="Директор по закупкам", freq="ежечасно"),
    dict(code="lead_time", name="Срок поставки, дн.", formula="Среднее(Дата_прихода - Дата_заказа)", unit="дн.",
         directions="Логистика,Закупки", good="down", builder=False, description="Среднее по заказам поставщикам.",
         source="ERP / закупки · fact_supplier_order", owner="Директор по закупкам", freq="ежечасно"),
    dict(code="defect", name="Брак при приёмке, %", formula="Брак_шт / (Принято_шт + Брак_шт)", unit="%",
         directions="Качество,Закупки,Логистика", good="down", builder=False,
         description="Доля забракованного товара при приёмке на РЦ.", source="WMS · fact_supplier_order",
         owner="Директор по качеству", freq="ежечасно"),
    dict(code="trips_on_time", name="Рейсы в окно ±30 мин, %", formula="Рейсы_вовремя / Рейсы", unit="%",
         directions="Логистика", good="up", builder=False,
         description="Рейсы РЦ → магазин, прибывшие не позже 30 минут после плана.", source="TMS · fact_delivery",
         owner="Директор по логистике", freq="каждые 15 минут"),
]

_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div)


class FormulaError(ValueError):
    pass


def parse(formula: str) -> tuple[ast.Expression, set[str]]:
    text = (formula or "").strip().replace("×", "*").replace("÷", "/").replace("−", "-").replace(",", ".")
    if not text:
        raise FormulaError("Формула пустая")
    if len(text) > 300:
        raise FormulaError("Формула длиннее 300 символов")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Синтаксическая ошибка: {e.msg}") from None
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load)) or isinstance(node, _ALLOWED_BIN):
            continue
        if isinstance(node, (ast.UAdd, ast.USub)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN):
                raise FormulaError("Разрешены только + − × / и скобки")
            continue
        if isinstance(node, ast.UnaryOp):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            continue
        if isinstance(node, ast.Name):
            if node.id not in BASE_FIELDS:
                raise FormulaError(f"Неизвестное поле «{node.id}». Доступны: {', '.join(BASE_FIELDS)}")
            used.add(node.id)
            continue
        raise FormulaError("В формуле есть недопустимая конструкция")
    if not used:
        raise FormulaError("Формула должна использовать хотя бы одно поле")
    return tree, used


def _eval(node, df: pd.DataFrame):
    if isinstance(node, ast.Expression):
        return _eval(node.body, df)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        col = BASE_FIELDS[node.id][0]
        if col not in df:
            return pd.Series(np.nan, index=df.index)
        return df[col].astype(float)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, df)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, df), _eval(node.right, df)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        with np.errstate(divide="ignore", invalid="ignore"):
            if isinstance(b, pd.Series):
                return a / b.where(b != 0)
            return a / b if b != 0 else np.nan
    raise FormulaError("Недопустимая конструкция")


def evaluate(formula: str, df: pd.DataFrame) -> pd.Series:
    tree, _ = parse(formula)
    out = _eval(tree, df)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index, dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)


def fields_needed(formula: str) -> set[str]:
    return {BASE_FIELDS[n][0] for n in parse(formula)[1]}


def slug(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", "_", name.strip().lower()).strip("_")
    return ("u_" + s)[:40] or "u_kpi"


def library(custom: pd.DataFrame | None = None, include_archived: bool = False) -> pd.DataFrame:
    """Стандартные показатели + пользовательские (из хранилища)."""
    std = pd.DataFrame(STANDARD)
    std["kind"] = "Стандартный"
    std["archived"] = 0
    if custom is not None and len(custom):
        c = custom.rename(columns={"created_by": "author"}).copy()
        c["kind"] = "Пользовательский"
        c["builder"] = True
        c["source"] = "Формула над витриной продаж"
        c["freq"] = "как у исходных полей"
        std = pd.concat([std, c[[k for k in std.columns if k in c.columns]]], ignore_index=True)
    std["archived"] = std["archived"].fillna(0).astype(int)
    if not include_archived:
        std = std[std["archived"] == 0]
    return std.reset_index(drop=True)


def fmt_value(v: float, unit: str) -> str:
    from bi import ui
    if v is None or not np.isfinite(v):
        return "—"
    if unit == "₽":
        return ui.rub(v)
    if unit == "%":
        return ui.pct(v)
    if unit == "дн.":
        return f"{ui.num(v, 1)} дн."
    return ui.big(v)
