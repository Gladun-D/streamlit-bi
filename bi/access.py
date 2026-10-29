"""Демо-модель доступа: роли, видимые страницы, ограничение данных (row-level security), права на экспорт.

ВАЖНО: в прототипе роль выбирается вручную — это демонстрация модели, а не защита данных.
В целевой системе пользователь определяется через SSO / Active Directory, а ограничения
применяются на стороне сервера при каждом запросе к витринам.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# Ключи страниц приложения
PAGE_TITLES = {
    "home": "Главная",
    "overview": "Обзор сети",
    "story": "Data story",
    "drill": "Детализация по иерархии",
    "compare": "Сравнение периодов",
    "builder": "Конструктор отчётов",
    "sales": "Продажи и промо",
    "assortment": "Ассортимент ABC/XYZ",
    "stock": "Запасы и пополнение",
    "logistics": "Логистика",
    "quality": "Качество и потери",
    "forecast": "Прогноз",
    "ai": "AI-помощник",
    "alerts": "Уведомления",
    "kpi": "Библиотека показателей",
    "admin": "Администрирование",
    "about": "О данных и методика",
}
ALL_PAGES = tuple(PAGE_TITLES)


@dataclass(frozen=True)
class Role:
    key: str
    title: str
    description: str
    pages: tuple[str, ...]
    scope_kind: str = "all"          # all | okrug | division | ugm | categories | dc
    can_export: bool = True
    can_build: bool = False          # создание и публикация отчётов в конструкторе
    is_admin: bool = False
    default_favorites: tuple[str, ...] = field(default_factory=tuple)


_MGMT = ("home", "overview", "story", "drill", "compare", "sales", "assortment", "stock", "logistics", "quality",
         "forecast", "ai", "alerts", "kpi", "about")
ROLES: dict[str, Role] = {r.key: r for r in [
    Role("network", "Директор сети", "Все сети, все регионы и РЦ", _MGMT,
         default_favorites=("overview", "story", "alerts")),
    Role("okrug", "Директор округа", "Сеть Б, один федеральный округ", _MGMT, scope_kind="okrug",
         default_favorites=("overview", "drill", "alerts")),
    Role("division", "Директор дивизиона", "Сеть Б, один дивизион", _MGMT, scope_kind="division",
         default_favorites=("drill", "sales", "alerts")),
    Role("ugm", "Управляющий группой магазинов (УГМ)", "Магазины своей группы; без экспорта",
         ("home", "overview", "drill", "sales", "stock", "alerts", "kpi", "about"), scope_kind="ugm",
         can_export=False, default_favorites=("drill", "alerts")),
    Role("category", "Категорийный менеджер", "Свои категории во всех магазинах",
         ("home", "overview", "drill", "compare", "builder", "sales", "assortment", "forecast", "ai", "alerts",
          "kpi", "about"), scope_kind="categories", can_build=True,
         default_favorites=("assortment", "sales", "builder")),
    Role("logist", "Логист РЦ", "Один РЦ и его магазины",
         ("home", "stock", "logistics", "quality", "forecast", "alerts", "kpi", "about"), scope_kind="dc",
         default_favorites=("logistics", "quality", "stock")),
    Role("analyst", "Аналитик", "Все данные, конструктор отчётов",
         _MGMT + ("builder",), can_build=True, default_favorites=("builder", "forecast", "ai")),
    Role("admin", "Администратор BI", "Все данные и администрирование", ALL_PAGES, can_build=True, is_admin=True,
         default_favorites=("admin", "kpi", "alerts")),
]}
DEFAULT_ROLE = "network"


@dataclass(frozen=True)
class Scope:
    """Ограничение данных для роли. Пустой кортеж — без ограничения по этому измерению."""
    label: str = "Все данные"
    chains: tuple[str, ...] = ()
    okrugs: tuple[str, ...] = ()
    divisions: tuple[str, ...] = ()
    ugms: tuple[str, ...] = ()
    dcs: tuple[int, ...] = ()
    cats: tuple[int, ...] = ()


def role_scope(role: Role, stores: pd.DataFrame, grants: pd.DataFrame | None = None,
               today: date | None = None) -> Scope:
    """Конкретные значения ограничения выбираются из справочника (демо-пользователь «назначен» на первый объект)."""
    p = stores[stores["chain"] == "P"]
    if role.scope_kind == "okrug":
        ok = "Центральный ФО"
        return Scope(f"Сеть Б · {ok}", chains=("P",), okrugs=(ok,))
    if role.scope_kind == "division":
        dv = "Средняя Волга"
        return Scope(f"Сеть Б · дивизион «{dv}»", chains=("P",), divisions=(dv,))
    if role.scope_kind == "ugm":
        ugms = sorted(p[p["division"] == "Средняя Волга"]["ugm"].unique())
        own = (ugms[0],)
        extra = _active_grants(grants, role.key, "ugm", today)
        label = f"{own[0]}" + (f" + временно {', '.join(extra)}" if extra else "")
        return Scope(label, ugms=tuple(dict.fromkeys(own + extra)))
    if role.scope_kind == "categories":
        return Scope("Категории: молочная продукция; мясо, птица, колбасы", cats=(2, 3))
    if role.scope_kind == "dc":
        return Scope("Сеть Б · РЦ Воронеж", chains=("P",), dcs=(3,))
    return Scope()


def _active_grants(grants: pd.DataFrame | None, role_key: str, kind: str, today: date | None) -> tuple[str, ...]:
    if grants is None or grants.empty:
        return ()
    g = grants[(grants["role_key"] == role_key) & (grants["scope_type"] == kind)]
    if today is not None:
        g = g[(pd.to_datetime(g["valid_from"]).dt.date <= today) & (pd.to_datetime(g["valid_to"]).dt.date >= today)]
    return tuple(g["scope_value"].astype(str))


def scope_store_mask(scope: Scope, stores: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=stores.index)
    if scope.chains:
        m &= stores["chain"].isin(scope.chains)
    if scope.okrugs:
        m &= stores["fed_district"].isin(scope.okrugs)
    if scope.divisions:
        m &= stores["division"].isin(scope.divisions)
    if scope.ugms:
        m &= stores["ugm"].isin(scope.ugms)
    if scope.dcs:
        m &= stores["dc_id"].isin(scope.dcs)
    return m
