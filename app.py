"""Товародвижение · BI (версия 2) — точка входа Streamlit.

Запуск: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import uuid
from datetime import date, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Товародвижение · BI v2", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="auto")

from bi import access as A  # noqa: E402
from bi import config as C  # noqa: E402
from bi import queries as Q  # noqa: E402
from bi import state, ui  # noqa: E402
from bi.seed import seed  # noqa: E402
from views import (about, admin, ai_page, alerts_page, assortment, builder, compare, drill,  # noqa: E402
                   forecast_page, home, kpi_library, logistics, overview, quality, sales, stock, story)

ui.inject_css()

try:
    Q.backend()
    D = Q.dims()
    D0, D1 = Q.data_bounds()
    state.init(seed)
except Exception as exc:  # понятное сообщение вместо трассировки
    st.error("Не удалось подключиться к источнику данных.", icon=":material/error:")
    st.exception(exc)
    st.stop()

if "sid" not in st.session_state:
    st.session_state["sid"] = uuid.uuid4().hex
    st.session_state["_new_session"] = True

stores, dcs, cats = D["store"], D["dc"], D["category"]
names = C.chain_names()

# ---------------------------------------------------------------------------
# Параметры ссылки (?ch=P&ok=...) — применяются один раз при открытии
# ---------------------------------------------------------------------------
QP_KEYS = {"ch": "f_chains", "ok": "f_okrugs", "dv": "f_divisions", "ug": "f_ugms", "rg": "f_regions",
           "st": "f_stores", "dc": "f_dcs", "ct": "f_cats", "zn": "f_zones", "sp": "f_sups", "sk": "f_skus"}
INT_KEYS = {"st", "dc", "ct", "sp", "sk"}
VALID = {
    "f_chains": set(stores["chain"]), "f_okrugs": set(stores["fed_district"]), "f_divisions": set(stores["division"]),
    "f_ugms": set(stores["ugm"]), "f_regions": set(stores["region"]), "f_stores": set(stores["store_id"]),
    "f_dcs": set(dcs["dc_id"]), "f_cats": set(cats["category_id"]), "f_zones": set(C.WAREHOUSE_ZONES),
    "f_sups": set(D["supplier"]["supplier_id"]), "f_skus": set(D["sku"]["sku_id"]),
}
PERIODS = ["Последние 7 дней", "Последние 4 недели", "Текущий месяц", "Прошлый месяц", "Последние 13 недель",
           "Текущий квартал", "С начала года", "Последние 52 недели", "Весь период", "Календарь", "Ползунок"]


def apply_payload(p: dict):
    """Ставит значения фильтров из словаря (пресет или ссылка), отбрасывая недопустимые значения."""
    mapping = {"chains": "f_chains", "okrugs": "f_okrugs", "divisions": "f_divisions", "ugms": "f_ugms",
               "regions": "f_regions", "stores": "f_stores", "dcs": "f_dcs", "cats": "f_cats", "zones": "f_zones",
               "suppliers": "f_sups", "skus": "f_skus"}
    for k, key in mapping.items():
        vals = p.get(k)
        st.session_state[key] = [v for v in (vals or []) if v in VALID[key]]
    if p.get("period") in PERIODS:
        st.session_state["f_period"] = p["period"]
    if p.get("d0") and p.get("d1"):
        try:
            st.session_state["f_range"] = (date.fromisoformat(p["d0"]), date.fromisoformat(p["d1"]))
            st.session_state["f_period"] = "Календарь"
        except ValueError:
            pass
    st.session_state["f_cond"] = p.get("condition", "") if p.get("condition", "") in Q.CONDITIONS else ""


if not st.session_state.get("_qp_applied"):
    st.session_state["_qp_applied"] = True
    qp = st.query_params.to_dict()
    if any(k in qp for k in list(QP_KEYS) + ["p", "d0", "cond"]):
        payload = {}
        inv = {v: k for k, v in {"chains": "ch", "okrugs": "ok", "divisions": "dv", "ugms": "ug", "regions": "rg",
                                 "stores": "st", "dcs": "dc", "cats": "ct", "zones": "zn", "suppliers": "sp",
                                 "skus": "sk"}.items()}
        for short, long_ in inv.items():
            if short in qp:
                vals = [v for v in qp[short].split(",") if v]
                payload[long_] = [int(v) for v in vals if v.isdigit()] if short in INT_KEYS else vals
        payload["period"] = qp.get("p")
        payload["d0"], payload["d1"] = qp.get("d0"), qp.get("d1")
        payload["condition"] = qp.get("cond", "")
        apply_payload(payload)

# ---------------------------------------------------------------------------
# Боковая панель: версия, роль, режим
# ---------------------------------------------------------------------------
links = C.version_links()
with st.sidebar:
    st.markdown('<div class="bi-brand">Товародвижение · BI</div>'
                '<div class="bi-sub">Продажи, запасы и логистика продуктовой розницы</div>', unsafe_allow_html=True)
    st.link_button("Открыть версию 1", links["1"], icon=":material/history:", width="stretch",
                   help="Первая версия прототипа (MVP) — откроется в новой вкладке")
    st.caption(f":material/check_circle: Вы в версии {C.APP_VERSION} · {C.APP_VERSION_DATE}")
    meta = Q.meta()
    loaded = meta["loaded_at"].max()
    st.markdown('<span class="bi-badge">Синтетические данные</span>'
                f'<span class="bi-badge blue">обновлены {loaded:%d.%m.%Y %H:%M}</span>'
                f'<span class="bi-badge green">по {D1:%d.%m.%Y}</span>', unsafe_allow_html=True)

    role_keys = list(A.ROLES)

    def _on_role():
        ui.audit("Смена роли", A.ROLES[st.session_state["role_key"]].title)
        st.session_state["favorites"] = list(A.ROLES[st.session_state["role_key"]].default_favorites)
        st.session_state.pop("drill_path", None)

    if "role_key" not in st.session_state:
        st.session_state["role_key"] = A.DEFAULT_ROLE
    st.selectbox("Демо-пользователь", role_keys, format_func=lambda k: A.ROLES[k].title, key="role_key",
                 on_change=_on_role,
                 help="Демонстрация модели доступа: роль определяет видимые страницы, данные (RLS) и права "
                      "на экспорт. В целевой системе пользователь определяется через SSO / Active Directory.")
    ROLE = A.ROLES[st.session_state["role_key"]]
    SCOPE = A.role_scope(ROLE, stores, state.grants(), D1)
    st.caption(f":material/lock: Доступ к данным: {SCOPE.label}")
    mobile = st.toggle("Мобильный режим", key="mobile", help="Упрощённый интерфейс для телефона: избранные "
                       "дашборды, уведомления и базовые фильтры. Конструктор — только в полной версии.")

if st.session_state.pop("_new_session", False):
    ui.audit("Вход", ROLE.title)

# ---------------------------------------------------------------------------
# Навигация с учётом прав (page-level security)
# ---------------------------------------------------------------------------
MODULES = {
    "home": (home, ":material/home:"), "overview": (overview, ":material/dashboard:"),
    "story": (story, ":material/auto_stories:"), "alerts": (alerts_page, ":material/notifications:"),
    "drill": (drill, ":material/account_tree:"), "compare": (compare, ":material/compare:"),
    "sales": (sales, ":material/point_of_sale:"), "assortment": (assortment, ":material/category:"),
    "stock": (stock, ":material/inventory_2:"), "logistics": (logistics, ":material/local_shipping:"),
    "quality": (quality, ":material/verified:"), "forecast": (forecast_page, ":material/trending_up:"),
    "ai": (ai_page, ":material/neurology:"), "builder": (builder, ":material/build:"),
    "kpi": (kpi_library, ":material/menu_book:"), "admin": (admin, ":material/admin_panel_settings:"),
    "about": (about, ":material/info:"),
}
SECTIONS = {
    "Главное": ["home", "overview", "story", "alerts"],
    "Анализ": ["drill", "compare", "sales", "assortment"],
    "Цепочка поставок": ["stock", "logistics", "quality"],
    "Прогноз и AI": ["forecast", "ai"],
    "Самообслуживание": ["builder", "kpi"],
    "Сервис": ["admin", "about"],
}
MOBILE_PAGES = ("home", "overview", "alerts", "drill")
allowed = [p for p in ROLE.pages if (not mobile or p in MOBILE_PAGES)]
st.session_state["allowed_pages"] = allowed


def _make(key: str):
    module = MODULES[key][0]

    def run():
        try:
            module.render(st.session_state["F"])
        except Exception as exc:
            ui.audit("Ошибка загрузки отчёта", A.PAGE_TITLES[key], str(exc)[:200])
            st.error("Отчёт не удалось построить. Ошибка записана в журнал мониторинга.", icon=":material/error:")
            st.exception(exc)
    run.__name__ = f"page_{key}"
    return run


PAGE_OBJ = {k: st.Page(_make(k), title=A.PAGE_TITLES[k], icon=MODULES[k][1], url_path=k, default=(k == "home"))
            for k in allowed}
st.session_state["page_obj"] = PAGE_OBJ
nav = st.navigation({sec: [PAGE_OBJ[k] for k in keys if k in PAGE_OBJ] for sec, keys in SECTIONS.items()
                     if any(k in PAGE_OBJ for k in keys)}, position="sidebar", expanded=True)


# ---------------------------------------------------------------------------
# Фильтры (общие для всех страниц)
# ---------------------------------------------------------------------------
def _reset():
    for k in ["f_period", "f_range", "f_slider", "f_cond", *VALID]:
        st.session_state.pop(k, None)


def _keep_valid(key: str, options):
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in set(options)]


def period_dates(name: str) -> tuple[date, date]:
    if name == "Последние 7 дней":
        return D1 - timedelta(days=6), D1
    if name == "Последние 4 недели":
        return D1 - timedelta(days=27), D1
    if name == "Текущий месяц":
        return D1.replace(day=1), D1
    if name == "Прошлый месяц":
        e = D1.replace(day=1) - timedelta(days=1)
        return e.replace(day=1), e
    if name == "Последние 13 недель":
        return D1 - timedelta(days=90), D1
    if name == "Текущий квартал":
        return date(D1.year, 3 * ((D1.month - 1) // 3) + 1, 1), D1
    if name == "С начала года":
        return date(D1.year, 1, 1), D1
    if name == "Последние 52 недели":
        return D1 - timedelta(days=363), D1
    if name == "Весь период":
        return D0, D1
    return D1 - timedelta(days=90), D1


def _parse_list(upload) -> list[int]:
    """Список магазинов из файла: коды (А-001, Б-123) или ID — по одному в строке или в первой колонке."""
    raw = upload.getvalue()
    if upload.name.lower().endswith((".xlsx", ".xls")):
        vals = pd.read_excel(io.BytesIO(raw), header=None).iloc[:, 0].astype(str).tolist()
    else:
        text = raw.decode("utf-8-sig", errors="replace")
        vals = [v.strip() for line in text.splitlines() for v in line.replace(";", ",").split(",")]
    by_code = dict(zip(stores["store_code"].str.upper(), stores["store_id"]))
    out = []
    for v in vals:
        v = v.strip().upper().replace("A-", "А-").replace("B-", "Б-")
        if v in by_code:
            out.append(int(by_code[v]))
        elif v.isdigit() and int(v) in VALID["f_stores"]:
            out.append(int(v))
    return sorted(set(out))


with st.sidebar:
    st.subheader("Фильтры", divider="gray")
    st.session_state.setdefault("f_period", PERIODS[4])
    period = st.selectbox("Период", PERIODS, key="f_period",
                          help="Быстрое переключение периодов; «Календарь» и «Ползунок» — выбор диапазона вручную")
    if period == "Календарь":
        st.session_state.setdefault("f_range", (D1 - timedelta(days=90), D1))
        rng = st.date_input("Даты", min_value=D0, max_value=D1, format="DD.MM.YYYY", key="f_range")
        start, end = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (D1 - timedelta(days=90), D1))
    elif period == "Ползунок":
        st.session_state.setdefault("f_slider", (D1 - timedelta(days=90), D1))
        start, end = st.slider("Диапазон", min_value=D0, max_value=D1, format="DD.MM.YY", key="f_slider")
    else:
        start, end = period_dates(period)
    start = max(start, D0)
    st.caption(f"{start:%d.%m.%Y} — {end:%d.%m.%Y}")

    chains = tuple(st.pills("Сеть", ["G", "P"], selection_mode="multi", format_func=lambda c: names[c],
                            key="f_chains", help="Пусто — обе сети") or ())
    base = stores[stores["chain"].isin(chains)] if chains else stores

    with st.expander("Оргструктура и география", icon=":material/account_tree:",
                     expanded=bool(st.session_state.get("f_okrugs") or st.session_state.get("f_stores"))):
        opts = sorted(base["fed_district"].unique())
        _keep_valid("f_okrugs", opts)
        okrugs = st.multiselect("Федеральный округ", opts, placeholder="Все округа", key="f_okrugs")
        b2 = base[base["fed_district"].isin(okrugs)] if okrugs else base
        opts = sorted(b2["division"].unique())
        _keep_valid("f_divisions", opts)
        divisions = st.multiselect("Дивизион", opts, placeholder="Все дивизионы", key="f_divisions")
        b3 = b2[b2["division"].isin(divisions)] if divisions else b2
        opts = sorted(b3["ugm"].unique())
        _keep_valid("f_ugms", opts)
        ugms = st.multiselect("УГМ (группа магазинов)", opts, placeholder="Все УГМ", key="f_ugms")
        b4 = b3[b3["ugm"].isin(ugms)] if ugms else b3
        opts = sorted(b4["region"].unique())
        _keep_valid("f_regions", opts)
        regions = st.multiselect("Регион", opts, placeholder="Все регионы", key="f_regions")
        b5 = b4[b4["region"].isin(regions)] if regions else b4
        up = st.file_uploader("Загрузить список магазинов", type=["csv", "txt", "xlsx"], key="f_upload",
                              help="Коды магазинов (А-001, Б-123) или ID — по одному в строке или в первой колонке")
        if up is not None and st.session_state.get("_upload_sig") != (up.name, up.size):
            st.session_state["_upload_sig"] = (up.name, up.size)
            ids = _parse_list(up)
            st.session_state["f_stores"] = [i for i in ids if i in set(b5["store_id"])]
            st.session_state["_upload_msg"] = f"Из файла выбрано магазинов: {len(st.session_state['f_stores'])}"
        if st.session_state.get("_upload_msg"):
            st.caption(st.session_state["_upload_msg"])
        _keep_valid("f_stores", b5["store_id"].tolist())
        name_by_id = stores.set_index("store_id")["store_name"]
        sel_stores = st.multiselect("Магазин", b5["store_id"].tolist(), format_func=lambda i: name_by_id[i],
                                    placeholder=f"Все магазины ({len(b5)})", key="f_stores")
        dc_opts = dcs[dcs["chain"].isin(chains)] if chains else dcs
        _keep_valid("f_dcs", dc_opts["dc_id"].tolist())
        sel_dcs = st.multiselect("РЦ", dc_opts["dc_id"].tolist(),
                                 format_func=lambda i: dcs.set_index("dc_id").loc[i, "name"],
                                 placeholder="Все РЦ", key="f_dcs")

    if not mobile:
        with st.expander("Товар и поставщик", icon=":material/inventory:",
                         expanded=bool(st.session_state.get("f_sups") or st.session_state.get("f_skus"))):
            zones = st.multiselect("Склад (температурная зона)", list(C.WAREHOUSE_ZONES), placeholder="Все склады",
                                   key="f_zones")
            zone_cats = {c for z in zones for c in C.WAREHOUSE_ZONES[z]}
            cat_opts = [c for c in cats["category_id"] if not zone_cats or c in zone_cats]
            _keep_valid("f_cats", cat_opts)
            cat_name = cats.set_index("category_id")["category"]
            sel_cats = st.multiselect("Категория", cat_opts, format_func=lambda i: cat_name[i],
                                      placeholder="Все категории", key="f_cats")
            sku = D["sku"]
            sk = sku[sku["category_id"].isin(sel_cats or cat_opts)]
            sup_name = D["supplier"].set_index("supplier_id")["supplier"]
            sup_opts = sorted(sk["supplier_id"].unique())
            _keep_valid("f_sups", sup_opts)
            sel_sups = st.multiselect("Поставщик", sup_opts, format_func=lambda i: sup_name[i],
                                      placeholder="Все поставщики", key="f_sups")
            if sel_sups:
                sk = sk[sk["supplier_id"].isin(sel_sups)]
            _keep_valid("f_skus", sk["sku_id"].tolist())
            sku_name = sku.set_index("sku_id")["sku"]
            sel_skus = st.multiselect("Товар (SKU)", sk["sku_id"].tolist(), format_func=lambda i: sku_name[i],
                                      placeholder=f"Все товары ({len(sk)})", key="f_skus")
            st.caption("Поставщик и SKU применяются к данным РЦ, закупкам и качеству; продажи магазинов в "
                       "прототипе детализированы до категории.")
        st.session_state.setdefault("f_cond", "")
        cond = st.selectbox("Условие по магазинам", list(Q.CONDITIONS), format_func=lambda k: Q.CONDITIONS[k],
                            key="f_cond", help="Контекстный фильтр: например, только магазины с падением РТО")
    else:
        zones, sel_cats, sel_sups, sel_skus, cond = [], [], [], [], ""

    if not mobile:
        with st.popover("Наборы фильтров", icon=":material/bookmarks:", width="stretch"):
            pr = state.presets_for(ROLE.key)
            if len(pr):
                pick = st.selectbox("Сохранённые наборы", pr["id"].tolist(),
                                    format_func=lambda i: pr.set_index("id").loc[i, "name"]
                                    + (" · общий" if pr.set_index("id").loc[i, "shared"] else " · мой"),
                                    key="preset_pick")

                def _apply_preset():
                    row = pr.set_index("id").loc[st.session_state["preset_pick"]]
                    apply_payload(json.loads(row["payload"]))
                    ui.audit("Применён набор фильтров", row["name"])
                st.button("Применить", on_click=_apply_preset, icon=":material/check:", key="preset_apply")
            st.divider()
            pname = st.text_input("Сохранить текущие фильтры как", max_chars=80, key="preset_name")
            pshared = st.checkbox("Доступен всем пользователям", value=True, key="preset_shared")
            if st.button("Сохранить набор", icon=":material/bookmark_add:", key="preset_save") and pname.strip():
                state.save_preset(pname, ROLE.key, pshared, {
                    "period": period, "d0": str(start) if period in ("Календарь", "Ползунок") else None,
                    "d1": str(end) if period in ("Календарь", "Ползунок") else None,
                    "chains": list(chains), "okrugs": okrugs, "divisions": divisions, "ugms": ugms,
                    "regions": regions, "stores": sel_stores, "dcs": sel_dcs, "cats": sel_cats, "zones": zones,
                    "suppliers": sel_sups, "skus": sel_skus, "condition": cond})
                ui.audit("Сохранён набор фильтров", pname)
                st.success("Набор сохранён")

        with st.popover("Пороги подсветки", icon=":material/tune:", width="stretch"):
            t = ui.thresholds()
            st.caption("Цвет в таблицах и статусы: 🟢 норма · 🟡 в пределах допуска · 🔴 отклонение")
            c1, c2 = st.columns(2)
            new = {
                "plan_green": c1.number_input("План: зелёный от, %", 50.0, 150.0, t["plan_green"] * 100, 0.5) / 100,
                "plan_yellow": c2.number_input("План: жёлтый от, %", 50.0, 150.0, t["plan_yellow"] * 100, 0.5) / 100,
                "lfl_yellow": c1.number_input("LFL: жёлтый ниже, %", -50.0, 50.0, t["lfl_yellow"] * 100, 0.5) / 100,
                "lfl_red": c2.number_input("LFL: красный ниже, %", -50.0, 50.0, t["lfl_red"] * 100, 0.5) / 100,
                "osa_green": c1.number_input("OSA: зелёный от, %", 50.0, 100.0, t["osa_green"] * 100, 0.5) / 100,
                "osa_red": c2.number_input("OSA: красный ниже, %", 50.0, 100.0, t["osa_red"] * 100, 0.5) / 100,
                "top_n": int(st.number_input("Подсвечивать лучших и худших, строк", 1, 50, int(t["top_n"]))),
            }
            st.session_state["thresholds"] = new

    st.button("Сбросить фильтры", icon=":material/restart_alt:", on_click=_reset, width="stretch")
    st.caption(f"Источник: {Q.backend().name}")

F = Q.resolve(Q.Filters(start=start, end=end, chains=chains, dcs=tuple(sel_dcs), regions=tuple(regions),
                        stores=tuple(sel_stores), cats=tuple(sel_cats), okrugs=tuple(okrugs),
                        divisions=tuple(divisions), ugms=tuple(ugms), suppliers=tuple(sel_sups),
                        skus=tuple(sel_skus), zones=tuple(zones), condition=cond),
              scope=SCOPE, thresholds=ui.thresholds())
st.session_state["F"] = F
st.session_state["SCOPE"] = SCOPE

share = {"p": period, "ch": ",".join(chains), "ok": ",".join(okrugs), "dv": ",".join(divisions),
         "ug": ",".join(ugms), "rg": ",".join(regions), "st": ",".join(map(str, sel_stores)),
         "dc": ",".join(map(str, sel_dcs)), "ct": ",".join(map(str, sel_cats)), "zn": ",".join(zones),
         "sp": ",".join(map(str, sel_sups)), "sk": ",".join(map(str, sel_skus)), "cond": cond}
if period in ("Календарь", "Ползунок"):
    share.update({"d0": str(start), "d1": str(end)})
st.session_state["_share_params"] = share
try:
    st.query_params.from_dict({k: v for k, v in share.items() if v})
except Exception:
    pass

nav.run()
