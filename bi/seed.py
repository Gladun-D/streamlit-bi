"""Демонстрационное наполнение служебного хранилища при первом запуске."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np

from bi import alerts, state
from bi import config as C
from bi import queries as Q


def seed() -> None:
    d = Q.dims()
    stores = d["store"]
    ugms = sorted(stores[(stores["chain"] == "P") & (stores["division"] == "Средняя Волга")]["ugm"].unique())
    end = C.DATA_END

    for kind, (name, thr, _) in alerts.KINDS.items():
        channel = "Telegram (целевая система)" if kind == "plan_forecast" else "Интерфейс BI"
        state.add_rule(name, kind, thr, "сеть", channel, "Администратор BI")

    presets = [
        ("Центр: магазины с падением РТО", {"period": "Последние 13 недель", "okrugs": ["Центральный ФО"],
                                            "condition": "lfl_neg"}),
        ("Сеть Б · фреш и овощи", {"period": "Последние 4 недели", "chains": ["P"],
                                   "zones": ["Охлаждённый (фреш)", "Фрукты и овощи"]}),
        ("Сеть А · текущий месяц", {"period": "Текущий месяц", "chains": ["G"]}),
        ("Низкая доступность на полке", {"period": "Последние 4 недели", "condition": "osa_low"}),
    ]
    for name, payload in presets:
        state.save_preset(name, "analyst", True, payload)

    state.add_comment("overview", "Отчёт целиком", "Проверить падение в Самарской обл.: совпадает ли с открытием "
                      "конкурента? Нужны данные по конкурентному окружению.", "Демо: Директор сети")
    state.add_comment("drill", "Отчёт целиком", "На уровне УГМ не хватает показателя «РТО на кв. м» — "
                      "нужна площадь магазинов из НСИ.", "Демо: Директор дивизиона")
    state.add_comment("logistics", "Отчёт целиком", "РЦ с летним провалом OTIF: разобрать поставщиков "
                      "охлаждённой группы.", "Демо: Логист РЦ")
    cm = state.comments("logistics")
    if len(cm):
        state.set_comment_status(int(cm["id"].iloc[0]), "решён")

    if len(ugms) > 1:
        state.add_grant("ugm", "ugm", ugms[1], str(end - timedelta(days=3)), str(end + timedelta(days=14)),
                        "Замещение на время отпуска коллеги", "Администратор BI")
    state.upsert_kpi("u_writeoff_to_rto", "Списания к РТО, %", "Списания / РТО", "%", ["Финансы", "Качество"],
                     "Списания в себестоимости к розничному товарообороту.", "Финансовый директор", "down",
                     "Администратор BI")
    for action, details in [("Назначена роль", "Директор округа → Центральный ФО"),
                            ("Назначена роль", "Логист РЦ → РЦ Воронеж"),
                            ("Изменён доступ к страницам", "УГМ: скрыты «Логистика» и «Конструктор отчётов»")]:
        state.journal("Администратор BI", action, details)

    # Демонстрационная история использования за 14 дней (помечена demo = 1)
    rng = np.random.default_rng(7)
    pages = ["Обзор сети", "Детализация по иерархии", "Продажи и промо", "Запасы и пополнение", "Логистика",
             "Уведомления", "Конструктор отчётов", "Прогноз", "AI-помощник", "Ассортимент ABC/XYZ", "Главная"]
    weights = np.array([18, 12, 10, 8, 7, 6, 4, 4, 3, 3, 25], dtype=float)
    roles = ["network", "okrug", "division", "ugm", "category", "logist", "analyst", "admin"]
    now = datetime.now()
    rows = []
    for i in range(14 * 45):
        ts = now - timedelta(minutes=int(rng.integers(30, 14 * 24 * 60)))
        role = str(rng.choice(roles, p=[0.12, 0.1, 0.12, 0.3, 0.12, 0.12, 0.08, 0.04]))
        u = rng.random()
        if u < 0.05:
            action, obj = "Экспорт", str(rng.choice(["obzor.xlsx", "detalizaciya.pdf", "sales.pptx"]))
        elif u < 0.065:
            action, obj = "Ошибка загрузки отчёта", str(rng.choice(pages))
        elif u < 0.09:
            action, obj = "Вход", ""
        else:
            action, obj = "Просмотр отчёта", str(rng.choice(pages, p=weights / weights.sum()))
        rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), f"demo{int(rng.integers(1, 60)):03d}", role, action, obj,
                     "демонстрационная история", 1))
    for r in rows:
        state.execute("INSERT INTO audit (ts, session, role, action, object, details, demo) VALUES (?,?,?,?,?,?,?)", r)


def payload_json(p: dict) -> str:
    return json.dumps(p, ensure_ascii=False)
