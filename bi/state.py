"""Хранилище пользовательских настроек прототипа (SQLite рядом с данными).

Хранит: наборы фильтров, сохранённые отчёты, комментарии, пользовательские показатели,
правила уведомлений, временные доступы, журнал прав и аудит действий.

В прототипе файл живёт до перезапуска сервера; в целевой системе это таблицы служебной БД.
Все тексты ограничены по длине, число записей — по количеству (приложение публичное).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from bi.db import data_dir

_LOCK = threading.Lock()
MAX_TEXT = 500
MAX_NAME = 80
LIMITS = {"presets": 300, "reports": 300, "comments": 1000, "kpis": 200, "alert_rules": 200, "grants": 200,
          "rights_journal": 2000, "audit": 20000}

SCHEMA = """
CREATE TABLE IF NOT EXISTS presets (id INTEGER PRIMARY KEY, name TEXT, owner TEXT, shared INTEGER, payload TEXT,
                                    created_at TEXT);
CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY, name TEXT, owner TEXT, shared INTEGER, payload TEXT,
                                    created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY, page TEXT, widget TEXT, text TEXT, author TEXT,
                                     created_at TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS kpis (code TEXT PRIMARY KEY, name TEXT, formula TEXT, unit TEXT, directions TEXT,
                                 description TEXT, owner TEXT, good TEXT, archived INTEGER, created_by TEXT,
                                 created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS alert_rules (id INTEGER PRIMARY KEY, name TEXT, kind TEXT, threshold REAL, level TEXT,
                                        channel TEXT, active INTEGER, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS grants (id INTEGER PRIMARY KEY, role_key TEXT, scope_type TEXT, scope_value TEXT,
                                   valid_from TEXT, valid_to TEXT, reason TEXT, created_by TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS rights_journal (id INTEGER PRIMARY KEY, ts TEXT, actor TEXT, action TEXT, details TEXT);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, ts TEXT, session TEXT, role TEXT, action TEXT,
                                  object TEXT, details TEXT, demo INTEGER DEFAULT 0);
"""


def _path() -> Path:
    p = data_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p / "app_state.sqlite"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(_path(), timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _clip(s: Any, n: int = MAX_TEXT) -> str:
    return str(s or "").strip()[:n]


def execute(sql: str, params: tuple = ()) -> int:
    with _LOCK, _conn() as con:
        cur = con.execute(sql, params)
        return cur.lastrowid


def fetch(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _LOCK, _conn() as con:
        return pd.read_sql_query(sql, con, params=params)


def _trim(table: str) -> None:
    limit = LIMITS[table]
    key = "code" if table == "kpis" else "id"
    execute(f"DELETE FROM {table} WHERE {key} IN (SELECT {key} FROM {table} ORDER BY rowid DESC LIMIT -1 OFFSET ?)",
            (limit,))


# ---------------------------------------------------------------------------
# Инициализация и демо-наполнение
# ---------------------------------------------------------------------------
_INIT_DONE = False


def init(seed_fn=None) -> None:
    """Создаёт таблицы; при пустой базе вызывает seed_fn() для демонстрационного наполнения."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    with _LOCK, _conn() as con:
        con.executescript(SCHEMA)
        empty = con.execute("SELECT COUNT(*) FROM alert_rules").fetchone()[0] == 0
    if empty and seed_fn is not None:
        seed_fn()
    _INIT_DONE = True


# ---------------------------------------------------------------------------
# Наборы фильтров и отчёты
# ---------------------------------------------------------------------------
def save_preset(name: str, owner: str, shared: bool, payload: dict) -> None:
    execute("INSERT INTO presets (name, owner, shared, payload, created_at) VALUES (?,?,?,?,?)",
            (_clip(name, MAX_NAME), owner, int(shared), json.dumps(payload, ensure_ascii=False, default=str), now()))
    _trim("presets")


def presets_for(owner: str) -> pd.DataFrame:
    return fetch("SELECT * FROM presets WHERE shared = 1 OR owner = ? ORDER BY id DESC", (owner,))


def delete_preset(pid: int, owner: str, admin: bool = False) -> None:
    if admin:
        execute("DELETE FROM presets WHERE id = ?", (pid,))
    else:
        execute("DELETE FROM presets WHERE id = ? AND owner = ?", (pid, owner))


def save_report(name: str, owner: str, shared: bool, payload: dict) -> None:
    ex = fetch("SELECT id FROM reports WHERE name = ? AND owner = ?", (_clip(name, MAX_NAME), owner))
    body = json.dumps(payload, ensure_ascii=False, default=str)
    if len(ex):
        execute("UPDATE reports SET shared = ?, payload = ?, updated_at = ? WHERE id = ?",
                (int(shared), body, now(), int(ex["id"].iloc[0])))
    else:
        execute("INSERT INTO reports (name, owner, shared, payload, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                (_clip(name, MAX_NAME), owner, int(shared), body, now(), now()))
    _trim("reports")


def reports_for(owner: str) -> pd.DataFrame:
    return fetch("SELECT * FROM reports WHERE shared = 1 OR owner = ? ORDER BY updated_at DESC", (owner,))


def delete_report(rid: int, owner: str, admin: bool = False) -> None:
    if admin:
        execute("DELETE FROM reports WHERE id = ?", (rid,))
    else:
        execute("DELETE FROM reports WHERE id = ? AND owner = ?", (rid, owner))


# ---------------------------------------------------------------------------
# Комментарии
# ---------------------------------------------------------------------------
def add_comment(page: str, widget: str, text: str, author: str) -> None:
    if not _clip(text):
        return
    execute("INSERT INTO comments (page, widget, text, author, created_at, status) VALUES (?,?,?,?,?,?)",
            (page, _clip(widget, MAX_NAME), _clip(text), author, now(), "новый"))
    _trim("comments")


def comments(page: str | None = None) -> pd.DataFrame:
    if page:
        return fetch("SELECT * FROM comments WHERE page = ? ORDER BY id DESC", (page,))
    return fetch("SELECT * FROM comments ORDER BY id DESC")


def set_comment_status(cid: int, status: str) -> None:
    execute("UPDATE comments SET status = ? WHERE id = ?", (status, cid))


# ---------------------------------------------------------------------------
# Пользовательские показатели
# ---------------------------------------------------------------------------
def custom_kpis(include_archived: bool = True) -> pd.DataFrame:
    df = fetch("SELECT * FROM kpis ORDER BY created_at")
    return df if include_archived else df[df["archived"] == 0]


def upsert_kpi(code: str, name: str, formula: str, unit: str, directions: list[str], description: str,
               owner: str, good: str, actor: str) -> None:
    ex = fetch("SELECT code FROM kpis WHERE code = ?", (code,))
    dirs = ",".join(directions)
    if len(ex):
        execute("UPDATE kpis SET name=?, formula=?, unit=?, directions=?, description=?, owner=?, good=?, "
                "updated_at=? WHERE code=?",
                (_clip(name, MAX_NAME), _clip(formula, 300), unit, dirs, _clip(description), _clip(owner, MAX_NAME),
                 good, now(), code))
    else:
        execute("INSERT INTO kpis VALUES (?,?,?,?,?,?,?,?,0,?,?,?)",
                (code, _clip(name, MAX_NAME), _clip(formula, 300), unit, dirs, _clip(description),
                 _clip(owner, MAX_NAME), good, actor, now(), now()))
    _trim("kpis")


def archive_kpi(code: str, archived: bool) -> None:
    execute("UPDATE kpis SET archived = ?, updated_at = ? WHERE code = ?", (int(archived), now(), code))


# ---------------------------------------------------------------------------
# Правила уведомлений
# ---------------------------------------------------------------------------
def alert_rules() -> pd.DataFrame:
    return fetch("SELECT * FROM alert_rules ORDER BY id")


def add_rule(name: str, kind: str, threshold: float, level: str, channel: str, actor: str) -> None:
    execute("INSERT INTO alert_rules (name, kind, threshold, level, channel, active, created_by, created_at) "
            "VALUES (?,?,?,?,?,1,?,?)", (_clip(name, MAX_NAME), kind, float(threshold), level, channel, actor, now()))
    _trim("alert_rules")


def set_rule_active(rid: int, active: bool) -> None:
    execute("UPDATE alert_rules SET active = ? WHERE id = ?", (int(active), rid))


# ---------------------------------------------------------------------------
# Доступы и журналы
# ---------------------------------------------------------------------------
def grants() -> pd.DataFrame:
    return fetch("SELECT * FROM grants ORDER BY id DESC")


def add_grant(role_key: str, scope_type: str, scope_value: str, valid_from: str, valid_to: str, reason: str,
              actor: str) -> None:
    execute("INSERT INTO grants (role_key, scope_type, scope_value, valid_from, valid_to, reason, created_by, "
            "created_at) VALUES (?,?,?,?,?,?,?,?)",
            (role_key, scope_type, _clip(scope_value, MAX_NAME), valid_from, valid_to, _clip(reason, 200), actor, now()))
    journal(actor, "Выдан временный доступ", f"{role_key}: {scope_type} = {scope_value}, до {valid_to}. {reason}")
    _trim("grants")


def revoke_grant(gid: int, actor: str) -> None:
    g = fetch("SELECT * FROM grants WHERE id = ?", (gid,))
    execute("DELETE FROM grants WHERE id = ?", (gid,))
    if len(g):
        r = g.iloc[0]
        journal(actor, "Отозван временный доступ", f"{r['role_key']}: {r['scope_type']} = {r['scope_value']}")


def journal(actor: str, action: str, details: str) -> None:
    execute("INSERT INTO rights_journal (ts, actor, action, details) VALUES (?,?,?,?)",
            (now(), actor, _clip(action, MAX_NAME), _clip(details)))
    _trim("rights_journal")


def rights_journal() -> pd.DataFrame:
    return fetch("SELECT * FROM rights_journal ORDER BY id DESC")


def audit(session: str, role: str, action: str, obj: str = "", details: str = "") -> None:
    try:
        execute("INSERT INTO audit (ts, session, role, action, object, details, demo) VALUES (?,?,?,?,?,?,0)",
                (now(), session[:12], role, action, _clip(obj, 120), _clip(details, 300)))
        if hash(session) % 50 == 0:
            _trim("audit")
    except Exception:
        pass   # аудит не должен ломать отчёт


def audit_log(days: int = 30) -> pd.DataFrame:
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    return fetch("SELECT * FROM audit WHERE ts >= ? ORDER BY id DESC", (since,))
