"""Источник данных: встроенный DuckDB поверх parquet (по умолчанию) или MySQL.

Выбор источника:
  DATA_BACKEND=duckdb  (по умолчанию) — данные генерируются при первом запуске в BI_DATA_DIR
  DATA_BACKEND=mysql   — подключение по MYSQL_URL, например
                         mysql+pymysql://user:password@host:3306/retail_bi?charset=utf8mb4
Параметры читаются из переменных окружения или из .streamlit/secrets.toml.

SQL во всём приложении написан на общем подмножестве (SUM/COUNT/AVG/CASE, GROUP BY, IN, BETWEEN),
поэтому одинаково работает в обеих СУБД. Параметры — в стиле :name, списки раскрываются автоматически.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from bi import synth

ROOT = Path(__file__).resolve().parents[1]


def setting(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def data_dir() -> Path:
    custom = setting("BI_DATA_DIR")
    if custom:
        return Path(custom)
    default = ROOT / "data" / "generated"
    try:
        default.mkdir(parents=True, exist_ok=True)
        probe = default / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return default
    except OSError:
        return Path(tempfile.gettempdir()) / "retail_bi_data"


def ensure_generated(target: Path) -> None:
    """Генерирует данные в отдельном процессе, чтобы память освободилась после генерации."""
    if synth.is_ready(target):
        return
    target.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, "-m", "bi.synth", str(target)], cwd=str(ROOT),
                          capture_output=True, text=True)
    if proc.returncode != 0 or not synth.is_ready(target):
        raise RuntimeError("Не удалось сгенерировать данные:\n" + (proc.stderr or proc.stdout)[-3000:])


_PARAM = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")


def expand_params(sql: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Раскрывает списки: `x IN :ids` -> `x IN (:ids_0, :ids_1, …)`."""
    out: dict[str, Any] = {}

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in params:
            return m.group(0)
        val = params[name]
        if isinstance(val, (list, tuple, set)):
            vals = list(val)
            if not vals:
                raise ValueError(f"Пустой список для параметра {name}")
            keys = [f"{name}_{i}" for i in range(len(vals))]
            out.update(dict(zip(keys, vals)))
            return "(" + ", ".join(":" + k for k in keys) + ")"
        out[name] = val
        return ":" + name

    return _PARAM.sub(repl, sql), out


class DuckDBBackend:
    name = "DuckDB (встроенная, синтетические данные)"
    kind = "duckdb"

    def __init__(self, folder: Path):
        import duckdb
        self.folder = folder
        self.con = duckdb.connect(database=":memory:")
        self.con.execute("SET threads TO 2")
        for t in synth.TABLES:
            path = (folder / f"{t}.parquet").as_posix().replace("'", "''")
            self.con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path}')")

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        sql, p = expand_params(sql, params or {})
        sql = _PARAM.sub(lambda m: "$" + m.group(1) if m.group(1) in p else m.group(0), sql)
        cur = self.con.cursor()   # отдельный курсор на запрос — потокобезопасно
        try:
            return cur.execute(sql, p).df()
        finally:
            cur.close()


class MySQLBackend:
    """Любая СУБД через SQLAlchemy: MySQL/MariaDB (проверено), PostgreSQL, ClickHouse, MS SQL (нужен драйвер)."""
    kind = "sql"

    def __init__(self, url: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800)
        dialect = {"mysql": "MySQL", "mariadb": "MariaDB", "postgresql": "PostgreSQL", "clickhouse": "ClickHouse",
                   "mssql": "MS SQL"}.get(self.engine.dialect.name, self.engine.dialect.name)
        self.name = f"{dialect} ({self.engine.url.host}/{self.engine.url.database})"

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        from sqlalchemy import text
        sql, p = expand_params(sql, params or {})
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=p)


def make_backend():
    kind = (setting("DATA_BACKEND", "duckdb") or "duckdb").lower()
    if kind in ("mysql", "sql"):
        url = setting("MYSQL_URL") if kind == "mysql" else setting("SQL_URL")
        if not url:
            raise RuntimeError(f"DATA_BACKEND={kind}, но не задан {'MYSQL_URL' if kind == 'mysql' else 'SQL_URL'}")
        return MySQLBackend(url)
    folder = data_dir()
    ensure_generated(folder)
    return DuckDBBackend(folder)
