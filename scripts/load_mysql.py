"""Загружает синтетические данные в MySQL (или MariaDB) и создаёт индексы.

Пример:
    python scripts/load_mysql.py --url "mysql+pymysql://user:pass@localhost:3306/retail_bi?charset=utf8mb4"

Если данные ещё не сгенерированы, скрипт сгенерирует их сам.
После загрузки запустите приложение с DATA_BACKEND=mysql и MYSQL_URL=<тот же адрес>.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.types import Date, DateTime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bi import synth  # noqa: E402
from bi.db import data_dir, ensure_generated  # noqa: E402

INDEXES = {
    "fact_sales": ["date, store_id", "date, category_id"],
    "fact_store_day": ["date, store_id"],
    "fact_plan": ["month, store_id"],
    "fact_sku_dc": ["date, dc_id", "sku_id, dc_id", "week_start"],
    "fact_supplier_order": ["receipt_date", "order_date", "supplier_id"],
    "fact_delivery": ["date, dc_id", "store_id"],
    "dim_store_history": ["store_id"],
    "dim_certificate": ["valid_to"],
}
DATE_COLS = {"date", "month", "week_start", "order_date", "promised_date", "receipt_date", "open_date",
             "changed_on", "valid_from", "valid_to", "data_until"}
TS_COLS = {"planned_ts", "actual_ts", "loaded_at"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="SQLAlchemy URL базы MySQL")
    ap.add_argument("--data-dir", default=None, help="Каталог parquet (по умолчанию — как у приложения)")
    ap.add_argument("--chunk", type=int, default=200_000, help="Строк в одной порции загрузки")
    ap.add_argument("--since", default=None,
                    help="Загрузить факты только начиная с даты ГГГГ-ММ-ДД (быстрая демо-загрузка)")
    args = ap.parse_args()

    folder = Path(args.data_dir) if args.data_dir else data_dir()
    ensure_generated(folder)
    engine = create_engine(args.url, pool_pre_ping=True)

    for table in synth.TABLES:
        t0 = time.time()
        pf = pq.ParquetFile(folder / f"{table}.parquet")
        first = True
        rows = 0
        for batch in pf.iter_batches(batch_size=args.chunk):
            df = batch.to_pandas()
            if args.since and table.startswith("fact_"):
                key = {"fact_plan": "month", "fact_supplier_order": "order_date"}.get(table, "date")
                df = df[df[key] >= pd.Timestamp(args.since)]
                if df.empty:
                    continue
            for c in df.columns:
                if df[c].dtype.name in ("string", "str") or str(df[c].dtype).startswith("string"):
                    df[c] = df[c].astype(object)
                if c in DATE_COLS:
                    df[c] = df[c].astype("datetime64[ns]").dt.date
            dtype = {c: Date() for c in df.columns if c in DATE_COLS}
            dtype |= {c: DateTime() for c in df.columns if c in TS_COLS}
            df.to_sql(table, engine, if_exists="replace" if first else "append", index=False,
                      chunksize=20_000, method="multi", dtype=dtype)
            first = False
            rows += len(df)
        with engine.begin() as conn:
            for i, cols in enumerate(INDEXES.get(table, [])):
                conn.execute(text(f"CREATE INDEX ix_{table}_{i} ON {table} ({cols})"))
        print(f"{table:22s} {rows:>10,d} строк  {time.time() - t0:6.1f} с", flush=True)
    print("Готово. Запуск приложения: DATA_BACKEND=mysql MYSQL_URL=<адрес> streamlit run app.py")


if __name__ == "__main__":
    main()
