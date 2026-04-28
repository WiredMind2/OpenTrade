"""
Price ingestion skeleton for Kaggle Yahoo CSV archives.

Expect local CSV files downloaded from Kaggle (e.g., `AAPL.csv`). The script ingests daily OHLCV into `price_daily`.

Usage:
  python ingest_prices.py --db data/backtest.db --csv_dir data/kaggle_yahoo
"""
import argparse
import os
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional
from script_logger import logger


def _configure_sqlite_for_ingest(conn: sqlite3.Connection) -> None:
    # These pragmas make bulk ingestion dramatically faster.
    # WAL helps concurrency (readers while writing); NORMAL is a good balance for local dev.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common Yahoo/Kaggle column naming variants.
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("date",):
            rename_map[c] = "date"
        elif cl in ("open",):
            rename_map[c] = "open"
        elif cl in ("high",):
            rename_map[c] = "high"
        elif cl in ("low",):
            rename_map[c] = "low"
        elif cl in ("close",):
            rename_map[c] = "close"
        elif cl in ("adj close", "adj_close", "adjusted_close", "adjclose"):
            rename_map[c] = "adjusted_close"
        elif cl in ("volume",):
            rename_map[c] = "volume"
        elif cl in ("symbol", "ticker", "company"):
            rename_map[c] = "ticker"
    df = df.rename(columns=rename_map)
    return df


def _iter_chunks(seq, chunk_size: int):
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


def ingest_csv_to_db(db_path: str, csv_path: str, ticker: Optional[str] = None):
    """
    Ingest a CSV file. If `ticker` is provided, treat the file as a single-ticker CSV.
    Otherwise, if the CSV contains a column like 'symbol' or 'ticker', ingest all tickers present in the file.
    """
    # Read once; parse date column if present.
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    parse_dates = [c for c in header_cols if c.strip().lower() == "date"]
    df = pd.read_csv(csv_path, parse_dates=parse_dates if parse_dates else None)
    df = _normalize_columns(df)

    if "date" not in df.columns:
        logger.warning("Skipping %s: missing date column", csv_path)
        return

    # Sort by date for more predictable ingestion.
    df = df.sort_values("date")

    # Use a longer timeout: large ingests can hold locks for a while.
    conn = sqlite3.connect(db_path, timeout=60)
    _configure_sqlite_for_ingest(conn)
    cur = conn.cursor()

    # Helper to insert ticker metadata
    def ensure_ticker(tkr: str):
        try:
            cur.execute('INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)', (tkr.upper(), None, None))
        except Exception as e:
            logger.warning("Failed to insert ticker %s: %s", tkr.upper(), e)

    # Determine if CSV is multi-ticker (has a ticker column) or single-ticker.
    is_multi = ("ticker" in df.columns) and (ticker is None)

    # Ensure all required numeric columns exist; fill missing with None.
    for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if col not in df.columns:
            df[col] = None
    df["adjusted_close"] = df["adjusted_close"].where(df["adjusted_close"].notna(), df["close"])

    # Convert to SQLite-friendly types in vectorized fashion.
    # Kaggle/Yahoo exports may include timezone offsets; normalize to UTC then format as YYYY-MM-DD.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    df = df[df["date"].notna()]

    for col in ("open", "high", "low", "close", "adjusted_close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round().astype("Int64")

    insert_sql = (
        "INSERT OR REPLACE INTO price_daily "
        "(ticker, date, open, high, low, close, adjusted_close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )

    batch_size = 50_000
    total_inserted = 0

    if is_multi:
        # Multi-ticker: expect a ticker column.
        df["ticker"] = df["ticker"].astype(str).str.upper()
        tickers = sorted(set(df["ticker"].dropna().tolist()))
        for tkr in tickers:
            ensure_ticker(tkr)

        records = list(
            zip(
                df["ticker"].tolist(),
                df["date"].tolist(),
                df["open"].tolist(),
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                df["adjusted_close"].tolist(),
                [None if pd.isna(v) else int(v) for v in df["volume"].tolist()],
            )
        )
    else:
        # Single ticker: infer from filename if not provided.
        if ticker is None:
            ticker = Path(csv_path).stem
        t = str(ticker).upper()
        ensure_ticker(t)

        records = list(
            zip(
                [t] * len(df),
                df["date"].tolist(),
                df["open"].tolist(),
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                df["adjusted_close"].tolist(),
                [None if pd.isna(v) else int(v) for v in df["volume"].tolist()],
            )
        )

    logger.info("Ingesting %d rows from %s", len(records), csv_path)
    try:
        for chunk in _iter_chunks(records, batch_size):
            cur.executemany(insert_sql, chunk)
            conn.commit()
            total_inserted += len(chunk)
            logger.info("... committed %d/%d rows from %s", total_inserted, len(records), os.path.basename(csv_path))
    finally:
        conn.close()

    logger.info("Ingested %d rows from %s", total_inserted, csv_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--csv_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kaggle_yahoo')))
    args = parser.parse_args()

    # recursively find CSVs
    paths = []
    for root, _, files in os.walk(args.csv_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                paths.append(os.path.join(root, f))
    if not paths:
        logger.warning('No CSV files found under %s', args.csv_dir)
        return
    for p in paths:
        ingest_csv_to_db(args.db, p, None)


if __name__ == '__main__':
    main()
