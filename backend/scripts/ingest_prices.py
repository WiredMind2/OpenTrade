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


def ingest_csv_to_db(db_path: str, csv_path: str, ticker: Optional[str] = None):
    """
    Ingest a CSV file. If `ticker` is provided, treat the file as a single-ticker CSV.
    Otherwise, if the CSV contains a column like 'symbol' or 'ticker', ingest all tickers present in the file.
    """
    df = pd.read_csv(csv_path, parse_dates=[col for col in ['Date', 'date'] if col in pd.read_csv(csv_path, nrows=0).columns])
    df = df.sort_values(df.columns[0])
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    def safe_float(x):
        try:
            return float(x) if pd.notna(x) else None
        except Exception:
            return None

    def safe_int(x):
        try:
            return int(x) if pd.notna(x) else None
        except Exception:
            return None

    # Helper to insert ticker metadata
    def ensure_ticker(tkr: str):
        try:
            cur.execute('INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)', (tkr.upper(), None, None))
        except Exception:
            pass

    # If dataframe has a symbol/ticker/company column, treat as combined
    symbol_cols = [c for c in df.columns if c.lower() in ('symbol', 'ticker', 'company')]
    if symbol_cols and ticker is None:
        sym_col = symbol_cols[0]
        grouped = df.groupby(sym_col)
        total = 0
        for tkr, group in grouped:
            t = str(tkr).upper()
            ensure_ticker(t)
            rows = []
            for _, row in group.iterrows():
                # try common column names
                date_val = None
                if 'Date' in row.index:
                    date_val = row['Date']
                elif 'date' in row.index:
                    date_val = row['date']
                else:
                    continue
                date = pd.to_datetime(date_val).strftime('%Y-%m-%d')
                open_p = safe_float(row.get('Open') if 'Open' in row.index else row.get('open'))
                high = safe_float(row.get('High') if 'High' in row.index else row.get('high'))
                low = safe_float(row.get('Low') if 'Low' in row.index else row.get('low'))
                close = safe_float(row.get('Close') if 'Close' in row.index else row.get('close'))
                adj = safe_float(row.get('Adj Close') if 'Adj Close' in row.index else row.get('Adj_Close')) or close
                volume = safe_int(row.get('Volume') if 'Volume' in row.index else row.get('volume'))
                try:
                    cur.execute(
                        'INSERT OR REPLACE INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (t, date, open_p, high, low, close, adj, volume)
                    )
                    total += 1
                except Exception as e:
                    print('Insert failed for', t, date, e)
        conn.commit()
        conn.close()
        print(f'Ingested {total} rows from combined CSV for {csv_path}')
        return

    # Otherwise, treat file as single-ticker CSV; attempt to infer ticker from filename if not provided
    if ticker is None:
        ticker = Path(csv_path).stem
    t = ticker.upper()
    ensure_ticker(t)
    inserted = 0
    for _, row in df.iterrows():
        # try multiple possible date column names
        if 'Date' in row.index:
            date_val = row['Date']
        elif 'date' in row.index:
            date_val = row['date']
        else:
            continue
        date = pd.to_datetime(date_val).strftime('%Y-%m-%d')
        open_p = safe_float(row.get('Open') if 'Open' in row.index else row.get('open'))
        high = safe_float(row.get('High') if 'High' in row.index else row.get('high'))
        low = safe_float(row.get('Low') if 'Low' in row.index else row.get('low'))
        close = safe_float(row.get('Close') if 'Close' in row.index else row.get('close'))
        adj = safe_float(row.get('Adj Close') if 'Adj Close' in row.index else row.get('Adj_Close')) or close
        volume = safe_int(row.get('Volume') if 'Volume' in row.index else row.get('volume'))
        try:
            cur.execute(
                'INSERT OR REPLACE INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (t, date, open_p, high, low, close, adj, volume)
            )
            inserted += 1
        except Exception as e:
            print('Insert failed for', t, date, e)
    conn.commit()
    conn.close()
    print(f'Ingested {inserted} rows for {t} from {csv_path}')


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
        print('No CSV files found under', args.csv_dir)
        return
    for p in paths:
        ingest_csv_to_db(args.db, p, None)


if __name__ == '__main__':
    main()
