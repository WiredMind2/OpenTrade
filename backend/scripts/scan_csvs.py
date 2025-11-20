"""
Scan `data/kaggle_yahoo/` for CSV files, insert tickers into the `tickers` table, and report which default tickers are present.

Usage:
  python scripts/scan_csvs.py --db data/backtest.db --csv_dir data/kaggle_yahoo
"""
import os
import sqlite3
import argparse
from pathlib import Path

DEFAULT_TICKERS = ['AAPL','MSFT','AMZN','GOOG','NVDA','META','TSLA','JPM','BAC','JNJ']


def scan_and_register(db_path: str, csv_dir: str):
    p = Path(csv_dir)
    if not p.exists():
        print(f'CSV dir {csv_dir} does not exist')
        return
    # Recursively find CSV files (the Kaggle dataset may unpack into nested folders)
    files = [f for f in p.rglob('*.csv') if f.is_file()]
    tickers = [f.stem for f in files]
    # debug: show a few sample files
    sample = files[:10]
    if sample:
        print('Sample CSV files found:')
        for s in sample:
            print(' -', s)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    inserted = 0
    for t in tickers:
        try:
            cur.execute('INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)', (t.upper(), None, None))
            inserted += cur.rowcount
        except Exception as e:
            print('Failed to insert ticker', t, e)
    conn.commit()
    # report
    present_defaults = [t for t in DEFAULT_TICKERS if t in tickers]
    missing_defaults = [t for t in DEFAULT_TICKERS if t not in tickers]
    print(f'Found {len(tickers)} CSV files, registered {inserted} new tickers into the DB')
    print('Default tickers present:', ','.join(present_defaults))
    if missing_defaults:
        print('Default tickers missing (will substitute if needed):', ','.join(missing_defaults))
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--csv_dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kaggle_yahoo')))
    args = parser.parse_args()
    scan_and_register(args.db, args.csv_dir)


if __name__ == '__main__':
    main()
