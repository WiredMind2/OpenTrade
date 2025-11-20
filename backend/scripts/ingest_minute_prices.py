"""
Minute-level price ingestion using yfinance.

Fetches 1-minute OHLCV data for tickers and stores in price_minute table.

Usage:
  python ingest_minute_prices.py --tickers AAPL,MSFT --period 7d
  python ingest_minute_prices.py --tickers AAPL --period 60d --interval 5m
"""

import argparse
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fetch_minute_data(ticker: str, period: str = "7d", interval: str = "1m"):
    """
    Fetch minute-level data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol
        period: Period to fetch (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

    Returns:
        DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"No data found for {ticker}")
            return None

        # Reset index to get datetime as column
        df = df.reset_index()

        # Rename columns to match our schema
        df = df.rename(columns={
            'Datetime': 'dt',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Ensure dt is in the right format
        if 'dt' in df.columns:
            df['dt'] = pd.to_datetime(df['dt']).dt.strftime('%Y-%m-%d %H:%M:%S')

        df['ticker'] = ticker.upper()

        return df[['ticker', 'dt', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def insert_minute_data(db_path: str, df: pd.DataFrame):
    """Insert minute data into the database."""
    if df is None or df.empty:
        return 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure ticker exists in tickers table
    ticker = df['ticker'].iloc[0]
    cur.execute('INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)',
                (ticker, None, None))

    inserted = 0
    for _, row in df.iterrows():
        try:
            cur.execute('''
                INSERT OR REPLACE INTO price_minute
                (ticker, dt, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['ticker'],
                row['dt'],
                float(row['open']) if pd.notna(row['open']) else None,
                float(row['high']) if pd.notna(row['high']) else None,
                float(row['low']) if pd.notna(row['low']) else None,
                float(row['close']) if pd.notna(row['close']) else None,
                int(row['volume']) if pd.notna(row['volume']) else None
            ))
            inserted += 1
        except Exception as e:
            print(f"Error inserting row for {row['ticker']} at {row['dt']}: {e}")

    conn.commit()
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description='Fetch and ingest minute-level price data')
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--tickers', required=True, help='Comma-separated list of tickers (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--period', default='7d', help='Period to fetch (default: 7d)')
    parser.add_argument('--interval', default='1m', choices=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h'],
                        help='Data interval (default: 1m)')

    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(',')]

    print(f"Fetching {args.interval} data for period {args.period}...")

    total_inserted = 0
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = fetch_minute_data(ticker, args.period, args.interval)
        if df is not None:
            inserted = insert_minute_data(args.db, df)
            print(f"Inserted {inserted} rows for {ticker}")
            total_inserted += inserted
        else:
            print(f"Failed to fetch data for {ticker}")

    print(f"Total rows inserted: {total_inserted}")


if __name__ == '__main__':
    main()