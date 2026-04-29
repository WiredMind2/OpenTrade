import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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
from backend.scripts.script_logger import logger

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
            logger.warning('No data found for %s', ticker)
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
        logger.error('Error fetching data for %s: %s', ticker, e)
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
            logger.error('Error inserting row for %s at %s: %s', row['ticker'], row['dt'], e)

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

    logger.info('Fetching %s data for period %s...', args.interval, args.period)

    total_inserted = 0
    for ticker in tickers:
        logger.info('Fetching data for %s...', ticker)
        df = fetch_minute_data(ticker, args.period, args.interval)
        if df is not None:
            inserted = insert_minute_data(args.db, df)
            logger.info('Inserted %d rows for %s', inserted, ticker)
            total_inserted += inserted
        else:
            logger.warning('Failed to fetch data for %s', ticker)

    logger.info('Total rows inserted: %d', total_inserted)


if __name__ == '__main__':
    main()