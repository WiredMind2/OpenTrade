"""
Quick bootstrap - Add tickers to database without fetching price data.

This is a fast alternative to bootstrap_tickers.py. It only adds ticker
entries to the tickers table. Price data will be fetched on-demand
when users view charts.

Usage:
    python -m backend.scripts.quick_bootstrap
"""
import sqlite3
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Popular US stock tickers
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK.B", "UNH", "JNJ", "V", "XOM", "JPM", "WMT", "MA",
    "PG", "HD", "CVX", "LLY", "ABBV", "MRK", "PEP", "KO",
    "COST", "AVGO", "TMO", "MCD", "CSCO", "ABT", "ACN"
]


def main():
    from backend.config import get_config
    from backend.main import app_state
    
    config = get_config()
    db_path = app_state.get('database_path') or config.database.path
    
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', db_path
        ))
    
    print(f"Database: {db_path}")
    
    if not os.path.exists(db_path):
        print("ERROR: Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get existing tickers
    cur.execute("SELECT ticker FROM tickers")
    existing = set(row[0] for row in cur.fetchall())
    
    added = 0
    for ticker in TICKERS:
        if ticker not in existing:
            cur.execute(
                "INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)",
                (ticker, None, "NASDAQ")
            )
            if cur.rowcount > 0:
                added += 1
                print(f"Added: {ticker}")
    
    conn.commit()
    conn.close()
    
    print(f"\nDone! Added {added} new tickers.")
    print("Price data will be fetched on-demand when charts are viewed.")


if __name__ == "__main__":
    main()