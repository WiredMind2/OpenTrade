"""
Bootstrap popular tickers into the database.

This script pre-loads the most traded stock tickers so users can immediately
view charts without waiting for external data fetches.

Run this once during initial setup or to refresh the ticker list.
"""
import sqlite3
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Popular US stock tickers to pre-load
POPULAR_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta (Facebook)
    "TSLA",   # Tesla
    "BRK.B",  # Berkshire Hathaway
    "UNH",    # UnitedHealth
    "JNJ",    # Johnson & Johnson
    "V",      # Visa
    "XOM",    # Exxon Mobil
    "JPM",    # JPMorgan Chase
    "WMT",    # Walmart
    "MA",     # Mastercard
    "PG",     # Procter & Gamble
    "HD",     # Home Depot
    "CVX",    # Chevron
    "LLY",    # Eli Lilly
    "ABBV",   # AbbVie
    "MRK",    # Merck
    "PEP",    # PepsiCo
    "KO",     # Coca-Cola
    "COST",   # Costco
    "AVGO",   # Broadcom
    "TMO",    # Thermo Fisher
    "MCD",    # McDonald's
    "CSCO",   # Cisco
    "ABT",    # Abbott
    "ACN",    # Accenture
]

POPULAR_TICKER_NAMES = {
    "AAPL": "Apple",
    "ABBV": "AbbVie",
    "ABT": "Abbott",
    "ACN": "Accenture",
    "AMZN": "Amazon",
    "AVGO": "Broadcom",
    "BRK.B": "Berkshire Hathaway",
    "COST": "Costco",
    "CSCO": "Cisco",
    "CVX": "Chevron",
    "GOOGL": "Alphabet",
    "HD": "Home Depot",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "LLY": "Eli Lilly",
    "MA": "Mastercard",
    "MCD": "McDonald's",
    "META": "Meta",
    "MRK": "Merck",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "PEP": "PepsiCo",
    "PG": "Procter & Gamble",
    "TMO": "Thermo Fisher",
    "TSLA": "Tesla",
    "UNH": "UnitedHealth Group",
    "V": "Visa",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
}


def get_db_path():
    """Get the database path from config or use default."""
    from backend.config import get_config
    from backend.main import app_state
    
    config = get_config()
    db_path = app_state.get('database_path') or config.database.path
    
    # Make path absolute if relative
    if not os.path.isabs(db_path):
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', db_path))
    
    return db_path


def ensure_tickers_in_db(db_path: str, tickers: list) -> dict:
    """Ensure tickers exist in the tickers table."""
    results = {"added": 0, "existing": 0, "errors": []}
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    for ticker in tickers:
        try:
            name = POPULAR_TICKER_NAMES.get(ticker)
            cur.execute(
                "INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)",
                (ticker, name, "NASDAQ")
            )
            if cur.rowcount > 0:
                results["added"] += 1
            else:
                results["existing"] += 1
                cur.execute(
                    "UPDATE tickers SET name = COALESCE(name, ?) WHERE ticker = ?",
                    (name, ticker),
                )
        except Exception as e:
            results["errors"].append(f"{ticker}: {str(e)}")
    
    conn.commit()
    conn.close()
    
    return results


def fetch_and_store_prices(db_path: str, ticker: str, days: int = 730) -> bool:
    """Fetch price data from Yahoo Finance and store in database."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
        return False
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Install with: pip install pandas")
        return False
    
    try:
        stock = yf.Ticker(ticker)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = stock.history(start=start_date, end=end_date, interval="1d")
        
        if df is None or df.empty:
            print(f"  No data available for {ticker}")
            return False
        
        # Reset index and rename columns
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adjusted_close',
            'Volume': 'volume'
        })
        
        # Format date
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df = df[df['date'].notna()]
        df['ticker'] = ticker
        
        # Insert into database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        inserted = 0
        for _, row in df.iterrows():
            try:
                cur.execute("""
                    INSERT OR REPLACE INTO price_daily
                    (ticker, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['ticker'],
                    row['date'],
                    float(row['open']) if pd.notna(row['open']) else None,
                    float(row['high']) if pd.notna(row['high']) else None,
                    float(row['low']) if pd.notna(row['low']) else None,
                    float(row['close']) if pd.notna(row['close']) else None,
                    float(row['adjusted_close']) if pd.notna(row.get('adjusted_close')) else None,
                    int(row['volume']) if pd.notna(row['volume']) else None
                ))
                inserted += 1
            except Exception as e:
                print(f"    Error inserting {ticker} {row.get('date')}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"  Inserted {inserted} price records for {ticker}")
        return inserted > 0
        
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return False


def main():
    """Main function to bootstrap tickers."""
    print("=" * 60)
    print("Bootstrap Popular Tickers")
    print("=" * 60)
    
    db_path = get_db_path()
    print(f"\nDatabase: {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return
    
    # Ensure tickers exist in tickers table
    print(f"\nAdding {len(POPULAR_TICKERS)} tickers to tickers table...")
    results = ensure_tickers_in_db(db_path, POPULAR_TICKERS)
    print(f"  Added: {results['added']}")
    print(f"  Existing: {results['existing']}")
    
    if results['errors']:
        print(f"  Errors: {results['errors'][:5]}")  # Show first 5 errors
    
    # Check which tickers already have price data
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM price_daily GROUP BY ticker")
    existing = set(row[0] for row in cur.fetchall())
    conn.close()
    
    print(f"\nExisting tickers with price data: {len(existing)}")
    
    # Fetch prices for tickers that don't have data
    tickers_to_fetch = [t for t in POPULAR_TICKERS if t not in existing]
    print(f"Tickers needing price data: {len(tickers_to_fetch)}")
    
    if tickers_to_fetch:
        print("\nFetching price data from Yahoo Finance...")
        print("(This may take a few minutes)\n")
        
        for i, ticker in enumerate(tickers_to_fetch, 1):
            print(f"[{i}/{len(tickers_to_fetch)}] Fetching {ticker}...", end=" ")
            fetch_and_store_prices(db_path, ticker)
    
    # Final summary
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT ticker, COUNT(*) as cnt FROM price_daily GROUP BY ticker ORDER BY cnt DESC")
    final_tickers = cur.fetchall()
    conn.close()
    
    print("\n" + "=" * 60)
    print("Bootstrap Complete!")
    print("=" * 60)
    print(f"\nTotal tickers with data: {len(final_tickers)}")
    print("\nTop 10 tickers by data points:")
    for ticker, cnt in final_tickers[:10]:
        print(f"  {ticker}: {cnt} records")


if __name__ == "__main__":
    main()
