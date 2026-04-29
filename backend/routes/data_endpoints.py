"""
Data endpoints for the Trading Backtester API.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from backend.logging_config import get_component_logger


logger = get_component_logger(__file__)
router = APIRouter()


def _resolve_db_path() -> str:
    """Resolve DB path regardless of import style used by the test/runtime."""
    # Some tests import `main` while runtime imports `backend.main`; support both.
    for module_name in ("main", "backend.main"):
        try:
            module = __import__(module_name, fromlist=["app_state"])
            app_state = getattr(module, "app_state", None)
            if isinstance(app_state, dict) and app_state.get("database_path"):
                return app_state["database_path"]
        except Exception:
            continue

    env_db = os.getenv("DB_PATH")
    if env_db:
        return env_db

    from backend.config import get_config
    return get_config().database.path


def _refresh_latest_daily_prices(db_path: str, ticker: str, end_date: Optional[datetime]) -> None:
    """Fetch and upsert missing recent daily bars into SQLite.

    The DB is initially populated from a historical Kaggle dump; this function keeps it
    current so charts show the latest candles.
    """
    try:
        import yfinance as yf
    except Exception:
        logger.warning("yfinance not available; skipping refresh for %s", ticker)
        return

    tkr = ticker.upper().strip()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT MAX(date) FROM price_daily WHERE ticker = ?", (tkr,))
            row = cur.fetchone()
            max_date_str = row[0] if row else None

        if max_date_str:
            last_dt = pd.to_datetime(max_date_str, errors="coerce").to_pydatetime()
            if not last_dt:
                return
            fetch_from = last_dt.date() + timedelta(days=1)
        else:
            # If no data exists, pull a reasonable default window.
            fetch_from = (datetime.utcnow().date() - timedelta(days=365 * 5))

        fetch_to = (end_date.date() if end_date else datetime.utcnow().date()) + timedelta(days=1)

        if fetch_from >= fetch_to:
            return

        stock = yf.Ticker(tkr)
        df = stock.history(start=fetch_from, end=fetch_to, interval="1d")
        if df is None or df.empty:
            return

        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )
        if "date" not in df.columns:
            return

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df[df["date"].notna()]

        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)",
                (tkr, None, None),
            )
            cur.executemany(
                """
                INSERT OR REPLACE INTO price_daily
                (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        tkr,
                        r["date"],
                        float(r["open"]) if pd.notna(r["open"]) else None,
                        float(r["high"]) if pd.notna(r["high"]) else None,
                        float(r["low"]) if pd.notna(r["low"]) else None,
                        float(r["close"]) if pd.notna(r["close"]) else None,
                        (
                            float(r["adjusted_close"])
                            if ("adjusted_close" in df.columns and pd.notna(r.get("adjusted_close")))
                            else (float(r["close"]) if pd.notna(r["close"]) else None)
                        ),
                        int(r["volume"]) if pd.notna(r["volume"]) else None,
                    )
                    for _, r in df.iterrows()
                ],
            )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to refresh latest prices for %s: %s", ticker, e)


@router.get("/data/prices/{ticker}", tags=["Data"])
async def get_price_data(
    ticker: str = Path(description="Stock ticker"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records")
):
    """Get historical price data."""
    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="Start date must be before or equal to end date")

    try:
        db_path = _resolve_db_path()

        # Best-effort refresh to ensure latest candles exist in DB.
        _refresh_latest_daily_prices(db_path, ticker, end_date)

        query = "SELECT date, open, high, low, close, adjusted_close, volume FROM price_daily WHERE ticker = ?"
        params = [ticker.upper()]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date().isoformat())

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date().isoformat())

        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)  # type: ignore[arg-type]

        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return {"ticker": ticker.upper(), "data": []}

        # Convert to list of dictionaries
        data = []
        for _, row in df.iterrows():
            data.append({
                "date": row['date'],
                "open": row['open'],
                "high": row['high'],
                "low": row['low'],
                "close": row['close'],
                "adjusted_close": row['adjusted_close'],
                "volume": row['volume']
            })

        return {
            "ticker": ticker.upper(),
            "count": len(data),
            "data": data
        }

    except Exception as e:
        logger.error(f"Failed to get price data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))