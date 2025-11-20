"""
Data endpoints for the Trading Backtester API.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query

from backend.logging_config import get_app_logger


logger = get_app_logger()
router = APIRouter()


@router.get("/data/prices/{ticker}", tags=["Data"])
async def get_price_data(
    ticker: str = Path(description="Stock ticker"),
    start_date: Optional[datetime] = Query(None, description="Start date"),
    end_date: Optional[datetime] = Query(None, description="End date"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records")
):
    """Get historical price data."""
    from backend.main import app_state  # Import here to avoid circular imports

    # Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="Start date must be before or equal to end date")

    try:
        conn = sqlite3.connect(app_state["database_path"])

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

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

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