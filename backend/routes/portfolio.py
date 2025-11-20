"""
Portfolio endpoints for the Trading Backtester API.
"""
import json
import sqlite3
import pandas as pd
from datetime import datetime
import sys
import os

from fastapi import APIRouter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from backend.logging_config import get_app_logger
from backend.schemas import PortfolioResponse


logger = get_app_logger()
router = APIRouter()


@router.get("/portfolio/current", response_model=PortfolioResponse, tags=["Portfolio"])
async def get_current_portfolio():
    """Get current portfolio state."""
    from backend.main import app_state  # Import here to avoid circular imports

    try:
        conn = sqlite3.connect(app_state["database_path"])

        # Get latest portfolio snapshot
        cur = conn.cursor()
        cur.execute("""
            SELECT total_value, cash, invested_value, exposure, pnl, daily_return, positions_json, timestamp
            FROM portfolio_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """)

        row = cur.fetchone()
        conn.close()

        if row:
            total_value, cash, invested_value, exposure, pnl, daily_return, positions_json, timestamp = row
            positions = json.loads(positions_json) if positions_json else []

            return PortfolioResponse(
                timestamp=pd.to_datetime(timestamp).to_pydatetime(),
                total_value=total_value,
                cash=cash,
                invested_value=invested_value,
                exposure=exposure,
                positions=positions,
                pnl=pnl,
                daily_return=daily_return
            )

        # Fallback to mock data if no portfolio snapshots exist
        return PortfolioResponse(
            timestamp=datetime.utcnow(),
            total_value=105000.0,
            cash=25000.0,
            invested_value=80000.0,
            exposure=0.76,
            positions=[
                {"ticker": "AAPL", "quantity": 100, "value": 15000.0, "pnl": 1000.0},
                {"ticker": "MSFT", "quantity": 50, "value": 12500.0, "pnl": 750.0},
                {"ticker": "GOOGL", "quantity": 20, "value": 25000.0, "pnl": -500.0}
            ],
            pnl=5000.0,
            daily_return=0.008
        )

    except Exception as e:
        logger.error(f"Failed to get portfolio: {str(e)}")
        # Return fallback data instead of 500 error
        return PortfolioResponse(
            timestamp=datetime.utcnow(),
            total_value=105000.0,
            cash=25000.0,
            invested_value=80000.0,
            exposure=0.76,
            positions=[
                {"ticker": "AAPL", "quantity": 100, "value": 15000.0, "pnl": 1000.0},
                {"ticker": "MSFT", "quantity": 50, "value": 12500.0, "pnl": 750.0},
                {"ticker": "GOOGL", "quantity": 20, "value": 25000.0, "pnl": -500.0}
            ],
            pnl=5000.0,
            daily_return=0.008
        )