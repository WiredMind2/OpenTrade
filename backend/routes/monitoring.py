"""
Monitoring and metrics endpoints for the Trading Backtester API.
"""
import psutil
import sqlite3
from datetime import datetime, timedelta
import sys
import os

from fastapi import APIRouter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from backend.logging_config import get_app_logger  # noqa: E402
from backend.schemas import SystemMetrics  # noqa: E402


logger = get_app_logger()
router = APIRouter()


@router.get("/metrics", response_model=SystemMetrics, tags=["Monitoring"])
async def get_system_metrics():
    """Get system performance metrics."""
    from backend.main import app_state  # Import here to avoid circular imports

    try:
        conn = sqlite3.connect(app_state["database_path"])
        cur = conn.cursor()

        # Get database connection count (approximate)
        cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        db_connections = cur.fetchone()[0]

        # Get recent predictions count (last 24 hours)
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        cur.execute("SELECT COUNT(*) FROM sentiment_predictions WHERE produced_at >= ?", (yesterday,))
        recent_predictions = cur.fetchone()[0]

        # Get recent backtests count
        cur.execute("SELECT COUNT(*) FROM backtest_runs WHERE completed_at >= ?", (yesterday,))
        recent_backtests = cur.fetchone()[0]

        conn.close()
    except Exception as e:
        logger.warning(f"Failed to get database metrics: {str(e)}")
        db_connections = 0
        recent_predictions = 0
        recent_backtests = 0

    return SystemMetrics(
        timestamp=datetime.utcnow(),
        cpu_percent=psutil.cpu_percent(),
        memory_percent=psutil.virtual_memory().percent,
        disk_usage_percent=psutil.disk_usage('/').percent,
        database_connections=db_connections,
        active_models=len(app_state["models_loaded"]),
        recent_predictions=recent_predictions,
        error_rate=0.0  # Would calculate from error logs in production
    )


@router.get("/monitoring/metrics", response_model=SystemMetrics, tags=["Monitoring"])
async def get_monitoring_metrics():
    """Get detailed monitoring metrics (alias for /metrics)."""
    return await get_system_metrics()