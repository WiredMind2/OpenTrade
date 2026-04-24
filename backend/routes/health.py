"""
Health check endpoints for the Trading Backtester API.
"""
from datetime import datetime
import sqlite3
import sys
import os

from fastapi import APIRouter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from backend.config import get_config
from backend.logging_config import get_component_logger
from backend.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    from backend.main import app_state as backend_app_state  # Import here to avoid circular imports
    try:
        # Tests commonly patch `main.app_state`; prefer it when available.
        from main import app_state as shim_app_state  # type: ignore
    except Exception:
        shim_app_state = None

    logger = get_component_logger(__file__)
    app_state = shim_app_state if isinstance(shim_app_state, dict) else backend_app_state

    config = get_config()
    start_time = app_state.get("start_time") or datetime.utcnow()
    uptime = (datetime.utcnow() - start_time).total_seconds()

    # Check services
    services = {
        "database": "healthy",
        "models": "healthy" if app_state["models_loaded"] else "no_models",
        "api": "healthy"
    }

    # Test database connection
    try:
        conn = sqlite3.connect(app_state.get("database_path", "data/backtest.db"))
        conn.execute("SELECT 1")
        conn.close()
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = "unhealthy"
        logger.warning("Database health check failed", error=e)

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=config.version,
        uptime_seconds=uptime,
        services=services,
        database="sqlite",
        models_loaded=len(app_state["models_loaded"])
    )