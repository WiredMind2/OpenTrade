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
    from backend.main import app_state  # Import here to avoid circular imports

    logger = get_component_logger(__file__)

    config = get_config()
    uptime = (datetime.utcnow() - app_state["start_time"]).total_seconds()

    # Check services
    services = {
        "database": "healthy",
        "models": "healthy" if app_state["models_loaded"] else "no_models",
        "api": "healthy"
    }

    # Test database connection
    try:
        conn = sqlite3.connect(app_state["database_path"])
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