"""
Main FastAPI application for the Trading Backtester API.

This module sets up the FastAPI application and includes all route modules.
"""
import sys
import os
# Add the parent directory to sys.path to allow relative imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from pathlib import Path

from backend.config import get_config
from backend.logging_config import get_component_logger
from backend.error_handling import TradingBacktesterError
from backend.models import ModelRegistry
from backend.strategies import strategy_registry
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import route modules (package-relative)
from backend.routes.health import router as health_router
from backend.routes.monitoring import router as monitoring_router
from backend.routes.predictions import router as predictions_router
from backend.routes.backtests import router as backtests_router
from backend.routes.models_endpoints import router as models_router
from backend.routes.portfolio import router as portfolio_router
from backend.routes.data_endpoints import router as data_router
from backend.routes.websocket import websocket_endpoint, broadcast_chart_update
from backend.routes.scripts import router as scripts_router
from backend.routes.udf import router as udf_router
from backend.routes.strategies import router as strategies_router
from backend.routes.strategy_analytics import router as strategy_analytics_router
from backend.ml.storage import ensure_ml_schema



logger = get_component_logger(__file__)


# Global state
app_state = {
    "start_time": datetime.utcnow(),
    "models_loaded": {},  # Keep for backward compatibility
    "model_registry": ModelRegistry(),
    "strategy_registry": strategy_registry,
    "database_path": None,
    "active_websockets": set(),
    "chart_broadcast_task": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Trading Backtester API")

    # Load configuration
    config = get_config()
    config.database.path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', config.database.path))
    if app_state["database_path"] is None:
        app_state["database_path"] = config.database.path

    # Initialize database connection
    await init_database()

    # Load models
    if not app_state["models_loaded"]:
        await load_models()

    # Start chart broadcasting task
    # Avoid background workers during tests to prevent file-handle leaks
    # (especially on Windows where SQLite files can't be unlinked while open).
    is_test_env = bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"))
    if not is_test_env:
        app_state["chart_broadcast_task"] = asyncio.create_task(chart_broadcast_worker())

    logger.info("Trading Backtester API started successfully")

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down Trading Backtester API")

        # Cancel chart broadcasting task
        if app_state["chart_broadcast_task"]:
            app_state["chart_broadcast_task"].cancel()
            try:
                await app_state["chart_broadcast_task"]
            except asyncio.CancelledError:
                pass

        await cleanup_resources()


# Create FastAPI application
app = FastAPI(
    title="Trading Backtester API",
    description="REST API for algorithmic trading backtesting and strategy evaluation",
    version="1.0.0",
    lifespan=lifespan  # type: ignore[type-abstract]
)

# Initialize rate limiter
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(SlowAPIMiddleware)

# Add rate limit exceeded handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include route modules
app.include_router(health_router)
app.include_router(monitoring_router)
app.include_router(predictions_router)
logger.info(f"Predictions router included, routes: {[route.path for route in predictions_router.routes]}")
app.include_router(backtests_router)
app.include_router(models_router, prefix="/api")
app.include_router(strategies_router, prefix="/api", tags=["strategies"])
app.include_router(strategy_analytics_router)
app.include_router(portfolio_router)
app.include_router(data_router)
app.include_router(scripts_router)
app.include_router(udf_router, prefix="/udf")

# Add WebSocket endpoint
app.add_api_websocket_route("/ws", websocket_endpoint)


async def init_database():
    """Initialize database connection."""
    import sqlite3

    try:
        conn = sqlite3.connect(app_state["database_path"])
        # Test connection
        conn.execute("SELECT 1")
        conn.close()
        logger.info("Database connection initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise TradingBacktesterError(f"Database initialization failed: {e}")


async def load_models():
    """Load trained models into memory using the model registry."""
    config = get_config()
    models_dir = Path(config.model.model_dir)
    models_pkg_dir = Path(__file__).parent / "models"

    # Discover and register models
    app_state["model_registry"].discover(models_dir, models_pkg_dir)

    # For backward compatibility, populate the old models_loaded dict
    registry = app_state["model_registry"]
    for model in registry.list():
        if hasattr(model, '_model_data'):
            # For joblib models, keep the old format
            app_state["models_loaded"][model.name] = model._model_data

    logger.info(f"Loaded {len(registry.list())} models via registry")


async def cleanup_resources():
    """Cleanup resources on shutdown."""
    # Close any open database connections
    # Save model states if needed
    # Clean up temporary files
    logger.info("Resources cleaned up")


async def chart_broadcast_worker():
    """Background worker that periodically broadcasts chart updates for subscribed symbols."""
    import sqlite3
    import pandas as pd
    from backend.routes.websocket import chart_subscriptions

    logger.info("Starting chart broadcast worker")

    while True:
        try:
            # Wait 30 seconds between broadcasts
            await asyncio.sleep(30)

            # Check if there are any active subscriptions
            if not chart_subscriptions:
                continue

            # Get unique symbol-resolution pairs that have subscriptions
            active_symbols = set()
            for subscriptions in chart_subscriptions.values():
                for symbol, resolution, _ in subscriptions:
                    active_symbols.add((symbol, resolution))

            if not active_symbols:
                continue

            logger.debug(f"Chart broadcast worker: checking {len(active_symbols)} symbol-resolution pairs")

            # For each active symbol-resolution, get the latest bar and broadcast
            for symbol, resolution in active_symbols:
                try:
                    # Determine which table to query based on resolution
                    # For simplicity, we'll use price_daily for now, but could be extended
                    table = 'price_daily'
                    date_col = 'date'

                    conn = sqlite3.connect(app_state["database_path"])

                    # Get the most recent bar for this symbol
                    query = f"""
                        SELECT {date_col} as date, open, high, low, close, volume
                        FROM {table}
                        WHERE ticker = ?
                        ORDER BY {date_col} DESC
                        LIMIT 1
                    """

                    df = pd.read_sql_query(query, conn, params=[symbol])
                    conn.close()

                    if not df.empty:
                        row = df.iloc[0]
                        # Convert to bar format expected by frontend
                        bar_data = {
                            "time": int(pd.to_datetime(row['date']).timestamp()),
                            "open": float(row['open']),
                            "high": float(row['high']),
                            "low": float(row['low']),
                            "close": float(row['close']),
                            "volume": int(row['volume']) if not pd.isna(row['volume']) else 0
                        }

                        # Broadcast the update
                        await broadcast_chart_update(symbol, resolution, bar_data)
                        logger.debug(f"Broadcast chart update for {symbol}:{resolution}")

                except Exception as e:
                    logger.error(f"Error broadcasting chart update for {symbol}:{resolution}: {e}")

        except asyncio.CancelledError:
            logger.info("Chart broadcast worker cancelled")
            break
        except Exception as e:
            logger.error(f"Chart broadcast worker error: {e}")
            # Continue running despite errors


# Error handlers
@app.exception_handler(TradingBacktesterError)
async def trading_backtester_exception_handler(request, exc):
    """Handle custom TradingBacktesterError exceptions."""
    logger.error(f"TradingBacktesterError: {exc.message}", extra={"error": exc})
    return JSONResponse(
        status_code=500,
        content={"error": exc.message, "error_code": exc.error_code}
    )


@app.exception_handler(Exception)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=getattr(exc, 'status_code', 500),
        content={"error": getattr(exc, 'detail', str(exc))}
    )


# Utility functions


if __name__ == "__main__":
    import uvicorn

    config = get_config()

    uvicorn.run(
        "backend.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers if config.environment == "production" else 1,
        reload=config.api.reload if config.environment == "development" else False,
        log_level=config.logging.level.lower()
    )