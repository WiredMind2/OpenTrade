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
from datetime import datetime, timezone
from pathlib import Path

from backend.config import get_config
from backend.logging_config import get_component_logger, setup_logging
from backend.error_handling import TradingBacktesterError
from backend.models import ModelRegistry
from backend.strategies import strategy_registry
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Initialize logging before route modules load so console output remains clear.
setup_logging()

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
from backend.routes.news import router as news_router
from backend.routes.monte_carlo import router as monte_carlo_router
from backend.ml.storage import ensure_ml_schema
from backend.db.variant_schema import ensure_variant_schema
from backend.services.news_auto_ingest import (
    daily_news_auto_ingest_worker,
    news_auto_ingest_disabled,
    parse_interval_sec,
    parse_query,
)



logger = get_component_logger(__file__)


# Global state
app_state = {
    "start_time": datetime.now(timezone.utc).replace(tzinfo=None),
    "models_loaded": {},  # Keep for backward compatibility
    "model_registry": ModelRegistry(),
    "strategy_registry": strategy_registry,
    "database_path": None,
    "active_websockets": set(),
    "chart_broadcast_task": None,
    "news_ingest_task": None,
}


def _ensure_legacy_columns(conn):
    """Patch minimal legacy schemas used by tests before applying full schema."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles'")
    if not cur.fetchone():
        return

    existing_columns = {row[1] for row in cur.execute("PRAGMA table_info(articles)").fetchall()}
    if "canonical_timestamp" not in existing_columns:
        cur.execute("ALTER TABLE articles ADD COLUMN canonical_timestamp TEXT")
        # Backfill from published_at when available so downstream queries still work.
        if "published_at" in existing_columns:
            cur.execute(
                "UPDATE articles SET canonical_timestamp = published_at WHERE canonical_timestamp IS NULL"
            )


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
    is_test_env = bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") or ("pytest" in sys.modules))
    if not is_test_env:
        app_state["chart_broadcast_task"] = asyncio.create_task(chart_broadcast_worker())
        news_key = config.newsapi_key or os.getenv("NEWSAPI_KEY")
        if news_key and not news_auto_ingest_disabled():
            app_state["news_ingest_task"] = asyncio.create_task(
                daily_news_auto_ingest_worker(
                    app_state["database_path"],
                    api_key=news_key,
                    interval_sec=parse_interval_sec(),
                    query=parse_query(),
                )
            )
            logger.info("Scheduled background news auto-ingest (NewsAPI)")

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

        if app_state.get("news_ingest_task"):
            app_state["news_ingest_task"].cancel()
            try:
                await app_state["news_ingest_task"]
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
app.include_router(news_router)
app.include_router(monte_carlo_router, prefix="/api/monte-carlo", tags=["Monte Carlo"])

# Add WebSocket endpoint
app.add_api_websocket_route("/ws", websocket_endpoint)


async def init_database():
    """Initialize database connection and create tables."""
    import sqlite3
    from pathlib import Path

    try:
        db_path = app_state["database_path"]
        if not db_path:
            raise TradingBacktesterError("Database path is not configured")

        # Ensure parent directory exists (common on fresh checkouts).
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        try:
            # Keep behavior consistent with schema.sql
            conn.execute("PRAGMA foreign_keys = ON;")
            _ensure_legacy_columns(conn)

            # Apply schema.sql (idempotent because it uses IF NOT EXISTS)
            schema_path = (Path(__file__).resolve().parent.parent / "db" / "schema.sql")
            if not schema_path.exists():
                raise TradingBacktesterError(f"Schema file not found at: {schema_path}")

            # Run migrations
            migration_path = (Path(__file__).resolve().parent.parent / "db" / "create_monte_carlo_tables.sql")
            if migration_path.exists():
                with open(migration_path, 'r') as f:
                    sql = f.read()
                conn.executescript(sql)

            schema_sql = schema_path.read_text(encoding="utf-8")
            try:
                conn.executescript(schema_sql)
            except sqlite3.DatabaseError as e:
                # If the DB file is in a broken state (common after an interrupted schema apply),
                # SQLite can refuse to even read sqlite_master. In that case, the safest recovery
                # is to delete the DB file and re-run initialization.
                msg = str(e).lower()
                if "malformed database schema" in msg or "database disk image is malformed" in msg:
                    raise TradingBacktesterError(
                        f"SQLite database file appears corrupted: {db_path}. "
                        f"Delete the file and restart to recreate tables from schema.sql. "
                        f"Original error: {e}"
                    )
                raise
            conn.commit()

            # Validate/patch ML schema expectations (adds missing columns if needed)
            ensure_ml_schema(conn)

            # Variant / params_hash columns and artifact FK backfill
            ensure_variant_schema(conn)

            # Test connection
            conn.execute("SELECT 1")
            logger.info("Database initialized and schema ensured")
        finally:
            conn.close()
    except Exception as e:
        logger.exception("Failed to initialize database")
        raise TradingBacktesterError(f"Database initialization failed: {e}")


async def load_models():
    """Load trained models into memory using the model registry."""
    config = get_config()
    models_dir = Path(config.model.model_dir)
    models_pkg_dir = Path(__file__).parent / "models"

    # Discover and register models
    app_state["model_registry"].discover(models_dir, models_pkg_dir)

    # For backward compatibility, populate the old models_loaded dict
    # Model filenames are like lightgbm_1d_20260428070915 but the route
    # looks up lightgbm_1d, so we register under the short key (first two parts).
    registry = app_state["model_registry"]
    for model in registry.list():
        if hasattr(model, '_model_data'):
            parts = model.name.split('_')
            short_key = '_'.join(parts[:2]) if len(parts) >= 3 else model.name
            app_state["models_loaded"][short_key] = model._model_data

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
                    logger.exception("Error broadcasting chart update for %s:%s", symbol, resolution)

        except asyncio.CancelledError:
            logger.info("Chart broadcast worker cancelled")
            break
        except Exception as e:
            logger.exception("Chart broadcast worker error")
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