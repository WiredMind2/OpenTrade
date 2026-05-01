"""
Strategy listing endpoints for the Trading Backtester API.
"""
from typing import List, Dict, Any, Optional
import sqlite3
import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List

from backend.logging_config import get_component_logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

router = APIRouter()

logger = get_component_logger(__file__)

# Rate limiter for heavy endpoints
limiter = Limiter(key_func=get_remote_address)


class TrainRequest(BaseModel):
    """Request model for training configuration."""
    dataset_path: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None  # Generic config for flexibility


class ProjectionRequest(BaseModel):
    """Request model for strategy projections."""
    symbol: str = Field(..., description="Stock symbol to project")
    startTime: str = Field(..., description="Start time for projection in ISO format")
    startPrice: float = Field(..., gt=0, description="Starting price for projection")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Strategy parameters")
    horizon: int = Field(default=30, ge=1, le=365, description="Number of days to project forward")


class ForecastRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    as_of: Optional[datetime] = Field(default=None, description="Anchor timestamp (defaults to utcnow)")
    current_price: Optional[float] = Field(default=None, gt=0, description="Optional override current price")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Strategy parameters")
    horizon_days: int = Field(default=5, ge=1, le=365)


class SignalRequest(BaseModel):
    symbols: List[str] = Field(..., min_length=1, description="Ticker symbols")
    as_of: Optional[datetime] = Field(default=None, description="Signal generation timestamp")
    current_prices: Optional[Dict[str, float]] = Field(default={}, description="Optional symbol->price overrides")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Strategy parameters")


def _get_db_connection():
    """Get database connection from app state."""
    db_path = None
    for module_name in ("backend.main", "main"):
        try:
            module = __import__(module_name, fromlist=["app_state"])
            app_state = getattr(module, "app_state", None)
            if isinstance(app_state, dict) and app_state.get("database_path"):
                db_path = app_state.get("database_path")
                break
        except Exception:
            continue
    if not db_path:
        from backend.config import get_config
        db_path = get_config().database.path
    return sqlite3.connect(db_path)


def _load_latest_prices(symbols: List[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        for symbol in symbols:
            cur.execute(
                """
                SELECT close
                FROM price_daily
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (symbol.upper(),),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                prices[symbol.upper()] = float(row[0])
    finally:
        conn.close()
    return prices


def validate_parameters(parameters: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Validate parameters against strategy's parameter schema."""
    for param_name, param_value in parameters.items():
        if param_name not in schema:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown parameter '{param_name}' for this strategy"
            )

        param_schema = schema[param_name]
        param_type = param_schema.get('type')

        # Type validation
        if param_type == 'int':
            if not isinstance(param_value, int):
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter '{param_name}' must be an integer"
                )
        elif param_type == 'float':
            if not isinstance(param_value, (int, float)):
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter '{param_name}' must be a number"
                )
        elif param_type == 'bool':
            if not isinstance(param_value, bool):
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter '{param_name}' must be a boolean"
                )
        elif param_type == 'str':
            if not isinstance(param_value, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter '{param_name}' must be a string"
                )

        # Range validation if specified
        if 'min' in param_schema and param_value < param_schema['min']:
            raise HTTPException(
                status_code=400,
                detail=f"Parameter '{param_name}' must be >= {param_schema['min']}"
            )
        if 'max' in param_schema and param_value > param_schema['max']:
            raise HTTPException(
                status_code=400,
                detail=f"Parameter '{param_name}' must be <= {param_schema['max']}"
            )

        # Custom validation for specific parameters
        if param_name == 'short_window' and 'long_window' in parameters:
            if parameters['short_window'] >= parameters['long_window']:
                raise HTTPException(
                    status_code=400,
                    detail="short_window must be less than long_window"
                )


@router.get("/strategies", response_model=List[Dict[str, Any]], tags=["Strategies"])
async def list_strategies():
    """List all registered strategies with their metadata."""
    from backend.main import app_state  # Import here to avoid circular imports

    logger.info("list_strategies endpoint called")

    registry = app_state.get("strategy_registry")
    if not registry:
        logger.warning("Strategy registry not found in app_state")
        return []

    strategies = registry.list()
    logger.info(f"Returning {len(strategies)} strategies", strategies=[s['name'] for s in strategies])
    return strategies


@router.get("/strategies/{name}", response_model=Dict[str, Any], tags=["Strategies"])
async def get_strategy(name: str):
    """Get detailed metadata for a specific strategy by name."""
    from backend.main import app_state  # Import here to avoid circular imports

    registry = app_state.get("strategy_registry")
    if not registry:
        logger.warning("Strategy registry not found in app_state")
        raise HTTPException(status_code=500, detail="Strategy registry not available")

    strategy = registry.get(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    # Get base metadata
    metadata = {
        'name': strategy.name,
        'description': strategy.description,
        'type': strategy.type,
        'parameters_schema': strategy.parameters_schema,
        'can_train': strategy.can_train
    }

    # For ML strategies, include model info (None for Phase 1)
    if strategy.type == 'ml':
        metadata['model_info'] = None  # Placeholder for future ML model info

    return metadata


@router.post("/strategies/{name}/train", response_model=Dict[str, str], tags=["Strategies"])
async def train_strategy(name: str, request: TrainRequest):
    """Train a strategy's model with the provided configuration."""
    from backend.main import app_state  # Import here to avoid circular imports

    registry = app_state.get("strategy_registry")
    if not registry:
        logger.warning("Strategy registry not found in app_state")
        raise HTTPException(status_code=500, detail="Strategy registry not available")

    strategy = registry.get(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    if not strategy.can_train:
        raise HTTPException(status_code=400, detail=f"Strategy '{name}' does not support training")

    try:
        # Prepare config dict from request
        config = request.config or {}
        if request.dataset_path:
            config['csv_path'] = request.dataset_path
        if request.hyperparameters:
            config.update(request.hyperparameters)

        # Call strategy.train() which handles job creation and background execution
        result = strategy.train(config)
        job_id = result.get("job_id") if isinstance(result, dict) else result

        return {"job_id": job_id}

    except Exception as e:
        logger.error(f"Error training strategy '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/model_jobs/{job_id}", response_model=Dict[str, Any], tags=["Jobs"])
async def get_model_job_status(job_id: str):
    """Get the status of a model training job."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, model_name, status, created_at, updated_at, config, result, error
            FROM model_jobs
            WHERE id = ?
        """, (job_id,))

        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        job_data = {
            "job_id": row[0],
            "model_name": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "config": row[5],
            "results": row[6],
            "logs": None,
            "error": row[7]
        }

        # Extract logs from results if available
        if job_data["results"]:
            try:
                results = json.loads(job_data["results"])
                if isinstance(results, dict) and "stdout" in results:
                    job_data["logs"] = results["stdout"]
            except:
                pass  # If parsing fails, leave logs as None

        return job_data

    finally:
        conn.close()


@router.post("/strategies/{name}/project", response_model=Dict[str, Any], tags=["Strategies"])
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute
async def project_strategy(request: Request, name: str, req: ProjectionRequest):
    """Project future performance of a strategy."""
    from backend.main import app_state  # Import here to avoid circular imports
    from backend.cache import chart_data_cache

    registry = app_state.get("strategy_registry")
    if not registry:
        logger.warning("Strategy registry not found in app_state")
        raise HTTPException(status_code=500, detail="Strategy registry not available")

    strategy = registry.get(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    # Validate parameters against strategy schema
    validate_parameters(req.params, strategy.parameters_schema)

    # Create cache key: strategy+symbol+start_time+params+horizon
    cache_key = f"{name}+{req.symbol}+{req.startTime}+{json.dumps(req.params, sort_keys=True)}+{req.horizon}"

    # Check cache first
    cached_result = chart_data_cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for projection: {cache_key}")
        return cached_result

    try:
        # Call strategy.project() method with updated parameters
        projection = strategy.project(
            parameters=req.params,
            projection_days=req.horizon,
            initial_capital=req.startPrice  # Use startPrice as initial capital
        )

        # Cache the result
        chart_data_cache.set(cache_key, projection)

        return projection

    except Exception as e:
        logger.error(f"Error projecting strategy '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Projection failed: {str(e)}")


@router.post("/strategies/{name}/forecast", response_model=Dict[str, Any], tags=["Strategies"])
async def forecast_strategy(name: str, req: ForecastRequest):
    """Generate structured forecast output for a single symbol."""
    from backend.main import app_state

    registry = app_state.get("strategy_registry")
    if not registry:
        raise HTTPException(status_code=500, detail="Strategy registry not available")
    strategy = registry.get(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    as_of = req.as_of or datetime.utcnow()
    symbol = req.symbol.upper()
    prices = _load_latest_prices([symbol])
    current_price = req.current_price or prices.get(symbol)
    if not current_price or current_price <= 0:
        raise HTTPException(status_code=400, detail=f"No valid current price available for {symbol}")

    try:
        forecast = strategy.forecast(
            parameters=req.params or {},
            symbol=symbol,
            as_of=as_of,
            current_price=float(current_price),
            horizon_days=req.horizon_days,
        )
        return {
            "symbol": forecast.symbol,
            "horizon_days": forecast.horizon_days,
            "predicted_return": forecast.predicted_return,
            "confidence": forecast.confidence,
            "predicted_path": forecast.predicted_path,
            "metadata": forecast.metadata,
        }
    except Exception as e:
        logger.error(f"Error forecasting strategy '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@router.post("/strategies/{name}/signals", response_model=Dict[str, Any], tags=["Strategies"])
async def generate_strategy_signals(name: str, req: SignalRequest):
    """Generate executable target allocations for requested symbols."""
    from backend.main import app_state

    registry = app_state.get("strategy_registry")
    if not registry:
        raise HTTPException(status_code=500, detail="Strategy registry not available")
    strategy = registry.get(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    symbols = [s.upper() for s in req.symbols]
    prices = _load_latest_prices(symbols)
    merged_prices = {**prices, **{k.upper(): float(v) for k, v in (req.current_prices or {}).items()}}
    as_of = req.as_of or datetime.utcnow()

    try:
        allocations = strategy.generate_target_allocations(
            parameters=req.params or {},
            symbols=symbols,
            as_of=as_of,
            current_prices=merged_prices,
        )
        return {
            "strategy": name,
            "as_of": as_of.isoformat(),
            "signals": [
                {
                    "ticker": a.ticker,
                    "target_pct": a.target_pct,
                    "reason": a.reason,
                    "confidence": a.confidence,
                    "timestamp": a.timestamp.isoformat(),
                    "metadata": a.metadata,
                }
                for a in allocations
            ],
        }
    except Exception as e:
        logger.error(f"Error generating signals for strategy '{name}': {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")


@router.get("/strategies/{name}/signals/schema", response_model=Dict[str, Any], tags=["Strategies"])
async def get_strategy_signal_schema(name: str):
    """Return input and output contract schema for signal generation."""
    from backend.main import app_state

    registry = app_state.get("strategy_registry")
    if not registry:
        raise HTTPException(status_code=500, detail="Strategy registry not available")
    strategy = registry.get(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    return {
        "strategy": name,
        "input": {
            "symbols": "string[]",
            "as_of": "datetime (optional)",
            "current_prices": "record<string, number> (optional)",
            "params": strategy.parameters_schema,
        },
        "output": {
            "signals": [
                {
                    "ticker": "string",
                    "target_pct": "number (-1..1)",
                    "reason": "string",
                    "confidence": "number (0..1)",
                    "timestamp": "datetime",
                    "metadata": "object",
                }
            ]
        },
    }