"""
Backtest endpoints for the Trading Backtester API.
"""
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import uuid

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks

from backend.logging_config import get_component_logger
from backend.schemas import BacktestRequest, BacktestResult
from .backtest_engine import insert_pending_backtest_run, run_backtest_background


logger = get_component_logger(__file__)
router = APIRouter()


def _json_object(value: str | None) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


@router.post("/backtest", response_model=BacktestResult, tags=["Backtests"])
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """Start a new backtest using Backtrader."""
    from backend.main import app_state  # Import here to avoid circular imports
    from backend.config import get_config

    try:
        config = get_config()
        # Validate date range before touching registry (cheap guard; keeps tests deterministic).
        if (request.end_date - request.start_date).days > 365 * 5:  # Max 5 years
            raise HTTPException(status_code=400, detail="Date range too large (max 5 years)")

        registry = app_state.get("strategy_registry")
        strategy = registry.get(request.strategy_name) if registry else None
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy '{request.strategy_name}' not found")

        # Generate unique backtest ID
        backtest_id = f"bt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"

        # Log backtest request
        logger.info(
            f"Starting backtest: {request.strategy_name} from {request.start_date} to {request.end_date}",
            extra={
                "strategy": request.strategy_name,
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "initial_capital": request.initial_capital,
                "backtest_id": backtest_id
            }
        )

        db_path = app_state.get("database_path") or config.database.path
        pending_run_id = insert_pending_backtest_run(
            db_path,
            strategy_name=request.strategy_name,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            parameters=request.parameters or {},
            backtest_id=backtest_id,
        )

        # Run backtest in background (preflight + execution); clients poll GET /backtest/{id}
        background_tasks.add_task(
            run_backtest_background,
            backtest_id,
            request.strategy_name,
            request.start_date,
            request.end_date,
            request.initial_capital,
            request.parameters,
            app_state,
            pending_run_id,
        )

        # Return immediate response with backtest ID
        return BacktestResult(
            strategy_name=request.strategy_name,
            start_date=request.start_date,
            end_date=request.end_date,
            completed_at=None,
            initial_capital=request.initial_capital,
            final_value=request.initial_capital,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            volatility=0.0,
            timestamp=datetime.utcnow(),
            metrics={"backtest_id": backtest_id, "status": "running"},
            equity_curve=[]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest/{backtest_id}", response_model=BacktestResult, tags=["Backtests"])
async def get_backtest_result(
    backtest_id: str = Path(description="Backtest ID")
):
    """Get backtest results by ID."""
    from backend.main import app_state  # Import here to avoid circular imports

    try:
        conn = sqlite3.connect(app_state["database_path"])

        cur = conn.cursor()
        cur.execute("""
            SELECT name, started_at, completed_at, initial_capital, final_value,
                   total_return, annualized_return, sharpe_ratio, max_drawdown,
                   win_rate, total_trades, avg_trade_return, volatility,
                   equity_curve, metrics
            FROM backtest_runs
            WHERE CAST(id AS TEXT) = ?
               OR json_extract(metrics, '$.backtest_id') = ?
            ORDER BY id DESC
            LIMIT 1
        """, (backtest_id, backtest_id))

        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Backtest not found")

        (name, started_at, completed_at, initial_capital, final_value,
         total_return, annualized_return, sharpe_ratio, max_drawdown,
         win_rate, total_trades, avg_trade_return, volatility,
         equity_curve_json, metrics_json) = row

        def _f(x, default: float = 0.0) -> float:
            try:
                return default if x is None else float(x)
            except Exception:
                return default

        if isinstance(equity_curve_json, str) and equity_curve_json.strip():
            equity_curve = json.loads(equity_curve_json)
        elif isinstance(equity_curve_json, list):
            equity_curve = equity_curve_json
        else:
            equity_curve = []
        if isinstance(metrics_json, dict):
            metrics = metrics_json
        elif isinstance(metrics_json, str) and metrics_json.strip():
            metrics = json.loads(metrics_json)
        else:
            metrics = {}

        sim_end = metrics.get("end_date") if isinstance(metrics, dict) else None
        end_ts = completed_at or sim_end or started_at

        return BacktestResult(
            strategy_name=name,
            start_date=pd.to_datetime(started_at).to_pydatetime(),
            end_date=pd.to_datetime(end_ts).to_pydatetime(),
            completed_at=pd.to_datetime(completed_at).to_pydatetime() if completed_at else None,
            initial_capital=_f(initial_capital, 100000.0),
            final_value=_f(final_value, _f(initial_capital, 100000.0)),
            total_return=_f(total_return),
            annualized_return=_f(annualized_return),
            sharpe_ratio=_f(sharpe_ratio),
            max_drawdown=_f(max_drawdown),
            win_rate=_f(win_rate),
            total_trades=int(total_trades or 0),
            avg_trade_return=_f(avg_trade_return),
            volatility=_f(volatility),
            timestamp=pd.to_datetime(completed_at or started_at).to_pydatetime(),
            metrics=metrics,
            equity_curve=equity_curve
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest result: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading/backtest", response_model=List[Dict[str, Any]], tags=["Backtests"])
async def list_backtests(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(5, ge=1, le=50, description="Items per page")
):
    """List recent backtests with pagination."""
    from backend.main import app_state  # Import here to avoid circular imports

    try:
        conn = sqlite3.connect(app_state["database_path"])

        offset = (page - 1) * limit
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, params, started_at, completed_at, initial_capital, final_value,
                   total_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
                   metrics, equity_curve, params
            FROM backtest_runs
            ORDER BY completed_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        backtests = []
        for row in cur.fetchall():
            (bt_id, name, params_json, started_at, completed_at, initial_capital, final_value,
             total_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
             metrics_json, equity_curve_json, params_json) = row

            params = _json_object(params_json)
            metrics = _json_object(metrics_json)
            execution_summary = metrics.get("execution_summary", {})
            run_ticker = None
            t = params.get("ticker")
            if isinstance(t, str) and t.strip():
                run_ticker = t.strip().upper()
            equity_curve = json.loads(equity_curve_json) if equity_curve_json else []
            chart_data = []
            for idx, point in enumerate(equity_curve):
                if not isinstance(point, dict):
                    continue
                value = point.get("value")
                if value is None:
                    continue
                pt = {"day": idx, "value": value}
                if isinstance(point.get("date"), str) and point.get("date"):
                    pt["date"] = point["date"]
                chart_data.append(pt)

            backtests.append({
                "id": bt_id,
                "strategy_name": name,
                "params": params,
                "ticker": run_ticker,
                "start_date": started_at,
                "end_date": completed_at,
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "status": metrics.get("status", "completed"),
                "error": metrics.get("error"),
                "metrics": metrics,
                "equity_curve": equity_curve,
                "execution_engine": execution_summary.get("engine"),
                "signals_emitted": execution_summary.get("signals_emitted", 0),
                "order_intents": execution_summary.get("order_intents", 0),
                "order_fills": execution_summary.get("order_fills", 0),
                "timestamp": completed_at
                if completed_at
                else started_at
                if started_at
                else datetime.utcnow().isoformat(),
                "chart_data": chart_data,
            })

        conn.close()
        return backtests

    except Exception as e:
        logger.error(f"Failed to list backtests: {str(e)}")
        # Return empty list instead of 500 error
        return []
