"""
Backtest endpoints for the Trading Backtester API.
"""
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks

from backend.logging_config import get_component_logger
from backend.schemas import BacktestRequest, BacktestResult
from .backtest_engine import run_backtest_background


logger = get_component_logger(__file__)
router = APIRouter()


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

        # Validate date range
        if (request.end_date - request.start_date).days > 365 * 5:  # Max 5 years
            raise HTTPException(status_code=400, detail="Date range too large (max 5 years)")

        # Generate unique backtest ID
        backtest_id = f"bt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

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

        # Run backtest in background
        background_tasks.add_task(
            run_backtest_background,
            backtest_id,
            request.strategy_name,
            request.start_date,
            request.end_date,
            request.initial_capital,
            request.parameters,
            app_state
        )

        # Return immediate response with backtest ID
        return BacktestResult(
            strategy_name=request.strategy_name,
            start_date=request.start_date,
            end_date=request.end_date,
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
            FROM backtest_runs WHERE id = ?
        """, (backtest_id,))

        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Backtest not found")

        (name, started_at, completed_at, initial_capital, final_value,
         total_return, annualized_return, sharpe_ratio, max_drawdown,
         win_rate, total_trades, avg_trade_return, volatility,
         equity_curve_json, metrics_json) = row

        equity_curve = json.loads(equity_curve_json) if equity_curve_json else []
        metrics = json.loads(metrics_json) if metrics_json else {}

        return BacktestResult(
            strategy_name=name,
            start_date=pd.to_datetime(started_at).to_pydatetime(),
            end_date=pd.to_datetime(completed_at).to_pydatetime(),
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            timestamp=pd.to_datetime(completed_at).to_pydatetime(),
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
            SELECT id, name, started_at, completed_at, initial_capital, final_value,
                   total_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
                   metrics
            FROM backtest_runs
            ORDER BY completed_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        backtests = []
        for row in cur.fetchall():
            (bt_id, name, started_at, completed_at, initial_capital, final_value,
             total_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
             metrics_json) = row

            metrics = json.loads(metrics_json) if metrics_json else {}

            backtests.append({
                "id": bt_id,
                "strategy_name": name,
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
                "timestamp": completed_at
            })

        conn.close()
        return backtests

    except Exception as e:
        logger.error(f"Failed to list backtests: {str(e)}")
        # Return empty list instead of 500 error
        return []