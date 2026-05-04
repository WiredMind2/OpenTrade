"""
Backtest endpoints for the Trading Backtester API.
"""
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.logging_config import get_component_logger


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


def _format_backtest_list_item(
    bt_id: Any,
    name: Any,
    params_json: Any,
    started_at: Any,
    completed_at: Any,
    initial_capital: Any,
    final_value: Any,
    total_return: Any,
    annualized_return: Any,
    sharpe_ratio: Any,
    max_drawdown: Any,
    win_rate: Any,
    total_trades: Any,
    avg_trade_return: Any,
    volatility: Any,
    metrics_json: Any,
    equity_curve_json: Any,
) -> Dict[str, Any]:
    """Shape one ``backtest_runs`` row for ``GET /trading/backtest`` (list or single-id lookup)."""
    params = _json_object(params_json) if isinstance(params_json, str) else (
        params_json if isinstance(params_json, dict) else {}
    )
    metrics = _json_object(metrics_json) if isinstance(metrics_json, str) else (
        metrics_json if isinstance(metrics_json, dict) else {}
    )
    execution_summary = metrics.get("execution_summary", {}) if isinstance(metrics, dict) else {}
    run_ticker = None
    t = params.get("ticker")
    if isinstance(t, str) and t.strip():
        run_ticker = t.strip().upper()

    if isinstance(equity_curve_json, str) and equity_curve_json.strip():
        try:
            equity_curve = json.loads(equity_curve_json)
        except (TypeError, json.JSONDecodeError):
            equity_curve = []
    elif isinstance(equity_curve_json, list):
        equity_curve = equity_curve_json
    else:
        equity_curve = []

    chart_data: List[Dict[str, Any]] = []
    for idx, point in enumerate(equity_curve):
        if not isinstance(point, dict):
            continue
        value = point.get("value")
        if value is None:
            continue
        pt: Dict[str, Any] = {"day": idx, "value": value}
        if isinstance(point.get("date"), str) and point.get("date"):
            pt["date"] = point["date"]
        chart_data.append(pt)

    def _f(x: Any, default: float = 0.0) -> float:
        try:
            return default if x is None else float(x)
        except (TypeError, ValueError):
            return default

    return {
        "id": bt_id,
        "strategy_name": name,
        "params": params,
        "ticker": run_ticker,
        "start_date": started_at,
        "end_date": completed_at,
        "initial_capital": _f(initial_capital, 100000.0),
        "final_value": _f(final_value),
        "total_return": _f(total_return),
        "annualized_return": _f(annualized_return),
        "sharpe_ratio": _f(sharpe_ratio),
        "max_drawdown": _f(max_drawdown),
        "win_rate": _f(win_rate),
        "total_trades": int(total_trades or 0),
        "avg_trade_return": _f(avg_trade_return),
        "volatility": _f(volatility),
        "status": metrics.get("status", "completed") if isinstance(metrics, dict) else "completed",
        "error": metrics.get("error") if isinstance(metrics, dict) else None,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "execution_engine": execution_summary.get("engine") if isinstance(execution_summary, dict) else None,
        "signals_emitted": execution_summary.get("signals_emitted", 0) if isinstance(execution_summary, dict) else 0,
        "order_intents": execution_summary.get("order_intents", 0) if isinstance(execution_summary, dict) else 0,
        "order_fills": execution_summary.get("order_fills", 0) if isinstance(execution_summary, dict) else 0,
        "timestamp": completed_at
        if completed_at
        else started_at
        if started_at
        else datetime.utcnow().isoformat(),
        "chart_data": chart_data,
    }


@router.get("/trading/backtest", response_model=List[Dict[str, Any]], tags=["Backtests"])
async def list_backtests(
    page: int = Query(1, ge=1, description="Page number (ignored when backtest_id is set)"),
    limit: int = Query(5, ge=1, le=50, description="Items per page (ignored when backtest_id is set)"),
    backtest_id: Optional[str] = Query(
        None,
        description="If set, return at most one run matching numeric id or metrics.backtest_id (same shape as list items).",
    ),
):
    """List recent backtests with pagination, or fetch one run by id / client backtest id."""
    from backend.main import app_state  # Import here to avoid circular imports

    id_filter = backtest_id is not None and str(backtest_id).strip() != ""

    try:
        conn = sqlite3.connect(app_state["database_path"])
        try:
            cur = conn.cursor()

            if id_filter:
                bid = str(backtest_id).strip()
                cur.execute(
                    """
                    SELECT id, name, params, started_at, completed_at, initial_capital, final_value,
                           total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
                           avg_trade_return, volatility, metrics, equity_curve
                    FROM backtest_runs
                    WHERE CAST(id AS TEXT) = ?
                       OR json_extract(metrics, '$.backtest_id') = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (bid, bid),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Backtest not found")
                return [_format_backtest_list_item(*row)]

            offset = (page - 1) * limit
            cur.execute(
                """
                SELECT id, name, params, started_at, completed_at, initial_capital, final_value,
                       total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
                       avg_trade_return, volatility, metrics, equity_curve
                FROM backtest_runs
                ORDER BY completed_at IS NULL DESC, datetime(completed_at) DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
            return [_format_backtest_list_item(*row) for row in rows]
        finally:
            conn.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list backtests: {str(e)}")
        if id_filter:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return []
