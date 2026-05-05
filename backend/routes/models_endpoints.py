"""
Model information endpoints for the Trading Backtester API.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import sys
import os
import uuid
import sqlite3
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from backend.schemas import ModelSummary, ModelPredictRequest, ModelPredictResponse
from backend.schemas.models import (
    RetrainRequest,
    RetrainResponse,
    SavedModelCreateRequest,
    SavedModelUpdateRequest,
    SavedModelEvaluateRequest,
    SavedModelBatchEvaluateRequest,
    SavedModelSignalsBatchRequest,
    SavedModelSignalResponse,
    SavedModelResponse,
    SavedModelEvaluationResponse,
)
from backend.domain.trading import TargetAllocation
from backend.utils.backtest_variants import compute_params_hash
from backend.routes.backtest_engine import evaluate_strategy_runtime_once
from backend.services.strategy_framework import StrategyOptimizerEngine
from backend.logging_config import get_component_logger


router = APIRouter()
logger = get_component_logger(__file__)


def _get_app_state() -> dict:
    """Return active app_state, preferring top-level shim in tests."""
    from backend.main import app_state as backend_app_state
    try:
        from main import app_state as shim_app_state  # type: ignore
        if isinstance(shim_app_state, dict):
            return shim_app_state
    except Exception:
        pass
    return backend_app_state


@router.get("/models", response_model=List[ModelSummary], tags=["Models"])
async def list_models():
    """List available models."""
    app_state = _get_app_state()
    registry = app_state.get("model_registry")
    models = []

    if registry is None:
        return models

    for model in registry.list():
        config_schema = model.get_config_schema()
        models.append(ModelSummary(
            name=model.name,
            type=model.type,
            version=model.version,
            description=model.description,
            capabilities=model.capabilities,
            config_schema=config_schema.model_json_schema()
        ))

    return models


@router.post("/models/{name}/predict", response_model=ModelPredictResponse, tags=["Models"])
async def predict_with_model(name: str, request: ModelPredictRequest):
    """Generate predictions using a specific model."""
    from fastapi import HTTPException
    app_state = _get_app_state()

    registry = app_state["model_registry"]
    model = registry.get(name)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    try:
        result = model.predict(request.inputs, request.config)
        return ModelPredictResponse(predictions=result["predictions"], meta=result["meta"])
    except KeyError as e:
        logger.error(f"Model prediction result structure error. Expected 'predictions' and 'meta' keys, got: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        logger.error(f"Full result: {result}")
        raise HTTPException(status_code=500, detail=f"Prediction result format error: missing key {e}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def require_admin():
    return True


def _get_db_connection():
    """Get database connection from app state."""
    app_state = _get_app_state()
    db_path = app_state.get("database_path", "data/backtest.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_jobs (
            id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            config TEXT,
            result TEXT,
            error TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            params_json TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            objective TEXT NOT NULL DEFAULT 'balanced',
            baseline_metrics_json TEXT,
            latest_metrics_json TEXT,
            latest_equity_curve_json TEXT,
            degrade_status TEXT NOT NULL DEFAULT 'healthy',
            degrade_reason TEXT,
            last_evaluated_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_saved_models_ticker_active ON saved_models(ticker, is_active)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_saved_models_strategy_ticker ON saved_models(strategy_name, ticker)")
    conn.commit()
    return conn


def _score_from_metrics(metrics: Dict[str, Any], objective: str) -> float:
    return float(StrategyOptimizerEngine.score(metrics or {}, objective or "balanced"))


def _compute_degrade_status(
    baseline_metrics: Optional[Dict[str, Any]],
    latest_metrics: Optional[Dict[str, Any]],
    drift_thresholds: Dict[str, float],
) -> tuple[str, Optional[str]]:
    if not baseline_metrics or not latest_metrics:
        return "healthy", None

    return_drop = float(drift_thresholds.get("return_drop", 0.05))
    sharpe_drop = float(drift_thresholds.get("sharpe_drop", 0.4))
    drawdown_increase = float(drift_thresholds.get("drawdown_increase", 0.05))

    b_ret = float(baseline_metrics.get("total_return", 0.0) or 0.0)
    l_ret = float(latest_metrics.get("total_return", 0.0) or 0.0)
    b_sh = float(baseline_metrics.get("sharpe_ratio", 0.0) or 0.0)
    l_sh = float(latest_metrics.get("sharpe_ratio", 0.0) or 0.0)
    b_dd = float(baseline_metrics.get("max_drawdown", 0.0) or 0.0)
    l_dd = float(latest_metrics.get("max_drawdown", 0.0) or 0.0)

    reasons: List[str] = []
    if l_ret < b_ret - return_drop:
        reasons.append("return_drop")
    if l_sh < b_sh - sharpe_drop:
        reasons.append("sharpe_drop")
    if l_dd > b_dd + drawdown_increase:
        reasons.append("drawdown_increase")

    if len(reasons) >= 2:
        return "degraded", ", ".join(reasons)
    if len(reasons) == 1:
        return "watch", reasons[0]
    return "healthy", None


def _saved_model_from_row(row: sqlite3.Row) -> SavedModelResponse:
    return SavedModelResponse(
        id=int(row["id"]),
        name=str(row["name"]),
        strategy_name=str(row["strategy_name"]),
        ticker=str(row["ticker"]),
        params=json.loads(row["params_json"] or "{}"),
        params_hash=str(row["params_hash"]),
        objective=str(row["objective"]),
        baseline_metrics=json.loads(row["baseline_metrics_json"]) if row["baseline_metrics_json"] else None,
        latest_metrics=json.loads(row["latest_metrics_json"]) if row["latest_metrics_json"] else None,
        latest_equity_curve=json.loads(row["latest_equity_curve_json"]) if row["latest_equity_curve_json"] else None,
        degrade_status=str(row["degrade_status"]),
        degrade_reason=row["degrade_reason"],
        last_evaluated_at=row["last_evaluated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        is_active=bool(row["is_active"]),
    )


def _store_job(job_id: str, model_name: str, status: str, config: dict = None, result: dict = None, error: str = None):
    """Store job in database."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO model_jobs (id, model_name, status, config, result, error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, model_name, status, str(config) if config else None,
              str(result) if result else None, error))
        conn.commit()
    finally:
        conn.close()


def _update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    """Update job status in database."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE model_jobs
            SET status = ?, updated_at = datetime('now'),
                result = ?, error = ?
            WHERE id = ?
        """, (status, str(result) if result else None, error, job_id))
        conn.commit()
    finally:
        conn.close()


async def _run_retrain_background(job_id: str, model_name: str, training_payload: dict, config: dict, options: dict):
    """Background task to run model retraining."""
    try:
        app_state = _get_app_state()

        # Update status to running
        _update_job_status(job_id, "running")

        # Get model
        registry = app_state["model_registry"]
        model = registry.get(model_name)

        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        # Run retrain
        result = model.retrain(training_payload, config, background=True)

        # Update status to completed
        _update_job_status(job_id, "completed", result=result)

    except Exception as e:
        # Update status to failed
        _update_job_status(job_id, "failed", error=str(e))


@router.post("/models/{name}/retrain", response_model=RetrainResponse, tags=["Models"])
async def retrain_model(name: str, request: RetrainRequest, background_tasks: BackgroundTasks):
    """Retrain a model with new data."""
    # Check admin access (placeholder)
    if not require_admin():
        raise HTTPException(status_code=403, detail="Admin access required")

    app_state = _get_app_state()

    # Get model
    registry = app_state["model_registry"]
    model = registry.get(name)

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    # Check if retrain is supported
    if "retrain" not in model.capabilities:
        raise HTTPException(status_code=400, detail=f"Model '{name}' does not support retraining")

    # Check if background execution is requested
    background = request.options.get("background", False)

    if background:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Store job as queued
        _store_job(job_id, name, "queued", config=request.config)

        # Add background task
        background_tasks.add_task(
            _run_retrain_background,
            job_id,
            name,
            request.training_payload,
            request.config,
            request.options
        )

        return RetrainResponse(job_id=job_id, status="queued")

    else:
        # Run retrain synchronously
        try:
            result = model.retrain(request.training_payload, request.config, background=False)
            return RetrainResponse(status="completed", model_meta=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@router.get("/jobs/{job_id}", tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get the status of a retraining job."""
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
            "id": row[0],
            "model_name": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "config": row[5],
            "result": row[6],
            "error": row[7]
        }

        return job_data

    finally:
        conn.close()


@router.post("/models/saved", response_model=SavedModelResponse, tags=["Saved Models"])
async def create_saved_model(request: SavedModelCreateRequest):
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        params = dict(request.params or {})
        params_hash = compute_params_hash(params)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO saved_models (
                name, strategy_name, ticker, params_json, params_hash, objective, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request.name.strip(),
                request.strategy_name.strip(),
                request.ticker.strip().upper(),
                json.dumps(params),
                params_hash,
                request.objective.strip().lower(),
                1 if request.is_active else 0,
            ),
        )
        conn.commit()
        new_id = int(cur.lastrowid)
        cur.execute("SELECT * FROM saved_models WHERE id = ?", (new_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=500, detail="Saved model insert failed")
        return _saved_model_from_row(row)
    finally:
        conn.close()


@router.get("/models/saved", response_model=List[SavedModelResponse], tags=["Saved Models"])
async def list_saved_models(
    ticker: Optional[str] = Query(None),
    strategy: Optional[str] = Query(None),
    active: Optional[bool] = Query(None),
    objective: Optional[str] = Query(None),
):
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        conditions: List[str] = []
        params: List[Any] = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.strip().upper())
        if strategy:
            conditions.append("strategy_name = ?")
            params.append(strategy.strip())
        if active is not None:
            conditions.append("is_active = ?")
            params.append(1 if active else 0)
        if objective:
            conditions.append("objective = ?")
            params.append(objective.strip().lower())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM saved_models
            {where_clause}
            ORDER BY updated_at DESC, id DESC
        """
        rows = conn.execute(query, params).fetchall()
        return [_saved_model_from_row(row) for row in rows]
    finally:
        conn.close()


@router.patch("/models/saved/{model_id}", response_model=SavedModelResponse, tags=["Saved Models"])
async def update_saved_model(model_id: int, request: SavedModelUpdateRequest):
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM saved_models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Saved model '{model_id}' not found")

        updates: List[str] = []
        params: List[Any] = []
        if request.name is not None:
            updates.append("name = ?")
            params.append(request.name.strip())
        if request.objective is not None:
            updates.append("objective = ?")
            params.append(request.objective.strip().lower())
        if request.is_active is not None:
            updates.append("is_active = ?")
            params.append(1 if request.is_active else 0)
        if request.params is not None:
            params_hash = compute_params_hash(request.params)
            updates.append("params_json = ?")
            params.append(json.dumps(request.params))
            updates.append("params_hash = ?")
            params.append(params_hash)

        if updates:
            updates.append("updated_at = datetime('now')")
            params.append(model_id)
            cur.execute(
                f"UPDATE saved_models SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        cur.execute("SELECT * FROM saved_models WHERE id = ?", (model_id,))
        updated = cur.fetchone()
        if updated is None:
            raise HTTPException(status_code=500, detail="Saved model update failed")
        return _saved_model_from_row(updated)
    finally:
        conn.close()


@router.delete("/models/saved/{model_id}", tags=["Saved Models"])
async def delete_saved_model(model_id: int):
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM saved_models WHERE id = ?", (model_id,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Saved model '{model_id}' not found")
        return {"deleted": True, "model_id": model_id}
    finally:
        conn.close()


async def _evaluate_saved_model_row(
    row: sqlite3.Row,
    request: SavedModelEvaluateRequest,
    app_state: Dict[str, Any],
) -> SavedModelEvaluationResponse:
    strategy_name = str(row["strategy_name"])
    ticker = str(row["ticker"]).upper()
    params = json.loads(row["params_json"] or "{}")
    objective = (request.objective or row["objective"] or "balanced").lower()
    start_date = datetime.fromisoformat(request.start_date)
    end_date = datetime.fromisoformat(request.end_date)

    result = await evaluate_strategy_runtime_once(
        strategy_name=strategy_name,
        ticker=ticker,
        parameters=params,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(request.initial_capital),
        app_state=app_state,
    )
    latest_metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    baseline_metrics = json.loads(row["baseline_metrics_json"]) if row["baseline_metrics_json"] else None
    degrade_status, degrade_reason = _compute_degrade_status(
        baseline_metrics=baseline_metrics,
        latest_metrics=latest_metrics,
        drift_thresholds=request.drift_thresholds or {},
    )
    if baseline_metrics is None and latest_metrics:
        baseline_metrics = latest_metrics

    return SavedModelEvaluationResponse(
        model_id=int(row["id"]),
        name=str(row["name"]),
        params=dict(params),
        status=str(result.get("status", "failed")),
        strategy_name=strategy_name,
        ticker=ticker,
        params_hash=str(row["params_hash"]),
        objective=objective,
        metrics=latest_metrics,
        equity_curve=result.get("equity_curve", []) or [],
        degrade_status=degrade_status,  # type: ignore[arg-type]
        degrade_reason=degrade_reason,
        evaluated_at=datetime.utcnow().isoformat(),
        error=result.get("error"),
    )


@router.post("/models/saved/{model_id}/evaluate", response_model=SavedModelEvaluationResponse, tags=["Saved Models"])
async def evaluate_saved_model(model_id: int, request: SavedModelEvaluateRequest):
    app_state = _get_app_state()
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM saved_models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Saved model '{model_id}' not found")

        evaluation = await _evaluate_saved_model_row(row, request, app_state)
        cur.execute(
            """
            UPDATE saved_models
            SET baseline_metrics_json = COALESCE(baseline_metrics_json, ?),
                latest_metrics_json = ?,
                latest_equity_curve_json = ?,
                degrade_status = ?,
                degrade_reason = ?,
                objective = ?,
                last_evaluated_at = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (
                json.dumps(evaluation.metrics),
                json.dumps(evaluation.metrics),
                json.dumps(evaluation.equity_curve),
                evaluation.degrade_status,
                evaluation.degrade_reason,
                evaluation.objective,
                evaluation.evaluated_at,
                model_id,
            ),
        )
        conn.commit()
        return evaluation
    finally:
        conn.close()


def _rows_not_excluded(candidates: List[sqlite3.Row], exclude: Set[int]) -> List[sqlite3.Row]:
    return [r for r in candidates if int(r["id"]) not in exclude]


def _stale_rank_select_rows(
    candidates: List[sqlite3.Row],
    *,
    objective: str,
    top_n: int,
    include: Set[int],
    exclude: Set[int],
) -> List[sqlite3.Row]:
    """Pinned includes are always kept; remaining slots are filled by descending stale score."""
    filtered = []
    for row in candidates:
        mid = int(row["id"])
        if mid in exclude:
            continue
        filtered.append(row)
    if not filtered:
        return []

    if include:
        pinned = [r for r in filtered if int(r["id"]) in include]
        pool = [r for r in filtered if int(r["id"]) not in include]
        ranked_pool: List[Tuple[float, sqlite3.Row]] = []
        for row in pool:
            metrics = json.loads(row["latest_metrics_json"]) if row["latest_metrics_json"] else {}
            ranked_pool.append((_score_from_metrics(metrics, objective), row))
        ranked_pool.sort(key=lambda x: (x[0], int(x[1]["id"])), reverse=True)
        rest_n = max(0, top_n - len(pinned))
        return pinned + [r for _, r in ranked_pool[:rest_n]]

    ranked: List[Tuple[float, sqlite3.Row]] = []
    for row in filtered:
        metrics = json.loads(row["latest_metrics_json"]) if row["latest_metrics_json"] else {}
        ranked.append((_score_from_metrics(metrics, objective), row))
    ranked.sort(key=lambda x: (x[0], int(x[1]["id"])), reverse=True)
    return [r for _, r in ranked[:top_n]]


def _fresh_rank_workset(
    candidates: List[sqlite3.Row],
    *,
    max_evaluate: int,
    include: Set[int],
    exclude: Set[int],
) -> List[sqlite3.Row]:
    """Rows to evaluate when ranking after fresh runs (cap by max_evaluate, always retain pins)."""
    base = _rows_not_excluded(candidates, exclude)
    if not base:
        return []
    if include:
        pinned = [r for r in base if int(r["id"]) in include]
        pool = [r for r in base if int(r["id"]) not in include]
        budget = max(0, max_evaluate - len(pinned))
        return pinned + pool[:budget]
    return base[:max_evaluate]


_REASON_EXIT_FRAGMENTS = ("bearish", "_sell", "exit", "sell_signal", "short", "overbought", "liquidat")
_REASON_ENTRY_FRAGMENTS = ("bullish", "_buy", "entry", "buy_signal", "long", "oversold")


def _allocation_to_user_action(
    alloc: Optional[TargetAllocation],
) -> Tuple[str, float, float, str]:
    """Map engine allocation to buy / sell / hold for the UI."""
    if alloc is None:
        return "hold", 0.0, 0.0, "no_change"

    tp = float(alloc.target_pct or 0.0)
    conf = float(alloc.confidence or 0.0)
    reason = str(alloc.reason or "")
    rl = reason.lower()

    if tp > 1e-6:
        return "buy", tp, conf, reason
    if tp < -1e-6:
        return "sell", tp, conf, reason

    if any(fragment in rl for fragment in _REASON_EXIT_FRAGMENTS):
        return "sell", tp, conf, reason
    if any(fragment in rl for fragment in _REASON_ENTRY_FRAGMENTS):
        return "buy", tp, conf, reason

    return "hold", tp, conf, reason


def _resolve_signal_anchor(
    conn: sqlite3.Connection,
    ticker: str,
    as_of_date: Optional[str],
) -> Tuple[str, float, datetime]:
    """Latest (or as-of) daily close for signal generation."""
    cur = conn.cursor()
    sym = ticker.strip().upper()
    if as_of_date and str(as_of_date).strip():
        cutoff = str(as_of_date).strip()[:10]
        cur.execute(
            """
            SELECT date, close
            FROM price_daily
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (sym, cutoff),
        )
    else:
        cur.execute(
            """
            SELECT date, close
            FROM price_daily
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (sym,),
        )
    row = cur.fetchone()
    if not row or row[1] is None:
        suffix = f" on or before {as_of_date}" if (as_of_date and str(as_of_date).strip()) else ""
        raise HTTPException(
            status_code=400,
            detail=f"No daily price bar found for {sym}{suffix}.",
        )
    date_iso = str(row[0])
    close = float(row[1])
    as_of_dt = datetime.strptime(date_iso, "%Y-%m-%d")
    return date_iso, close, as_of_dt


def _persist_evaluation(cur: sqlite3.Cursor, evaluation: SavedModelEvaluationResponse) -> None:
    cur.execute(
        """
        UPDATE saved_models
        SET baseline_metrics_json = COALESCE(baseline_metrics_json, ?),
            latest_metrics_json = ?,
            latest_equity_curve_json = ?,
            degrade_status = ?,
            degrade_reason = ?,
            objective = ?,
            last_evaluated_at = ?,
            updated_at = datetime('now')
        WHERE id = ?
        """,
        (
            json.dumps(evaluation.metrics),
            json.dumps(evaluation.metrics),
            json.dumps(evaluation.equity_curve),
            evaluation.degrade_status,
            evaluation.degrade_reason,
            evaluation.objective,
            evaluation.evaluated_at,
            evaluation.model_id,
        ),
    )


@router.post("/models/saved/evaluate-batch", response_model=List[SavedModelEvaluationResponse], tags=["Saved Models"])
async def evaluate_saved_models_batch(request: SavedModelBatchEvaluateRequest):
    app_state = _get_app_state()
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM saved_models
            WHERE ticker = ? AND is_active = 1
            ORDER BY updated_at DESC, id DESC
            """,
            (request.ticker.strip().upper(),),
        )
        candidates = list(cur.fetchall())
        if not candidates:
            return []

        include = {int(v) for v in request.include_model_ids or []}
        exclude = {int(v) for v in request.exclude_model_ids or []}

        single_req = SavedModelEvaluateRequest(
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            objective=request.objective,
            drift_thresholds=request.drift_thresholds,
        )
        objective = request.objective or "balanced"

        if request.rank_after_evaluation:
            workset = _fresh_rank_workset(
                candidates,
                max_evaluate=request.max_evaluate,
                include=include,
                exclude=exclude,
            )
        else:
            workset = _stale_rank_select_rows(
                candidates,
                objective=objective,
                top_n=request.top_n,
                include=include,
                exclude=exclude,
            )

        results: List[SavedModelEvaluationResponse] = []
        for row in workset:
            evaluation = await _evaluate_saved_model_row(row, single_req, app_state)
            _persist_evaluation(cur, evaluation)
            results.append(evaluation)

        if request.rank_after_evaluation:
            results.sort(
                key=lambda ev: (_score_from_metrics(ev.metrics, objective), ev.model_id),
                reverse=True,
            )
            results = results[: request.top_n]

        conn.commit()
        return results
    finally:
        conn.close()


@router.post("/models/saved/signals-batch", response_model=List[SavedModelSignalResponse], tags=["Saved Models"])
async def saved_models_signals_batch(request: SavedModelSignalsBatchRequest):
    """
    Rank saved models by last stored metrics, then run signal logic only at the latest daily bar
    (``generate_target_allocations`` → buy / sell / hold). No historical backtest pass.
    """
    app_state = _get_app_state()
    registry = app_state.get("strategy_registry")
    if not registry:
        raise HTTPException(status_code=500, detail="Strategy registry not available")

    ticker = request.ticker.strip().upper()
    conn = _get_db_connection()
    conn.row_factory = sqlite3.Row
    try:
        anchor_date_iso, last_price, as_of_dt = _resolve_signal_anchor(conn, ticker, request.as_of_date)
        prices: Dict[str, float] = {ticker: last_price}

        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM saved_models
            WHERE ticker = ? AND is_active = 1
            ORDER BY updated_at DESC, id DESC
            """,
            (ticker,),
        )
        candidates = list(cur.fetchall())
        if not candidates:
            return []

        include_set = {int(v) for v in request.include_model_ids or []}
        exclude_set = {int(v) for v in request.exclude_model_ids or []}
        objective = request.objective or "balanced"
        selected_rows = _stale_rank_select_rows(
            candidates,
            objective=objective,
            top_n=request.top_n,
            include=include_set,
            exclude=exclude_set,
        )

        results: List[SavedModelSignalResponse] = []
        for row in selected_rows:
            sid = int(row["id"])
            strat_name = str(row["strategy_name"])
            raw_params = json.loads(row["params_json"] or "{}")
            strategy_impl = registry.get(strat_name)
            action = "hold"
            tgt = 0.0
            conf = 0.0
            reason = "unknown_strategy"
            err: Optional[str] = None

            if strategy_impl is None:
                err = f"Strategy '{strat_name}' not found"
            else:
                try:
                    allocations = strategy_impl.generate_target_allocations(
                        raw_params,
                        [ticker],
                        as_of_dt,
                        prices,
                        db_conn=conn,
                    )
                    match = next(
                        (a for a in allocations if str(a.ticker).upper() == ticker),
                        None,
                    )
                    action, tgt, conf, reason = _allocation_to_user_action(match)
                except Exception as ex:
                    err = str(ex)
                    action, tgt, conf, reason = "hold", 0.0, 0.0, "signal_error"

            results.append(
                SavedModelSignalResponse(
                    model_id=sid,
                    name=str(row["name"]),
                    strategy_name=strat_name,
                    ticker=ticker,
                    params=dict(raw_params),
                    params_hash=str(row["params_hash"]),
                    as_of=anchor_date_iso,
                    last_price=last_price,
                    action=action,  # type: ignore[arg-type]
                    target_pct=tgt,
                    confidence=conf,
                    reason=reason,
                    degrade_status=str(row["degrade_status"]),  # type: ignore[arg-type]
                    degrade_reason=row["degrade_reason"],
                    error=err,
                )
            )
        return results
    finally:
        conn.close()