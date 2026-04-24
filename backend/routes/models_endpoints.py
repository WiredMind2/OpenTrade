"""
Model information endpoints for the Trading Backtester API.
"""
from datetime import datetime
from typing import List
import sys
import os
import uuid
import sqlite3
from fastapi import APIRouter, HTTPException, BackgroundTasks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from backend.schemas import ModelInfo, ModelSummary, ModelPredictRequest, ModelPredictResponse
from backend.schemas.models import RetrainRequest, RetrainResponse


router = APIRouter()


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
        # Log the actual result structure for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Model prediction result structure error. Expected 'predictions' and 'meta' keys, got: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        logger.error(f"Full result: {result}")
        raise HTTPException(status_code=500, detail=f"Prediction result format error: missing key {e}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def require_admin():
    """Placeholder admin authentication check."""
    # TODO: Implement proper authentication
    # For now, assume admin access
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
    conn.commit()
    return conn


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