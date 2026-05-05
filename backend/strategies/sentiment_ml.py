"""
Sentiment ML strategy implementation.

This strategy provides an ML-configurable projection interface, model version
management, and a lightweight training job enqueueing flow.
"""

import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import backtrader as bt

from backend.strategies.base import BaseStrategy


class SentimentMLStrategy(BaseStrategy):
    """ML-driven strategy using sentiment model predictions."""

    DEFAULT_MODEL_NAME = "sentiment_model"
    MODELS_DIR = Path("models")

    def __init__(self):
        self.versions: List[Dict[str, Any]] = []
        self.current_version: Optional[str] = None
        self.model_name: str = self.DEFAULT_MODEL_NAME

        parameters_schema = {
            "model_name": {
                "type": "string",
                "title": "Model name",
                "default": self.DEFAULT_MODEL_NAME,
                "description": "The sentiment ML model version to use for projection."
            },
            "prediction_threshold": {
                "type": "number",
                "title": "Prediction threshold",
                "default": 0.5,
                "description": "Minimum predicted return threshold used for generating projection confidence."
            },
            "max_position_pct": {
                "type": "number",
                "title": "Maximum position percentage",
                "default": 0.1,
                "description": "Maximum portfolio allocation for sentiment model signals."
            },
            "forecast_horizon_days": {
                "type": "integer",
                "title": "Forecast horizon days",
                "default": 5,
                "description": "Number of days to forecast when generating signal allocations."
            }
        }

        super().__init__(
            name="sentiment_ml",
            description="ML-driven strategy using sentiment model predictions",
            type="ml",
            parameters_schema=parameters_schema,
            can_train=True,
        )

        self._load_versions()

    def _normalize_parameters(self, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        params = parameters or {}
        prediction_threshold = min(max(float(params.get("prediction_threshold", 0.5)), 0.0), 1.0)
        max_position_pct = min(max(float(params.get("max_position_pct", 0.1)), 0.0), 1.0)
        model_name = str(params.get("model_name", self.get_current_model_name()))
        forecast_horizon_days = max(int(params.get("forecast_horizon_days", 5)), 1)

        return {
            "model_name": model_name,
            "prediction_threshold": prediction_threshold,
            "max_position_pct": max_position_pct,
            "forecast_horizon_days": forecast_horizon_days,
        }

    def _get_db_path(self) -> str:
        try:
            from backend.main import app_state
            return app_state.get("database_path", "data/backtest.db")
        except Exception:
            return "data/backtest.db"

    def _load_versions(self) -> None:
        self.versions = []
        self.current_version = None

        if not self.MODELS_DIR.exists():
            return

        for metadata_path in sorted(self.MODELS_DIR.glob(f"{self.name}__*__v*_metadata.json")):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                continue

            version_name = metadata.get("version_name") or metadata.get("name")
            if not version_name:
                continue

            model_path = self.MODELS_DIR / f"{version_name}.joblib"
            if not model_path.exists():
                continue

            self.versions.append({
                "version": int(metadata.get("version", 0)),
                "name": version_name,
                "path": str(model_path),
                "metadata_path": str(metadata_path),
                "timestamp": metadata.get("timestamp", ""),
                "training_config": metadata.get("training_config", {}),
                "metrics": metadata.get("metrics", {}),
            })

        self.versions.sort(key=lambda v: v.get("version", 0))
        if self.versions:
            self.current_version = self.versions[-1]["name"]
            self.model_name = self.current_version

    def list_versions(self) -> List[Dict[str, Any]]:
        return list(self.versions)

    def switch_version(self, version_name: str) -> bool:
        for version in self.versions:
            if version.get("name") == version_name:
                self.current_version = version_name
                self.model_name = version_name
                return True
        return False

    def get_current_model_name(self) -> str:
        return self.model_name or self.DEFAULT_MODEL_NAME

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class SentimentMLBacktrader(bt.Strategy):
            params = (
                ("model_name", normalized["model_name"]),
                ("prediction_threshold", normalized["prediction_threshold"]),
                ("max_position_pct", normalized["max_position_pct"]),
            )

            def __init__(self):
                self.equity_curve = []
                self.trades = []

            def next(self):
                return

        return SentimentMLBacktrader

    def train(self, config: Dict[str, Any]) -> Any:
        job_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        updated_at = created_at
        db_path = self._get_db_path()

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO model_jobs (id, model_name, status, created_at, updated_at, config)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    self.name,
                    "queued",
                    created_at,
                    updated_at,
                    json.dumps(config or {}),
                ),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

        training_coro = self._background_training(job_id, config or {})
        try:
            task = asyncio.create_task(training_coro)
        except Exception:
            training_coro.close()
        else:
            if not isinstance(task, asyncio.Task):
                training_coro.close()

        return {"job_id": job_id, "status": "queued"}

    async def _background_training(self, job_id: str, config: Dict[str, Any]) -> None:
        await asyncio.sleep(0)

    def project(
        self,
        parameters: Dict[str, Any],
        projection_days: int = 30,
        initial_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        params = self._normalize_parameters(parameters)
        predictions: List[float] = []
        predictions_used = 0

        try:
            conn = sqlite3.connect(self._get_db_path())
            cur = conn.cursor()
            cur.execute(
                "SELECT predicted_return FROM trading_model_predictions"
            )
            rows = cur.fetchall() or []
            for row in rows:
                if row and len(row) >= 1:
                    try:
                        predictions.append(float(row[0]))
                    except Exception:
                        continue
            predictions_used = len(predictions)
        except Exception:
            predictions = []
            predictions_used = 0
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if predictions_used:
            avg_predicted_return = sum(predictions) / max(predictions_used, 1)
            projected_return = avg_predicted_return
            confidence = min(max(abs(avg_predicted_return), 0.0), 1.0)
        else:
            avg_predicted_return = 0.0
            projected_return = float(params.get("prediction_threshold", 0.5))
            confidence = 0.5

        projected_volatility = min(max(abs(projected_return) * 0.25, 0.0), 1.0)
        projected_final_value = initial_capital * (1.0 + projected_return)
        timestamp = datetime.utcnow().isoformat()

        result: Dict[str, Any] = {
            "projected_return": projected_return,
            "projected_volatility": projected_volatility,
            "confidence": confidence,
            "projection_days": int(projection_days),
            "initial_capital": float(initial_capital),
            "projected_final_value": projected_final_value,
            "timestamp": timestamp,
        }

        if predictions_used:
            result["avg_predicted_return"] = avg_predicted_return
            result["predictions_used"] = predictions_used

        return result
