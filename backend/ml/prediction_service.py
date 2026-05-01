"""
Prediction service for realtime multi-horizon inference.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict

import joblib
from pathlib import Path

from backend.logging_config import get_component_logger
from backend.ml.calibration import DirectionalCalibrator
from backend.ml.contracts import HORIZONS, ModelMetadata, PredictionIntervals, PredictionResult
from backend.ml.feature_pipeline import FeatureInput, FeaturePipeline


logger = get_component_logger(__file__)

_MODEL_DIR = Path("models")


class PredictionService:
    """Service to run deterministic realtime inference and persistence."""

    def __init__(self, database_path: str, models_loaded: Dict[str, Any]):
        self.database_path = database_path
        self.models_loaded = models_loaded
        self.pipeline = FeaturePipeline()
        self.calibrators = self._load_calibrators()

    def _load_calibrators(self) -> Dict[str, DirectionalCalibrator]:
        calibrators: Dict[str, DirectionalCalibrator] = {}
        for horizon in ("1d", "3d", "7d"):
            path = _MODEL_DIR / f"lightgbm_{horizon}.calibration.json"
            if path.exists():
                calibrators[horizon] = DirectionalCalibrator.load(path)
                logger.info("Loaded calibrator for %s (n=%d)", horizon, calibrators[horizon].n_samples)
            else:
                logger.warning("No calibration file for %s — using heuristic confidence", horizon)
        return calibrators

    def _calibrated_confidence(self, horizon: str, predicted_return: float) -> float:
        cal = self.calibrators.get(horizon)
        if cal is not None:
            return cal.confidence(predicted_return)
        # Fallback heuristic (pre-calibration behaviour)
        return max(0.1, min(0.95, 1.0 - abs(predicted_return) * 2.0))

    def _resolve_model_key(self, horizon: str) -> str:
        if horizon not in HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}")
        canonical = f"lightgbm_{horizon}"
        if canonical in self.models_loaded:
            return canonical
        for key in self.models_loaded.keys():
            if key.startswith(canonical):
                return key
        raise KeyError(f"No model available for horizon {horizon}")

    def predict(self, ticker: str, horizon: str) -> PredictionResult:
        model_key = self._resolve_model_key(horizon)
        model_data = self.models_loaded[model_key]
        model = model_data.get("lgbm", model_data)
        now = datetime.utcnow()
        horizon_steps = int(horizon.replace("d", ""))

        with sqlite3.connect(self.database_path) as conn:
            # Build feature vector once with the production pipeline (14 features)
            vector = self.pipeline.build_vector(conn, FeatureInput(ticker=ticker.upper(), as_of=now))
            predicted_return = float(model.predict(vector)[0])
            confidence = self._calibrated_confidence(horizon, predicted_return)
            band = abs(predicted_return) * max(0.15, (1.0 - confidence))

            # Build a price path by linear interpolation toward the predicted final price.
            # This is for display only — each intermediate step is not an independent prediction.
            current_price = self._latest_close(conn, ticker)
            path_prices, path_targets = self._build_price_path(current_price, predicted_return, horizon_steps)

            metadata = {
                "request_id": f"req_{now.timestamp()}",
                "model_key": model_key,
                "prediction_latency_ms": 0,
                "predicted_path_targets": path_targets,
                "predicted_path_prices": path_prices,
            }
            result = PredictionResult(
                ticker=ticker.upper(),
                horizon=horizon,  # type: ignore[arg-type]
                predicted_return=predicted_return,
                confidence=confidence,
                model=ModelMetadata(
                    model_name=model_key,
                    model_version=model_key,
                    horizon=horizon,  # type: ignore[arg-type]
                    feature_schema_version=self.pipeline.schema_version,
                ),
                features_used=list(self.pipeline.feature_names),
                intervals=PredictionIntervals(
                    lower=predicted_return - band,
                    upper=predicted_return + band,
                ),
                metadata=metadata,
            )
            self._persist_prediction(conn, result)
            return result

    def _latest_close(self, conn: sqlite3.Connection, ticker: str) -> float | None:
        cur = conn.cursor()
        cur.execute(
            "SELECT close FROM price_daily WHERE ticker = ? ORDER BY date DESC LIMIT 1",
            (ticker.upper(),),
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] else None

    def _build_price_path(
        self,
        current_price: float | None,
        predicted_return: float,
        horizon_steps: int,
    ) -> tuple[list[float], list[float]]:
        """Linear interpolation of price and return from now to the predicted final value."""
        if current_price is None or current_price == 0:
            return [], [predicted_return]
        final_price = current_price * (1.0 + predicted_return)
        path_prices = [
            current_price + (final_price - current_price) * (i + 1) / horizon_steps
            for i in range(horizon_steps)
        ]
        path_targets = [
            (p - current_price) / current_price for p in path_prices
        ]
        return path_prices, path_targets

    def _persist_prediction(self, conn: sqlite3.Connection, result: PredictionResult) -> None:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info('sentiment_predictions')")
        cols = [r[1] for r in cur.fetchall()]
        confidence_column = "predicted_confidence" if "predicted_confidence" in cols else "confidence"
        insert_columns = [
            "ticker",
            "horizon",
            "predicted_return",
            confidence_column,
            "produced_at",
            "model",
            "features_used",
            "metadata",
        ]
        values = [
            result.ticker,
            result.horizon,
            result.predicted_return,
            result.confidence,
            result.timestamp.isoformat(),
            result.model.model_name,
            ",".join(result.features_used),
            json.dumps(
                {
                    **result.metadata,
                    "model_version": result.model.model_version,
                    "feature_schema_version": result.model.feature_schema_version,
                    "intervals": result.intervals.model_dump() if result.intervals else None,
                }
            ),
        ]
        if "model_version" in cols:
            insert_columns.append("model_version")
            values.append(result.model.model_version)
        if "feature_schema_version" in cols:
            insert_columns.append("feature_schema_version")
            values.append(result.model.feature_schema_version)
        if "prediction_latency_ms" in cols:
            insert_columns.append("prediction_latency_ms")
            values.append(float(result.metadata.get("prediction_latency_ms", 0)))

        placeholders = ",".join(["?"] * len(insert_columns))
        cur.execute(
            f"INSERT INTO sentiment_predictions ({','.join(insert_columns)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    @staticmethod
    def load_bundle(bundle_path: str) -> Dict[str, Any]:
        return joblib.load(bundle_path)
