"""
Prediction service for realtime multi-horizon inference.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd

from backend.logging_config import get_component_logger
from backend.ml.contracts import HORIZONS, ModelMetadata, PredictionIntervals, PredictionResult
from backend.ml.feature_pipeline import FeatureInput, FeaturePipeline
from backend.ml.forecasting import FeatureBuilder, Preprocessor, RecursiveForecaster, TargetMode, RecursionMode
from backend.ml.forecasting.model_adapter import ModelAdapter
from backend.ml.forecasting.datasource import DataSource


logger = get_component_logger(__file__)


class PredictionService:
    """Service to run deterministic realtime inference and persistence."""

    def __init__(self, database_path: str, models_loaded: Dict[str, Any]):
        self.database_path = database_path
        self.models_loaded = models_loaded
        self.pipeline = FeaturePipeline()

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
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        with sqlite3.connect(self.database_path) as conn:
            vector = self.pipeline.build_vector(conn, FeatureInput(ticker=ticker.upper(), as_of=now))
            predicted_return = float(model.predict(vector)[0])
            confidence = max(0.1, min(0.95, 1.0 - abs(predicted_return) * 2.0))
            band = abs(predicted_return) * max(0.15, (1.0 - confidence))
            horizon_steps = int(horizon.replace("d", ""))
            path_targets = [predicted_return]
            path_prices = []
            try:
                ds = DataSource(self.database_path)
                hist = ds.load_ohlcv(ticker=ticker.upper(), end_date=now.date().isoformat())
                if not hist.empty and horizon_steps > 1:
                    fb = FeatureBuilder()
                    pre = Preprocessor(use_scaler=False)
                    fitted = fb.build(hist).dropna(subset=fb.feature_columns)
                    if not fitted.empty:
                        pre.fit(fitted[fb.feature_columns])
                        adapter = ModelAdapter(name=model_key, model=model)
                        forecaster = RecursiveForecaster(
                            model=adapter,
                            preprocessor=pre,
                            feature_builder=fb,
                            target_mode=TargetMode.log_return_1,
                            recursion_mode=RecursionMode.strict_recursive,
                        )
                        fc = forecaster.forecast(hist, horizon=horizon_steps, model_version=model_key)
                        path_targets = fc.predicted_targets
                        path_prices = fc.predicted_prices
                        predicted_return = float(path_targets[-1])
                        confidence = max(0.1, min(0.95, 1.0 - abs(float(pd.Series(path_targets).std()))))
                        band = abs(predicted_return) * max(0.15, (1.0 - confidence))
            except Exception as exc:
                logger.warning("Recursive path generation failed: %s", exc)

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
