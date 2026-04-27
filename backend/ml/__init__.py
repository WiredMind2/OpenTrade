"""
Shared ML utilities for prediction training and inference.
"""

from backend.ml.contracts import (
    HORIZONS,
    FeatureContract,
    ModelMetadata,
    PredictionIntervals,
    PredictionResult,
)
from backend.ml.feature_pipeline import FeaturePipeline
from backend.ml.prediction_service import PredictionService
from backend.ml.forecasting import (
    ForecastConfig,
    ForecastResult,
    BacktestBridge,
    WalkForwardSplitter,
    RecursiveForecaster,
)

__all__ = [
    "HORIZONS",
    "FeatureContract",
    "FeaturePipeline",
    "ModelMetadata",
    "PredictionIntervals",
    "PredictionResult",
    "PredictionService",
    "ForecastConfig",
    "ForecastResult",
    "BacktestBridge",
    "WalkForwardSplitter",
    "RecursiveForecaster",
]
