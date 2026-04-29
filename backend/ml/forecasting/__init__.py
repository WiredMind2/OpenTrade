"""
Recursive multi-step forecasting framework.
"""

from backend.ml.forecasting.contracts import (
    ForecastConfig,
    ForecastResult,
    HorizonMetrics,
    RecursionMode,
    TargetMode,
)
from backend.ml.forecasting.datasource import DataSource
from backend.ml.forecasting.feature_builder import FeatureBuilder
from backend.ml.forecasting.target_builder import TargetBuilder
from backend.ml.forecasting.splitter import WalkForwardSplitter, WalkForwardSplit
from backend.ml.forecasting.preprocessor import Preprocessor
from backend.ml.forecasting.model_adapter import ModelAdapter, build_model_adapter
from backend.ml.forecasting.recursive_forecaster import RecursiveForecaster
from backend.ml.forecasting.evaluator import Evaluator
from backend.ml.forecasting.backtest_bridge import BacktestBridge
from backend.ml.forecasting.xgb_intraday_strategy import (
    XGBIntradayConfig,
    XGBIntradayStrategyRunner,
)

__all__ = [
    "BacktestBridge",
    "DataSource",
    "Evaluator",
    "FeatureBuilder",
    "ForecastConfig",
    "ForecastResult",
    "HorizonMetrics",
    "ModelAdapter",
    "Preprocessor",
    "RecursionMode",
    "RecursiveForecaster",
    "TargetBuilder",
    "TargetMode",
    "WalkForwardSplit",
    "WalkForwardSplitter",
    "XGBIntradayConfig",
    "XGBIntradayStrategyRunner",
    "build_model_adapter",
]
