"""
Contracts for recursive forecasting stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TargetMode(str, Enum):
    log_return_1 = "log_return_1"
    return_1 = "return_1"
    delta_1 = "delta_1"
    price_1 = "price_1"
    residual_return_1 = "residual_return_1"


class RecursionMode(str, Enum):
    strict_recursive = "strict_recursive"
    semi_recursive = "semi_recursive"


@dataclass
class ForecastConfig:
    target_mode: TargetMode = TargetMode.log_return_1
    recursion_mode: RecursionMode = RecursionMode.strict_recursive
    horizon: int = 5
    model_name: str = "ridge"
    min_train_size: int = 500
    test_size: int = 20
    step_size: int = 20
    gap: int = 5
    retrain_cadence: int = 20
    threshold: float = 0.0


@dataclass
class ForecastResult:
    origin_time: datetime
    horizon: int
    target_mode: str
    predicted_targets: List[float]
    predicted_prices: List[float]
    feature_snapshot: Dict[str, float]
    model_version: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HorizonMetrics:
    horizon: int
    rmse: float
    mae: float
    directional_accuracy: float
    correlation: float
    r2_oos: float
    mape: Optional[float] = None
