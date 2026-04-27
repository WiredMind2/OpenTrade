"""
Pluggable model adapters for one-step regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge


@dataclass
class ModelAdapter:
    name: str
    model: Any

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x)).ravel()


def build_model_adapter(name: str, params: Dict[str, Any] | None = None) -> ModelAdapter:
    p = params or {}
    key = name.lower()
    if key == "linear":
        return ModelAdapter("linear", LinearRegression(**p))
    if key == "ridge":
        return ModelAdapter("ridge", Ridge(**({"alpha": 1.0} | p)))
    if key == "lasso":
        return ModelAdapter("lasso", Lasso(**({"alpha": 0.001} | p)))
    if key == "elastic_net":
        return ModelAdapter("elastic_net", ElasticNet(**({"alpha": 0.001, "l1_ratio": 0.5} | p)))
    if key == "random_forest":
        return ModelAdapter("random_forest", RandomForestRegressor(**({"n_estimators": 200, "random_state": 42} | p)))
    if key in {"lightgbm", "lgbm"}:
        import lightgbm as lgb

        return ModelAdapter("lightgbm", lgb.LGBMRegressor(**({"n_estimators": 300} | p)))
    raise ValueError(f"Unsupported model: {name}")
