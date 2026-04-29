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
    if key in {"xgboost", "xgb"}:
        from xgboost import XGBRegressor

        defaults = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "booster": "gbtree",
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "max_bin": 256,
            "random_state": 42,
            "n_jobs": -1,
        }
        return ModelAdapter("xgboost", XGBRegressor(**(defaults | p)))
    raise ValueError(f"Unsupported model: {name}")
