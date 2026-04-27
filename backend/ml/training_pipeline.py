"""
Training pipeline utilities for multi-horizon models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from backend.ml.feature_pipeline import DEFAULT_FEATURE_NAMES


@dataclass
class TrainedArtifact:
    model_path: str
    metadata_path: str
    metrics: Dict[str, float]


def train_horizon_model(
    df: pd.DataFrame,
    horizon: str,
    outdir: str,
    model_prefix: str = "lightgbm",
) -> TrainedArtifact:
    os.makedirs(outdir, exist_ok=True)
    X = df[DEFAULT_FEATURE_NAMES].fillna(0.0).values
    y = df[f"label_{horizon}"].astype(float).values
    split = max(1, int(len(df) * 0.8))
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]
    if len(y_valid) == 0:
        X_valid, y_valid = X_train, y_train

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)],
    )
    preds = np.asarray(model.predict(X_valid)).ravel()
    rmse = float(mean_squared_error(y_valid, preds) ** 0.5)
    mae = float(mean_absolute_error(y_valid, preds))

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    version = f"{model_prefix}_{horizon}_{timestamp}"
    model_path = os.path.join(outdir, f"{version}.joblib")
    metadata_path = os.path.join(outdir, f"{version}.metadata.json")
    joblib.dump(
        {
            "lgbm": model,
            "feature_names": list(DEFAULT_FEATURE_NAMES),
            "feature_schema_version": "ml_features_v1",
            "horizon": horizon,
            "trained_at": datetime.utcnow().isoformat(),
        },
        model_path,
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": version,
                "model_version": version,
                "horizon": horizon,
                "feature_schema_version": "ml_features_v1",
                "metrics": {"rmse": rmse, "mae": mae},
                "trained_at": datetime.utcnow().isoformat(),
            },
            f,
            indent=2,
        )
    return TrainedArtifact(model_path=model_path, metadata_path=metadata_path, metrics={"rmse": rmse, "mae": mae})


def ensure_training_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize input dataframe to expected feature/label column names.
    """
    normalized = df.copy()
    rename_map = {}
    for name in DEFAULT_FEATURE_NAMES:
        if name not in normalized.columns:
            normalized[name] = 0.0
            rename_map[name] = "generated_default"
    for horizon in ("1d", "3d", "7d"):
        col = f"label_{horizon}"
        if col not in normalized.columns:
            if "label" in normalized.columns and horizon == "1d":
                normalized[col] = normalized["label"]
            else:
                normalized[col] = 0.0
            rename_map[col] = "generated_default"
    return normalized, rename_map
