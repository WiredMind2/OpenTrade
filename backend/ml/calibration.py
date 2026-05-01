"""
Directional confidence calibration for LightGBM prediction models.

Maps a raw predicted log-return to a calibrated P(correct direction) using
logistic regression fitted on walk-forward backtest results.

A calibrated confidence of 0.65 means the model was right ~65% of the time
when it made predictions with that magnitude on held-out data.
Range is [0.5, 1.0]: 0.5 = pure coin-flip, 1.0 = perfect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


class DirectionalCalibrator:
    """Logistic regression: P(direction_correct) = sigmoid(a * |predicted_return| + b)."""

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0
        self.n_samples = 0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        y_pred: predicted log-returns (float array)
        y_true: actual log-returns (float array)
        Fits P(sign match) as a function of abs(predicted_return).
        """
        from sklearn.linear_model import LogisticRegression

        correct = (np.sign(y_pred) == np.sign(y_true)).astype(int)
        # Feature: absolute predicted return (larger → should mean more confident)
        X = np.abs(y_pred).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X, correct)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.n_samples = int(len(y_pred))

    def confidence(self, predicted_return: float) -> float:
        """Return calibrated P(direction correct) in [0.5, 1.0]."""
        x = abs(predicted_return)
        p = 1.0 / (1.0 + np.exp(-(self.a * x + self.b)))
        # Clamp: a calibrated model should never claim below 50% on its own predictions
        return float(max(0.5, min(0.99, p)))

    def to_dict(self) -> Dict[str, float | int]:
        return {"a": self.a, "b": self.b, "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, d: dict) -> "DirectionalCalibrator":
        cal = cls()
        cal.a = float(d["a"])
        cal.b = float(d["b"])
        cal.n_samples = int(d.get("n_samples", 0))
        return cal

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DirectionalCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))
