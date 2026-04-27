"""
Bridge forecasts into backtest-consumable signal primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from backend.ml.forecasting.contracts import ForecastResult


@dataclass
class BacktestBridge:
    threshold: float = 0.0

    def to_signal(self, forecast: ForecastResult) -> Dict[str, float]:
        targets = np.asarray(forecast.predicted_targets, dtype=float)
        mean_ret = float(np.mean(targets)) if targets.size else 0.0
        terminal_ret = float(targets[-1]) if targets.size else 0.0
        slope = float(np.polyfit(range(len(targets)), targets, 1)[0]) if len(targets) > 1 else 0.0
        long_signal = 1.0 if (mean_ret > self.threshold and terminal_ret > 0.0) else 0.0
        return {
            "h1_prediction": float(targets[0]) if targets.size else 0.0,
            "mean_prediction": mean_ret,
            "terminal_prediction": terminal_ret,
            "forecast_slope": slope,
            "long_signal": long_signal,
        }
