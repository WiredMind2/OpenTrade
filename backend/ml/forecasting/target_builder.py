"""
Target generation for one-step-ahead forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.ml.forecasting.contracts import TargetMode


class TargetBuilder:
    def __init__(self, mode: TargetMode = TargetMode.log_return_1):
        self.mode = mode

    def build(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].astype(float)
        next_close = close.shift(-1)
        if self.mode == TargetMode.log_return_1:
            return np.log(next_close / close)
        if self.mode == TargetMode.return_1:
            return (next_close - close) / close
        if self.mode == TargetMode.delta_1:
            return next_close - close
        if self.mode == TargetMode.price_1:
            return next_close
        if self.mode == TargetMode.residual_return_1:
            ret = (next_close - close) / close
            baseline = ret.rolling(20, min_periods=5).mean()
            return ret - baseline
        raise ValueError(f"Unsupported target mode: {self.mode}")
