"""
Train-only preprocessing transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    use_scaler: bool = True
    winsor_clip: Optional[float] = None

    def __post_init__(self):
        self.scaler = StandardScaler() if self.use_scaler else None
        self._fitted = False

    def fit(self, x_train: pd.DataFrame) -> None:
        arr = x_train.values.astype(float)
        if self.winsor_clip is not None:
            arr = np.clip(arr, -self.winsor_clip, self.winsor_clip)
        if self.scaler is not None:
            self.scaler.fit(arr)
        self._fitted = True

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit on training data first.")
        arr = x.values.astype(float)
        if self.winsor_clip is not None:
            arr = np.clip(arr, -self.winsor_clip, self.winsor_clip)
        if self.scaler is not None:
            return self.scaler.transform(arr)
        return arr
