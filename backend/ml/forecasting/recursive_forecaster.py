"""
Recursive multi-step forecasting from a one-step model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.ml.forecasting.contracts import ForecastResult, RecursionMode, TargetMode
from backend.ml.forecasting.feature_builder import FeatureBuilder
from backend.ml.forecasting.model_adapter import ModelAdapter
from backend.ml.forecasting.preprocessor import Preprocessor


class RecursiveForecaster:
    def __init__(
        self,
        model: ModelAdapter,
        preprocessor: Preprocessor,
        feature_builder: FeatureBuilder,
        target_mode: TargetMode = TargetMode.log_return_1,
        recursion_mode: RecursionMode = RecursionMode.strict_recursive,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_builder = feature_builder
        self.target_mode = target_mode
        self.recursion_mode = recursion_mode

    def forecast(self, history_df: pd.DataFrame, horizon: int, model_version: str = "v1") -> ForecastResult:
        work = history_df.copy().reset_index(drop=True)
        featured = self.feature_builder.build(work)
        predicted_targets: List[float] = []
        predicted_prices: List[float] = []
        last_price = float(work["close"].iloc[-1])
        origin_time = pd.to_datetime(work["date"].iloc[-1]).to_pydatetime() if "date" in work else datetime.utcnow()

        for step in range(horizon):
            current_featured = self.feature_builder.build(work)
            x_row = current_featured[self.feature_builder.feature_columns].iloc[[-1]].fillna(0.0)
            x_model = self.preprocessor.transform(x_row)
            pred = float(self.model.predict(x_model)[0])
            predicted_targets.append(pred)

            next_close = self._next_close(last_price, pred)
            predicted_prices.append(next_close)
            last_price = next_close

            next_row = self._next_row(work.iloc[-1], next_close, step)
            work = pd.concat([work, pd.DataFrame([next_row])], ignore_index=True)

        return ForecastResult(
            origin_time=origin_time,
            horizon=horizon,
            target_mode=self.target_mode.value,
            predicted_targets=predicted_targets,
            predicted_prices=predicted_prices,
            feature_snapshot=featured[self.feature_builder.feature_columns].iloc[-1].fillna(0.0).to_dict(),
            model_version=model_version,
            metadata={
                "recursion_mode": self.recursion_mode.value,
                "model_name": self.model.name,
            },
        )

    def _next_close(self, current_price: float, pred: float) -> float:
        if self.target_mode == TargetMode.log_return_1:
            return max(0.01, current_price * float(np.exp(pred)))
        if self.target_mode in {TargetMode.return_1, TargetMode.residual_return_1}:
            return max(0.01, current_price * (1.0 + pred))
        if self.target_mode == TargetMode.delta_1:
            return max(0.01, current_price + pred)
        if self.target_mode == TargetMode.price_1:
            return max(0.01, pred)
        return max(0.01, current_price)

    def _next_row(self, last_row: pd.Series, next_close: float, step: int) -> Dict:
        if self.recursion_mode == RecursionMode.strict_recursive:
            open_px = next_close
            high_px = next_close
            low_px = next_close
        else:
            open_px = float(last_row.get("open", next_close))
            high_px = float(last_row.get("high", next_close))
            low_px = float(last_row.get("low", next_close))
        volume = float(last_row.get("volume", 0.0))
        date_value = pd.to_datetime(last_row.get("date")) + pd.Timedelta(days=1) if "date" in last_row else None
        row = {
            "open": open_px,
            "high": max(high_px, next_close),
            "low": min(low_px, next_close),
            "close": next_close,
            "volume": volume,
        }
        if date_value is not None:
            row["date"] = date_value
        return row
