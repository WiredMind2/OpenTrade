"""
Offline walk-forward training and evaluation runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from backend.ml.forecasting.contracts import ForecastConfig
from backend.ml.forecasting.evaluator import Evaluator
from backend.ml.forecasting.feature_builder import FeatureBuilder
from backend.ml.forecasting.model_adapter import build_model_adapter
from backend.ml.forecasting.preprocessor import Preprocessor
from backend.ml.forecasting.recursive_forecaster import RecursiveForecaster
from backend.ml.forecasting.splitter import WalkForwardSplitter
from backend.ml.forecasting.target_builder import TargetBuilder


@dataclass
class WalkForwardRunner:
    config: ForecastConfig

    def run(self, df: pd.DataFrame) -> Dict[str, List]:
        fb = FeatureBuilder()
        tb = TargetBuilder(self.config.target_mode)
        featured = fb.build(df)
        featured["target"] = tb.build(featured)
        featured = featured.dropna(subset=fb.feature_columns + ["target"]).reset_index(drop=True)
        splitter = WalkForwardSplitter(
            train_mode="expanding",
            min_train_size=self.config.min_train_size,
            test_size=self.config.test_size,
            step_size=self.config.step_size,
            gap=self.config.gap,
        )
        y_true_by_h: Dict[int, List[float]] = {h: [] for h in range(1, self.config.horizon + 1)}
        y_pred_by_h: Dict[int, List[float]] = {h: [] for h in range(1, self.config.horizon + 1)}

        for split in splitter.split(len(featured)):
            train_df = featured.iloc[split.train_start:split.train_end]
            test_df = featured.iloc[split.test_start:split.test_end]
            x_train = train_df[fb.feature_columns]
            y_train = train_df["target"]
            prep = Preprocessor(use_scaler=self.config.model_name in {"ridge", "lasso", "elastic_net", "linear"})
            prep.fit(x_train)
            model = build_model_adapter(self.config.model_name)
            model.fit(prep.transform(x_train), y_train.values)
            forecaster = RecursiveForecaster(
                model=model,
                preprocessor=prep,
                feature_builder=fb,
                target_mode=self.config.target_mode,
                recursion_mode=self.config.recursion_mode,
            )
            for i in range(len(test_df)):
                origin_idx = split.test_start + i - 1
                if origin_idx < 0:
                    continue
                history = featured.iloc[: origin_idx + 1][["date", "open", "high", "low", "close", "volume"]].copy()
                fc = forecaster.forecast(history, horizon=self.config.horizon)
                for h in range(1, self.config.horizon + 1):
                    target_idx = origin_idx + h
                    if target_idx >= len(featured):
                        continue
                    y_true_by_h[h].append(float(featured.iloc[target_idx]["target"]))
                    y_pred_by_h[h].append(float(fc.predicted_targets[h - 1]))

        metrics = Evaluator().compute_horizon_metrics(y_true_by_h, y_pred_by_h)
        return {"metrics": metrics}
