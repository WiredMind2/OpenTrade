import numpy as np
import pandas as pd

from backend.ml.forecasting import (
    FeatureBuilder,
    Preprocessor,
    RecursiveForecaster,
    RecursionMode,
    TargetMode,
    WalkForwardSplitter,
)
from backend.ml.forecasting.model_adapter import ModelAdapter


class ConstantModel:
    def predict(self, x):
        return np.array([0.001] * len(x))


def _sample_df(n: int = 600) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(100, 120, n) + np.sin(np.arange(n) / 10)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.linspace(1000, 2000, n),
        }
    )


def test_walk_forward_splitter_respects_gap():
    splitter = WalkForwardSplitter(train_mode="expanding", min_train_size=500, test_size=20, step_size=20, gap=5)
    first = next(splitter.split(700))
    assert first.train_end == 500
    assert first.test_start == 505
    assert first.test_end == 525


def test_recursive_forecaster_generates_path():
    df = _sample_df(620)
    fb = FeatureBuilder()
    features = fb.build(df).dropna(subset=fb.feature_columns)
    pre = Preprocessor(use_scaler=False)
    pre.fit(features[fb.feature_columns])
    model = ModelAdapter(name="constant", model=ConstantModel())
    fc = RecursiveForecaster(
        model=model,
        preprocessor=pre,
        feature_builder=fb,
        target_mode=TargetMode.log_return_1,
        recursion_mode=RecursionMode.strict_recursive,
    ).forecast(df, horizon=5, model_version="vtest")
    assert len(fc.predicted_targets) == 5
    assert len(fc.predicted_prices) == 5
    assert fc.origin_time is not None
