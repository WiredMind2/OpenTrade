import numpy as np
import pandas as pd

from backend.ml.forecasting.xgb_intraday_strategy import XGBIntradayConfig, XGBIntradayStrategyRunner


def _sample_df(n: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0.05, 0.8, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + np.random.default_rng(43).normal(0.0, 0.1, n),
            "high": close + np.abs(np.random.default_rng(44).normal(0.3, 0.1, n)),
            "low": close - np.abs(np.random.default_rng(45).normal(0.3, 0.1, n)),
            "close": close,
            "volume": np.random.default_rng(46).integers(10_000, 50_000, n),
        }
    )


def test_target_alignment_uses_forward_horizon():
    df = _sample_df(30)
    cfg = XGBIntradayConfig(horizon_bars=3)
    runner = XGBIntradayStrategyRunner(cfg)
    target = runner.build_target(df)
    expected = np.log(df["close"].shift(-3) / df["close"])
    assert np.allclose(target.iloc[:-3], expected.iloc[:-3], equal_nan=True)
    assert target.iloc[-3:].isna().all()


def test_feature_builder_is_past_only_for_lag_features():
    df = _sample_df(60)
    cfg = XGBIntradayConfig()
    runner = XGBIntradayStrategyRunner(cfg)
    featured = runner.build_features(df)
    idx = 20
    assert np.isclose(featured.loc[idx, "ret_2"], np.log(df.loc[idx, "close"] / df.loc[idx - 2, "close"]))


def test_execution_occurs_after_signal_bar():
    df = _sample_df(40)
    cfg = XGBIntradayConfig(max_holding_bars=2)
    runner = XGBIntradayStrategyRunner(cfg)
    preds = df.copy()
    preds["target"] = 0.0
    preds["pred"] = 0.0
    preds["signal"] = 0
    preds.loc[5, "signal"] = 1
    trades = runner._simulate_execution(preds)
    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == preds.iloc[6]["date"]
