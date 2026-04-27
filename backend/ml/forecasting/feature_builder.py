"""
Causal feature engineering for recursive forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureBuilder:
    """
    Builds causal features using only history up to each timestamp.
    """

    def __init__(self):
        self.feature_columns: list[str] = []

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["ret_1"] = work["close"].pct_change()
        work["log_ret_1"] = np.log(work["close"] / work["close"].shift(1))
        work["delta_1"] = work["close"].diff()
        for lag in (1, 2, 3, 5, 10, 20):
            work[f"ret_lag_{lag}"] = work["ret_1"].shift(lag)
            work[f"log_ret_lag_{lag}"] = work["log_ret_1"].shift(lag)
            work[f"delta_lag_{lag}"] = work["delta_1"].shift(lag)
            work[f"sign_lag_{lag}"] = np.sign(work["ret_1"].shift(lag))
        for window in (3, 5, 10, 20):
            work[f"ret_mean_{window}"] = work["ret_1"].rolling(window).mean()
        for window in (5, 10, 20, 50):
            work[f"ret_std_{window}"] = work["ret_1"].rolling(window).std()
        work["ret_skew_20"] = work["ret_1"].rolling(20).skew()
        work["ret_kurt_20"] = work["ret_1"].rolling(20).kurt()
        work["ret_z_20"] = (work["ret_1"] - work["ret_1"].rolling(20).mean()) / work["ret_1"].rolling(20).std()
        work["dist_to_roll_low_20"] = work["close"] - work["close"].rolling(20).min()
        work["dist_to_roll_high_20"] = work["close"].rolling(20).max() - work["close"]
        for p in (3, 5, 10, 20):
            work[f"momentum_{p}"] = work["close"] / work["close"].shift(p) - 1.0
        work["sma_5"] = work["close"].rolling(5).mean()
        work["sma_20"] = work["close"].rolling(20).mean()
        work["ema_10"] = work["close"].ewm(span=10, adjust=False).mean()
        work["ema_50"] = work["close"].ewm(span=50, adjust=False).mean()
        work["sma_spread_5_20"] = work["sma_5"] - work["sma_20"]
        work["ema_spread_10_50"] = work["ema_10"] - work["ema_50"]
        work["close_sma_ratio_20"] = work["close"] / work["sma_20"]
        work["vol_ret_1"] = work["volume"].pct_change()
        work["vol_mean_20"] = work["volume"].rolling(20).mean()
        work["vol_z_20"] = (work["volume"] - work["vol_mean_20"]) / work["volume"].rolling(20).std()
        work["hl_range"] = (work["high"] - work["low"]) / work["close"].replace(0, np.nan)
        tr = pd.concat(
            [
                (work["high"] - work["low"]),
                (work["high"] - work["close"].shift(1)).abs(),
                (work["low"] - work["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        work["true_range"] = tr
        work["atr_14"] = tr.rolling(14).mean()
        work["close_location"] = (work["close"] - work["low"]) / (work["high"] - work["low"]).replace(0, np.nan)

        feature_cols = [c for c in work.columns if c not in {"date", "open", "high", "low", "close", "volume"}]
        self.feature_columns = feature_cols
        return work

    def latest_vector(self, featured_df: pd.DataFrame) -> pd.Series:
        if not self.feature_columns:
            raise ValueError("FeatureBuilder has no known feature columns.")
        return featured_df[self.feature_columns].iloc[-1]
