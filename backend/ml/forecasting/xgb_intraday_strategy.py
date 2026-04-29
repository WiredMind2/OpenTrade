"""
Leakage-safe walk-forward XGBoost intraday strategy runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class XGBIntradayConfig:
    horizon_bars: int = 3
    long_threshold: float = 0.0008
    short_threshold: float = 0.0008
    tree_method: str = "hist"
    objective: str = "reg:squarederror"
    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    max_bin: int = 256
    random_state: int = 42
    min_train_size: int = 500
    test_size: int = 100
    step_size: int = 100
    gap: int = 1
    retrain_cadence: int = 1
    commission_bps: float = 0.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 2.0
    max_holding_bars: int = 3
    feature_columns: List[str] = field(default_factory=list)


class XGBIntradayStrategyRunner:
    def __init__(self, config: XGBIntradayConfig | None = None):
        self.config = config or XGBIntradayConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy().sort_values("date").reset_index(drop=True)
        eps = 1e-9

        work["ret_1"] = np.log(work["close"] / work["close"].shift(1))
        for n in (2, 3, 5, 10):
            work[f"ret_{n}"] = np.log(work["close"] / work["close"].shift(n))

        # Session-free fallback for daily/intraday mixed datasets.
        session_open = work["open"]
        work["gap_from_open"] = np.log(work["close"] / session_open.replace(0.0, np.nan))
        work["range_1"] = (work["high"] - work["low"]) / work["close"].replace(0.0, np.nan)
        work["body_frac"] = (work["close"] - work["open"]) / (work["high"] - work["low"]).clip(lower=eps)
        work["close_to_high"] = (work["high"] - work["close"]) / work["close"].replace(0.0, np.nan)
        work["close_to_low"] = (work["close"] - work["low"]) / work["close"].replace(0.0, np.nan)

        for n in (5, 10, 20):
            roll = work["ret_1"].rolling(n)
            work[f"rolling_mean_ret_{n}"] = roll.mean()
            work[f"rolling_std_ret_{n}"] = roll.std()
        work["ret_z_20"] = (
            (work["ret_1"] - work["ret_1"].rolling(20).mean())
            / work["ret_1"].rolling(20).std().replace(0.0, np.nan)
        )

        for n in (5, 10, 20, 50):
            ema = work["close"].ewm(span=n, adjust=False).mean()
            work[f"ema_dist_{n}"] = (work["close"] - ema) / ema.replace(0.0, np.nan)
            work[f"ema_slope_{n}"] = ema.pct_change(3)

        work["rsi_5"] = _rsi(work["close"], 5)
        work["rsi_14"] = _rsi(work["close"], 14)

        for n in (5, 20):
            work[f"vol_ratio_{n}"] = work["volume"] / work["volume"].rolling(n).mean()
            work[f"vol_std_{n}"] = work["volume"].rolling(n).std()

        typical_price = (work["high"] + work["low"] + work["close"]) / 3.0
        vwap_proxy = (typical_price * work["volume"]).cumsum() / work["volume"].cumsum().replace(0.0, np.nan)
        work["vwap_dist"] = (work["close"] - vwap_proxy) / vwap_proxy.replace(0.0, np.nan)

        if pd.api.types.is_datetime64_any_dtype(work["date"]):
            work["minutes_since_open"] = work["date"].dt.hour * 60 + work["date"].dt.minute
        else:
            work["minutes_since_open"] = 0.0
        work["minutes_to_close"] = np.maximum(390.0 - work["minutes_since_open"], 0.0)

        feature_cols = [
            c
            for c in work.columns
            if c
            not in {
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            }
        ]
        work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan)
        self.config.feature_columns = feature_cols
        return work

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        h = self.config.horizon_bars
        return np.log(df["close"].shift(-h) / df["close"])

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        from backend.ml.forecasting.splitter import WalkForwardSplitter

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for XGBIntradayStrategyRunner") from exc

        featured = self.build_features(df)
        featured["target"] = self.build_target(featured)
        featured = featured.dropna(subset=self.config.feature_columns + ["target"]).reset_index(drop=True)
        if featured.empty:
            raise ValueError("No usable rows after feature/label generation.")

        n_rows = len(featured)
        min_train = min(self.config.min_train_size, max(30, int(n_rows * 0.6)))
        test_size = min(self.config.test_size, max(10, int(n_rows * 0.2)))
        step_size = min(self.config.step_size, test_size)

        splitter = WalkForwardSplitter(
            train_mode="expanding",
            min_train_size=min_train,
            test_size=test_size,
            step_size=step_size,
            gap=self.config.gap,
        )

        prediction_rows: List[Dict[str, Any]] = []
        for split_idx, split in enumerate(splitter.split(len(featured))):
            if split_idx % max(self.config.retrain_cadence, 1) != 0:
                continue

            train_df = featured.iloc[split.train_start:split.train_end]
            test_df = featured.iloc[split.test_start:split.test_end].copy()
            if train_df.empty or test_df.empty:
                continue

            model = XGBRegressor(
                objective=self.config.objective,
                tree_method=self.config.tree_method,
                booster="gbtree",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                gamma=self.config.gamma,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                max_bin=self.config.max_bin,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            model.fit(train_df[self.config.feature_columns], train_df["target"])
            test_df["pred"] = model.predict(test_df[self.config.feature_columns])
            test_df["signal"] = 0
            test_df.loc[test_df["pred"] > self.config.long_threshold, "signal"] = 1
            test_df.loc[test_df["pred"] < -self.config.short_threshold, "signal"] = -1
            prediction_rows.extend(test_df.to_dict("records"))

        if not prediction_rows:
            holdout = int(n_rows * 0.8)
            if holdout <= 10 or holdout >= n_rows - 5:
                raise ValueError("No walk-forward predictions produced; dataset too small after feature/label generation.")
            train_df = featured.iloc[:holdout]
            test_df = featured.iloc[holdout:].copy()
            model = XGBRegressor(
                objective=self.config.objective,
                tree_method=self.config.tree_method,
                booster="gbtree",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                gamma=self.config.gamma,
                reg_alpha=self.config.reg_alpha,
                reg_lambda=self.config.reg_lambda,
                max_bin=self.config.max_bin,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
            model.fit(train_df[self.config.feature_columns], train_df["target"])
            test_df["pred"] = model.predict(test_df[self.config.feature_columns])
            test_df["signal"] = 0
            test_df.loc[test_df["pred"] > self.config.long_threshold, "signal"] = 1
            test_df.loc[test_df["pred"] < -self.config.short_threshold, "signal"] = -1
            prediction_rows.extend(test_df.to_dict("records"))

        predictions = pd.DataFrame(prediction_rows).sort_values("date").reset_index(drop=True)
        trades = self._simulate_execution(predictions)
        return {
            "forecast_metrics": self._forecast_metrics(predictions),
            "trading_metrics": self._trading_metrics(trades),
            "predictions": predictions,
            "trades": trades,
            "feature_columns": list(self.config.feature_columns),
        }

    def _simulate_execution(self, preds: pd.DataFrame) -> pd.DataFrame:
        roundtrip_bps = (
            self.config.commission_bps + self.config.half_spread_bps * 2.0 + self.config.slippage_bps * 2.0
        )
        cost_frac = roundtrip_bps / 10000.0
        hold = max(1, self.config.max_holding_bars)

        rows: List[Dict[str, Any]] = []
        next_entry_idx = 1
        for idx in range(len(preds) - hold - 1):
            if idx < next_entry_idx:
                continue
            signal = int(preds.iloc[idx]["signal"])
            if signal == 0:
                continue

            entry_idx = idx + 1
            exit_idx = min(entry_idx + hold, len(preds) - 1)
            entry_open = float(preds.iloc[entry_idx]["open"])
            exit_close = float(preds.iloc[exit_idx]["close"])
            if entry_open <= 0.0:
                continue

            gross_ret = signal * ((exit_close / entry_open) - 1.0)
            net_ret = gross_ret - cost_frac
            rows.append(
                {
                    "signal_time": preds.iloc[idx]["date"],
                    "entry_time": preds.iloc[entry_idx]["date"],
                    "exit_time": preds.iloc[exit_idx]["date"],
                    "side": signal,
                    "entry_open": entry_open,
                    "exit_close": exit_close,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "pred_return": float(preds.iloc[idx]["pred"]),
                }
            )
            next_entry_idx = exit_idx + 1

        return pd.DataFrame(rows)

    def _forecast_metrics(self, preds: pd.DataFrame) -> Dict[str, float]:
        y_true = preds["target"].to_numpy(dtype=float)
        y_pred = preds["pred"].to_numpy(dtype=float)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        sign_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
        corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
        return {
            "rmse": rmse,
            "mae": mae,
            "sign_accuracy": sign_accuracy,
            "correlation": corr,
        }

    def _trading_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        if trades.empty:
            return {
                "trades": 0.0,
                "net_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_return": 0.0,
                "sharpe_like": 0.0,
            }
        rets = trades["net_return"].to_numpy(dtype=float)
        wins = rets[rets > 0.0].sum()
        losses = -rets[rets < 0.0].sum()
        std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
        sharpe_like = float(np.sqrt(len(rets)) * np.mean(rets) / std) if std > 0 else 0.0
        return {
            "trades": float(len(rets)),
            "net_return": float(np.sum(rets)),
            "win_rate": float(np.mean(rets > 0.0)),
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "avg_trade_return": float(np.mean(rets)),
            "sharpe_like": sharpe_like,
        }
