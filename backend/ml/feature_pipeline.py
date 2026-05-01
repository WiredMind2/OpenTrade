"""
Unified feature pipeline for model training and realtime inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import sqlite3


DEFAULT_FEATURE_NAMES: List[str] = [
    # Returns
    "return_1d",
    "avg_return_5d",
    "avg_return_20d",
    "return_volatility_20d",
    "return_skew_20d",
    # Volume
    "volume_ratio",
    "volume_trend_20d",
    # Momentum / trend
    "rsi_14",
    "macd_hist",
    "ema_ratio_9_21",
    "ema_ratio_21_50",
    # Volatility
    "bb_pct_b",
    "atr_pct",
    # Seasonality
    "day_of_week",
]


@dataclass
class FeatureInput:
    ticker: str
    as_of: datetime


# ── Pure numpy helpers ────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> float:
    """Exponential moving average of the last value in arr."""
    if arr.size == 0:
        return 0.0
    alpha = 2.0 / (period + 1)
    val = float(arr[0])
    for x in arr[1:]:
        val = alpha * float(x) + (1.0 - alpha) * val
    return val


def _ema_series(arr: np.ndarray, period: int) -> np.ndarray:
    """Full EMA series (same length as arr)."""
    out = np.empty_like(arr, dtype=float)
    if arr.size == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = float(arr[0])
    for i in range(1, len(arr)):
        out[i] = alpha * float(arr[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if closes.size < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _macd_hist(closes: np.ndarray) -> float:
    """MACD histogram = (EMA12 - EMA26) - EMA9(MACD)."""
    if closes.size < 26:
        return 0.0
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = ema12 - ema26
    signal = _ema_series(macd_line, 9)
    return float(macd_line[-1] - signal[-1])


def _bb_pct_b(closes: np.ndarray, period: int = 20) -> float:
    """Bollinger Band %B: position of last price within the band."""
    if closes.size < period:
        return 0.5
    window = closes[-period:]
    mid = window.mean()
    std = window.std()
    if std == 0.0:
        return 0.5
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    return float((closes[-1] - lower) / (upper - lower))


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """ATR as a percentage of the last close price."""
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return 0.0
    highs, lows, closes = highs[-n:], lows[-n:], closes[-n:]
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr = tr[-period:].mean() if len(tr) >= period else tr.mean()
    last_close = closes[-1]
    return float(atr / last_close) if last_close != 0.0 else 0.0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class FeaturePipeline:
    """Builds deterministic numeric feature vectors for a ticker."""

    schema_version = "ml_features_v3"

    def __init__(self, feature_names: Sequence[str] | None = None):
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)

    def build_vector(self, conn: sqlite3.Connection, payload: FeatureInput) -> np.ndarray:
        price_data = self._load_prices(conn, payload.ticker, payload.as_of)
        feature_map = self._compute_feature_map(price_data, payload.as_of)
        self._validate_features(feature_map)
        return np.array([feature_map[name] for name in self.feature_names], dtype=float).reshape(1, -1)

    def _load_prices(
        self,
        conn: sqlite3.Connection,
        ticker: str,
        as_of: datetime,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Return (open, high, low, close, volume) rows, newest first, up to 300 days."""
        cur = conn.cursor()
        since = (as_of - timedelta(days=300)).date().isoformat()
        cur.execute(
            """
            SELECT open, high, low, close, volume
            FROM price_daily
            WHERE ticker = ? AND date <= ? AND date >= ?
            ORDER BY date DESC
            LIMIT 300
            """,
            (ticker.upper(), as_of.date().isoformat(), since),
        )
        return cur.fetchall()

    def _compute_feature_map(
        self,
        price_data: List[Tuple[float, float, float, float, float]],
        as_of: datetime,
    ) -> dict:
        # ── Price arrays (oldest → newest) ─────────────────────────────────
        rows = list(reversed(price_data))
        opens   = np.array([float(r[0]) for r in rows if r[0] is not None], dtype=float)
        highs   = np.array([float(r[1]) for r in rows if r[1] is not None], dtype=float)
        lows    = np.array([float(r[2]) for r in rows if r[2] is not None], dtype=float)
        closes  = np.array([float(r[3]) for r in rows if r[3] is not None], dtype=float)
        volumes = np.array([float(r[4]) for r in rows if r[4] is not None], dtype=float)

        if closes.size < 2:
            returns = np.array([0.0])
        else:
            returns = np.diff(closes) / np.where(closes[:-1] != 0, closes[:-1], 1.0)

        # ── Return features ────────────────────────────────────────────────
        return_1d           = float(returns[-1])        if returns.size >= 1  else 0.0
        avg_return_5d       = float(returns[-5:].mean()) if returns.size >= 5  else float(returns.mean())
        avg_return_20d      = float(returns[-20:].mean()) if returns.size >= 20 else float(returns.mean())
        return_volatility_20d = float(returns[-20:].std()) if returns.size >= 20 else float(returns.std())
        return_skew_20d     = float(_skew(returns[-20:])) if returns.size >= 20 else 0.0

        # ── Volume features ────────────────────────────────────────────────
        vol_mean_20d = float(volumes[-20:].mean()) if volumes.size >= 20 else float(volumes.mean()) if volumes.size else 1.0
        volume_ratio = float(volumes[-1] / vol_mean_20d) if vol_mean_20d != 0.0 and volumes.size else 1.0
        volume_trend_20d = (
            float(np.polyfit(np.arange(min(20, volumes.size)), volumes[-20:], 1)[0])
            if volumes.size > 1 else 0.0
        )

        # ── Momentum / trend ───────────────────────────────────────────────
        rsi_14 = _rsi(closes)
        macd_h = _macd_hist(closes)

        ema9  = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        ema50 = _ema(closes, 50)
        ema_ratio_9_21  = float(ema9  / ema21  - 1.0) if ema21  != 0.0 else 0.0
        ema_ratio_21_50 = float(ema21 / ema50  - 1.0) if ema50  != 0.0 else 0.0

        # ── Volatility ─────────────────────────────────────────────────────
        bb_pct_b = _bb_pct_b(closes)
        atr_p    = _atr_pct(highs, lows, closes)

        # ── Seasonality ────────────────────────────────────────────────────
        day_of_week = float(as_of.weekday())  # 0=Mon … 4=Fri

        return {
            "return_1d":            return_1d,
            "avg_return_5d":        avg_return_5d,
            "avg_return_20d":       avg_return_20d,
            "return_volatility_20d": return_volatility_20d,
            "return_skew_20d":      return_skew_20d,
            "volume_ratio":         volume_ratio,
            "volume_trend_20d":     volume_trend_20d,
            "rsi_14":               rsi_14,
            "macd_hist":            macd_h,
            "ema_ratio_9_21":       ema_ratio_9_21,
            "ema_ratio_21_50":      ema_ratio_21_50,
            "bb_pct_b":             bb_pct_b,
            "atr_pct":              atr_p,
            "day_of_week":          day_of_week,
        }

    def _validate_features(self, feature_map: dict) -> None:
        missing = [n for n in self.feature_names if n not in feature_map]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        for name in self.feature_names:
            val = feature_map[name]
            if np.isnan(val) or np.isinf(val):
                feature_map[name] = 0.0


def _skew(arr: np.ndarray) -> float:
    if arr.size < 3:
        return 0.0
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0.0:
        return 0.0
    return float(((arr - mu) ** 3).mean() / sigma ** 3)
