"""
Bootstrap training: trains baseline lightgbm models for 1d/3d/7d horizons
directly from price_daily (no articles required).

Usage:
    python -m backend.scripts.bootstrap_train --db data/backtest.db --outdir models
"""
from __future__ import annotations

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

from backend.ml.feature_pipeline import DEFAULT_FEATURE_NAMES
from backend.ml.training_pipeline import train_horizon_model


def _ema_series(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _per_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    close  = g["close"].astype(float)
    high   = g["high"].astype(float)
    low    = g["low"].astype(float)
    volume = g["volume"].astype(float).fillna(0.0)
    ret    = close.pct_change()

    # Returns
    g["return_1d"]            = ret
    g["avg_return_5d"]        = ret.rolling(5).mean()
    g["avg_return_20d"]       = ret.rolling(20).mean()
    g["return_volatility_20d"] = ret.rolling(20).std()
    g["return_skew_20d"]      = ret.rolling(20).skew()

    # Volume
    vol_mean = volume.rolling(20).mean()
    g["volume_ratio"]    = volume / vol_mean.replace(0, np.nan)
    def _slope(w: pd.Series) -> float:
        x = np.arange(len(w), dtype=float)
        y = w.to_numpy(dtype=float)
        return float(np.polyfit(x, y, 1)[0]) if not np.allclose(y, y[0]) else 0.0
    g["volume_trend_20d"] = volume.rolling(20).apply(_slope, raw=False)

    # Momentum / trend
    g["rsi_14"] = _rsi(close)
    ema12 = _ema_series(close, 12)
    ema26 = _ema_series(close, 26)
    macd_line = ema12 - ema26
    signal    = _ema_series(macd_line, 9)
    g["macd_hist"]       = macd_line - signal
    ema9  = _ema_series(close, 9)
    ema21 = _ema_series(close, 21)
    ema50 = _ema_series(close, 50)
    g["ema_ratio_9_21"]  = ema9  / ema21.replace(0, np.nan) - 1.0
    g["ema_ratio_21_50"] = ema21 / ema50.replace(0, np.nan) - 1.0

    # Bollinger Band %B
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_width = (upper - lower).replace(0, np.nan)
    g["bb_pct_b"] = (close - lower) / band_width

    # ATR %
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    g["atr_pct"] = tr.rolling(14).mean() / close.replace(0, np.nan)

    # Seasonality
    g["day_of_week"] = pd.to_datetime(g["date"]).dt.dayofweek.astype(float)

    # Sentiment placeholders
    g["avg_sentiment"]        = 0.0
    g["sentiment_volatility"] = 0.0
    g["article_count"]        = 0.0

    for h, days in (("1d", 1), ("3d", 3), ("7d", 7)):
        g[f"label_{h}"] = close.shift(-days) / close - 1.0

    return g


def build_dataset(db_path: str, min_history: int = 60) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume FROM price_daily ORDER BY ticker, date",
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError("price_daily is empty; cannot bootstrap")

    counts = df.groupby("ticker").size()
    eligible = counts[counts >= min_history].index
    df = df[df["ticker"].isin(eligible)].copy()

    parts = []
    for ticker, group in df.groupby("ticker", sort=False):
        feat = _per_ticker_features(group)
        feat["ticker"] = ticker
        parts.append(feat)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    needed = list(DEFAULT_FEATURE_NAMES) + ["label_1d", "label_3d", "label_7d"]
    out = out.dropna(subset=needed)

    for col in DEFAULT_FEATURE_NAMES:
        out[col] = out[col].astype(float).fillna(0.0)

    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/backtest.db")
    parser.add_argument("--outdir", default="models")
    parser.add_argument("--min-history", type=int, default=60)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Building dataset from {args.db}...")
    df = build_dataset(args.db, min_history=args.min_history)
    print(f"Dataset rows: {len(df)} across {df['ticker'].nunique()} tickers")

    for horizon in ("1d", "3d", "7d"):
        artifact = train_horizon_model(df, horizon=horizon, outdir=args.outdir)
        print(f"[{horizon}] -> {artifact.model_path} metrics={artifact.metrics}")


if __name__ == "__main__":
    main()
