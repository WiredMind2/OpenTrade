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


def _per_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    close = g["close"].astype(float)
    volume = g["volume"].astype(float).fillna(0.0)
    ret = close.pct_change()

    g["avg_return_5d"] = ret.rolling(5).mean()
    g["avg_return_20d"] = ret.rolling(20).mean()
    g["return_volatility_20d"] = ret.rolling(20).std()
    g["volume_mean_20d"] = volume.rolling(20).mean()

    def _slope(window: pd.Series) -> float:
        x = np.arange(len(window), dtype=float)
        y = window.to_numpy(dtype=float)
        if np.allclose(y, y[0]):
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    g["volume_trend_20d"] = volume.rolling(20).apply(_slope, raw=False)

    g["avg_sentiment"] = 0.0
    g["sentiment_volatility"] = 0.0
    g["article_count"] = 0.0

    for h, days in (("1d", 1), ("3d", 3), ("7d", 7)):
        g[f"label_{h}"] = close.shift(-days) / close - 1.0

    return g


def build_dataset(db_path: str, min_history: int = 60) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT ticker, date, close, volume FROM price_daily ORDER BY ticker, date",
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
