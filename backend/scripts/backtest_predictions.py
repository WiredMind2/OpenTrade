"""
Walk-forward backtest for the LightGBM prediction models.

For each (ticker, date) in the evaluation window, this script:
  1. Builds features as-of that date using the production FeaturePipeline
  2. Runs the loaded model to get a predicted log-return
  3. Looks up the actual close price H trading-days later
  4. Computes actual log-return and compares with the prediction

Metrics reported per horizon:
  - Directional Accuracy (DA)   : % of correct sign predictions
  - MAE                         : mean absolute error on log-returns
  - RMSE                        : root mean squared error
  - Naive baseline DA           : always-predict-zero directional accuracy
  - Pearson correlation

Usage:
    cd /home/leo/4IF/SMART/OpenTrade
    python -m backend.scripts.backtest_predictions [--tickers AAPL MSFT] [--days 180]
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from backend.ml.feature_pipeline import FeatureInput, FeaturePipeline

DB_PATH = "data/backtest.db"
MODEL_DIR = Path("models")
HORIZONS = {"1d": 1, "3d": 3, "7d": 7}

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "GOLD"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_models() -> Dict[str, object]:
    # Sort descending so the most recent model wins for each horizon
    models: Dict[str, object] = {}
    for path in sorted(MODEL_DIR.glob("lightgbm_*.joblib"), reverse=True):
        bundle = joblib.load(path)
        model = bundle.get("lgbm", bundle)
        for h in HORIZONS:
            if f"lightgbm_{h}_" in path.stem or path.stem == f"lightgbm_{h}":
                models.setdefault(f"lightgbm_{h}", model)
    return models


def trading_dates(conn: sqlite3.Connection, ticker: str, start: str, end: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT date FROM price_daily WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
        (ticker, start, end),
    )
    return [r[0] for r in cur.fetchall()]


def close_on_or_after(
    conn: sqlite3.Connection, ticker: str, ref_date: str, offset_days: int
) -> Tuple[float | None, str | None]:
    """Return the close price and date that is offset_days *trading* days after ref_date."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, close FROM price_daily
        WHERE ticker = ? AND date > ?
        ORDER BY date
        LIMIT ?
        """,
        (ticker, ref_date, offset_days),
    )
    rows = cur.fetchall()
    if len(rows) < offset_days:
        return None, None
    return float(rows[-1][1]), rows[-1][0]


def actual_log_return(conn: sqlite3.Connection, ticker: str, date: str, horizon_days: int) -> float | None:
    cur = conn.cursor()
    cur.execute(
        "SELECT close FROM price_daily WHERE ticker = ? AND date = ?",
        (ticker, date),
    )
    row = cur.fetchone()
    if not row or row[0] is None or row[0] == 0:
        return None
    close_now = float(row[0])
    future_close, _ = close_on_or_after(conn, ticker, date, horizon_days)
    if future_close is None or future_close == 0:
        return None
    return float(np.log(future_close / close_now))


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    tickers: List[str],
    lookback_days: int = 180,
    verbose: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns nested dict:  results[horizon_label][metric_key] = list of values
    where metric_key is 'y_true' or 'y_pred'.
    """
    models = load_models()
    if not models:
        raise RuntimeError(f"No models found in {MODEL_DIR}")

    pipeline = FeaturePipeline()
    conn = sqlite3.connect(DB_PATH)

    end_date = datetime.utcnow().date()
    # Leave max(horizons) extra days so we always have a realized return
    eval_end = end_date - timedelta(days=max(HORIZONS.values()) + 5)
    eval_start = eval_end - timedelta(days=lookback_days)

    results: Dict[str, Dict[str, List[float]]] = {
        h: {"y_true": [], "y_pred": []} for h in HORIZONS
    }

    for ticker in tickers:
        dates = trading_dates(conn, ticker, eval_start.isoformat(), eval_end.isoformat())
        if verbose:
            print(f"  {ticker}: {len(dates)} dates in evaluation window")

        for date_str in dates:
            as_of = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=21)
            try:
                vector = pipeline.build_vector(conn, FeatureInput(ticker=ticker, as_of=as_of))
            except Exception:
                continue

            for h_label, h_days in HORIZONS.items():
                model = models.get(f"lightgbm_{h_label}")
                if model is None:
                    continue
                y_true = actual_log_return(conn, ticker, date_str, h_days)
                if y_true is None:
                    continue
                y_pred = float(model.predict(vector)[0])
                results[h_label]["y_true"].append(y_true)
                results[h_label]["y_pred"].append(y_pred)

    conn.close()
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return {}
    da = float(np.mean(np.sign(yt) == np.sign(yp)))
    # Naive baseline: always predict zero (→ always wrong on direction if market drifts)
    naive_da = float(np.mean(np.sign(yt) == 0))  # fraction of flat days
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if n > 1 else 0.0
    # Naive MAE: always predict 0
    naive_mae = float(np.mean(np.abs(yt)))
    return {
        "n": n,
        "da": da,
        "naive_da": naive_da,
        "mae": mae,
        "naive_mae": naive_mae,
        "rmse": rmse,
        "corr": corr,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(results: Dict[str, Dict[str, List[float]]]) -> None:
    print()
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("=" * 70)
    print(f"  {'Horizon':<10} {'N':>6} {'DA':>8} {'Naive DA':>10} {'MAE':>10} {'Naive MAE':>11} {'RMSE':>10} {'Corr':>8}")
    print("-" * 70)
    for h_label in ["1d", "3d", "7d"]:
        data = results.get(h_label, {})
        yt = data.get("y_true", [])
        yp = data.get("y_pred", [])
        if not yt:
            print(f"  {h_label:<10}  (no data)")
            continue
        m = compute_metrics(yt, yp)
        da_marker = " ✓" if m["da"] > 0.53 else " ✗"
        print(
            f"  {h_label:<10} {m['n']:>6} "
            f"{m['da']:>7.1%}{da_marker} "
            f"{m['naive_da']:>9.1%}  "
            f"{m['mae']:>9.5f}  "
            f"{m['naive_mae']:>9.5f}  "
            f"{m['rmse']:>9.5f}  "
            f"{m['corr']:>7.3f}"
        )
    print("=" * 70)
    print()
    print("  Legend:")
    print("    DA       = directional accuracy (% correct sign predictions)")
    print("    Naive DA = % of days market was flat (random-walk baseline)")
    print("    MAE      = mean absolute error on log-returns")
    print("    Naive MAE= MAE of always-predict-zero baseline")
    print("    ✓ / ✗    = DA above/below 53% acceptance threshold")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward prediction backtest")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=180, help="Evaluation window in calendar days")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\nBacktesting on: {args.tickers}")
    print(f"Evaluation window: last {args.days} calendar days")
    if args.verbose:
        print()

    results = run_backtest(args.tickers, lookback_days=args.days, verbose=args.verbose)
    print_report(results)


if __name__ == "__main__":
    main()
