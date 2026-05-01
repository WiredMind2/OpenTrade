"""
Fit and save directional confidence calibrators for each horizon.

Runs a walk-forward backtest to collect (predicted_return, actual_return) pairs,
fits a DirectionalCalibrator per horizon, and saves the result next to each model.

Calibration files: models/lightgbm_{h}.calibration.json

Usage:
    python -m backend.scripts.calibrate_models [--days 365] [--tickers AAPL MSFT ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.ml.calibration import DirectionalCalibrator
from backend.scripts.backtest_predictions import (
    DEFAULT_TICKERS,
    HORIZONS,
    MODEL_DIR,
    run_backtest,
)


def calibration_path(horizon: str) -> Path:
    return MODEL_DIR / f"lightgbm_{horizon}.calibration.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="Calibration window in calendar days")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    print(f"Collecting backtest data over {args.days} days on {args.tickers}...")
    results = run_backtest(args.tickers, lookback_days=args.days)

    print()
    print(f"{'Horizon':<10} {'N':>6}  {'Raw DA':>8}  {'Calib a':>10}  {'Calib b':>10}  Saved")
    print("-" * 60)

    for h_label in ["1d", "3d", "7d"]:
        data = results.get(h_label, {})
        y_true = np.asarray(data.get("y_true", []))
        y_pred = np.asarray(data.get("y_pred", []))

        if len(y_true) < 30:
            print(f"{h_label:<10} {'—':>6}  (not enough data, skipping)")
            continue

        raw_da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        cal = DirectionalCalibrator()
        cal.fit(y_pred, y_true)

        path = calibration_path(h_label)
        cal.save(path)

        # Sanity check: calibrated confidence on typical predictions
        typical = float(np.mean(np.abs(y_pred)))
        conf_typical = cal.confidence(typical)

        print(
            f"{h_label:<10} {len(y_true):>6}  {raw_da:>7.1%}  "
            f"{cal.a:>10.4f}  {cal.b:>10.4f}  -> {path.name}"
        )
        print(f"           Typical |pred|={typical:.5f} → calibrated confidence={conf_typical:.1%}")

    print()
    print("Done. Calibration files written to models/")


if __name__ == "__main__":
    main()
