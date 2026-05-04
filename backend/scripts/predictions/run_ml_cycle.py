"""
Weekly ML retraining cycle: retrain LightGBM models + refit calibrators.

Steps:
  1. Retrain lightgbm_1d / 3d / 7d from price_daily (bootstrap_train)
  2. Refit directional calibrators from a walk-forward backtest (calibrate_models)
  3. Prune old model files, keeping the 2 most recent per horizon
  4. Log the run to ml_run_log

Usage:
    python -m backend.scripts.predictions.run_ml_cycle [--db data/backtest.db] [--outdir models]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from backend.logging_config import get_component_logger
from backend.scripts.bootstrap_train import build_dataset, main as _bootstrap_main
from backend.ml.training_pipeline import train_horizon_model

logger = get_component_logger(__file__)

DEFAULT_DB = "data/backtest.db"
DEFAULT_OUTDIR = "models"
KEEP_VERSIONS = 2  # number of recent model files to keep per horizon


def _prune_old_models(outdir: Path, keep: int = KEEP_VERSIONS) -> list[str]:
    """Delete old model + metadata files, keeping the `keep` most recent per horizon."""
    pruned = []
    for horizon in ("1d", "3d", "7d"):
        versions = sorted(outdir.glob(f"lightgbm_{horizon}_*.joblib"), reverse=True)
        for old in versions[keep:]:
            old.unlink(missing_ok=True)
            meta = old.with_suffix("").with_suffix(".metadata.json")
            meta.unlink(missing_ok=True)
            pruned.append(old.name)
            logger.info("Pruned old model: %s", old.name)
    return pruned


def _log_run(
    conn: sqlite3.Connection,
    status: str,
    metrics: dict,
    started_at: datetime,
) -> None:
    conn.execute(
        "INSERT INTO ml_run_log(run_type, status, metadata, started_at, finished_at) VALUES (?,?,?,?,?)",
        (
            "retrain",
            status,
            json.dumps(metrics),
            started_at.isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def run_cycle(db_path: str = DEFAULT_DB, outdir: str = DEFAULT_OUTDIR) -> dict:
    outdir_path = Path(outdir)
    started_at = datetime.now(timezone.utc)
    logger.info("=== ML retraining cycle started ===")

    # ── Step 1: retrain models ────────────────────────────────────────────────
    logger.info("Building dataset from %s ...", db_path)
    df = build_dataset(db_path)
    logger.info("Dataset: %d rows, %d tickers", len(df), df["ticker"].nunique())

    horizon_metrics = {}
    for horizon in ("1d", "3d", "7d"):
        artifact = train_horizon_model(df, horizon=horizon, outdir=outdir)
        horizon_metrics[horizon] = artifact.metrics
        logger.info("[%s] rmse=%.5f  mae=%.5f  -> %s", horizon, artifact.metrics["rmse"], artifact.metrics["mae"], artifact.model_path)

    # ── Step 2: refit calibrators ─────────────────────────────────────────────
    logger.info("Refitting calibrators ...")
    from backend.scripts.calibrate_models import calibration_path
    from backend.scripts.backtest_predictions import run_backtest, DEFAULT_TICKERS
    from backend.ml.calibration import DirectionalCalibrator
    import numpy as np

    results = run_backtest(DEFAULT_TICKERS, lookback_days=365)
    calibration_metrics = {}
    for h_label in ("1d", "3d", "7d"):
        data = results.get(h_label, {})
        y_true = np.asarray(data.get("y_true", []))
        y_pred = np.asarray(data.get("y_pred", []))
        if len(y_true) < 30:
            logger.warning("Not enough calibration data for %s, skipping", h_label)
            continue
        cal = DirectionalCalibrator()
        cal.fit(y_pred, y_true)
        cal.save(calibration_path(h_label))
        da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        calibration_metrics[h_label] = {"da": da, "n": len(y_true)}
        logger.info("[%s] calibrated  da=%.1f%%  n=%d", h_label, da * 100, len(y_true))

    # ── Step 3: prune old model files ─────────────────────────────────────────
    pruned = _prune_old_models(outdir_path)

    summary = {
        "horizon_metrics": horizon_metrics,
        "calibration_metrics": calibration_metrics,
        "pruned_models": pruned,
        "dataset_rows": len(df),
        "tickers": df["ticker"].nunique(),
    }

    # ── Step 4: log to database ───────────────────────────────────────────────
    with sqlite3.connect(db_path) as conn:
        _log_run(conn, "completed", summary, started_at)

    logger.info("=== ML retraining cycle completed ===")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly ML retraining cycle")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    summary = run_cycle(db_path=args.db, outdir=args.outdir)

    print("\n── Cycle summary ──────────────────────────────")
    for h, m in summary["horizon_metrics"].items():
        da = summary["calibration_metrics"].get(h, {}).get("da")
        da_str = f"  DA={da:.1%}" if da else ""
        print(f"  {h}  rmse={m['rmse']:.5f}  mae={m['mae']:.5f}{da_str}")
    if summary["pruned_models"]:
        print(f"  Pruned: {', '.join(summary['pruned_models'])}")
    print(f"  Dataset: {summary['dataset_rows']} rows / {summary['tickers']} tickers")
    print()


if __name__ == "__main__":
    main()
