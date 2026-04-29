"""
Run walk-forward XGBoost intraday strategy on OHLCV data.
"""

from __future__ import annotations

import argparse
import json

from backend.ml.forecasting.datasource import DataSource
from backend.ml.forecasting.xgb_intraday_strategy import XGBIntradayConfig, XGBIntradayStrategyRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="SQLite path containing price_daily")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--long-threshold", type=float, default=0.0008)
    parser.add_argument("--short-threshold", type=float, default=0.0008)
    args = parser.parse_args()

    ds = DataSource(args.db)
    df = ds.load_ohlcv(args.ticker)
    cfg = XGBIntradayConfig(
        horizon_bars=args.horizon_bars,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )
    runner = XGBIntradayStrategyRunner(cfg)
    out = runner.run(df)

    payload = {
        "forecast_metrics": out["forecast_metrics"],
        "trading_metrics": out["trading_metrics"],
        "feature_columns": out["feature_columns"],
        "predictions_rows": len(out["predictions"]),
        "trades_rows": len(out["trades"]),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
