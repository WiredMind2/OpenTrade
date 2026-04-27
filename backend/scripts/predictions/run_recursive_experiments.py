"""
Run recursive forecasting ablation experiments.
"""

import argparse
import json
from dataclasses import asdict

from backend.ml.forecasting.contracts import ForecastConfig, TargetMode
from backend.ml.forecasting.datasource import DataSource
from backend.ml.forecasting.runner import WalkForwardRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ds = DataSource(args.db)
    df = ds.load_ohlcv(args.ticker)
    configs = [
        ForecastConfig(target_mode=TargetMode.log_return_1, model_name="ridge", horizon=5),
        ForecastConfig(target_mode=TargetMode.return_1, model_name="ridge", horizon=5),
        ForecastConfig(target_mode=TargetMode.log_return_1, model_name="lightgbm", horizon=5),
    ]
    results = []
    for cfg in configs:
        out = WalkForwardRunner(cfg).run(df)
        results.append(
            {
                "config": asdict(cfg),
                "metrics": [asdict(m) for m in out["metrics"]],
            }
        )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
