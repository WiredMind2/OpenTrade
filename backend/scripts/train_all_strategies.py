"""
Run signal-parameter optimization (training) for every catalog strategy that supports it,
for a single ticker and date range.

Usage:
  python -m backend.scripts.train_all_strategies --ticker AAPL --start-date 2024-01-01 --end-date 2024-12-31
  python -m backend.scripts.train_all_strategies --db data/backtest.db --ticker MSFT --start-date 2023-01-01 --end-date 2023-12-31 --pair-ticker GOOGL

``pairs_trading`` is skipped unless ``--pair-ticker`` is set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import HTTPException

from backend.config import get_config
from backend.routes.strategies import _optimize_signal_strategy
from backend.scripts.script_logger import logger
from backend.services.strategy_framework import SIGNAL_PARAMETER_TRAINABLE_STRATEGIES
from backend.strategies import strategy_registry


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def run_batch(
    db_path: str,
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    *,
    initial_capital: float,
    objective: str,
    max_evals: int,
    optimizer_mode: str,
    random_seed: Optional[int],
    pair_ticker: Optional[str],
    universe_limit: int,
    stop_on_error: bool,
) -> Tuple[int, List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Returns (exit_code, successes, failures) where failures is list of (strategy, detail).
    """
    sym = ticker.strip().upper()
    successes: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    ordered = sorted(SIGNAL_PARAMETER_TRAINABLE_STRATEGIES)
    planned: List[str] = []
    for name in ordered:
        st = strategy_registry.get(name)
        if st is None or not getattr(st, "catalog_visible", True):
            logger.info("Skipping %s: not registered or not catalog-visible", name)
            continue
        if name == "pairs_trading" and not (pair_ticker or "").strip():
            logger.warning(
                "Skipping pairs_trading: provide --pair-ticker (second leg) to include it in the batch.",
            )
            continue
        planned.append(name)

    print(
        json.dumps(
            {"batch_plan": True, "strategies": planned, "total": len(planned), "ticker": sym},
        ),
        flush=True,
    )

    for name in planned:
        st = strategy_registry.get(name)
        if st is None:
            continue
        logger.info(
            "=== Training %s on %s (%s → %s) ===",
            name,
            sym,
            start_date.date().isoformat(),
            end_date.date().isoformat(),
        )
        try:
            result = _optimize_signal_strategy(
                strategy=st,
                strategy_name=name,
                db_path=db_path,
                ticker=sym,
                start_date=start_date,
                end_date=end_date,
                initial_capital=float(initial_capital),
                objective=objective,
                max_evals=int(max_evals),
                optimizer_mode=optimizer_mode,
                random_seed=random_seed,
                pair_ticker=(pair_ticker.strip().upper() if pair_ticker else None),
                universe_limit=int(universe_limit),
            )
            successes.append(result)
            print(
                json.dumps({"strategy": name, "status": "ok", "best_metrics": result.get("best_metrics")}),
                flush=True,
            )
        except HTTPException as e:
            detail = str(e.detail) if e.detail is not None else str(e)
            failures.append((name, detail))
            logger.error("Training failed for %s: %s", name, detail)
            print(json.dumps({"strategy": name, "status": "error", "detail": detail}), flush=True)
            if stop_on_error:
                return 1, successes, failures

    if successes and not failures:
        return 0, successes, failures
    if successes and failures:
        return 0, successes, failures
    if not successes and failures:
        return 1, successes, failures
    return 0, successes, failures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Batch signal-parameter training for all supported strategies.")
    parser.add_argument("--db", default=None, help="SQLite database path (default: config database.path)")
    parser.add_argument("--ticker", required=True, help="Primary ticker symbol")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument(
        "--objective",
        default="balanced",
        choices=("sharpe", "return", "drawdown", "balanced"),
    )
    parser.add_argument("--max-evals", type=int, default=8)
    parser.add_argument("--optimizer-mode", default="grid", choices=("grid", "random"))
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--pair-ticker", default=None, help="Second leg for pairs_trading (optional)")
    parser.add_argument("--universe-limit", type=int, default=8)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch on the first failed strategy (default: continue).",
    )
    args = parser.parse_args(argv)

    db_path = args.db or get_config().database.path
    db_path = os.path.abspath(db_path)

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if end_date <= start_date:
        logger.error("end-date must be after start-date")
        return 1

    code, ok, bad = run_batch(
        db_path,
        args.ticker,
        start_date,
        end_date,
        initial_capital=args.initial_capital,
        objective=args.objective,
        max_evals=max(1, min(int(args.max_evals), 50)),
        optimizer_mode=args.optimizer_mode,
        random_seed=args.random_seed,
        pair_ticker=args.pair_ticker,
        universe_limit=max(2, min(int(args.universe_limit), 15)),
        stop_on_error=args.stop_on_error,
    )
    logger.info(
        "Batch finished: %d succeeded, %d failed, exit=%s",
        len(ok),
        len(bad),
        code,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
