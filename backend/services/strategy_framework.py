from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, List
import itertools
import random
import sqlite3

import pandas as pd

# Strategies that support synchronous signal-engine parameter search via ``/strategies/{name}/train``.
SIGNAL_PARAMETER_TRAINABLE_STRATEGIES: FrozenSet[str] = frozenset(
    {
        "moving_average",
        "mean_reversion",
        "ts_momentum",
        "pairs_trading",
        "cross_sectional_ls",
        "macd",
        "rl_directional",
        "rl_portfolio_allocator",
        "volatility_targeting",
    }
)


def strategy_supports_signal_parameter_training(strategy_name: str) -> bool:
    return strategy_name in SIGNAL_PARAMETER_TRAINABLE_STRATEGIES


@dataclass
class PreflightIssue:
    code: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    ready: bool
    issues: List[PreflightIssue] = field(default_factory=list)
    warnings: List[PreflightIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message, "details": i.details}
                for i in self.issues
            ],
            "warnings": [
                {"code": i.code, "severity": i.severity, "message": i.message, "details": i.details}
                for i in self.warnings
            ],
            "suggestions": self.suggestions,
            "diagnostics": self.diagnostics,
        }


class StrategyPreflightService:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def evaluate(
        self,
        strategy_name: str,
        strategy: Any,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> PreflightResult:
        profile = strategy.get_capability_profile() if hasattr(strategy, "get_capability_profile") else {}
        min_history_bars = int(profile.get("min_history_bars", 30) or 30)
        requires_predictions = bool(profile.get("requires_predictions", False))
        required_horizons = list(profile.get("required_prediction_horizons", []))

        issues: List[PreflightIssue] = []
        warnings: List[PreflightIssue] = []
        suggestions: List[str] = []
        diagnostics: Dict[str, Any] = {"strategy": strategy_name, "ticker": ticker.upper()}

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            price_count = int(
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM price_daily
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    """,
                    (ticker.upper(), start_date.date().isoformat(), end_date.date().isoformat()),
                ).fetchone()[0]
                or 0
            )
            diagnostics["price_bars"] = price_count
            if price_count < min_history_bars:
                issues.append(
                    PreflightIssue(
                        code="INSUFFICIENT_HISTORY",
                        severity="error",
                        message=f"Insufficient price history for {ticker.upper()} ({price_count} bars, required {min_history_bars}).",
                        details={"required": min_history_bars, "available": price_count},
                    )
                )

            if requires_predictions:
                pred_total = int(
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM trading_model_predictions
                        WHERE ticker = ? AND dt >= ? AND dt <= ?
                        """,
                        (ticker.upper(), start_date.date().isoformat(), end_date.date().isoformat()),
                    ).fetchone()[0]
                    or 0
                )
                diagnostics["prediction_rows"] = pred_total
                if pred_total <= 0:
                    issues.append(
                        PreflightIssue(
                            code="PREDICTION_GAP",
                            severity="error",
                            message=(
                                f"No predictions found for {ticker.upper()} in requested period. "
                                "Run ML prediction pipeline first."
                            ),
                            details={"required_horizons": required_horizons},
                        )
                    )
                    suggestions.append("Run prediction generation pipeline before backtesting/training this strategy.")
                elif pred_total < max(20, min_history_bars // 2):
                    warnings.append(
                        PreflightIssue(
                            code="LOW_PREDICTION_COVERAGE",
                            severity="warning",
                            message=f"Prediction coverage is low ({pred_total} rows).",
                            details={"rows": pred_total},
                        )
                    )
        finally:
            conn.close()

        return PreflightResult(
            ready=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            diagnostics=diagnostics,
        )


class StrategyOptimizerEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def candidate_grid(self, strategy_name: str) -> List[Dict[str, Any]]:
        if strategy_name == "moving_average":
            return [
                {"short_window": s, "long_window": l, "max_position_pct": p}
                for s, l, p in itertools.product([5, 10, 15, 20], [30, 50, 80, 120], [0.05, 0.08, 0.1])
                if s < l
            ]
        if strategy_name == "mean_reversion":
            out: List[Dict[str, Any]] = []
            for s, l, ez, xz, p in itertools.product(
                [5, 10, 15],
                [25, 40, 55],
                [1.5, 2.0, 2.5],
                [0.25, 0.4, 0.55],
                [0.08, 0.1],
            ):
                if s >= l or xz >= ez:
                    continue
                out.append(
                    {
                        "short_window": s,
                        "long_window": l,
                        "entry_z": float(ez),
                        "exit_z": float(xz),
                        "max_position_pct": float(p),
                    }
                )
            return out
        if strategy_name == "ts_momentum":
            return [
                {"short_window": s, "long_window": l, "max_position_pct": p}
                for s, l, p in itertools.product([3, 8, 12], [20, 35, 50], [0.06, 0.1])
                if s < l
            ]
        if strategy_name == "macd":
            return [
                {
                    "macd_fast": fast,
                    "macd_slow": slow,
                    "macd_signal": sig,
                    "ema_period": ema,
                    "risk_pct": risk,
                }
                for fast, slow, sig, ema, risk in itertools.product(
                    [8, 12, 16],
                    [21, 26, 35],
                    [7, 9],
                    [100, 200],
                    [0.005, 0.01],
                )
                if fast < slow
            ]
        if strategy_name == "pairs_trading":
            out = []
            for s, l, ze, zx, p in itertools.product(
                [8, 15],
                [40, 70],
                [1.5, 2.0, 2.5],
                [0.35, 0.5],
                [0.08, 0.1],
            ):
                if s >= l or zx >= ze:
                    continue
                out.append(
                    {
                        "short_window": s,
                        "long_window": l,
                        "z_entry": float(ze),
                        "z_exit": float(zx),
                        "max_position_pct": float(p),
                        "hedge_mode": "ols_log",
                    }
                )
            return out
        if strategy_name == "cross_sectional_ls":
            return [
                {
                    "short_window": s,
                    "long_window": l,
                    "top_frac": tf,
                    "bottom_frac": bf,
                    "max_gross_exposure": g,
                    "max_position_pct": mp,
                    "single_name_use_momentum": False,
                }
                for s, l, tf, bf, g, mp in itertools.product(
                    [5, 10],
                    [28, 45],
                    [0.1, 0.2],
                    [0.1, 0.2],
                    [0.8, 1.0],
                    [0.08, 0.1],
                )
                if s < l
            ]
        if strategy_name == "rl_portfolio_allocator":
            out = []
            for s, l, mg, temp in itertools.product(
                [5, 10],
                [25, 45],
                [0.8, 1.0],
                [0.7, 1.2],
            ):
                if s >= l:
                    continue
                lb = max(l + 5, 35)
                out.append(
                    {
                        "short_window": s,
                        "long_window": l,
                        "lookbacks": lb,
                        "max_gross": float(mg),
                        "temperature": float(temp),
                    }
                )
            return out
        if strategy_name == "rl_directional":
            out = []
            for s, l, mp, eps, lr in itertools.product(
                [3, 5, 8],
                [20, 35, 50],
                [0.05, 0.1],
                [0.05, 0.12, 0.2],
                [0.1, 0.2],
            ):
                if s >= l:
                    continue
                out.append(
                    {
                        "short_window": s,
                        "long_window": l,
                        "max_position_pct": float(mp),
                        "epsilon": float(eps),
                        "learning_rate": float(lr),
                        "gamma": 0.99,
                        "n_return_bins": 5,
                        "return_clip": 0.05,
                        "online_q_updates": True,
                    }
                )
            return out
        if strategy_name == "volatility_targeting":
            out = []
            for vol_lb, mom_lb, target_vol, mp in itertools.product(
                [10, 20, 30],
                [10, 20, 40],
                [0.10, 0.15, 0.20],
                [0.05, 0.1],
            ):
                out.append(
                    {
                        "short_window": max(5, min(vol_lb, mom_lb)),
                        "long_window": max(vol_lb, mom_lb) + 40,
                        "target_ann_vol": float(target_vol),
                        "vol_lookback": int(vol_lb),
                        "momentum_lookback": int(mom_lb),
                        "max_position_pct": float(mp),
                    }
                )
            return out
        return []

    def build_candidates(
        self,
        strategy_name: str,
        optimizer_mode: str,
        max_evals: int,
        *,
        random_seed: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return up to ``max_evals`` parameter dicts to evaluate.

        ``optimizer_mode``:
        - ``grid``: deterministic order from the full candidate grid.
        - ``random``: shuffled subset of the same grid (reproducible if ``random_seed`` is set).
        """
        grid = self.candidate_grid(strategy_name)
        if not grid or max_evals <= 0:
            return []
        mode = (optimizer_mode or "grid").strip().lower()
        if mode == "random":
            rng = random.Random(random_seed)
            order = grid[:]
            rng.shuffle(order)
            return order[:max_evals]
        return grid[:max_evals]

    @staticmethod
    def score(metrics: Dict[str, float], objective: str) -> float:
        sharpe = float(metrics.get("sharpe_ratio", 0.0) or 0.0)
        total_return = float(metrics.get("total_return", 0.0) or 0.0)
        max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
        if objective == "sharpe":
            return sharpe
        if objective == "return":
            return total_return
        if objective == "drawdown":
            return -max_drawdown
        return (sharpe * 10.0) + (total_return * 2.0) - max_drawdown

    def load_price_frame(self, ticker: str, start_date: datetime, end_date: datetime, lookback_days: int = 365):
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                """
                SELECT date, open, high, low, close, volume
                FROM price_daily
                WHERE ticker = ? AND date >= date(?, ? || ' days') AND date <= ?
                ORDER BY date ASC
                """,
                conn,
                params=[
                    ticker.upper(),
                    start_date.date().isoformat(),
                    f"-{lookback_days}",
                    end_date.date().isoformat(),
                ],
            )
        finally:
            conn.close()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date")
