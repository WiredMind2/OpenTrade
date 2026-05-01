from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import itertools
import sqlite3

import pandas as pd


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
        if strategy_name == "recursive_forecast":
            return [
                {"prediction_threshold": t, "forecast_horizon_days": h, "max_position_pct": p}
                for t, h, p in itertools.product([0.0005, 0.001, 0.002, 0.003], [1, 3, 5, 7], [0.05, 0.08, 0.1])
            ]
        return []

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
