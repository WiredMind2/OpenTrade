"""
Concrete standalone recursive forecast strategy.

This strategy exposes the shared RecursiveForecastStrategy runtime as a
first-class strategy in the registry and frontend.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Any, Dict

import numpy as np
from backend.strategies.recursive_forecast import RecursiveForecastStrategy


class RecursiveForecastStandaloneStrategy(RecursiveForecastStrategy):
    def __init__(self):
        parameters_schema = {
            "prediction_threshold": {
                "type": "float",
                "default": 0.002,
                "description": "Minimum absolute forecast return to open a position",
            },
            "max_position_pct": {
                "type": "float",
                "default": 0.10,
                "description": "Maximum absolute portfolio exposure",
            },
            "forecast_horizon_days": {
                "type": "int",
                "default": 5,
                "description": "Recursive horizon used for path-aware signals",
            },
        }
        super().__init__(
            name="recursive_forecast",
            description="Standalone recursive multi-step forecasting strategy",
            type="ml",
            parameters_schema=parameters_schema,
            can_train=False,
        )
        self.model_name = "recursive_forecast"

    def project(
        self,
        parameters: Dict[str, Any],
        projection_days: int = 30,
        initial_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        getcontext().prec = 10
        ticker = str(parameters.get("symbol", "AAPL")).upper()
        horizon_days = max(1, int(parameters.get("forecast_horizon_days", 5)))
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=max(30, projection_days))

        predicted_returns = []
        confidences = []
        try:
            from backend.main import app_state
            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT predicted_return, COALESCE(predicted_confidence, confidence, 0.5)
                FROM sentiment_predictions
                WHERE ticker = ? AND horizon = ? AND produced_at >= ? AND produced_at <= ?
                ORDER BY produced_at DESC
                LIMIT 100
                """,
                (ticker, f"{min(7, horizon_days)}d", start_date.isoformat(), end_date.isoformat()),
            )
            rows = cur.fetchall()
            conn.close()
            predicted_returns = [float(r[0]) for r in rows if r[0] is not None]
            confidences = [float(r[1]) for r in rows if r[1] is not None]
        except Exception:
            predicted_returns = []
            confidences = []

        if not predicted_returns:
            predicted_returns = [0.0005] * min(projection_days, 10)
        if not confidences:
            confidences = [0.55]

        avg_predicted_return = float(np.mean(predicted_returns))
        avg_confidence = float(np.mean(confidences))
        return_volatility = float(np.std(predicted_returns)) if len(predicted_returns) > 1 else 0.015

        initial_capital_dec = Decimal(str(initial_capital))
        projection_days_dec = Decimal(str(projection_days))
        avg_predicted_return_dec = Decimal(str(avg_predicted_return))
        avg_confidence_dec = Decimal(str(avg_confidence))
        total_return = avg_predicted_return_dec * projection_days_dec * avg_confidence_dec
        projected_final_value = initial_capital_dec * (Decimal("1") + total_return)
        projected_volatility = return_volatility * np.sqrt(max(1, projection_days))

        return {
            "projected_return": float(total_return),
            "projected_volatility": round(float(projected_volatility), 6),
            "confidence": round(avg_confidence, 4),
            "projection_days": projection_days,
            "initial_capital": float(initial_capital_dec),
            "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
            "avg_predicted_return": float(avg_predicted_return_dec),
            "predictions_used": len(predicted_returns),
            "model_version": self.model_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
