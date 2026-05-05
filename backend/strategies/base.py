"""
Base strategy interface for the trading backtesting system.

This module defines the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, Any, Type, Literal, List, Optional

import backtrader as bt

from backend.domain.trading import ForecastOutput, TargetAllocation


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str, description: str, type: Literal['rule', 'ml'],
                 parameters_schema: Dict[str, Any], can_train: bool):
        self.name = name
        self.description = description
        self.type = type
        self.parameters_schema = parameters_schema
        self.can_train = can_train

    def get_capability_profile(self) -> Dict[str, Any]:
        """Strategy execution contract used by generic preflight/optimizer flows."""
        return {
            "requires_predictions": False,
            "required_prediction_horizons": [],
            "supports_signal_execution": True,
            "supports_backtrader_execution": True,
            "preferred_optimizer_engine": "backtrader",
            "min_history_bars": 30,
            "supported_objectives": ["balanced", "sharpe", "return", "drawdown"],
        }

    def preferred_optimizer_engine(self) -> str:
        """
        Canonical optimizer engine selector.

        Strategies can override when they still need the legacy signal simulator.
        """
        return "backtrader"

    @abstractmethod
    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        """Create and return a Backtrader strategy class with the given parameters."""
        pass

    def train(self, config: Dict[str, Any]) -> Any:
        """Train the strategy (for ML strategies). Raises NotImplementedError for rule strategies."""
        raise NotImplementedError("Training not supported for rule-based strategies")

    @abstractmethod
    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Project future performance of the strategy.

        Args:
            parameters: Strategy-specific parameters
            projection_days: Number of days to project forward
            initial_capital: Starting capital for projection

        Returns:
            Dict containing projected performance metrics
        """
        pass

    def project_series(
        self,
        parameters: Dict[str, Any],
        anchor_time: datetime,
        anchor_price: float,
        projection_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Project a time series of prices from an anchor point.

        The default implementation adapts summary output from ``project()`` to a
        deterministic daily path. Strategies can override this with richer logic.
        """
        summary = self.project(
            parameters=parameters,
            projection_days=projection_days,
            initial_capital=anchor_price,
        )
        total_return = float(summary.get("projected_return", 0.0))
        confidence = float(summary.get("confidence", 0.5))
        daily_return = total_return / max(projection_days, 1)

        points: List[Dict[str, Any]] = []
        price = anchor_price
        for day in range(projection_days):
            current_time = anchor_time + timedelta(days=day)
            price = max(0.01, price * (1 + daily_return))
            band_width = abs(price * (1 - confidence) * 0.2)
            points.append(
                {
                    "time": current_time.isoformat(),
                    "price": round(price, 4),
                    "confidence": confidence,
                    "upperBound": round(price + band_width, 4),
                    "lowerBound": round(max(0.01, price - band_width), 4),
                }
            )

        return points

    def forecast(
        self,
        parameters: Dict[str, Any],
        symbol: str,
        as_of: datetime,
        current_price: float,
        horizon_days: int = 5,
    ) -> ForecastOutput:
        """Generate a forecast artifact for signal generation."""
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if current_price <= 0:
            raise ValueError("current_price must be positive")
        points = self.project_series(
            parameters={**(parameters or {}), "symbol": symbol},
            anchor_time=as_of,
            anchor_price=current_price,
            projection_days=horizon_days,
        )
        if points:
            final_price = float(points[-1].get("price", current_price))
            confidence = float(points[-1].get("confidence", 0.5))
        else:
            final_price = current_price
            confidence = 0.5
        predicted_return = 0.0 if current_price == 0 else (final_price - current_price) / current_price
        confidence = max(0.0, min(1.0, confidence))
        return ForecastOutput(
            symbol=symbol.upper(),
            horizon_days=horizon_days,
            predicted_return=predicted_return,
            confidence=confidence,
            predicted_path=points,
            metadata={"strategy": self.name},
        )

    def generate_target_allocations(
        self,
        parameters: Dict[str, Any],
        symbols: List[str],
        as_of: datetime,
        current_prices: Dict[str, float],
        *,
        db_conn: Optional[sqlite3.Connection] = None,
    ) -> List[TargetAllocation]:
        """Convert forecast outputs to executable target allocations."""
        _ = db_conn  # optional shared SQLite connection (used by signal-mode backtest overrides)
        params = parameters or {}
        threshold = max(float(params.get("prediction_threshold", 0.002)), 0.0)
        max_position_pct = min(max(float(params.get("max_position_pct", 0.1)), 0.0), 1.0)
        horizon_days = int(params.get("forecast_horizon_days", 5))

        allocations: List[TargetAllocation] = []
        for symbol in symbols:
            price = float(current_prices.get(symbol, 0.0) or 0.0)
            if price <= 0:
                continue
            forecast = self.forecast(
                parameters=params,
                symbol=symbol,
                as_of=as_of,
                current_price=price,
                horizon_days=horizon_days,
            )
            if abs(forecast.predicted_return) < threshold:
                target_pct = 0.0
                reason = "below_threshold"
            else:
                direction = 1.0 if forecast.predicted_return > 0 else -1.0
                scaled = min(abs(forecast.predicted_return) / max(threshold, 1e-9), 1.0)
                target_pct = direction * max_position_pct * scaled
                reason = "forecast_signal"
            allocations.append(
                TargetAllocation(
                    ticker=symbol.upper(),
                    target_pct=max(-max_position_pct, min(max_position_pct, target_pct)),
                    reason=reason,
                    confidence=forecast.confidence,
                    timestamp=as_of,
                    metadata={"predicted_return": forecast.predicted_return, "strategy": self.name},
                )
            )
        return allocations