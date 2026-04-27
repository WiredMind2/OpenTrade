"""
Base strategy interface for the trading backtesting system.

This module defines the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Type, Literal, List
import backtrader as bt


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, name: str, description: str, type: Literal['rule', 'ml'],
                 parameters_schema: Dict[str, Any], can_train: bool):
        self.name = name
        self.description = description
        self.type = type
        self.parameters_schema = parameters_schema
        self.can_train = can_train

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