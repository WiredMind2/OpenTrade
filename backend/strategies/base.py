"""
Base strategy interface for the trading backtesting system.

This module defines the abstract base class that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Literal
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