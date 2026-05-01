"""
Domain models for forecast, signal, and execution layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal


@dataclass
class ForecastOutput:
    symbol: str
    horizon_days: int
    predicted_return: float
    confidence: float
    predicted_path: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetAllocation:
    ticker: str
    target_pct: float
    reason: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderIntent:
    ticker: str
    side: Literal["buy", "sell"]
    notional_delta: float
    reason: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSnapshot:
    timestamp: datetime
    cash: float
    market_value: float
    total_value: float
    positions: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    min_trade_notional: float = 100.0
    commission_per_share: float = 0.005
    slippage_bps: float = 0.0
    max_gross_exposure: float = 1.0
    rebalance_frequency: Literal["daily", "weekly"] = "daily"


@dataclass
class SignalBatch:
    strategy_name: str
    as_of: datetime
    allocations: List[TargetAllocation] = field(default_factory=list)


@dataclass
class ExecutionReport:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

