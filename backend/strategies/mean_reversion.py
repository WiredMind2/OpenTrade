"""
Single-asset mean reversion (z-score vs rolling mean).

Fades extremes: long when price is sufficiently below the rolling mean; flat when
the z-score returns near zero. Multiple data feeds are handled independently.
"""

import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Type

import backtrader as bt
import numpy as np

from backend.strategies.base import BaseStrategy
from backend.strategies.bt_decision_markers import DecisionRecordingStrategy
from backend.strategies.support import capability_profile, param_float, param_int
from backend.domain.trading import TargetAllocation


class MeanReversionStrategy(BaseStrategy):
    """Z-score mean reversion vs rolling SMA and rolling std (Bollinger-style)."""

    def __init__(self):
        parameters_schema = {
            "short_window": param_int(
                10,
                "Shorter lookback (paired with long_window for engine preflight / lookback).",
            ),
            "long_window": param_int(
                30,
                "Rolling window for mean and volatility used in the z-score.",
            ),
            "entry_z": param_float(2.0, "Enter toward mean when z falls below -entry_z."),
            "exit_z": param_float(
                0.5,
                "Exit when |z| is below this (near the rolling mean) while in a position.",
            ),
            "max_position_pct": param_float(
                0.1, "Maximum position size as fraction of portfolio value."
            ),
        }

        super().__init__(
            name="mean_reversion",
            description="Single-asset mean reversion on z-score vs rolling mean",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, float]:
        params = parameters or {}
        short_window = max(2, int(params.get("short_window", 10)))
        long_window = max(short_window + 1, int(params.get("long_window", 30)))
        max_position_pct = min(max(float(params.get("max_position_pct", 0.1)), 0.0), 1.0)
        entry_z = max(float(params.get("entry_z", 2.0)), 0.5)
        exit_z = max(float(params.get("exit_z", 0.5)), 0.05)
        if exit_z >= entry_z:
            exit_z = max(entry_z * 0.25, 0.05)
        return {
            "short_window": float(short_window),
            "long_window": float(long_window),
            "max_position_pct": max_position_pct,
            "entry_z": entry_z,
            "exit_z": exit_z,
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=120)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class MeanReversionZScore(DecisionRecordingStrategy):
            params = (
                ("short_window", int(normalized["short_window"])),
                ("long_window", int(normalized["long_window"])),
                ("max_position_pct", normalized["max_position_pct"]),
                ("entry_z", normalized["entry_z"]),
                ("exit_z", normalized["exit_z"]),
            )

            def __init__(self):
                self.equity_curve = []
                self.trades = []
                self.rolling_means = [
                    bt.indicators.SMA(d.close, period=self.p.long_window) for d in self.datas
                ]
                self.rolling_stds = [
                    bt.indicators.StdDev(d.close, period=self.p.long_window) for d in self.datas
                ]

            def _z(self, i: int) -> float:
                data = self.datas[i]
                mean = float(self.rolling_means[i][0])
                std = float(self.rolling_stds[i][0])
                close = float(data.close[0])
                if std <= 1e-12 or np.isnan(std) or np.isnan(mean):
                    return float("nan")
                return (close - mean) / std

            def next(self):
                allocations: Dict[str, float] = {}

                for i, data in enumerate(self.datas):
                    ticker = data._name
                    z = self._z(i)
                    if z != z:  # NaN
                        continue
                    pos_size = self.getposition(data).size
                    if z < -self.p.entry_z:
                        allocations[ticker] = self.p.max_position_pct
                    elif pos_size != 0 and (
                        abs(z) < self.p.exit_z or z > self.p.entry_z
                    ):
                        allocations[ticker] = 0.0
                    elif pos_size != 0:
                        allocations[ticker] = self.p.max_position_pct
                    else:
                        allocations[ticker] = 0.0

                total_exposure = sum(abs(pct) for pct in allocations.values())
                if total_exposure > self.p.max_position_pct:
                    scale = self.p.max_position_pct / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                for ticker, target_pct in allocations.items():
                    data = None
                    for d in self.datas:
                        if hasattr(d, "_name") and d._name == ticker:
                            data = d
                            break
                    if data is None:
                        continue

                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    portfolio_value = self.broker.getvalue()
                    target_value = target_pct * portfolio_value

                    if abs(current_value - target_value) < 100:
                        continue

                    if target_value > current_value:
                        shares_to_buy = int((target_value - current_value) / data.close[0])
                        if shares_to_buy > 0:
                            self.buy(data=data, size=shares_to_buy)
                    elif target_value < current_value:
                        shares_to_sell = int((current_value - target_value) / data.close[0])
                        if shares_to_sell > 0:
                            self.sell(data=data, size=shares_to_sell)

                current_date = self.datas[0].datetime.date(0).isoformat()
                self.equity_curve.append(
                    {"date": current_date, "value": self.broker.getvalue()}
                )

            def notify_trade(self, trade):
                if trade.isclosed:
                    self.trades.append(
                        {
                            "size": trade.size,
                            "price": trade.price,
                            "value": trade.value,
                            "pnl": trade.pnl,
                            "pnlcomm": trade.pnlcomm,
                        }
                    )

        return MeanReversionZScore

    def generate_target_allocations(
        self,
        parameters: Dict[str, Any],
        symbols: List[str],
        as_of: datetime,
        current_prices: Dict[str, float],
        *,
        db_conn: Optional[sqlite3.Connection] = None,
    ) -> List[TargetAllocation]:
        from backend.main import app_state

        normalized = self._normalize_parameters(parameters)
        long_window = int(normalized["long_window"])
        max_position_pct = float(normalized["max_position_pct"])
        entry_z = float(normalized["entry_z"])
        exit_z = float(normalized["exit_z"])
        db_path = app_state.get("database_path") or "data/backtest.db"

        allocations: List[TargetAllocation] = []
        own_conn = db_conn is None
        conn = sqlite3.connect(db_path) if own_conn else db_conn
        try:
            cur = conn.cursor()
            for symbol in symbols:
                cur.execute(
                    """
                    SELECT close
                    FROM price_daily
                    WHERE ticker = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT ?
                    """,
                    (symbol.upper(), as_of.date().isoformat(), long_window + 1),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                if len(rows) < long_window + 1:
                    allocations.append(
                        TargetAllocation(
                            ticker=symbol.upper(),
                            target_pct=0.0,
                            reason="insufficient_history",
                            confidence=0.0,
                            timestamp=as_of,
                            metadata={"strategy": self.name},
                        )
                    )
                    continue
                closes = np.array(rows[::-1], dtype=float)
                window = closes[-long_window:]
                mean = float(np.mean(window))
                std = float(np.std(window, ddof=0))
                last = float(closes[-1])
                if std <= 1e-12:
                    allocations.append(
                        TargetAllocation(
                            ticker=symbol.upper(),
                            target_pct=0.0,
                            reason="zero_volatility",
                            confidence=0.0,
                            timestamp=as_of,
                            metadata={"strategy": self.name, "mean": mean},
                        )
                    )
                    continue
                z = (last - mean) / std

                if z < -entry_z:
                    target_pct = max_position_pct
                    reason = "mean_reversion_oversold"
                    confidence = min(1.0, (-z - entry_z) / max(entry_z, 1e-9))
                elif abs(z) < exit_z:
                    target_pct = 0.0
                    reason = "mean_reversion_flat"
                    confidence = min(1.0, 1.0 - abs(z) / max(exit_z, 1e-9))
                else:
                    continue

                allocations.append(
                    TargetAllocation(
                        ticker=symbol.upper(),
                        target_pct=target_pct,
                        reason=reason,
                        confidence=float(confidence),
                        timestamp=as_of,
                        metadata={
                            "strategy": self.name,
                            "z": z,
                            "mean": mean,
                            "std": std,
                            "current_price": float(
                                current_prices.get(symbol, 0.0) or 0.0
                            ),
                        },
                    )
                )
        finally:
            if own_conn:
                conn.close()
        return allocations

    def train(self, config: Dict[str, Any]) -> Any:
        raise NotImplementedError("Training not supported for rule-based strategies")

    def project_series(
        self,
        parameters: Dict[str, Any],
        anchor_time: datetime,
        anchor_price: float,
        projection_days: int = 30,
    ) -> List[Dict[str, Any]]:
        from backend.main import app_state

        try:
            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT close
                FROM price_daily
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 120
                """,
                (parameters.get("symbol", "AAPL"),),
            )
            closes = [row[0] for row in cur.fetchall()][::-1]
            conn.close()
        except Exception:
            closes = []

        if len(closes) > 1:
            returns = np.diff(closes) / closes[:-1]
            drift = float(np.mean(returns))
            vol = float(np.std(returns))
        else:
            drift = 0.0
            vol = 0.01

        points: List[Dict[str, Any]] = []
        price = anchor_price
        for day in range(projection_days):
            t = anchor_time + timedelta(days=day)
            damped = drift * 0.35
            oscillation = vol * 0.35 * np.sin(day / 3.0)
            next_return = damped + oscillation
            price = max(0.01, price * (1 + next_return))
            confidence = max(0.35, min(0.75, 0.72 - day * 0.012))
            band = abs(price * vol * 1.35)
            points.append(
                {
                    "time": t.isoformat(),
                    "price": round(price, 4),
                    "confidence": round(confidence, 4),
                    "upperBound": round(price + band, 4),
                    "lowerBound": round(max(0.01, price - band), 4),
                }
            )
        return points

    def project(
        self,
        parameters: Dict[str, Any],
        projection_days: int = 30,
        initial_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        getcontext().prec = 10

        try:
            from backend.main import app_state

            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)

            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=90)

            cur = conn.cursor()
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            portfolio_data: Dict[str, Any] = {}
            for ticker in tickers:
                cur.execute(
                    """
                    SELECT date, close, volume
                    FROM price_daily
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT 60
                    """,
                    (ticker, start_date.isoformat(), end_date.isoformat()),
                )
                rows = cur.fetchall()
                if rows:
                    closes = [row[1] for row in rows[::-1]]
                    if len(closes) > 1:
                        returns = np.diff(closes) / closes[:-1]
                        avg_return = float(np.mean(returns)) if returns.size > 0 else 0.0
                        volatility = (
                            float(np.std(returns)) if returns.size > 0 else 0.0
                        )
                        current_price = closes[-1]
                        portfolio_data[ticker] = {
                            "current_price": Decimal(str(current_price)),
                            "avg_daily_return": Decimal(str(avg_return)),
                            "volatility": Decimal(str(volatility)),
                            "weight": Decimal("1") / Decimal(str(len(tickers))),
                        }

            conn.close()

            if not portfolio_data:
                initial_capital_dec = Decimal(str(initial_capital))
                projected_return_dec = Decimal("0.015")
                projected_final_value = initial_capital_dec * (
                    Decimal("1") + projected_return_dec
                )
                return {
                    "projected_return": float(projected_return_dec),
                    "projected_volatility": 0.14,
                    "confidence": 0.45,
                    "projection_days": projection_days,
                    "initial_capital": float(initial_capital_dec),
                    "projected_final_value": float(
                        projected_final_value.quantize(Decimal("0.01"))
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            portfolio_return = sum(
                data["avg_daily_return"] * data["weight"]
                for data in portfolio_data.values()
            )
            portfolio_volatility = sum(
                data["volatility"] * data["weight"] for data in portfolio_data.values()
            )

            projection_days_dec = Decimal(str(projection_days))
            initial_capital_dec = Decimal(str(initial_capital))
            total_return = portfolio_return * projection_days_dec
            strategy_multiplier = Decimal("0.45")
            adjusted_return = total_return * strategy_multiplier
            adjusted_volatility = float(portfolio_volatility * Decimal("0.85"))
            projected_final_value = initial_capital_dec * (
                Decimal("1") + adjusted_return
            )

            return {
                "projected_return": float(adjusted_return),
                "projected_volatility": round(adjusted_volatility, 6),
                "confidence": 0.62,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(
                    projected_final_value.quantize(Decimal("0.01"))
                ),
                "market_return": float(total_return),
                "strategy_multiplier": float(strategy_multiplier),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            initial_capital_dec = Decimal(str(initial_capital))
            fallback_return = Decimal("0.01")
            projected_final_value = initial_capital_dec * (
                Decimal("1") + fallback_return
            )
            return {
                "projected_return": float(fallback_return),
                "projected_volatility": 0.12,
                "confidence": 0.28,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(
                    projected_final_value.quantize(Decimal("0.01"))
                ),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
