"""
Moving Average Crossover Strategy

This module implements a simple moving average crossover trading strategy.
"""

import sqlite3
from datetime import datetime
from typing import Dict, Any, Type, List
import backtrader as bt
import numpy as np

from backend.strategies.base import BaseStrategy
from backend.domain.trading import TargetAllocation


class MovingAverageStrategy(BaseStrategy):
    """Moving average crossover strategy implementation."""

    def __init__(self):
        parameters_schema = {
            'short_window': {
                'type': 'int',
                'default': 10,
                'description': 'Short moving average window in days'
            },
            'long_window': {
                'type': 'int',
                'default': 30,
                'description': 'Long moving average window in days'
            },
            'max_position_pct': {
                'type': 'float',
                'default': 0.1,
                'description': 'Maximum position size as percentage of portfolio value'
            }
        }

        super().__init__(
            name="moving_average",
            description="Simple moving average crossover strategy",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False
        )

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        """Create and return a Backtrader strategy class with MA crossover logic."""

        class MovingAverageCrossover(bt.Strategy):
            params = (
                ("short_window", int(parameters.get("short_window", 10))),
                ("long_window", int(parameters.get("long_window", 30))),
                ("max_position_pct", float(parameters.get("max_position_pct", 0.1))),
            )

            def __init__(self):
                self.equity_curve = []
                self.trades = []

                # Create indicators for each data feed
                self.short_mas = [
                    bt.indicators.SMA(d, period=self.p.short_window)
                    for d in self.datas
                ]
                self.long_mas = [
                    bt.indicators.SMA(d, period=self.p.long_window)
                    for d in self.datas
                ]
                self.crossovers = [
                    bt.indicators.CrossOver(short_ma, long_ma)
                    for short_ma, long_ma in zip(self.short_mas, self.long_mas)
                ]

            def next(self):
                allocations = {}

                # Generate signals for each ticker
                for i, data in enumerate(self.datas):
                    ticker = data._name
                    cross = self.crossovers[i][0]

                    if cross > 0:
                        # Buy signal: allocate max_position_pct
                        allocations[ticker] = self.p.max_position_pct
                    elif cross < 0:
                        # Sell signal: close position
                        allocations[ticker] = 0.0
    
                # Enforce position limits (scale down if total exposure exceeds max)
                total_exposure = sum(abs(pct) for pct in allocations.values())
                if total_exposure > self.p.max_position_pct:
                    scale = self.p.max_position_pct / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                # Execute trades
                for ticker, target_pct in allocations.items():
                    # Find data feed for this ticker
                    data = None
                    for d in self.datas:
                        if hasattr(d, '_name') and d._name == ticker:
                            data = d
                            break

                    if data is None:
                        continue

                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    portfolio_value = self.broker.getvalue()
                    target_value = target_pct * portfolio_value

                    if abs(current_value - target_value) < 100:  # Minimum trade size
                        continue

                    if target_value > current_value:
                        # Buy
                        shares_to_buy = int((target_value - current_value) / data.close[0])
                        if shares_to_buy > 0:
                            self.buy(data=data, size=shares_to_buy)
                    elif target_value < current_value:
                        # Sell
                        shares_to_sell = int((current_value - target_value) / data.close[0])
                        if shares_to_sell > 0:
                            self.sell(data=data, size=shares_to_sell)

                # Record equity curve
                current_date = self.datas[0].datetime.date(0).isoformat()
                self.equity_curve.append({
                    'date': current_date,
                    'value': self.broker.getvalue()
                })

            def notify_trade(self, trade):
                if trade.isclosed:
                    self.trades.append({
                        'size': trade.size,
                        'price': trade.price,
                        'value': trade.value,
                        'pnl': trade.pnl,
                        'pnlcomm': trade.pnlcomm
                    })

        return MovingAverageCrossover

    def generate_target_allocations(
        self,
        parameters: Dict[str, Any],
        symbols: List[str],
        as_of: datetime,
        current_prices: Dict[str, float],
    ) -> List[TargetAllocation]:
        from backend.main import app_state

        params = parameters or {}
        short_window = int(params.get("short_window", 10))
        long_window = int(params.get("long_window", 30))
        max_position_pct = float(params.get("max_position_pct", 0.1))
        db_path = app_state.get("database_path") or "data/backtest.db"

        allocations: List[TargetAllocation] = []
        conn = sqlite3.connect(db_path)
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
                    (symbol.upper(), as_of.date().isoformat(), max(long_window, short_window)),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                if len(rows) < max(short_window, long_window):
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
                short_ma = float(np.mean(closes[-short_window:]))
                long_ma = float(np.mean(closes[-long_window:]))
                if short_ma > long_ma:
                    target_pct = max_position_pct
                    reason = "ma_bullish"
                    confidence = min(1.0, abs(short_ma - long_ma) / max(long_ma, 1e-9))
                elif short_ma < long_ma:
                    target_pct = 0.0
                    reason = "ma_bearish"
                    confidence = min(1.0, abs(short_ma - long_ma) / max(long_ma, 1e-9))
                else:
                    target_pct = 0.0
                    reason = "ma_flat"
                    confidence = 0.1
                allocations.append(
                    TargetAllocation(
                        ticker=symbol.upper(),
                        target_pct=target_pct,
                        reason=reason,
                        confidence=float(confidence),
                        timestamp=as_of,
                        metadata={
                            "strategy": self.name,
                            "short_ma": short_ma,
                            "long_ma": long_ma,
                            "current_price": float(current_prices.get(symbol, 0.0) or 0.0),
                        },
                    )
                )
        finally:
            conn.close()
        return allocations

    def train(self, config: Dict[str, Any]) -> Any:
        """Training not supported for rule-based strategies."""
        raise NotImplementedError("Training not supported for rule-based strategies")

    def project_series(
        self,
        parameters: Dict[str, Any],
        anchor_time,
        anchor_price: float,
        projection_days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Generate anchored price series using recent daily return statistics."""
        from datetime import timedelta
        import sqlite3
        import numpy as np
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
            drift = 0.0007
            vol = 0.01

        points: List[Dict[str, Any]] = []
        price = anchor_price
        for day in range(projection_days):
            t = anchor_time + timedelta(days=day)
            trend_adj = drift * 0.7  # strategy generally follows trend with damping
            cyclical = vol * 0.2 * np.sin(day / 4)
            next_return = trend_adj + cyclical
            price = max(0.01, price * (1 + next_return))
            confidence = max(0.35, min(0.8, 0.78 - day * 0.01))
            band = abs(price * vol * 1.5)
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

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Project future performance using recent market data and strategy logic."""
        from datetime import datetime, timedelta
        from decimal import Decimal, getcontext
        import numpy as np

        # Set precision for financial calculations
        getcontext().prec = 10

        # Get recent price data for projection
        try:
            from backend.main import app_state
            import sqlite3

            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)

            # Get recent price data (last 90 days for trend analysis)
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=90)

            cur = conn.cursor()
            # Get data for major tickers to simulate portfolio
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Sample portfolio

            portfolio_data = {}
            for ticker in tickers:
                cur.execute("""
                    SELECT date, close, volume
                    FROM price_daily
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT 60
                """, (ticker, start_date.isoformat(), end_date.isoformat()))

                rows = cur.fetchall()
                if rows:
                    # Calculate recent returns and volatility
                    closes = [row[1] for row in rows[::-1]]  # Reverse to chronological order
                    if len(closes) > 1:
                        returns = np.diff(closes) / closes[:-1]
                        avg_return = np.mean(returns) if returns.size > 0 else 0.0
                        volatility = np.std(returns) if returns.size > 0 else 0.0
                        current_price = closes[-1]

                        portfolio_data[ticker] = {
                            'current_price': Decimal(str(current_price)),
                            'avg_daily_return': Decimal(str(avg_return)),
                            'volatility': Decimal(str(volatility)),
                            'weight': Decimal('1') / Decimal(str(len(tickers)))  # Equal weight
                        }

            conn.close()

            if not portfolio_data:
                # Fallback projection if no data
                initial_capital_dec = Decimal(str(initial_capital))
                projected_return_dec = Decimal('0.02')
                projected_final_value = initial_capital_dec * (Decimal('1') + projected_return_dec)

                return {
                    'projected_return': float(projected_return_dec),
                    'projected_volatility': 0.15,
                    'confidence': 0.5,
                    'projection_days': projection_days,
                    'initial_capital': float(initial_capital_dec),
                    'projected_final_value': float(projected_final_value.quantize(Decimal('0.01'))),
                    'timestamp': datetime.utcnow().isoformat()
                }

            # Calculate portfolio-level projections using Decimal
            portfolio_return = sum(data['avg_daily_return'] * data['weight'] for data in portfolio_data.values())
            portfolio_volatility = sum(data['volatility'] * data['weight'] for data in portfolio_data.values())

            # Project forward
            projection_days_dec = Decimal(str(projection_days))
            initial_capital_dec = Decimal(str(initial_capital))
            total_return = portfolio_return * projection_days_dec
            projected_final_value = initial_capital_dec * (Decimal('1') + total_return)

            # Adjust for strategy (MA crossover typically captures trends)
            # Assume 70% of market return with lower volatility
            strategy_multiplier = Decimal('0.7')
            adjusted_return = total_return * strategy_multiplier
            adjusted_volatility = float(portfolio_volatility * Decimal('0.8'))  # Lower volatility due to timing
            projected_final_value = initial_capital_dec * (Decimal('1') + adjusted_return)

            return {
                'projected_return': float(adjusted_return),
                'projected_volatility': round(adjusted_volatility, 6),
                'confidence': 0.7,  # Moderate confidence for rule-based strategy
                'projection_days': projection_days,
                'initial_capital': float(initial_capital_dec),
                'projected_final_value': float(projected_final_value.quantize(Decimal('0.01'))),
                'market_return': float(total_return),
                'strategy_multiplier': float(strategy_multiplier),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            # Fallback on error
            initial_capital_dec = Decimal(str(initial_capital))
            fallback_return = Decimal('0.01')
            projected_final_value = initial_capital_dec * (Decimal('1') + fallback_return)

            return {
                'projected_return': float(fallback_return),
                'projected_volatility': 0.12,
                'confidence': 0.3,
                'projection_days': projection_days,
                'initial_capital': float(initial_capital_dec),
                'projected_final_value': float(projected_final_value.quantize(Decimal('0.01'))),
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }