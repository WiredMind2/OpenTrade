"""
Time-series momentum (TCN/Transformer-style edge approximated without ML deps).

Uses short-horizon vs long-horizon mean of past returns and a logistic map to
approximate next-bar directional probability; trades when above/below fixed
thresholds (long / short / flat).
"""

import math
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import backtrader as bt
import numpy as np

from backend.domain.trading import TargetAllocation
from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_float, param_int

# Table 5.2: long if p(up) > 0.55, short if < 0.45
_PROB_LONG = 0.55
_PROB_SHORT = 0.45


class TsMomentumStrategy(BaseStrategy):
    """Rolling return momentum with logistic edge (no torch / gym)."""

    def __init__(self) -> None:
        parameters_schema = {
            "short_window": param_int(10, "Short horizon (bars) for mean daily return"),
            "long_window": param_int(40, "Long baseline (bars) for mean/vol of returns"),
            "max_position_pct": param_float(
                0.1, "Maximum absolute position size as fraction of portfolio value"
            ),
        }
        super().__init__(
            name="ts_momentum",
            description=(
                "Time-series momentum: short- vs long-horizon return drift mapped through "
                "a scaled logistic score to approximate next-bar directional edge (long/short/flat)."
            ),
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        params = parameters or {}
        short_window = max(2, int(params.get("short_window", 10)))
        long_window = max(short_window + 1, int(params.get("long_window", 40)))
        max_position_pct = min(max(float(params.get("max_position_pct", 0.1)), 0.0), 1.0)
        return {
            "short_window": short_window,
            "long_window": long_window,
            "max_position_pct": max_position_pct,
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=150)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class TsMomentumBt(bt.Strategy):
            params = (
                ("short_window", normalized["short_window"]),
                ("long_window", normalized["long_window"]),
                ("max_position_pct", normalized["max_position_pct"]),
            )

            def __init__(self) -> None:
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []

                self.ret_lines = [
                    bt.indicators.PctChange(d, period=1) for d in self.datas
                ]
                self.short_mu = [
                    bt.indicators.SMA(r, period=self.p.short_window)
                    for r in self.ret_lines
                ]
                self.long_mu = [
                    bt.indicators.SMA(r, period=self.p.long_window) for r in self.ret_lines
                ]
                self.ret_vol = [
                    bt.indicators.StdDev(r, period=self.p.long_window)
                    for r in self.ret_lines
                ]

            def next(self) -> None:
                allocations: Dict[str, float] = {}

                for i, data in enumerate(self.datas):
                    ticker = data._name
                    sm = float(self.short_mu[i][0])
                    lm = float(self.long_mu[i][0])
                    sig = float(self.ret_vol[i][0])
                    if any(math.isnan(x) for x in (sm, lm, sig)):
                        continue
                    z = (sm - lm) / (sig + 1e-9)
                    z = max(-6.0, min(6.0, z))
                    p_up = 1.0 / (1.0 + math.exp(-z))
                    if p_up > _PROB_LONG:
                        allocations[ticker] = self.p.max_position_pct
                    elif p_up < _PROB_SHORT:
                        allocations[ticker] = -self.p.max_position_pct
                    else:
                        allocations[ticker] = 0.0

                total_exposure = sum(abs(pct) for pct in allocations.values())
                if total_exposure > self.p.max_position_pct and total_exposure > 0:
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

            def notify_trade(self, trade: bt.Trade) -> None:
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

        return TsMomentumBt

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
        short_window = int(normalized["short_window"])
        long_window = int(normalized["long_window"])
        max_position_pct = float(normalized["max_position_pct"])
        db_path = app_state.get("database_path") or "data/backtest.db"

        allocations: List[TargetAllocation] = []
        own_conn = db_conn is None
        conn = sqlite3.connect(db_path) if own_conn else db_conn
        try:
            cur = conn.cursor()
            need = long_window + 2
            for symbol in symbols:
                cur.execute(
                    """
                    SELECT close
                    FROM price_daily
                    WHERE ticker = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT ?
                    """,
                    (symbol.upper(), as_of.date().isoformat(), need),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                if len(rows) < need:
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
                rets = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
                if len(rets) < long_window:
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

                short_mu = float(np.mean(rets[-short_window:]))
                long_mu = float(np.mean(rets[-long_window:]))
                sig = float(np.std(rets[-long_window:], ddof=0))
                z = (short_mu - long_mu) / (sig + 1e-9)
                z = max(-6.0, min(6.0, z))
                p_up = float(1.0 / (1.0 + np.exp(-z)))

                if p_up > _PROB_LONG:
                    target_pct = max_position_pct
                    reason = "ts_momentum_long"
                elif p_up < _PROB_SHORT:
                    target_pct = -max_position_pct
                    reason = "ts_momentum_short"
                else:
                    target_pct = 0.0
                    reason = "ts_momentum_flat"

                confidence = min(1.0, abs(p_up - 0.5) * 4.0)
                allocations.append(
                    TargetAllocation(
                        ticker=symbol.upper(),
                        target_pct=max(-max_position_pct, min(max_position_pct, target_pct)),
                        reason=reason,
                        confidence=confidence,
                        timestamp=as_of,
                        metadata={
                            "strategy": self.name,
                            "p_up": p_up,
                            "short_mu": short_mu,
                            "long_mu": long_mu,
                            "ret_vol": sig,
                            "current_price": float(current_prices.get(symbol, 0.0) or 0.0),
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
        from datetime import timedelta

        import sqlite3
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
            arr = np.array(closes, dtype=float)
            rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
            drift = float(np.mean(rets))
            vol = float(np.std(rets))
        else:
            drift = 0.0007
            vol = 0.01

        points: List[Dict[str, Any]] = []
        price = anchor_price
        for day in range(projection_days):
            t = anchor_time + timedelta(days=day)
            trend_adj = drift * 0.85
            cyclical = vol * 0.25 * np.sin(day / 3.5)
            next_return = trend_adj + cyclical
            price = max(0.01, price * (1 + next_return))
            confidence = max(0.35, min(0.82, 0.8 - day * 0.012))
            band = abs(price * vol * 1.55)
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
        from datetime import timedelta
        from decimal import Decimal, getcontext

        getcontext().prec = 10

        try:
            from backend.main import app_state
            import sqlite3

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
                        closes_arr = np.array(closes, dtype=float)
                        rets = np.diff(closes_arr) / np.maximum(closes_arr[:-1], 1e-12)
                        avg_return = float(np.mean(rets)) if rets.size > 0 else 0.0
                        volatility = float(np.std(rets)) if rets.size > 0 else 0.0
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
                projected_return_dec = Decimal("0.02")
                projected_final_value = initial_capital_dec * (
                    Decimal("1") + projected_return_dec
                )
                return {
                    "projected_return": float(projected_return_dec),
                    "projected_volatility": 0.15,
                    "confidence": 0.5,
                    "projection_days": projection_days,
                    "initial_capital": float(initial_capital_dec),
                    "projected_final_value": float(
                        projected_final_value.quantize(Decimal("0.01"))
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            portfolio_return = sum(
                data["avg_daily_return"] * data["weight"] for data in portfolio_data.values()
            )
            portfolio_volatility = sum(
                data["volatility"] * data["weight"] for data in portfolio_data.values()
            )

            projection_days_dec = Decimal(str(projection_days))
            initial_capital_dec = Decimal(str(initial_capital))
            total_return = portfolio_return * projection_days_dec

            strategy_multiplier = Decimal("0.72")
            adjusted_return = total_return * strategy_multiplier
            adjusted_volatility = float(portfolio_volatility * Decimal("0.82"))
            projected_final_value = initial_capital_dec * (Decimal("1") + adjusted_return)

            return {
                "projected_return": float(adjusted_return),
                "projected_volatility": round(adjusted_volatility, 6),
                "confidence": 0.68,
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
            projected_final_value = initial_capital_dec * (Decimal("1") + fallback_return)
            return {
                "projected_return": float(fallback_return),
                "projected_volatility": 0.12,
                "confidence": 0.3,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(
                    projected_final_value.quantize(Decimal("0.01"))
                ),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
