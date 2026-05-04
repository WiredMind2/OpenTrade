"""
Volatility targeting: scale position size ~ 1 / realized vol toward a target
annualized vol; direction from simple momentum (ROC sign) per ticker.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Any, Dict, Type

import backtrader as bt
import numpy as np

from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_float, param_int


class VolatilityTargetingStrategy(BaseStrategy):
    """Risk-parity style sizing on forecast vol with momentum direction."""

    def __init__(self) -> None:
        parameters_schema = {
            "short_window": param_int(10, "Feed warm-up: short horizon (bars); must be < long_window"),
            "long_window": param_int(60, "Feed warm-up: long horizon (bars); raised to cover vol/momentum"),
            "target_ann_vol": param_float(
                0.15,
                "Target annualized volatility (fraction, e.g. 0.15 = 15%)",
            ),
            "vol_lookback": param_int(20, "Bars for StdDev of daily returns (vol forecast)"),
            "momentum_lookback": param_int(20, "ROC period for momentum sign"),
            "max_position_pct": param_float(
                0.1,
                "Cap on absolute |weight| per name before portfolio exposure scaling",
            ),
        }
        super().__init__(
            name="volatility_targeting",
            description="Vol targeting: size ~ target_vol / realized vol; sign from momentum",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        vol_lookback = max(5, int(p.get("vol_lookback", 20)))
        momentum_lookback = max(2, int(p.get("momentum_lookback", 20)))
        short_window = max(2, int(p.get("short_window", 10)))
        long_window = max(
            short_window + 1,
            int(p.get("long_window", 60)),
            vol_lookback + 3,
            momentum_lookback + 3,
        )
        target_ann_vol = max(float(p.get("target_ann_vol", 0.15)), 1e-6)
        max_position_pct = min(max(float(p.get("max_position_pct", 0.1)), 0.0), 1.0)
        return {
            "short_window": short_window,
            "long_window": long_window,
            "target_ann_vol": target_ann_vol,
            "vol_lookback": vol_lookback,
            "momentum_lookback": momentum_lookback,
            "max_position_pct": max_position_pct,
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=120)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        n = self._normalize_parameters(parameters)
        target_daily = float(n["target_ann_vol"]) / float(np.sqrt(252.0))

        class VolatilityTargeting(bt.Strategy):
            params = (
                ("short_window", n["short_window"]),
                ("long_window", n["long_window"]),
                ("target_ann_vol", n["target_ann_vol"]),
                ("vol_lookback", n["vol_lookback"]),
                ("momentum_lookback", n["momentum_lookback"]),
                ("max_position_pct", n["max_position_pct"]),
            )

            def __init__(self) -> None:
                self.equity_curve: list[dict[str, Any]] = []
                self.trades: list[dict[str, Any]] = []
                self._target_daily = target_daily
                self._rets: list[bt.IndicatorBase] = []
                self._vols: list[bt.IndicatorBase] = []
                self._rocs: list[bt.IndicatorBase] = []
                for d in self.datas:
                    r = bt.indicators.PercentChange(d.close, period=1)
                    self._rets.append(r)
                    self._vols.append(
                        bt.indicators.StdDev(r, period=self.p.vol_lookback, movav=bt.indicators.SMA)
                    )
                    self._rocs.append(bt.indicators.ROC(d.close, period=self.p.momentum_lookback))

            def next(self) -> None:
                allocations: Dict[str, float] = {}
                for i, data in enumerate(self.datas):
                    ticker = data._name
                    vol = float(self._vols[i][0])
                    roc = float(self._rocs[i][0])
                    if vol <= 0 or vol != vol or roc != roc:
                        continue
                    vol = max(vol, 1e-8)
                    raw_w = self._target_daily / vol
                    cap = min(raw_w, self.p.max_position_pct)
                    if roc > 0:
                        allocations[ticker] = cap
                    elif roc < 0:
                        allocations[ticker] = -cap
                    else:
                        allocations[ticker] = 0.0

                total_exposure = sum(abs(v) for v in allocations.values())
                if total_exposure > self.p.max_position_pct and total_exposure > 0:
                    scale = self.p.max_position_pct / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                for ticker, target_pct in allocations.items():
                    data = next(
                        (d for d in self.datas if hasattr(d, "_name") and d._name == ticker),
                        None,
                    )
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
                self.equity_curve.append({"date": current_date, "value": self.broker.getvalue()})

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

        return VolatilityTargeting

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Rough projection using recent realized vol and momentum from sample tickers."""
        getcontext().prec = 10
        n = self._normalize_parameters(parameters)
        target_daily = float(n["target_ann_vol"]) / float(np.sqrt(252.0))

        try:
            from backend.main import app_state
            import sqlite3

            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=90)
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            cur = conn.cursor()
            blended: list[float] = []
            for ticker in tickers:
                cur.execute(
                    """
                    SELECT close
                    FROM price_daily
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date ASC
                    LIMIT 200
                    """,
                    (ticker, start_date.isoformat(), end_date.isoformat()),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                if len(rows) < n["vol_lookback"] + n["momentum_lookback"] + 2:
                    continue
                closes = np.array(rows, dtype=float)
                rets = np.diff(closes) / closes[:-1]
                if rets.size < n["vol_lookback"]:
                    continue
                sigma = float(np.std(rets[-(n["vol_lookback"]) :]))
                mom = (closes[-1] - closes[-1 - n["momentum_lookback"]]) / max(
                    abs(closes[-1 - n["momentum_lookback"]]), 1e-9
                )
                if sigma <= 0:
                    continue
                w = min(target_daily / sigma, n["max_position_pct"])
                sign = 1.0 if mom > 0 else (-1.0 if mom < 0 else 0.0)
                blended.append(sign * w * float(np.mean(rets[-min(20, rets.size) :])))
            conn.close()

            if not blended:
                initial_capital_dec = Decimal(str(initial_capital))
                projected_return_dec = Decimal("0.02")
                projected_final_value = initial_capital_dec * (Decimal("1") + projected_return_dec)
                return {
                    "projected_return": float(projected_return_dec),
                    "projected_volatility": float(n["target_ann_vol"]),
                    "confidence": 0.5,
                    "projection_days": projection_days,
                    "initial_capital": float(initial_capital_dec),
                    "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            avg_exposure = float(np.mean(np.abs(blended))) if blended else 0.0
            drift = float(np.mean(blended))
            projection_days_dec = Decimal(str(projection_days))
            initial_capital_dec = Decimal(str(initial_capital))
            damped = drift * Decimal("0.65")
            total_return = damped * projection_days_dec
            projected_final_value = initial_capital_dec * (Decimal("1") + total_return)
            est_ann_vol = min(float(n["target_ann_vol"]), float(avg_exposure * np.sqrt(252) * 0.25 + 0.05))

            return {
                "projected_return": float(total_return),
                "projected_volatility": round(est_ann_vol, 6),
                "confidence": 0.62,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
                "target_ann_vol": float(n["target_ann_vol"]),
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
                "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
