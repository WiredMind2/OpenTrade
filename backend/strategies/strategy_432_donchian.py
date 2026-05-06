"""
Strategy 432 BTC - Donchian Breakout Continuation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Type

import backtrader as bt

from backend.strategies.base import BaseStrategy
from backend.strategies.bt_decision_markers import DecisionRecordingStrategy
from backend.strategies.support import capability_profile, param_bool, param_float, param_int


class Strategy432DonchianStrategy(BaseStrategy):
    def __init__(self) -> None:
        parameters_schema = {
            "dc_len": param_int(30, "Donchian Length"),
            "atr_len": param_int(14, "ATR Length"),
            "adx_len": param_int(14, "ADX Length"),
            "ema200_len": param_int(200, "EMA 200 Length"),
            "min_break_atr": param_float(0.5, "Min Breakout ATR"),
            "adx_min": param_float(22.0, "ADX minimum"),
            "hold_bars": param_int(20, "Maximum bars in position"),
            "allow_shorts": param_bool(False, "Allow short entries"),
            "max_position_pct": param_float(0.95, "Position size as fraction of equity"),
        }
        super().__init__(
            name="strategy_432_donchian",
            description="BTC Donchian breakout continuation with ATR magnitude, ADX strength, EMA200 filter, and hold/opposite exits",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        return {
            "dc_len": max(1, int(p.get("dc_len", 30))),
            "atr_len": max(1, int(p.get("atr_len", 14))),
            "adx_len": max(1, int(p.get("adx_len", 14))),
            "ema200_len": max(1, int(p.get("ema200_len", 200))),
            "min_break_atr": max(0.0, float(p.get("min_break_atr", 0.5))),
            "adx_min": max(0.0, float(p.get("adx_min", 22.0))),
            "hold_bars": max(1, int(p.get("hold_bars", 20))),
            "allow_shorts": bool(p.get("allow_shorts", False)),
            "max_position_pct": min(max(float(p.get("max_position_pct", 0.95)), 0.0), 1.0),
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=260)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class Strategy432DonchianBt(DecisionRecordingStrategy):
            params = tuple((k, v) for k, v in normalized.items())

            def __init__(self) -> None:
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []
                self._entry_bar: Dict[bt.LineSeries, int] = {}

                self.atr = [bt.ind.ATR(d, period=self.p.atr_len) for d in self.datas]
                self.adx = [bt.ind.ADX(d, period=self.p.adx_len) for d in self.datas]
                self._min_bars = max(self.p.dc_len + 2, self.p.atr_len + 2, self.p.adx_len + 2, 20)

            def _donchian_high_prev(self, data: bt.LineSeries) -> float:
                return max(float(data.high[-i]) for i in range(1, self.p.dc_len + 1))

            def _donchian_low_prev(self, data: bt.LineSeries) -> float:
                return min(float(data.low[-i]) for i in range(1, self.p.dc_len + 1))

            def _position_size(self, close: float) -> int:
                if close <= 0:
                    return 0
                target_value = self.broker.getvalue() * float(self.p.max_position_pct)
                return max(0, int(target_value / close))

            def _ema_proxy(self, data: bt.LineSeries, period: int) -> float:
                use = max(2, min(period, len(data)))
                alpha = 2.0 / (use + 1.0)
                ema = float(data.close[-(use - 1)])
                for j in range(use - 2, -1, -1):
                    px = float(data.close[-j])
                    ema = alpha * px + (1.0 - alpha) * ema
                return ema

            def next(self) -> None:
                self.equity_curve.append({"date": self.datetime.date(0).isoformat(), "value": self.broker.getvalue()})
                for i, data in enumerate(self.datas):
                    if len(data) < self._min_bars:
                        continue

                    close = float(data.close[0])
                    atr_value = float(self.atr[i][0])
                    adx_value = float(self.adx[i][0])
                    ema200 = self._ema_proxy(data, int(self.p.ema200_len))
                    dc_high = self._donchian_high_prev(data)
                    dc_low = self._donchian_low_prev(data)
                    in_long = self.getposition(data).size > 0
                    in_short = self.getposition(data).size < 0
                    flat = self.getposition(data).size == 0

                    long_breakout_mag = (close - dc_high) / atr_value if close > dc_high and atr_value > 0 else 0.0
                    short_breakout_mag = (dc_low - close) / atr_value if close < dc_low and atr_value > 0 else 0.0

                    indicators_ready = atr_value > 0
                    long_entry = (
                        indicators_ready
                        and flat
                        and close > dc_high
                        and adx_value >= self.p.adx_min
                        and close > ema200
                        and long_breakout_mag >= self.p.min_break_atr
                    )
                    short_entry = (
                        indicators_ready
                        and self.p.allow_shorts
                        and flat
                        and close < dc_low
                        and adx_value >= self.p.adx_min
                        and close < ema200
                        and short_breakout_mag >= self.p.min_break_atr
                    )

                    bars_held = len(data) - int(self._entry_bar[data]) if data in self._entry_bar else None
                    long_exit = in_long and (close < dc_low or (bars_held is not None and bars_held >= self.p.hold_bars))
                    short_exit = in_short and (close > dc_high or (bars_held is not None and bars_held >= self.p.hold_bars))

                    if long_entry:
                        size = self._position_size(close)
                        if size > 0:
                            self.buy(data=data, size=size)
                            self._entry_bar[data] = len(data)
                    if short_entry:
                        size = self._position_size(close)
                        if size > 0:
                            self.sell(data=data, size=size)
                            self._entry_bar[data] = len(data)
                    if long_exit:
                        self.close(data=data)
                        self._entry_bar.pop(data, None)
                    if short_exit:
                        self.close(data=data)
                        self._entry_bar.pop(data, None)

            def notify_trade(self, trade: bt.Trade) -> None:
                if trade.isclosed:
                    self.trades.append(
                        {
                            "date": self.datetime.datetime(0).isoformat(),
                            "ref": trade.ref,
                            "pnl": trade.pnl,
                            "pnlcomm": trade.pnlcomm,
                            "size": trade.size,
                        }
                    )

        return Strategy432DonchianBt

    def project(
        self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        return {
            "projected_return": 0.0,
            "projected_volatility": 0.0,
            "confidence": 0.5,
            "projection_days": int(projection_days),
            "initial_capital": float(initial_capital),
            "projected_final_value": float(initial_capital),
            "timestamp": datetime.utcnow().isoformat(),
        }
