"""
Charter Execution Model [JOAT] strategy port.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import backtrader as bt

from backend.strategies.base import BaseStrategy
from backend.strategies.bt_decision_markers import DecisionRecordingStrategy
from backend.strategies.support import capability_profile, param_bool, param_float, param_int


class CharterExecutionModelStrategy(BaseStrategy):
    def __init__(self) -> None:
        parameters_schema = {
            "fast_length": param_int(34, "Fast Length"),
            "slow_length": param_int(89, "Slow Length"),
            "atr_length": param_int(100, "ATR Length"),
            "heat_length": param_int(70, "Heat Window"),
            "regime_floor": param_float(34.0, "Regime Strength Floor"),
            "regime_persistence_floor": param_int(8, "Regime Persistence Floor"),
            "liquidity_lookback": param_int(180, "Liquidity Lookback"),
            "liquidity_bins": param_int(24, "Liquidity Bins"),
            "liquidity_bias_floor": param_float(0.04, "Liquidity Bias Floor"),
            "pivot_length": param_int(4, "Pivot Length"),
            "swing_lookback": param_int(24, "Swing Lookback"),
            "gap_sigma": param_float(0.25, "Gap Sigma Filter"),
            "shift_momentum": param_int(9, "Shift Momentum Length"),
            "shift_rsi": param_int(14, "Shift RSI Length"),
            "displacement_floor": param_float(46.0, "Displacement Floor"),
            "allow_continuation_triggers": param_bool(True, "Allow Continuation Triggers"),
            "continuation_floor": param_float(50.0, "Continuation Pressure Floor"),
            "stop_atr": param_float(1.2, "Stop ATR Multiplier"),
            "target_one_r": param_float(1.5, "Target 1 R"),
            "target_two_r": param_float(2.5, "Target 2 R"),
            "use_trailing_in_strong_regime": param_bool(True, "Trail In Strong Regime"),
            "max_position_pct": param_float(0.10, "Position % of equity"),
        }
        super().__init__(
            name="charter_execution_model",
            description="Charter Execution Model [JOAT] with regime/liquidity/structure/trigger stack and staged ATR exits",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        fast = max(10, int(p.get("fast_length", 34)))
        slow = max(20, int(p.get("slow_length", 89)))
        if slow <= fast:
            slow = fast + 1
        return {
            "fast_length": fast,
            "slow_length": slow,
            "atr_length": max(20, int(p.get("atr_length", 100))),
            "heat_length": max(20, int(p.get("heat_length", 70))),
            "regime_floor": max(10.0, min(100.0, float(p.get("regime_floor", 34.0)))),
            "regime_persistence_floor": max(3, int(p.get("regime_persistence_floor", 8))),
            "liquidity_lookback": max(80, int(p.get("liquidity_lookback", 180))),
            "liquidity_bins": max(10, int(p.get("liquidity_bins", 24))),
            "liquidity_bias_floor": max(0.01, float(p.get("liquidity_bias_floor", 0.04))),
            "pivot_length": max(2, int(p.get("pivot_length", 4))),
            "swing_lookback": max(10, int(p.get("swing_lookback", 24))),
            "gap_sigma": max(0.10, float(p.get("gap_sigma", 0.25))),
            "shift_momentum": max(3, int(p.get("shift_momentum", 9))),
            "shift_rsi": max(5, int(p.get("shift_rsi", 14))),
            "displacement_floor": max(20.0, float(p.get("displacement_floor", 46.0))),
            "allow_continuation_triggers": bool(p.get("allow_continuation_triggers", True)),
            "continuation_floor": max(20.0, min(100.0, float(p.get("continuation_floor", 50.0)))),
            "stop_atr": max(0.5, float(p.get("stop_atr", 1.2))),
            "target_one_r": max(0.5, float(p.get("target_one_r", 1.5))),
            "target_two_r": max(1.0, float(p.get("target_two_r", 2.5))),
            "use_trailing_in_strong_regime": bool(p.get("use_trailing_in_strong_regime", True)),
            "max_position_pct": min(max(float(p.get("max_position_pct", 0.10)), 0.0), 1.0),
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=260)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class CharterExecutionModelBt(DecisionRecordingStrategy):
            params = tuple((k, v) for k, v in normalized.items())

            def __init__(self) -> None:
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []
                self.ema_fast = [bt.ind.EMA(d.close, period=self.p.fast_length) for d in self.datas]
                self.ema_slow = [bt.ind.EMA(d.close, period=self.p.slow_length) for d in self.datas]
                self.hma_fast = [bt.ind.HullMovingAverage(d.close, period=max(10, self.p.fast_length - 8)) for d in self.datas]
                self.hma_slow = [bt.ind.HullMovingAverage(d.close, period=max(20, self.p.slow_length - 13)) for d in self.datas]
                self.atr = [bt.ind.ATR(d, period=self.p.atr_length) for d in self.datas]
                self.reference_ema = [bt.ind.EMA(d.close, period=55) for d in self.datas]

                self._regime_persistence: Dict[bt.LineSeries, int] = {d: 0 for d in self.datas}
                self._prev_bull_regime: Dict[bt.LineSeries, Optional[bool]] = {d: None for d in self.datas}
                self._recent_pivot_high: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._recent_pivot_low: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._plans: Dict[bt.LineSeries, Dict[str, Any]] = {}

                self._min_bars = max(
                    self.p.atr_length + 2,
                    self.p.heat_length + 2,
                    self.p.liquidity_lookback + 2,
                    self.p.swing_lookback + 2,
                    self.p.shift_momentum + self.p.shift_rsi + 5,
                    self.p.slow_length + 5,
                    205,
                )

            def _rolling_max(self, line: bt.LineSeries, lookback: int) -> float:
                use = min(lookback, len(line) - 1)
                return max(float(line[-i]) for i in range(use + 1))

            def _rolling_min(self, line: bt.LineSeries, lookback: int) -> float:
                use = min(lookback, len(line) - 1)
                return min(float(line[-i]) for i in range(use + 1))

            def _safe_div(self, a: float, b: float) -> float:
                return 0.0 if abs(b) < 1e-12 else a / b

            def _clamp(self, v: float, lo: float, hi: float) -> float:
                return max(lo, min(hi, v))

            def _rsi_from_series(self, values: List[float], period: int) -> float:
                if len(values) < period + 1:
                    return 50.0
                gains = 0.0
                losses = 0.0
                for j in range(len(values) - period, len(values)):
                    delta = values[j] - values[j - 1]
                    if delta >= 0:
                        gains += delta
                    else:
                        losses += -delta
                avg_gain = gains / period
                avg_loss = losses / period
                if avg_loss <= 1e-12 and avg_gain <= 1e-12:
                    return 50.0
                if avg_loss <= 1e-12:
                    return 100.0
                rs = avg_gain / avg_loss
                return 100.0 - (100.0 / (1.0 + rs))

            def _update_pivots(self, data: bt.LineSeries) -> None:
                p = int(self.p.pivot_length)
                if len(data) < (2 * p + 1):
                    return
                highs = [float(data.high[-(2 * p - j)]) for j in range(2 * p + 1)]
                lows = [float(data.low[-(2 * p - j)]) for j in range(2 * p + 1)]
                center_high = highs[p]
                center_low = lows[p]
                if center_high >= max(highs):
                    self._recent_pivot_high[data] = float(data.high[-p])
                if center_low <= min(lows):
                    self._recent_pivot_low[data] = float(data.low[-p])

            def _position_size(self, close: float) -> int:
                if close <= 0:
                    return 0
                target_value = self.broker.getvalue() * float(self.p.max_position_pct)
                return max(0, int(target_value / close))

            def next(self) -> None:
                self.equity_curve.append({"date": self.datetime.date(0).isoformat(), "value": self.broker.getvalue()})

                for i, data in enumerate(self.datas):
                    if len(data) < self._min_bars:
                        continue
                    self._update_pivots(data)

                    close = float(data.close[0])
                    low = float(data.low[0])
                    high = float(data.high[0])
                    atr_value = max(float(self.atr[i][0]), 1e-9)
                    ema_fast = float(self.ema_fast[i][0])
                    ema_slow = float(self.ema_slow[i][0])
                    hma_fast = float(self.hma_fast[i][0])
                    hma_slow = float(self.hma_slow[i][0])
                    directional_mid = (ema_fast + hma_fast) * 0.5
                    structural_mid = (ema_slow + hma_slow) * 0.5
                    spread = directional_mid - structural_mid
                    spread_norm = self._safe_div(abs(spread), atr_value) * 100.0

                    range_high = self._rolling_max(data.high, int(self.p.heat_length))
                    range_low = self._rolling_min(data.low, int(self.p.heat_length))
                    heat_norm = self._safe_div(close - range_low, max(range_high - range_low, 1e-9)) * 100.0

                    bull_regime = directional_mid > structural_mid
                    prev_bull = self._prev_bull_regime[data]
                    if prev_bull is None:
                        persistence = 0
                    elif prev_bull == bull_regime:
                        persistence = self._regime_persistence[data] + 1
                    else:
                        persistence = 0
                    self._prev_bull_regime[data] = bull_regime
                    self._regime_persistence[data] = persistence

                    regime_strength = self._clamp(spread_norm * 0.60 + abs(heat_norm - 50.0) * 0.80, 0.0, 100.0)
                    mature_bull_regime = bull_regime and regime_strength >= self.p.regime_floor and persistence >= self.p.regime_persistence_floor
                    mature_bear_regime = (not bull_regime) and regime_strength >= self.p.regime_floor and persistence >= self.p.regime_persistence_floor

                    # Liquidity model
                    liq_high = self._rolling_max(data.high, int(self.p.liquidity_lookback))
                    liq_low = self._rolling_min(data.low, int(self.p.liquidity_lookback))
                    liq_range = liq_high - liq_low
                    liq_bin_size = liq_range / max(int(self.p.liquidity_bins), 1)
                    effective_lb = min(int(self.p.liquidity_lookback), len(data))
                    buy_bins = [0.0 for _ in range(int(self.p.liquidity_bins))]
                    sell_bins = [0.0 for _ in range(int(self.p.liquidity_bins))]
                    if liq_range > 0 and effective_lb > 0 and liq_bin_size > 0:
                        for offset in range(effective_lb):
                            hlc3 = (float(data.high[-offset]) + float(data.low[-offset]) + float(data.close[-offset])) / 3.0
                            raw_index = int(math.floor((hlc3 - liq_low) / liq_bin_size))
                            bin_index = max(0, min(int(self.p.liquidity_bins) - 1, raw_index))
                            vol = float(data.volume[-offset])
                            if float(data.close[-offset]) >= float(data.open[-offset]):
                                buy_bins[bin_index] += vol
                            else:
                                sell_bins[bin_index] += vol
                    total_buy = sum(buy_bins)
                    total_sell = sum(sell_bins)
                    liquidity_bias = self._safe_div(total_buy - total_sell, max(total_buy + total_sell, 1.0))
                    reference_price = float(self.reference_ema[i][0])
                    bullish_liquidity = liquidity_bias > self.p.liquidity_bias_floor and close >= reference_price
                    bearish_liquidity = liquidity_bias < -self.p.liquidity_bias_floor and close <= reference_price

                    # Structure
                    swing_low = self._rolling_min(data.low, int(self.p.swing_lookback))
                    swing_high = self._rolling_max(data.high, int(self.p.swing_lookback))
                    recent_pivot_low = self._recent_pivot_low[data] if self._recent_pivot_low[data] is not None else swing_low
                    recent_pivot_high = self._recent_pivot_high[data] if self._recent_pivot_high[data] is not None else swing_high
                    bullish_structure = close > float(recent_pivot_low) and close > ema_slow
                    bearish_structure = close < float(recent_pivot_high) and close < ema_slow

                    # Trigger stack
                    enough_gap_history = len(data) >= 3
                    bull_gap_sigma = 0.0
                    bear_gap_sigma = 0.0
                    if enough_gap_history:
                        bull_gap_series = []
                        bear_gap_series = []
                        available = min(200, len(data) - 2)
                        for k in range(available):
                            bull_gap_series.append(float(data.low[-k]) - float(data.high[-k - 2]))
                            bear_gap_series.append(float(data.low[-k - 2]) - float(data.high[-k]))
                        bull_gap_den = max(statistics.pstdev(bull_gap_series) if len(bull_gap_series) > 1 else 0.0, 1e-9)
                        bear_gap_den = max(statistics.pstdev(bear_gap_series) if len(bear_gap_series) > 1 else 0.0, 1e-9)
                        bull_gap_sigma = (low - float(data.high[-2])) / bull_gap_den
                        bear_gap_sigma = (float(data.low[-2]) - high) / bear_gap_den

                    bull_gap = enough_gap_history and low > float(data.high[-2]) and float(data.high[-1]) > float(data.high[-2]) and bull_gap_sigma > self.p.gap_sigma
                    bear_gap = enough_gap_history and high < float(data.low[-2]) and float(data.low[-1]) < float(data.low[-2]) and bear_gap_sigma > self.p.gap_sigma

                    momentum = close - float(data.close[-int(self.p.shift_momentum)])
                    momentum_hist = [
                        float(data.close[-(int(self.p.shift_momentum) + j)]) - float(data.close[-(2 * int(self.p.shift_momentum) + j)])
                        for j in range(int(self.p.shift_rsi) + 1)
                        if len(data) > (2 * int(self.p.shift_momentum) + j)
                    ]
                    pressure_rsi = self._rsi_from_series(momentum_hist[::-1], int(self.p.shift_rsi))
                    displacement = self._clamp(
                        self._safe_div(abs(ema_fast - ema_slow), atr_value) * 40.0 + abs(pressure_rsi - 50.0),
                        0.0,
                        100.0,
                    )
                    bull_shift = displacement >= self.p.displacement_floor and pressure_rsi > 54 and close > ema_fast and spread > 0
                    bear_shift = displacement >= self.p.displacement_floor and pressure_rsi < 46 and close < ema_fast and spread < 0

                    long_context = mature_bull_regime and bullish_liquidity and bullish_structure
                    short_context = mature_bear_regime and bearish_liquidity and bearish_structure
                    bull_continuation = (
                        self.p.allow_continuation_triggers
                        and pressure_rsi >= self.p.continuation_floor
                        and close > directional_mid
                        and low <= ema_fast
                    )
                    bear_continuation = (
                        self.p.allow_continuation_triggers
                        and pressure_rsi <= (100.0 - self.p.continuation_floor)
                        and close < directional_mid
                        and high >= ema_fast
                    )
                    long_trigger = bull_gap or bull_shift or bull_continuation
                    short_trigger = bear_gap or bear_shift or bear_continuation

                    pos = self.getposition(data)
                    long_signal = long_context and long_trigger and pos.size <= 0
                    short_signal = short_context and short_trigger and pos.size >= 0

                    # Risk model
                    long_invalidation = min(float(recent_pivot_low), low)
                    short_invalidation = max(float(recent_pivot_high), high)
                    long_stop = min(close - atr_value * float(self.p.stop_atr), long_invalidation - atr_value * 0.15)
                    short_stop = max(close + atr_value * float(self.p.stop_atr), short_invalidation + atr_value * 0.15)
                    long_risk = max(close - long_stop, 1e-6)
                    short_risk = max(short_stop - close, 1e-6)
                    long_t1 = close + long_risk * float(self.p.target_one_r)
                    long_t2 = close + long_risk * float(self.p.target_two_r)
                    short_t1 = close - short_risk * float(self.p.target_one_r)
                    short_t2 = close - short_risk * float(self.p.target_two_r)

                    if long_signal:
                        if pos.size < 0:
                            self.close(data=data)
                        size = self._position_size(close)
                        if size > 0:
                            self.buy(data=data, size=size)
                            self._plans[data] = {"dir": "long", "stop": long_stop, "t1": long_t1, "t2": long_t2, "t1_hit": False}
                        continue
                    if short_signal:
                        if pos.size > 0:
                            self.close(data=data)
                        size = self._position_size(close)
                        if size > 0:
                            self.sell(data=data, size=size)
                            self._plans[data] = {"dir": "short", "stop": short_stop, "t1": short_t1, "t2": short_t2, "t1_hit": False}
                        continue

                    if pos.size == 0:
                        self._plans.pop(data, None)
                        continue
                    plan = self._plans.get(data)
                    if not plan:
                        continue

                    if pos.size > 0 and plan["dir"] == "long":
                        active_stop = float(plan["stop"])
                        if self.p.use_trailing_in_strong_regime and regime_strength >= 60.0:
                            active_stop = max(active_stop, self._rolling_min(data.low, 5) - atr_value * 0.20)
                        plan["stop"] = active_stop
                        if close <= active_stop:
                            self.close(data=data)
                            self._plans.pop(data, None)
                            continue
                        if (not plan["t1_hit"]) and close >= float(plan["t1"]):
                            trim = max(1, int(abs(pos.size) * 0.5))
                            self.sell(data=data, size=min(trim, abs(pos.size)))
                            plan["t1_hit"] = True
                        if close >= float(plan["t2"]):
                            self.close(data=data)
                            self._plans.pop(data, None)
                    elif pos.size < 0 and plan["dir"] == "short":
                        active_stop = float(plan["stop"])
                        if self.p.use_trailing_in_strong_regime and regime_strength >= 60.0:
                            active_stop = min(active_stop, self._rolling_max(data.high, 5) + atr_value * 0.20)
                        plan["stop"] = active_stop
                        if close >= active_stop:
                            self.close(data=data)
                            self._plans.pop(data, None)
                            continue
                        if (not plan["t1_hit"]) and close <= float(plan["t1"]):
                            trim = max(1, int(abs(pos.size) * 0.5))
                            self.buy(data=data, size=min(trim, abs(pos.size)))
                            plan["t1_hit"] = True
                        if close <= float(plan["t2"]):
                            self.close(data=data)
                            self._plans.pop(data, None)

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

        return CharterExecutionModelBt

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
