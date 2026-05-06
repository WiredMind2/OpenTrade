"""
Concordance Regime Synthesis [JOAT] strategy port.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import backtrader as bt

from backend.strategies.base import BaseStrategy
from backend.strategies.bt_decision_markers import DecisionRecordingStrategy
from backend.strategies.support import capability_profile, param_bool, param_float, param_int, param_str


class ConcordanceRegimeSynthesisStrategy(BaseStrategy):
    def __init__(self) -> None:
        parameters_schema = {
            "regime_swing_length": param_int(10, "Regime Swing Length"),
            "regime_band_mult": param_float(2.0, "Regime Band Multiplier"),
            "regime_center_smooth": param_int(10, "Regime Center Smooth"),
            "regime_band_smooth": param_int(20, "Regime Band Smooth"),
            "regime_hold_bars": param_int(2, "Regime Hold Bars"),
            "ma_type": param_str("EMA", "Pressure MA Type"),
            "ma_length": param_int(55, "Pressure MA Length"),
            "pressure_normalize": param_int(80, "Pressure Normalize Length"),
            "pressure_signal_length": param_int(8, "Pressure Signal Length"),
            "delta_wave_length": param_int(20, "Delta Wave Length"),
            "delta_smooth_length": param_int(5, "Delta Wave Smooth"),
            "lattice_lookback": param_int(120, "Participation Lookback"),
            "lattice_rows": param_int(40, "Participation Rows"),
            "confluence_threshold": param_int(6, "Confluence Threshold"),
            "structure_impulse": param_float(1.0, "Structure Impulse ATR"),
            "structure_lookback": param_int(20, "Structure Validity Bars"),
            "use_htf_bias": param_bool(True, "Use HTF Bias Filter"),
            "use_session_filter": param_bool(False, "Use Session Filter"),
            "cooldown_bars": param_int(3, "Cooldown Bars After Entry"),
            "stop_atr_mult": param_float(1.8, "Stop ATR Multiplier"),
            "target_atr_mult": param_float(3.2, "Target ATR Multiplier"),
            "trail_atr_mult": param_float(1.2, "Trail ATR Multiplier"),
            "use_trailing": param_bool(True, "Use Trailing Stop"),
            "max_position_pct": param_float(0.10, "Position % of equity"),
        }
        super().__init__(
            name="concordance_regime_synthesis",
            description="Concordance Regime Synthesis [JOAT] with regime/pressure/participation/structure confluence and ATR risk controls",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        ma_type = str(p.get("ma_type", "EMA")).upper()
        if ma_type not in {"SMA", "EMA", "RMA", "WMA", "VWMA"}:
            ma_type = "EMA"
        return {
            "regime_swing_length": max(2, int(p.get("regime_swing_length", 10))),
            "regime_band_mult": max(0.5, float(p.get("regime_band_mult", 2.0))),
            "regime_center_smooth": max(2, int(p.get("regime_center_smooth", 10))),
            "regime_band_smooth": max(2, int(p.get("regime_band_smooth", 20))),
            "regime_hold_bars": max(0, int(p.get("regime_hold_bars", 2))),
            "ma_type": ma_type,
            "ma_length": max(2, int(p.get("ma_length", 55))),
            "pressure_normalize": max(20, int(p.get("pressure_normalize", 80))),
            "pressure_signal_length": max(1, int(p.get("pressure_signal_length", 8))),
            "delta_wave_length": max(5, int(p.get("delta_wave_length", 20))),
            "delta_smooth_length": max(1, int(p.get("delta_smooth_length", 5))),
            "lattice_lookback": max(20, int(p.get("lattice_lookback", 120))),
            "lattice_rows": max(10, int(p.get("lattice_rows", 40))),
            "confluence_threshold": max(3, min(10, int(p.get("confluence_threshold", 6)))),
            "structure_impulse": max(0.2, float(p.get("structure_impulse", 1.0))),
            "structure_lookback": max(2, int(p.get("structure_lookback", 20))),
            "use_htf_bias": bool(p.get("use_htf_bias", True)),
            "use_session_filter": bool(p.get("use_session_filter", False)),
            "cooldown_bars": max(0, int(p.get("cooldown_bars", 3))),
            "stop_atr_mult": max(0.5, float(p.get("stop_atr_mult", 1.8))),
            "target_atr_mult": max(0.5, float(p.get("target_atr_mult", 3.2))),
            "trail_atr_mult": max(0.5, float(p.get("trail_atr_mult", 1.2))),
            "use_trailing": bool(p.get("use_trailing", True)),
            "max_position_pct": min(max(float(p.get("max_position_pct", 0.10)), 0.0), 1.0),
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=260)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class ConcordanceRegimeSynthesisBt(DecisionRecordingStrategy):
            params = tuple((k, v) for k, v in normalized.items())

            def __init__(self) -> None:
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []

                self.atr_base = [bt.ind.ATR(d, period=55) for d in self.datas]
                self.atr_14 = [bt.ind.ATR(d, period=14) for d in self.datas]
                self.htf_fast = [bt.ind.EMA(d.close, period=21) for d in self.datas]
                self.htf_slow = [bt.ind.EMA(d.close, period=55) for d in self.datas]

                self._last_pivot_high: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._last_pivot_low: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._pending_state: Dict[bt.LineSeries, int] = {d: 0 for d in self.datas}
                self._pending_since: Dict[bt.LineSeries, Optional[int]] = {d: None for d in self.datas}
                self._regime_state: Dict[bt.LineSeries, int] = {d: 0 for d in self.datas}
                self._last_entry_bar: Dict[bt.LineSeries, Optional[int]] = {d: None for d in self.datas}
                self._last_bull_structure: Dict[bt.LineSeries, Optional[int]] = {d: None for d in self.datas}
                self._last_bear_structure: Dict[bt.LineSeries, Optional[int]] = {d: None for d in self.datas}
                self._pressure_ema: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._smooth_buy: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._smooth_sell: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._smooth_total: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._delta_wave: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._long_stop: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._long_target: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._short_stop: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}
                self._short_target: Dict[bt.LineSeries, Optional[float]] = {d: None for d in self.datas}

                self._min_bars = max(
                    60,
                    self.p.lattice_lookback + 5,
                    self.p.pressure_normalize + 5,
                    self.p.ma_length + 5,
                    (2 * self.p.regime_swing_length) + 5,
                )

            def _rolling_max(self, line: bt.LineSeries, lookback: int) -> float:
                use = min(lookback, len(line) - 1)
                return max(float(line[-i]) for i in range(use + 1))

            def _rolling_min(self, line: bt.LineSeries, lookback: int) -> float:
                use = min(lookback, len(line) - 1)
                return min(float(line[-i]) for i in range(use + 1))

            def _sma(self, values: List[float]) -> float:
                return sum(values) / max(len(values), 1)

            def _ema_step(self, prev: Optional[float], value: float, length: int) -> float:
                alpha = 2.0 / (length + 1.0)
                return value if prev is None else (alpha * value + (1.0 - alpha) * prev)

            def _rma_from_values(self, values: List[float], length: int) -> float:
                if not values:
                    return 0.0
                alpha = 1.0 / max(length, 1)
                out = values[0]
                for v in values[1:]:
                    out = alpha * v + (1.0 - alpha) * out
                return out

            def _wma_from_values(self, values: List[float]) -> float:
                n = len(values)
                if n == 0:
                    return 0.0
                wsum = sum((idx + 1) * v for idx, v in enumerate(values))
                return wsum / (n * (n + 1) / 2.0)

            def _ma(self, data: bt.LineSeries, length: int, ma_type: str) -> float:
                use = min(length, len(data))
                closes = [float(data.close[-(use - 1 - j)]) for j in range(use)]
                vols = [float(data.volume[-(use - 1 - j)]) for j in range(use)]
                if ma_type == "SMA":
                    return self._sma(closes)
                if ma_type == "RMA":
                    return self._rma_from_values(closes, use)
                if ma_type == "WMA":
                    return self._wma_from_values(closes)
                if ma_type == "VWMA":
                    den = sum(vols)
                    return self._sma(closes) if den <= 1e-12 else sum(c * v for c, v in zip(closes, vols)) / den
                # EMA
                out: Optional[float] = None
                for c in closes:
                    out = self._ema_step(out, c, use)
                return float(out if out is not None else 0.0)

            def _position_size(self, close: float) -> int:
                if close <= 0:
                    return 0
                target_value = self.broker.getvalue() * float(self.p.max_position_pct)
                return max(0, int(target_value / close))

            def _update_pivots(self, data: bt.LineSeries) -> None:
                s = int(self.p.regime_swing_length)
                if len(data) < (2 * s + 1):
                    return
                highs = [float(data.high[-(2 * s - j)]) for j in range(2 * s + 1)]
                lows = [float(data.low[-(2 * s - j)]) for j in range(2 * s + 1)]
                if highs[s] >= max(highs):
                    self._last_pivot_high[data] = float(data.high[-s])
                if lows[s] <= min(lows):
                    self._last_pivot_low[data] = float(data.low[-s])

            def next(self) -> None:
                self.equity_curve.append({"date": self.datetime.date(0).isoformat(), "value": self.broker.getvalue()})

                for i, data in enumerate(self.datas):
                    if len(data) < self._min_bars:
                        continue
                    self._update_pivots(data)

                    close = float(data.close[0])
                    open_px = float(data.open[0])
                    high = float(data.high[0])
                    low = float(data.low[0])
                    bar_idx = len(data) - 1
                    atr_base = max(float(self.atr_base[i][0]), 1e-9)
                    atr14_prev = float(self.atr_14[i][-1]) if len(data) > 1 else atr_base

                    # Regime engine
                    lp_h = self._last_pivot_high[data] if self._last_pivot_high[data] is not None else high
                    lp_l = self._last_pivot_low[data] if self._last_pivot_low[data] is not None else low
                    midpoint = 0.5 * (lp_h + lp_l)

                    # Smooth center and band using rolling SMA on reconstructed series
                    center_series = []
                    span_center = min(int(self.p.regime_center_smooth), len(data))
                    for j in range(span_center):
                        hh = self._last_pivot_high[data] if self._last_pivot_high[data] is not None else float(data.high[-j])
                        ll = self._last_pivot_low[data] if self._last_pivot_low[data] is not None else float(data.low[-j])
                        center_series.append(0.5 * (hh + ll))
                    regime_center = self._sma(center_series)

                    upper_raw = regime_center + atr_base * float(self.p.regime_band_mult)
                    lower_raw = regime_center - atr_base * float(self.p.regime_band_mult)
                    upper_hist = [upper_raw]
                    lower_hist = [lower_raw]
                    for j in range(1, min(int(self.p.regime_band_smooth), len(data))):
                        c_j = float(data.close[-j])
                        upper_hist.append(c_j + atr_base * float(self.p.regime_band_mult))
                        lower_hist.append(c_j - atr_base * float(self.p.regime_band_mult))
                    regime_upper = self._sma(upper_hist)
                    regime_lower = self._sma(lower_hist)

                    pending_state = self._pending_state[data]
                    pending_since = self._pending_since[data]
                    regime_state = self._regime_state[data]
                    if close > regime_upper and float(data.close[-1]) <= regime_upper:
                        pending_state = 1
                        pending_since = bar_idx
                    if close < regime_lower and float(data.close[-1]) >= regime_lower:
                        pending_state = -1
                        pending_since = bar_idx
                    pending_ready = (
                        pending_state != 0
                        and pending_since is not None
                        and (bar_idx - pending_since) >= int(self.p.regime_hold_bars)
                    )
                    if pending_ready:
                        regime_state = pending_state
                        pending_state = 0
                        pending_since = None
                    if regime_state == 0:
                        regime_state = 1 if close >= regime_center else -1
                    self._pending_state[data] = pending_state
                    self._pending_since[data] = pending_since
                    self._regime_state[data] = regime_state
                    bull_regime = regime_state == 1
                    bear_regime = regime_state == -1
                    regime_normalized = (close - regime_center) / atr_base
                    regime_strength = abs(regime_normalized)

                    # Pressure engine
                    ref_ma = self._ma(data, int(self.p.ma_length), str(self.p.ma_type))
                    raw_distance = close - ref_ma
                    abs_distances = [abs(float(data.close[-j]) - self._ma(data, int(self.p.ma_length), str(self.p.ma_type))) for j in range(min(int(self.p.pressure_normalize), len(data)))]
                    max_distance = max(max(abs_distances) if abs_distances else 0.0, 1e-9)
                    pressure_osc = (raw_distance / max_distance) * 100.0
                    pressure_signal = self._ema_step(self._pressure_ema[data], pressure_osc, int(self.p.pressure_signal_length))
                    self._pressure_ema[data] = pressure_signal

                    # Participation axis
                    range_high = self._rolling_max(data.high, int(self.p.lattice_lookback))
                    range_low = self._rolling_min(data.low, int(self.p.lattice_lookback))
                    step = max((range_high - range_low) / max(int(self.p.lattice_rows), 1), 1e-9)
                    lattice = [0.0 for _ in range(int(self.p.lattice_rows))]
                    effective_lb = min(int(self.p.lattice_lookback), len(data))
                    for j in range(effective_lb):
                        sample_price = float(data.close[-j])
                        idx = int(math.floor((sample_price - range_low) / step))
                        idx = min(max(idx, 0), int(self.p.lattice_rows) - 1)
                        lattice[idx] += float(data.volume[-j])
                    poc_index = max(range(len(lattice)), key=lambda k: lattice[k]) if lattice else 0
                    core_axis = range_low + step * (poc_index + 0.5)
                    deviation_axis = (close - core_axis) / atr_base

                    # Delta wave
                    buy_volume = float(data.volume[0]) if close > open_px else 0.0
                    sell_volume = float(data.volume[0]) if close < open_px else 0.0
                    smooth_buy = self._ema_step(self._smooth_buy[data], buy_volume, int(self.p.delta_wave_length))
                    smooth_sell = self._ema_step(self._smooth_sell[data], sell_volume, int(self.p.delta_wave_length))
                    smooth_total = self._ema_step(self._smooth_total[data], max(float(data.volume[0]), 1.0), int(self.p.delta_wave_length))
                    self._smooth_buy[data] = smooth_buy
                    self._smooth_sell[data] = smooth_sell
                    self._smooth_total[data] = smooth_total
                    delta_raw = ((smooth_buy - smooth_sell) / max(smooth_total, 1e-9)) * 100.0
                    delta_wave = self._ema_step(self._delta_wave[data], delta_raw, int(self.p.delta_smooth_length))
                    delta_slope = 0.0 if self._delta_wave[data] is None else delta_wave - float(self._delta_wave[data])
                    self._delta_wave[data] = delta_wave

                    # Structure engine
                    base_bear = len(data) > 2 and float(data.close[-2]) < float(data.open[-2])
                    base_bull = len(data) > 2 and float(data.close[-2]) > float(data.open[-2])
                    impulse_body = abs(float(data.close[-1]) - float(data.open[-1])) if len(data) > 1 else 0.0
                    impulse_atr = atr14_prev * float(self.p.structure_impulse)
                    demand_impulse = base_bear and len(data) > 1 and float(data.close[-1]) > float(data.open[-1]) and impulse_body >= impulse_atr
                    supply_impulse = base_bull and len(data) > 1 and float(data.close[-1]) < float(data.open[-1]) and impulse_body >= impulse_atr
                    chart_bull_gap = len(data) > 2 and low > float(data.high[-2]) and float(data.close[-1]) > float(data.high[-2])
                    chart_bear_gap = len(data) > 2 and high < float(data.low[-2]) and float(data.close[-1]) < float(data.low[-2])
                    if demand_impulse or chart_bull_gap:
                        self._last_bull_structure[data] = bar_idx
                    if supply_impulse or chart_bear_gap:
                        self._last_bear_structure[data] = bar_idx
                    bull_age = 100000 if self._last_bull_structure[data] is None else bar_idx - int(self._last_bull_structure[data])
                    bear_age = 100000 if self._last_bear_structure[data] is None else bar_idx - int(self._last_bear_structure[data])
                    recent_bull_structure = bull_age <= int(self.p.structure_lookback)
                    recent_bear_structure = bear_age <= int(self.p.structure_lookback)

                    # HTF proxy (same-timeframe approximation)
                    htf_valid = True
                    htf_fast = float(self.htf_fast[i][-1]) if len(data) > 1 else float(self.htf_fast[i][0])
                    htf_slow = float(self.htf_slow[i][-1]) if len(data) > 1 else float(self.htf_slow[i][0])
                    htf_close = float(data.close[-1]) if len(data) > 1 else close
                    htf_bull_bias = htf_valid and htf_fast > htf_slow and htf_close > htf_slow
                    htf_bear_bias = htf_valid and htf_fast < htf_slow and htf_close < htf_slow

                    # Session / cooldown
                    session_ok = not bool(self.p.use_session_filter)
                    last_entry_bar = self._last_entry_bar[data]
                    cooldown_ok = last_entry_bar is None or (bar_idx - int(last_entry_bar)) >= int(self.p.cooldown_bars)

                    long_score = 0
                    long_score += 2 if bull_regime else 0
                    long_score += 1 if regime_normalized >= 0 else 0
                    long_score += 1 if regime_strength >= 0.50 else 0
                    long_score += 1 if pressure_signal > 0 else 0
                    long_score += 1 if delta_wave > 0 else 0
                    long_score += 1 if delta_slope >= 0 else 0
                    long_score += 1 if recent_bull_structure else 0
                    long_score += 1 if close > core_axis else 0
                    long_score += 1 if deviation_axis > 0 else 0
                    long_score += 1 if (not self.p.use_htf_bias or not htf_valid or htf_bull_bias) else 0

                    short_score = 0
                    short_score += 2 if bear_regime else 0
                    short_score += 1 if regime_normalized <= 0 else 0
                    short_score += 1 if regime_strength >= 0.50 else 0
                    short_score += 1 if pressure_signal < 0 else 0
                    short_score += 1 if delta_wave < 0 else 0
                    short_score += 1 if delta_slope <= 0 else 0
                    short_score += 1 if recent_bear_structure else 0
                    short_score += 1 if close < core_axis else 0
                    short_score += 1 if deviation_axis < 0 else 0
                    short_score += 1 if (not self.p.use_htf_bias or not htf_valid or htf_bear_bias) else 0

                    score_spread = long_score - short_score
                    pos = self.getposition(data)
                    long_setup = (
                        session_ok
                        and cooldown_ok
                        and long_score >= int(self.p.confluence_threshold)
                        and score_spread >= 2
                        and pos.size <= 0
                    )
                    short_setup = (
                        session_ok
                        and cooldown_ok
                        and short_score >= int(self.p.confluence_threshold)
                        and score_spread <= -2
                        and pos.size >= 0
                    )

                    if pos.size == 0 and len(data) > 1 and self.getposition(data).size == 0:
                        pass

                    if long_setup:
                        self._long_stop[data] = min(close - atr_base * float(self.p.stop_atr_mult), regime_center - atr_base * 0.25)
                        self._long_target[data] = close + atr_base * float(self.p.target_atr_mult)
                        self._short_stop[data] = None
                        self._short_target[data] = None
                        self._last_entry_bar[data] = bar_idx
                        if pos.size < 0:
                            self.close(data=data)
                        size = self._position_size(close)
                        if size > 0:
                            self.buy(data=data, size=size)
                        continue

                    if short_setup:
                        self._short_stop[data] = max(close + atr_base * float(self.p.stop_atr_mult), regime_center + atr_base * 0.25)
                        self._short_target[data] = close - atr_base * float(self.p.target_atr_mult)
                        self._long_stop[data] = None
                        self._long_target[data] = None
                        self._last_entry_bar[data] = bar_idx
                        if pos.size > 0:
                            self.close(data=data)
                        size = self._position_size(close)
                        if size > 0:
                            self.sell(data=data, size=size)
                        continue

                    if pos.size > 0:
                        base_stop = self._long_stop[data] if self._long_stop[data] is not None else close - atr_base * float(self.p.stop_atr_mult)
                        trail_stop = close - atr_base * float(self.p.trail_atr_mult)
                        self._long_stop[data] = max(base_stop, trail_stop) if self.p.use_trailing else base_stop
                        long_target = self._long_target[data] if self._long_target[data] is not None else close + atr_base * float(self.p.target_atr_mult)
                        long_risk_exit = bear_regime or score_spread <= 0 or pressure_signal < 0 or close < regime_center
                        if close <= float(self._long_stop[data]) or close >= float(long_target) or long_risk_exit:
                            self.close(data=data)
                    elif pos.size < 0:
                        base_stop = self._short_stop[data] if self._short_stop[data] is not None else close + atr_base * float(self.p.stop_atr_mult)
                        trail_stop = close + atr_base * float(self.p.trail_atr_mult)
                        self._short_stop[data] = min(base_stop, trail_stop) if self.p.use_trailing else base_stop
                        short_target = self._short_target[data] if self._short_target[data] is not None else close - atr_base * float(self.p.target_atr_mult)
                        short_risk_exit = bull_regime or score_spread >= 0 or pressure_signal > 0 or close > regime_center
                        if close >= float(self._short_stop[data]) or close <= float(short_target) or short_risk_exit:
                            self.close(data=data)
                    else:
                        self._long_stop[data] = None
                        self._long_target[data] = None
                        self._short_stop[data] = None
                        self._short_target[data] = None

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

        return ConcordanceRegimeSynthesisBt

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
