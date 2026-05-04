"""
Cross-sectional long/short: rank universes by recent momentum, long top slice, short bottom slice.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Type

import backtrader as bt
import numpy as np

from backend.domain.trading import TargetAllocation
from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_bool, param_float, param_int


def _ceil_pos(n: int, frac: float) -> int:
    return max(1, int(math.ceil(float(n) * float(frac))))


class CrossSectionalLSStrategy(BaseStrategy):
    """Cross-sectional momentum long/short (decile-style sleeves, dollar-neutral target, per-name cap)."""

    def __init__(self) -> None:
        parameters_schema = {
            "short_window": param_int(10, "Lookback bars for short-horizon return used in the ranking signal"),
            "long_window": param_int(30, "Minimum history (bars); long window should exceed short_window"),
            "top_frac": param_float(
                0.1,
                "Fraction of names to long (e.g. 0.1 = top decile by signal)",
                min=0.01,
                max=0.5,
            ),
            "bottom_frac": param_float(
                0.1,
                "Fraction of names to short (e.g. 0.1 = bottom decile)",
                min=0.01,
                max=0.5,
            ),
            "max_gross_exposure": param_float(
                1.0,
                "Target gross notional as fraction of portfolio (long+|short|); 1.0 ≈ 50/50 sleeves before caps",
                min=0.0,
                max=2.0,
            ),
            "max_position_pct": param_float(
                0.1,
                "Per-name cap on absolute target weight (approximates dollar-neutral sizing when binding)",
                min=0.0,
                max=1.0,
            ),
            "single_name_use_momentum": param_bool(
                False,
                "With a single feed only: if true, go long max capped weight when short-window momentum is positive; "
                "otherwise hold cash.",
            ),
        }
        super().__init__(
            name="cross_sectional_ls",
            description=(
                "Each bar, rank all feeds by short-horizon return (close/close[-short_window]-1). "
                "Long the top top_frac slice, short the bottom bottom_frac slice, with equal weight "
                "within each sleeve and target dollar neutrality (gross split max_gross_exposure/2 per side), "
                "each name capped at max_position_pct. "
                "Requires at least two symbols in the backtest: with one feed only, the strategy holds cash "
                "(no short) unless you enable single-name fallback via single_name_use_momentum: then it goes "
                "long max_position_pct if momentum is positive, else flat."
            ),
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        short_window = max(2, int(p.get("short_window", 10)))
        long_window = max(short_window + 1, int(p.get("long_window", 30)))
        top_frac = float(p.get("top_frac", 0.1))
        bottom_frac = float(p.get("bottom_frac", 0.1))
        top_frac = min(max(top_frac, 0.01), 0.5)
        bottom_frac = min(max(bottom_frac, 0.01), 0.5)
        max_gross = min(max(float(p.get("max_gross_exposure", 1.0)), 0.0), 2.0)
        max_position_pct = min(max(float(p.get("max_position_pct", 0.1)), 0.0), 1.0)
        single_name_use_momentum = bool(p.get("single_name_use_momentum", False))
        return {
            "short_window": short_window,
            "long_window": long_window,
            "top_frac": top_frac,
            "bottom_frac": bottom_frac,
            "max_gross_exposure": max_gross,
            "max_position_pct": max_position_pct,
            "single_name_use_momentum": single_name_use_momentum,
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        n = self._normalize_parameters({})
        return capability_profile(min_history_bars=max(n["long_window"] + 5, 40))

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        norm = self._normalize_parameters(parameters)

        class CrossSectionalLS(bt.Strategy):
            params = (
                ("short_window", norm["short_window"]),
                ("long_window", norm["long_window"]),
                ("top_frac", norm["top_frac"]),
                ("bottom_frac", norm["bottom_frac"]),
                ("max_gross_exposure", norm["max_gross_exposure"]),
                ("max_position_pct", norm["max_position_pct"]),
                ("single_name_use_momentum", norm["single_name_use_momentum"]),
            )

            def __init__(self) -> None:
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []

            def _signal(self, data: Any) -> Optional[float]:
                sw = int(self.p.short_window)
                lw = int(self.p.long_window)
                if len(data) <= lw:
                    return None
                try:
                    past = float(data.close[-sw])
                except Exception:
                    return None
                if past <= 0:
                    return None
                return float(data.close[0]) / past - 1.0

            def _target_weights(self) -> Dict[str, float]:
                datas = list(self.datas)
                n = len(datas)
                gross = float(self.p.max_gross_exposure)
                cap = float(self.p.max_position_pct)
                top_f = float(self.p.top_frac)
                bot_f = float(self.p.bottom_frac)

                if n < 1:
                    return {}
                if n == 1:
                    if not bool(self.p.single_name_use_momentum):
                        return {datas[0]._name: 0.0}
                    s = self._signal(datas[0])
                    if s is None:
                        return {datas[0]._name: 0.0}
                    w = min(cap, gross * 0.5) if s > 0 else 0.0
                    return {datas[0]._name: w}

                scored: List[tuple[float, int]] = []
                for i, d in enumerate(datas):
                    sig = self._signal(d)
                    if sig is not None:
                        scored.append((sig, i))
                if len(scored) < 2:
                    return {d._name: 0.0 for d in datas}

                scored.sort(key=lambda x: -x[0])
                m = len(scored)
                n_top = min(_ceil_pos(m, top_f), m)
                n_bot = min(_ceil_pos(m, bot_f), m)
                order = [idx for _, idx in scored]
                long_idx = set(order[:n_top])
                short_idx = set(order[-n_bot:])
                overlap = long_idx & short_idx
                long_idx -= overlap
                short_idx -= overlap

                half = 0.5 * gross
                n_l = max(len(long_idx), 1) if long_idx else 0
                n_s = max(len(short_idx), 1) if short_idx else 0
                raw_long = (half / n_l) if n_l else 0.0
                raw_short = (half / n_s) if n_s else 0.0
                w_long = min(cap, raw_long) if n_l else 0.0
                w_short = min(cap, raw_short) if n_s else 0.0

                out: Dict[str, float] = {d._name: 0.0 for d in datas}
                for i in long_idx:
                    out[datas[i]._name] = w_long
                for i in short_idx:
                    out[datas[i]._name] = -w_short
                return out

            def _apply_targets(self, allocations: Dict[str, float]) -> None:
                for ticker, target_pct in allocations.items():
                    data = next((d for d in self.datas if getattr(d, "_name", None) == ticker), None)
                    if data is None:
                        continue
                    current_position = self.getposition(data).size
                    current_value = float(current_position) * float(data.close[0])
                    portfolio_value = float(self.broker.getvalue())
                    if portfolio_value <= 0:
                        continue
                    target_value = float(target_pct) * portfolio_value
                    if abs(current_value - target_value) < 100:
                        continue
                    px = float(data.close[0])
                    if px <= 0:
                        continue
                    if target_value > current_value:
                        shares_to_buy = int((target_value - current_value) / px)
                        if shares_to_buy > 0:
                            self.buy(data=data, size=shares_to_buy)
                    elif target_value < current_value:
                        shares_to_sell = int((current_value - target_value) / px)
                        if shares_to_sell > 0:
                            self.sell(data=data, size=shares_to_sell)

            def next(self) -> None:
                targets = self._target_weights()
                self._apply_targets(targets)
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

        return CrossSectionalLS

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

        norm = self._normalize_parameters(parameters)
        short_window = int(norm["short_window"])
        long_window = int(norm["long_window"])
        top_frac = float(norm["top_frac"])
        bottom_frac = float(norm["bottom_frac"])
        gross = float(norm["max_gross_exposure"])
        cap = float(norm["max_position_pct"])
        single_fb = bool(norm["single_name_use_momentum"])

        db_path = app_state.get("database_path") or "data/backtest.db"
        own_conn = db_conn is None
        conn = sqlite3.connect(db_path) if own_conn else db_conn
        cur = conn.cursor()

        def closes_for(sym: str) -> Optional[np.ndarray]:
            cur.execute(
                """
                SELECT close
                FROM price_daily
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (sym.upper(), as_of.date().isoformat(), long_window + 1),
            )
            rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
            if len(rows) < long_window + 1:
                return None
            return np.array(rows[::-1], dtype=float)

        try:
            scores: Dict[str, float] = {}
            for sym in symbols:
                arr = closes_for(sym)
                if arr is None or len(arr) < short_window + 1:
                    continue
                past = float(arr[-(short_window + 1)])
                if past <= 0:
                    continue
                cur_px = float(arr[-1])
                scores[sym.upper()] = cur_px / past - 1.0

            out: List[TargetAllocation] = []
            syms_u = [s.upper() for s in symbols]
            n = len(scores)

            if n == 0:
                for sym in syms_u:
                    out.append(
                        TargetAllocation(
                            ticker=sym,
                            target_pct=0.0,
                            reason="insufficient_history",
                            confidence=0.0,
                            timestamp=as_of,
                            metadata={"strategy": self.name},
                        )
                    )
                return out

            if n == 1:
                only = next(iter(scores))
                s = scores[only]
                if not single_fb:
                    tgt = 0.0
                    reason = "single_symbol_hold_cash"
                elif s > 0:
                    tgt = min(cap, 0.5 * gross)
                    reason = "single_symbol_momentum_long"
                else:
                    tgt = 0.0
                    reason = "single_symbol_momentum_flat"
                for sym in syms_u:
                    out.append(
                        TargetAllocation(
                            ticker=sym,
                            target_pct=tgt if sym == only else 0.0,
                            reason=reason,
                            confidence=0.5,
                            timestamp=as_of,
                            metadata={"strategy": self.name, "signal": s},
                        )
                    )
                return out

            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            m = len(ranked)
            n_top = min(_ceil_pos(m, top_frac), m)
            n_bot = min(_ceil_pos(m, bottom_frac), m)
            order = [t for t, _ in ranked]
            long_set = set(order[:n_top])
            short_set = set(order[-n_bot:])
            overlap = long_set & short_set
            long_set -= overlap
            short_set -= overlap

            half = 0.5 * gross
            n_l = max(len(long_set), 1) if long_set else 0
            n_s = max(len(short_set), 1) if short_set else 0
            w_long = min(cap, half / n_l) if n_l else 0.0
            w_short = min(cap, half / n_s) if n_s else 0.0

            weights: Dict[str, float] = {s: 0.0 for s in syms_u}
            for t in long_set:
                weights[t] = w_long
            for t in short_set:
                weights[t] = -w_short

            for sym in syms_u:
                sig = scores.get(sym)
                out.append(
                    TargetAllocation(
                        ticker=sym,
                        target_pct=float(weights.get(sym, 0.0)),
                        reason="cross_sectional_ls",
                        confidence=0.55 if sig is not None else 0.0,
                        timestamp=as_of,
                        metadata={
                            "strategy": self.name,
                            "signal": sig,
                            "in_long": sym in long_set,
                            "in_short": sym in short_set,
                        },
                    )
                )
            return out
        finally:
            if own_conn:
                conn.close()

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        getcontext().prec = 10
        from backend.main import app_state

        norm = self._normalize_parameters(parameters)
        gross = float(norm["max_gross_exposure"])
        cap = float(norm["max_position_pct"])
        try:
            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            rets: List[float] = []
            for tk in tickers:
                cur.execute(
                    """
                    SELECT close FROM price_daily
                    WHERE ticker = ?
                    ORDER BY date DESC
                    LIMIT 61
                    """,
                    (tk,),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                if len(rows) > int(norm["long_window"]):
                    chrono = rows[::-1]
                    sw = int(norm["short_window"])
                    lw = int(norm["long_window"])
                    if len(chrono) > sw and chrono[-(sw + 1)] > 0:
                        mom = chrono[-1] / chrono[-(sw + 1)] - 1.0
                        rets.append(mom)
            conn.close()

            if len(rets) >= 2:
                spread = float(np.std(rets))
                avg_m = float(np.mean(rets))
            else:
                spread = 0.012
                avg_m = 0.0005

            sleeve = min(0.5 * gross * 0.5, cap)
            daily_alpha = sleeve * float(np.tanh(avg_m * 50.0)) * 0.15
            vol_scale = max(0.08, min(0.22, spread * 8.0))
            days_dec = Decimal(str(projection_days))
            init_dec = Decimal(str(initial_capital))
            adj_ret = Decimal(str(daily_alpha)) * days_dec
            final_val = init_dec * (Decimal("1") + adj_ret)

            return {
                "projected_return": float(adj_ret),
                "projected_volatility": round(vol_scale, 6),
                "confidence": 0.55,
                "projection_days": projection_days,
                "initial_capital": float(init_dec),
                "projected_final_value": float(final_val.quantize(Decimal("0.01"))),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            init_dec = Decimal(str(initial_capital))
            fb = Decimal("0.015")
            return {
                "projected_return": float(fb),
                "projected_volatility": 0.14,
                "confidence": 0.35,
                "projection_days": projection_days,
                "initial_capital": float(init_dec),
                "projected_final_value": float((init_dec * (Decimal("1") + fb)).quantize(Decimal("0.01"))),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def train(self, config: Dict[str, Any]) -> Any:
        raise NotImplementedError("Training not supported for rule-based strategies")
