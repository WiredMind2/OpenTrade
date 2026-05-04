"""
Pairs trading: two-leg spread with rolling z-score mean reversion.

Spread definition uses a hedge ratio (rolling level ratio, OLS on log prices, or 1:1).
Only the primary ticker and one partner leg are traded; any additional Backtrader data
feeds are ignored for signals and sizing.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Tuple, Type

import backtrader as bt
import numpy as np

from backend.domain.trading import TargetAllocation
from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_float, param_int, param_str

logger = logging.getLogger(__name__)

_HEDGE_MODES = frozenset({"rolling_ratio", "ols_log", "unit"})


def _resolve_pair_tickers(
    parameters: Dict[str, Any], feed_names: List[str]
) -> Tuple[str, Optional[str]]:
    """Primary from parameters['ticker']; partner from pair_ticker or first other feed name (sorted)."""
    primary = str(parameters.get("ticker") or "AAPL").upper().strip()
    names = [n.upper() for n in feed_names if n]
    hint = parameters.get("pair_ticker") or parameters.get("pair")
    if hint is not None and str(hint).strip():
        sec = str(hint).upper().strip()
        if sec != primary and sec in names:
            return primary, sec
    others = sorted(n for n in names if n != primary)
    return primary, (others[0] if others else None)


def _beta_and_spread(
    p1: np.ndarray,
    p2: np.ndarray,
    mode: str,
) -> Tuple[float, float]:
    """Return (beta, current_spread) using last long_window points (p1, p2 chronologically)."""
    lx = np.log(np.maximum(p1, 1e-12))
    ly = np.log(np.maximum(p2, 1e-12))
    if mode == "unit":
        beta = 1.0
        spread = float(lx[-1] - beta * ly[-1])
        return beta, spread
    if mode == "rolling_ratio":
        m1 = float(np.mean(p1))
        m2 = float(np.mean(p2))
        beta = m1 / max(m2, 1e-12)
        spread = float(p1[-1] - beta * p2[-1])
        return beta, spread
    # ols_log
    x = lx
    y = ly
    vx = float(np.var(x))
    if vx < 1e-16:
        beta = 1.0
    else:
        beta = float(np.cov(x, y, ddof=0)[0, 1] / vx)
    spread = float(lx[-1] - beta * ly[-1])
    return beta, spread


def _zscore_from_spread_history(
    spreads: List[float], short_window: int, long_window: int
) -> Optional[float]:
    if len(spreads) < long_window:
        return None
    w = spreads[-long_window:]
    mu_s = float(np.mean(w[-short_window:]))
    sigma = float(np.std(w, ddof=0))
    if sigma < 1e-12:
        return None
    return (w[-1] - mu_s) / sigma


class PairsTradingStrategy(BaseStrategy):
    """Z-score mean reversion on a two-asset spread."""

    def __init__(self) -> None:
        parameters_schema = {
            "short_window": param_int(
                10,
                "Shorter rolling span for spread mean in z-score (bars); must be < long_window",
            ),
            "long_window": param_int(
                60,
                "Rolling window for hedge ratio and spread volatility in z-score (bars)",
            ),
            "max_position_pct": param_float(
                0.1,
                "Cap on combined gross exposure across the two legs (fraction of equity)",
            ),
            "ticker": param_str("AAPL", "Primary leg ticker (must match a data feed name)"),
            "pair_ticker": param_str(
                "",
                "Second leg; leave empty to use the first other feed ticker alphabetically",
            ),
            "z_entry": param_float(2.0, "Absolute z-score at or beyond which to open a spread"),
            "z_exit": param_float(
                0.5,
                "Mean-reversion band: exit when z crosses back inside this magnitude",
            ),
            "hedge_mode": param_str(
                "ols_log",
                "How to estimate hedge ratio: ols_log | rolling_ratio | unit",
            ),
        }
        super().__init__(
            name="pairs_trading",
            description="Pairs spread z-score mean reversion (two legs only; extra feeds ignored)",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )
        self._signal_regime: int = 0

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        short_window = max(2, int(p.get("short_window", 10)))
        long_window = max(short_window + 1, int(p.get("long_window", 60)))
        max_position_pct = min(max(float(p.get("max_position_pct", 0.1)), 0.0), 1.0)
        z_entry = max(float(p.get("z_entry", 2.0)), 0.25)
        z_exit = min(max(float(p.get("z_exit", 0.5)), 0.05), z_entry - 1e-6)
        mode = str(p.get("hedge_mode", "ols_log")).lower().strip()
        if mode not in _HEDGE_MODES:
            mode = "ols_log"
        return {
            "short_window": short_window,
            "long_window": long_window,
            "max_position_pct": max_position_pct,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "hedge_mode": mode,
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=150)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        normalized = self._normalize_parameters(parameters)

        class PairsZScore(bt.Strategy):
            params = (
                ("short_window", normalized["short_window"]),
                ("long_window", normalized["long_window"]),
                ("max_position_pct", normalized["max_position_pct"]),
                ("z_entry", normalized["z_entry"]),
                ("z_exit", normalized["z_exit"]),
                ("hedge_mode", normalized["hedge_mode"]),
            )

            def __init__(self) -> None:
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []
                self.regime: int = 0
                self.spread_hist: List[float] = []
                self.d0: Optional[bt.LineRoot] = None
                self.d1: Optional[bt.LineRoot] = None
                self._beta: float = 1.0
                names = [getattr(d, "_name", "") or "" for d in self.datas]
                primary, secondary = _resolve_pair_tickers(parameters, names)
                by_name = {getattr(d, "_name", "").upper(): d for d in self.datas}
                self.d0 = by_name.get(primary)
                if secondary:
                    self.d1 = by_name.get(secondary)
                if len(self.datas) > 2:
                    logger.info(
                        "pairs_trading: %d feeds attached; only trading %s and %s; others ignored",
                        len(self.datas),
                        primary,
                        secondary or "?",
                    )

            def _maybe_step_regime(self, z: float) -> None:
                z_in = self.p.z_entry
                z_out = self.p.z_exit
                if self.regime == 0:
                    if z <= -z_in:
                        self.regime = 1
                    elif z >= z_in:
                        self.regime = -1
                elif self.regime == 1:
                    if z >= -z_out:
                        self.regime = 0
                elif self.regime == -1:
                    if z <= z_out:
                        self.regime = 0

            def _close_both(self) -> None:
                if self.d0 is not None:
                    pos = self.getposition(self.d0).size
                    if pos > 0:
                        self.sell(data=self.d0, size=pos)
                    elif pos < 0:
                        self.buy(data=self.d0, size=-pos)
                if self.d1 is not None:
                    pos = self.getposition(self.d1).size
                    if pos > 0:
                        self.sell(data=self.d1, size=pos)
                    elif pos < 0:
                        self.buy(data=self.d1, size=-pos)

            def _open_long_spread(self, p0: float, p1: float) -> None:
                v = self.broker.getvalue()
                each = v * self.p.max_position_pct * 0.5
                n0 = int(max(each / max(p0, 1e-9), 0))
                n1 = int(max(each / max(p1, 1e-9), 0))
                if n0 > 0:
                    self.buy(data=self.d0, size=n0)
                if n1 > 0 and self.d1 is not None:
                    self.sell(data=self.d1, size=n1)

            def _open_short_spread(self, p0: float, p1: float) -> None:
                v = self.broker.getvalue()
                each = v * self.p.max_position_pct * 0.5
                n0 = int(max(each / max(p0, 1e-9), 0))
                n1 = int(max(each / max(p1, 1e-9), 0))
                if n0 > 0 and self.d0 is not None:
                    self.sell(data=self.d0, size=n0)
                if n1 > 0 and self.d1 is not None:
                    self.buy(data=self.d1, size=n1)

            def _sync_positions_to_regime(self, p0: float, p1: float) -> None:
                if self.d0 is None or self.d1 is None:
                    return
                if self.regime == 0:
                    self._close_both()
                    return
                p0s = self.getposition(self.d0).size
                p1s = self.getposition(self.d1).size
                if self.regime == 1:
                    ok = p0s > 0 and p1s < 0
                else:
                    ok = p0s < 0 and p1s > 0
                if not ok:
                    self._close_both()
                    if self.regime == 1:
                        self._open_long_spread(p0, p1)
                    else:
                        self._open_short_spread(p0, p1)

            def next(self) -> None:
                if self.d0 is None or self.d1 is None:
                    return
                p0 = float(self.d0.close[0])
                p1 = float(self.d1.close[0])
                lw = self.p.long_window
                if len(self.d0) < lw or len(self.d1) < lw:
                    return
                p1_arr = np.array([self.d0.close[-i] for i in range(lw, 0, -1)], dtype=float)
                p2_arr = np.array([self.d1.close[-i] for i in range(lw, 0, -1)], dtype=float)
                beta, sp = _beta_and_spread(p1_arr, p2_arr, self.p.hedge_mode)
                self._beta = beta
                self.spread_hist.append(sp)
                cap = max(lw * 3, 200)
                if len(self.spread_hist) > cap:
                    self.spread_hist = self.spread_hist[-cap:]
                z = _zscore_from_spread_history(
                    self.spread_hist, self.p.short_window, self.p.long_window
                )
                if z is None:
                    return
                prev = self.regime
                self._maybe_step_regime(z)
                if self.regime != prev:
                    self._sync_positions_to_regime(p0, p1)
                elif self.regime != 0:
                    self._sync_positions_to_regime(p0, p1)

                cur = self.datas[0].datetime.date(0).isoformat()
                self.equity_curve.append({"date": cur, "value": self.broker.getvalue()})

            def notify_trade(self, trade: Any) -> None:
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

        return PairsZScore

    def generate_target_allocations(
        self,
        parameters: Dict[str, Any],
        symbols: List[str],
        as_of: datetime,
        current_prices: Dict[str, float],
        *,
        db_conn: Optional[sqlite3.Connection] = None,
    ) -> List[TargetAllocation]:
        normalized = self._normalize_parameters(parameters)
        short_w = int(normalized["short_window"])
        long_w = int(normalized["long_window"])
        max_pct = float(normalized["max_position_pct"])
        z_entry = float(normalized["z_entry"])
        z_exit = float(normalized["z_exit"])
        mode = str(normalized["hedge_mode"])
        primary, secondary = _resolve_pair_tickers(parameters, symbols)
        meta_base = {"strategy": self.name, "hedge_mode": mode}

        if not secondary or secondary not in current_prices or primary not in current_prices:
            return [
                TargetAllocation(
                    ticker=primary,
                    target_pct=0.0,
                    reason="no_partner_leg",
                    confidence=0.0,
                    timestamp=as_of,
                    metadata=meta_base,
                )
            ]

        from backend.main import app_state

        db_path = app_state.get("database_path") or "data/backtest.db"
        own = db_conn is None
        conn = sqlite3.connect(db_path) if own else db_conn
        need = long_w + 5

        def _series(sym: str) -> List[Tuple[str, float]]:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT date, close
                FROM price_daily
                WHERE ticker = ? AND date <= ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (sym.upper(), as_of.date().isoformat(), need),
            )
            rows = [(str(r[0]), float(r[1])) for r in cur.fetchall() if r[1] is not None]
            return list(reversed(rows))

        try:
            s1 = _series(primary)
            s2 = _series(secondary)
        finally:
            if own:
                conn.close()

        m1 = {d: c for d, c in s1}
        m2 = {d: c for d, c in s2}
        common = sorted(set(m1) & set(m2))
        if len(common) < long_w:
            self._signal_regime = 0
            return [
                TargetAllocation(
                    ticker=primary,
                    target_pct=0.0,
                    reason="insufficient_history",
                    confidence=0.0,
                    timestamp=as_of,
                    metadata={**meta_base, "partner": secondary},
                ),
                TargetAllocation(
                    ticker=secondary,
                    target_pct=0.0,
                    reason="insufficient_history",
                    confidence=0.0,
                    timestamp=as_of,
                    metadata={**meta_base, "partner": primary},
                ),
            ]

        spreads: List[float] = []
        for i in range(long_w, len(common) + 1):
            sl = common[i - long_w : i]
            a1 = np.array([m1[d] for d in sl], dtype=float)
            a2 = np.array([m2[d] for d in sl], dtype=float)
            _, sp = _beta_and_spread(a1, a2, mode)
            spreads.append(sp)

        z = _zscore_from_spread_history(spreads, short_w, long_w)
        if z is None:
            self._signal_regime = 0
            t0 = t1 = 0.0
            reason = "warmup"
        else:
            prev = self._signal_regime
            if prev == 0:
                if z <= -z_entry:
                    self._signal_regime = 1
                elif z >= z_entry:
                    self._signal_regime = -1
            elif prev == 1:
                if z >= -z_exit:
                    self._signal_regime = 0
            elif prev == -1:
                if z <= z_exit:
                    self._signal_regime = 0

            if self._signal_regime == 0:
                t0 = t1 = 0.0
                reason = "flat"
            elif self._signal_regime == 1:
                t0 = max_pct * 0.5
                t1 = -max_pct * 0.5
                reason = "long_spread"
            else:
                t0 = -max_pct * 0.5
                t1 = max_pct * 0.5
                reason = "short_spread"
            conf = min(1.0, abs(z) / max(z_entry, 1e-9))
            return [
                TargetAllocation(
                    ticker=primary,
                    target_pct=max(-max_pct, min(max_pct, t0)),
                    reason=reason,
                    confidence=float(conf),
                    timestamp=as_of,
                    metadata={**meta_base, "z": z, "partner": secondary},
                ),
                TargetAllocation(
                    ticker=secondary,
                    target_pct=max(-max_pct, min(max_pct, t1)),
                    reason=reason,
                    confidence=float(conf),
                    timestamp=as_of,
                    metadata={**meta_base, "z": z, "partner": primary},
                ),
            ]

        return [
            TargetAllocation(
                ticker=primary,
                target_pct=t0,
                reason=reason,
                confidence=0.0,
                timestamp=as_of,
                metadata={**meta_base, "partner": secondary},
            ),
            TargetAllocation(
                ticker=secondary,
                target_pct=t1,
                reason=reason,
                confidence=0.0,
                timestamp=as_of,
                metadata={**meta_base, "partner": primary},
            ),
        ]

    def train(self, config: Dict[str, Any]) -> Any:
        raise NotImplementedError("Training not supported for rule-based strategies")

    def project(
        self,
        parameters: Dict[str, Any],
        projection_days: int = 30,
        initial_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        getcontext().prec = 10
        normalized = self._normalize_parameters(parameters)
        primary = str(parameters.get("ticker", "AAPL")).upper()
        pair = parameters.get("pair_ticker") or parameters.get("pair") or ""
        pair = str(pair).upper().strip() if pair else ""

        end_date = datetime.utcnow().date()
        spread_vols: List[float] = []
        try:
            from backend.main import app_state

            conn = sqlite3.connect(app_state.get("database_path", "data/backtest.db"))
            cur = conn.cursor()
            for sym in [primary, pair] if pair else [primary]:
                cur.execute(
                    """
                    SELECT close FROM price_daily
                    WHERE ticker = ? AND date <= ?
                    ORDER BY date DESC LIMIT 120
                    """,
                    (sym, end_date.isoformat()),
                )
                closes = [float(r[0]) for r in cur.fetchall() if r[0] is not None][::-1]
                if len(closes) > 5:
                    r = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
                    spread_vols.append(float(np.std(r)))
            conn.close()
        except Exception:
            spread_vols = []

        vol = float(np.mean(spread_vols)) if spread_vols else 0.012
        mr_edge = vol * 0.15
        days_dec = Decimal(str(max(1, projection_days)))
        initial_dec = Decimal(str(initial_capital))
        projected_return = Decimal(str(mr_edge)) * days_dec * Decimal("0.6")
        projected_final = initial_dec * (Decimal("1") + projected_return)
        confidence = 0.55 if vol < 0.02 else 0.45

        return {
            "projected_return": float(projected_return),
            "projected_volatility": round(vol * float(days_dec) ** 0.5, 6),
            "confidence": confidence,
            "projection_days": projection_days,
            "initial_capital": float(initial_dec),
            "projected_final_value": float(projected_final.quantize(Decimal("0.01"))),
            "primary": primary,
            "pair": pair or None,
            "hedge_mode": normalized["hedge_mode"],
            "timestamp": datetime.utcnow().isoformat(),
        }
