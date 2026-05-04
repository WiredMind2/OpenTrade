"""
RL-style portfolio allocator (torch-free).

Policy-gradient *heuristic*: per-asset Sharpe-like scores → softmax logits
(temperature-scaled), daily rebalance, deterministic turnover smoothing.

**Exposure model (bounded gross, long-only):** target weights are nonnegative
and renormalized so ``sum_i w_i = max_gross``. The stack runs a long-only
broker (no ``shortcash``), so dollar-neutral (sum w ≈ 0) targets are not used
here; see module docstring / strat_report alignment.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Optional, Type

import backtrader as bt
import numpy as np

from backend.domain.trading import TargetAllocation
from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_float, param_int


# Fixed deterministic turnover penalty (larger → smaller step toward new targets).
_TURNOVER_LAMBDA = 1.0


def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, float]:
    p = parameters or {}
    short_window = max(2, int(p.get("short_window", 10)))
    long_window = max(short_window + 1, int(p.get("long_window", 60)))
    lookbacks = max(long_window + 2, short_window + 2, int(p.get("lookbacks", long_window + 2)))
    max_gross = min(max(float(p.get("max_gross", 1.0)), 1e-6), 10.0)
    temperature = max(float(p.get("temperature", 1.0)), 1e-9)
    return {
        "short_window": float(short_window),
        "long_window": float(long_window),
        "lookbacks": float(lookbacks),
        "max_gross": max_gross,
        "temperature": temperature,
    }


def _closes_to_returns(closes: np.ndarray) -> np.ndarray:
    if closes.size < 2:
        return np.array([], dtype=float)
    c = np.asarray(closes, dtype=float)
    prev = c[:-1]
    nxt = c[1:]
    mask = prev > 0
    r = np.zeros_like(prev)
    r[mask] = (nxt[mask] - prev[mask]) / prev[mask]
    return r


def _sharpe_like_scores(
    closes_by_asset: List[np.ndarray],
    short_window: int,
    long_window: int,
) -> np.ndarray:
    """One Sharpe-like score per asset; NaN where insufficient data."""
    n = len(closes_by_asset)
    scores = np.full(n, np.nan, dtype=float)
    for i, closes in enumerate(closes_by_asset):
        r = _closes_to_returns(closes)
        if r.size < long_window:
            continue
        r_short = r[-short_window:]
        r_long = r[-long_window:]
        mu = float(np.mean(r_short))
        sig = float(np.std(r_long, ddof=0))
        scores[i] = mu / (sig + 1e-8)
    return scores


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    valid = np.isfinite(scores)
    if not np.any(valid):
        k = scores.size
        return np.ones(k, dtype=float) / max(k, 1)
    z = scores.copy()
    z[~valid] = -1e18
    z = (z - np.nanmax(z)) / temperature
    ex = np.exp(np.clip(z, -60.0, 60.0))
    ex[~valid] = 0.0
    s = ex.sum()
    if s <= 0:
        k = scores.size
        return np.ones(k, dtype=float) / max(k, 1)
    return ex / s


def _allocate_long_only(
    scores: np.ndarray,
    max_gross: float,
    temperature: float,
    w_prev: Optional[np.ndarray],
) -> np.ndarray:
    p = _softmax_weights(scores, temperature)
    w_star = max_gross * p
    n = w_star.size
    if w_prev is None or w_prev.shape != w_star.shape:
        w_prev = np.full(n, max_gross / max(n, 1), dtype=float)
    l1 = float(np.sum(np.abs(w_star - w_prev)))
    mix = 1.0 / (1.0 + _TURNOVER_LAMBDA * l1)
    w = w_prev + mix * (w_star - w_prev)
    tot = float(np.sum(w))
    if tot > 1e-12:
        w = w * (max_gross / tot)
    else:
        w = np.full(n, max_gross / max(n, 1), dtype=float)
    return w


class RLPortfolioAllocatorStrategy(BaseStrategy):
    """
    Deterministic multi-asset allocator: softmax over Sharpe-like scores,
    gross exposure capped at ``max_gross``, daily rebalance with turnover
    smoothing (fixed λ). No torch / no learned weights.
    """

    def __init__(self):
        parameters_schema = {
            "max_gross": param_float(
                1.0,
                "Sum of nonnegative target weights (gross long exposure cap).",
                minimum=0.01,
                maximum=10.0,
            ),
            "temperature": param_float(
                1.0,
                "Softmax temperature; higher → closer to equal-weight.",
                minimum=1e-6,
                maximum=100.0,
            ),
            "lookbacks": param_int(
                62,
                "Minimum aligned bars required before trading (warmup floor).",
                minimum=5,
                maximum=5000,
            ),
            "short_window": param_int(
                10,
                "Return window (bars) for the Sharpe-like numerator (mean return).",
                minimum=2,
                maximum=500,
            ),
            "long_window": param_int(
                60,
                "Return window (bars) for the Sharpe-like denominator (volatility).",
                minimum=3,
                maximum=2000,
            ),
        }
        super().__init__(
            name="rl_portfolio_allocator",
            description=(
                "Torch-free RL-style allocator: softmax over rolling Sharpe-like "
                "scores, bounded gross long-only exposure, daily rebalance with "
                "deterministic turnover penalty."
            ),
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    def get_capability_profile(self) -> Dict[str, Any]:
        cfg = _normalize_parameters({})
        return capability_profile(min_history_bars=int(cfg["lookbacks"]))

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        norm = _normalize_parameters(parameters)
        short_window = int(norm["short_window"])
        long_window = int(norm["long_window"])
        lookbacks = int(norm["lookbacks"])
        max_gross = float(norm["max_gross"])
        temperature = float(norm["temperature"])
        min_bars = max(lookbacks, long_window + 1, short_window + 1)

        class RLPortfolioAllocator(bt.Strategy):
            params = (
                ("short_window", short_window),
                ("long_window", long_window),
                ("lookbacks", lookbacks),
                ("max_gross", max_gross),
                ("temperature", temperature),
            )

            def __init__(self):
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []
                self._w_prev: Optional[np.ndarray] = None
                self._min_bars = min_bars

            def next(self):
                if len(self) < self._min_bars:
                    return
                n = len(self.datas)
                closes_by: List[np.ndarray] = []
                tickers: List[str] = []
                for d in self.datas:
                    if len(d) < self.p.long_window + 1:
                        return
                    tickers.append(getattr(d, "_name", "unknown"))
                    buf = max(self.p.long_window + 1, self.p.short_window + 1)
                    arr = np.array([d.close[-j] for j in range(buf - 1, -1, -1)], dtype=float)
                    closes_by.append(arr)

                scores = _sharpe_like_scores(closes_by, self.p.short_window, self.p.long_window)
                w = _allocate_long_only(scores, self.p.max_gross, self.p.temperature, self._w_prev)
                self._w_prev = w.copy()
                allocations = {tickers[i]: float(w[i]) for i in range(n)}

                for ticker, target_pct in allocations.items():
                    data = next(
                        (x for x in self.datas if getattr(x, "_name", None) == ticker),
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

        return RLPortfolioAllocator

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

        norm = _normalize_parameters(parameters)
        short_window = int(norm["short_window"])
        long_window = int(norm["long_window"])
        max_gross = float(norm["max_gross"])
        temperature = float(norm["temperature"])
        need = max(long_window + 1, short_window + 1)
        db_path = app_state.get("database_path") or "data/backtest.db"

        own_conn = db_conn is None
        conn = sqlite3.connect(db_path) if own_conn else db_conn
        closes_by: List[np.ndarray] = []
        ordered_syms: List[str] = []
        try:
            cur = conn.cursor()
            for symbol in symbols:
                sym = symbol.upper()
                ordered_syms.append(sym)
                cur.execute(
                    """
                    SELECT close
                    FROM price_daily
                    WHERE ticker = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT ?
                    """,
                    (sym, as_of.date().isoformat(), need),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                if len(rows) < need:
                    closes_by.append(np.array([], dtype=float))
                else:
                    closes_by.append(np.array(rows[::-1], dtype=float))
        finally:
            if own_conn:
                conn.close()

        scores = _sharpe_like_scores(closes_by, short_window, long_window)
        w = _allocate_long_only(scores, max_gross, temperature, None)
        valid = np.array([c.size >= need for c in closes_by], dtype=bool)
        w = w * valid.astype(float)
        tot = float(np.sum(w))
        if tot > 1e-12:
            w = w * (max_gross / tot)
        else:
            w = np.zeros_like(w)
        out: List[TargetAllocation] = []
        for i, sym in enumerate(ordered_syms):
            if closes_by[i].size < need:
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
                continue
            finite = scores[np.isfinite(scores)]
            if finite.size:
                lo, hi = float(np.min(finite)), float(np.max(finite))
                conf = float(min(1.0, max(0.0, (scores[i] - lo) / (hi - lo + 1e-9))))
            else:
                conf = 0.5
            out.append(
                TargetAllocation(
                    ticker=sym,
                    target_pct=float(w[i]),
                    reason="rl_softmax_sharpe",
                    confidence=conf,
                    timestamp=as_of,
                    metadata={
                        "strategy": self.name,
                        "score": float(scores[i]) if np.isfinite(scores[i]) else None,
                        "weight": float(w[i]),
                        "current_price": float(current_prices.get(sym, 0.0) or 0.0),
                    },
                )
            )
        return out

    def train(self, config: Dict[str, Any]) -> Any:
        raise NotImplementedError("Training not supported for rule-based strategies")

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        getcontext().prec = 10
        norm = _normalize_parameters(parameters)
        max_gross = float(norm["max_gross"])
        temperature = float(norm["temperature"])

        try:
            from backend.main import app_state

            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
            short_w = int(norm["short_window"])
            long_w = int(norm["long_window"])
            need = max(long_w + 1, short_w + 1)
            closes_by: List[np.ndarray] = []
            for t in tickers:
                cur.execute(
                    """
                    SELECT close FROM price_daily
                    WHERE ticker = ? ORDER BY date DESC LIMIT ?
                    """,
                    (t, need),
                )
                rows = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
                closes_by.append(np.array(rows[::-1], dtype=float) if len(rows) >= need else np.array([]))
            conn.close()

            scores = _sharpe_like_scores(closes_by, short_w, long_w)
            w = _allocate_long_only(scores, max_gross, temperature, None)
            # Heuristic projected portfolio drift from score dispersion (deterministic).
            disp = float(np.nanstd(scores)) if np.any(np.isfinite(scores)) else 0.0
            edge = min(0.03, max(-0.02, disp * 0.002 * (max_gross / max(temperature, 0.5))))
            proj_ret = Decimal(str(edge)) * Decimal(str(projection_days)) / Decimal("252")

            initial_capital_dec = Decimal(str(initial_capital))
            projected_final = initial_capital_dec * (Decimal("1") + proj_ret)

            return {
                "projected_return": float(proj_ret),
                "projected_volatility": round(float(disp * 0.5), 6),
                "confidence": 0.55,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(projected_final.quantize(Decimal("0.01"))),
                "allocator_weights_preview": [float(x) for x in w.tolist()],
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            initial_capital_dec = Decimal(str(initial_capital))
            fallback = Decimal("0.01")
            return {
                "projected_return": float(fallback),
                "projected_volatility": 0.12,
                "confidence": 0.35,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float((initial_capital_dec * (Decimal("1") + fallback)).quantize(Decimal("0.01"))),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
