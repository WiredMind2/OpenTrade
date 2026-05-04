"""
Tabular RL-style directional baseline (§5.2 strat_report).

Discretized recent-return state, three discrete actions {long, flat, short}
mapped to target weights +max_position_pct, 0, -max_position_pct per ticker.
Online numpy Q-updates inside Backtrader ``next()`` (lightweight baseline, not
offline DQN training with torch/gym).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Any, Dict, List, Type

import backtrader as bt
import numpy as np

from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_float, param_int, param_bool


ACTION_SHORT = 0
ACTION_FLAT = 1
ACTION_LONG = 2


def _discretize_return(r: float, n_bins: int, clip: float) -> int:
    if n_bins <= 1:
        return 0
    c = max(float(clip), 1e-9)
    x = float(np.clip(r, -c, c))
    # Map [-clip, clip] -> [0, n_bins - 1]
    t = (x + c) / (2.0 * c)
    idx = int(t * n_bins)
    return int(np.clip(idx, 0, n_bins - 1))


def _state_index(r_short: float, r_long: float, n_bins: int, clip: float) -> int:
    i_s = _discretize_return(r_short, n_bins, clip)
    i_l = _discretize_return(r_long, n_bins, clip)
    return i_s * n_bins + i_l


class RLDirectionalStrategy(BaseStrategy):
    """Tabular Q-style directional agent (online updates in Backtrader only)."""

    def __init__(self) -> None:
        parameters_schema = {
            "short_window": param_int(5, "Lookback bars for short-horizon cumulative return (state feature)"),
            "long_window": param_int(20, "Lookback bars for long-horizon cumulative return (state feature)"),
            "max_position_pct": param_float(0.1, "Absolute target weight per ticker for long/short legs"),
            "epsilon": param_float(0.12, "Epsilon-greedy exploration probability"),
            "learning_rate": param_float(0.2, "TD learning rate for tabular Q updates"),
            "gamma": param_float(0.99, "Discount factor for one-step TD backup"),
            "n_return_bins": param_int(5, "Bins per return axis (state space size = n_return_bins^2)"),
            "return_clip": param_float(0.05, "Clip cumulative returns to [-clip, clip] before binning"),
            "online_q_updates": param_bool(
                True,
                "If false, use fixed epsilon-greedy over a randomly initialized Q table (no TD updates)",
            ),
        }
        super().__init__(
            name="rl_directional",
            description=(
                "Lightweight tabular RL-style baseline: discretized momentum state, "
                "three actions (long / flat / short) as ±max position or flat, with optional "
                "online TD(0) Q-updates in Backtrader next(). Not full DQN / offline RL training."
            ),
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False,
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        p = parameters or {}
        short_window = max(2, int(p.get("short_window", 5)))
        long_window = max(short_window + 1, int(p.get("long_window", 20)))
        max_position_pct = min(max(float(p.get("max_position_pct", 0.1)), 0.0), 1.0)
        epsilon = min(max(float(p.get("epsilon", 0.12)), 0.0), 1.0)
        learning_rate = min(max(float(p.get("learning_rate", 0.2)), 1e-6), 1.0)
        gamma = min(max(float(p.get("gamma", 0.99)), 0.0), 0.9999)
        n_return_bins = max(2, min(int(p.get("n_return_bins", 5)), 32))
        return_clip = max(float(p.get("return_clip", 0.05)), 1e-6)
        online_q_updates = bool(p.get("online_q_updates", True))
        return {
            "short_window": short_window,
            "long_window": long_window,
            "max_position_pct": max_position_pct,
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "n_return_bins": n_return_bins,
            "return_clip": return_clip,
            "online_q_updates": online_q_updates,
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=120)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        n = self._normalize_parameters(parameters)

        class TabularRLDirectional(bt.Strategy):
            params = (
                ("short_window", n["short_window"]),
                ("long_window", n["long_window"]),
                ("max_position_pct", n["max_position_pct"]),
                ("epsilon", n["epsilon"]),
                ("learning_rate", n["learning_rate"]),
                ("gamma", n["gamma"]),
                ("n_return_bins", n["n_return_bins"]),
                ("return_clip", n["return_clip"]),
                ("online_q_updates", n["online_q_updates"]),
            )

            def __init__(self):
                self.equity_curve: List[Dict[str, Any]] = []
                self.trades: List[Dict[str, Any]] = []
                try:
                    self.broker.set_shortcash(True)
                except Exception:
                    pass
                nb = int(self.p.n_return_bins)
                n_states = nb * nb
                rng = np.random.default_rng()
                self._Q = [
                    rng.standard_normal((n_states, 3)).astype(np.float64) * 0.01
                    for _ in self.datas
                ]
                self._prev_state: List[int | None] = [None] * len(self.datas)
                self._prev_action: List[int | None] = [None] * len(self.datas)

            def _horizon_return(self, data: bt.LineRoot, lookback: int) -> float | None:
                if len(data) <= lookback:
                    return None
                c0 = float(data.close[0])
                c_past = float(data.close[-lookback])
                if c_past <= 0 or c0 <= 0:
                    return None
                return c0 / c_past - 1.0

            def _pick_action(self, qi: int, state_idx: int) -> int:
                if np.random.random() < float(self.p.epsilon):
                    return int(np.random.randint(0, 3))
                qrow = self._Q[qi][state_idx]
                return int(np.argmax(qrow))

            def _action_to_target_pct(self, action: int) -> float:
                m = float(self.p.max_position_pct)
                if action == ACTION_LONG:
                    return m
                if action == ACTION_SHORT:
                    return -m
                return 0.0

            def _reward_mult(self, action: int) -> float:
                if action == ACTION_LONG:
                    return 1.0
                if action == ACTION_SHORT:
                    return -1.0
                return 0.0

            def next(self):
                current_date = self.datas[0].datetime.date(0).isoformat()
                allocations: Dict[str, float] = {}
                nb = int(self.p.n_return_bins)
                clip = float(self.p.return_clip)
                alpha = float(self.p.learning_rate)
                gamma = float(self.p.gamma)
                learn = bool(self.p.online_q_updates)

                for qi, data in enumerate(self.datas):
                    ticker = getattr(data, "_name", f"d{qi}")
                    sw, lw = int(self.p.short_window), int(self.p.long_window)
                    r_s = self._horizon_return(data, sw)
                    r_l = self._horizon_return(data, lw)
                    if r_s is None or r_l is None:
                        continue

                    state_idx = _state_index(r_s, r_l, nb, clip)

                    prev_s = self._prev_state[qi]
                    prev_a = self._prev_action[qi]
                    if prev_s is not None and prev_a is not None and len(data) > 1:
                        bar_ret = float(data.close[0] / data.close[-1] - 1.0)
                        r_step = self._reward_mult(prev_a) * bar_ret
                        if learn:
                            max_next = float(np.max(self._Q[qi][state_idx]))
                            td_err = r_step + gamma * max_next - float(self._Q[qi][prev_s, prev_a])
                            self._Q[qi][prev_s, prev_a] += alpha * td_err

                    action = self._pick_action(qi, state_idx)
                    self._prev_state[qi] = state_idx
                    self._prev_action[qi] = action
                    allocations[str(ticker)] = self._action_to_target_pct(action)

                total_exposure = sum(abs(v) for v in allocations.values())
                max_gross = float(self.p.max_position_pct)
                if total_exposure > max_gross and total_exposure > 0:
                    scale = max_gross / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                portfolio_value = self.broker.getvalue()
                for ticker, target_pct in allocations.items():
                    data = next(
                        (d for d in self.datas if getattr(d, "_name", None) == ticker),
                        None,
                    )
                    if data is None:
                        continue
                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    target_value = target_pct * portfolio_value
                    if abs(current_value - target_value) < 100:
                        continue
                    if target_value > current_value:
                        shares = int((target_value - current_value) / data.close[0])
                        if shares > 0:
                            self.buy(data=data, size=shares)
                    else:
                        shares = int((current_value - target_value) / data.close[0])
                        if shares > 0:
                            self.sell(data=data, size=shares)

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

        return TabularRLDirectional

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        """Projection heuristic (same shape as other rule strategies; not RL rollouts)."""
        getcontext().prec = 10
        n = self._normalize_parameters(parameters)
        _ = n
        try:
            from backend.main import app_state
            import sqlite3

            db_path = app_state.get("database_path", "data/backtest.db")
            conn = sqlite3.connect(db_path)
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=90)
            cur = conn.cursor()
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            portfolio_data: Dict[str, Dict[str, Any]] = {}
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
                        volatility = float(np.std(returns)) if returns.size > 0 else 0.0
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
                projected_final_value = initial_capital_dec * (Decimal("1") + projected_return_dec)
                return {
                    "projected_return": float(projected_return_dec),
                    "projected_volatility": 0.16,
                    "confidence": 0.45,
                    "projection_days": projection_days,
                    "initial_capital": float(initial_capital_dec),
                    "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
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
            total_return = portfolio_return * float(projection_days_dec)
            # Directional / exploratory agent: damp vs buy-and-hold
            strategy_multiplier = Decimal("0.62")
            adjusted_return = Decimal(str(total_return)) * strategy_multiplier
            adjusted_volatility = float(portfolio_volatility * Decimal("0.95"))
            projected_final_value = initial_capital_dec * (Decimal("1") + adjusted_return)

            return {
                "projected_return": float(adjusted_return),
                "projected_volatility": round(adjusted_volatility, 6),
                "confidence": 0.55,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
                "market_return": float(Decimal(str(total_return))),
                "strategy_multiplier": float(strategy_multiplier),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            initial_capital_dec = Decimal(str(initial_capital))
            fallback_return = Decimal("0.01")
            projected_final_value = initial_capital_dec * (Decimal("1") + fallback_return)
            return {
                "projected_return": float(fallback_return),
                "projected_volatility": 0.14,
                "confidence": 0.3,
                "projection_days": projection_days,
                "initial_capital": float(initial_capital_dec),
                "projected_final_value": float(projected_final_value.quantize(Decimal("0.01"))),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
