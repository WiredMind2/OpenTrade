"""
Shared recursive-forecast strategy implementation.

This module provides a reusable strategy base that translates forecast paths
into portfolio allocations. Concrete strategies can inherit from this class
to avoid duplicating runtime signal logic.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import backtrader as bt

from backend.domain.trading import TargetAllocation
from backend.logging_config import get_component_logger
from backend.ml.forecasting import BacktestBridge
from backend.ml.prediction_service import PredictionService
from backend.strategies.base import BaseStrategy
from backend.strategies.bt_decision_markers import DecisionRecordingStrategy


logger = get_component_logger(__file__)


class RecursiveForecastStrategy(BaseStrategy):
    """
    Reusable strategy runtime that supports:
    - DB-backed prediction lookup
    - recursive prediction fallback via PredictionService
    - path-to-signal conversion via BacktestBridge
    """

    def get_capability_profile(self) -> Dict[str, Any]:
        return {
            # Forecasts are produced at runtime from price history + loaded horizon models;
            # trading_model_predictions rows are optional (DB cache / offline batch only).
            "requires_predictions": False,
            "required_prediction_horizons": ["1d", "3d", "7d"],
            "supports_signal_execution": True,
            "supports_backtrader_execution": True,
            "min_history_bars": 60,
            "supported_objectives": ["balanced", "sharpe", "return", "drawdown"],
        }

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        model_name = getattr(self, "model_name", self.name)

        class RecursiveForecastBacktrader(DecisionRecordingStrategy):
            params = (
                ("model_name", "sentiment_model"),
                ("prediction_threshold", 0.002),
                ("max_position_pct", 0.1),
                ("forecast_horizon_days", 5),
            )

            def __init__(self):
                self.equity_curve = []
                self.trades = []
                self.model_name = model_name
                self.prediction_threshold = self.p.prediction_threshold
                self.max_position_pct = self.p.max_position_pct
                self.forecast_horizon_days = self.p.forecast_horizon_days
                self.db_path = getattr(self, "db_path", "data/backtest.db")
                self.signal_bridge = BacktestBridge(threshold=self.prediction_threshold)
                self._prediction_cache: Dict[tuple, Dict[str, Any]] = {}
                self._missing_model_log_keys = set()

            def next(self):
                current_date = self.datas[0].datetime.date(0).isoformat()
                allocations = {}

                for data in self.datas:
                    ticker = data._name
                    prediction = self._get_prediction(ticker, current_date)
                    if prediction is None:
                        continue

                    predicted_return = prediction.get("predicted_return", 0.0)
                    suggested_position_pct = prediction.get("suggested_position_pct", 0.0)

                    if abs(predicted_return) >= self.prediction_threshold:
                        signal_pct = suggested_position_pct * (1 if predicted_return > 0 else -1)
                        allocations[ticker] = signal_pct
                    else:
                        allocations[ticker] = 0.0

                total_exposure = sum(abs(pct) for pct in allocations.values())
                if total_exposure > self.max_position_pct and total_exposure > 0:
                    scale = self.max_position_pct / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                for ticker, target_pct in allocations.items():
                    data = next((d for d in self.datas if hasattr(d, "_name") and d._name == ticker), None)
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

                self.equity_curve.append({"date": current_date, "value": self.broker.getvalue()})

            def _resolve_horizon(self) -> str:
                horizon_days = max(1, int(self.forecast_horizon_days))
                if horizon_days <= 1:
                    return "1d"
                if horizon_days <= 3:
                    return "3d"
                return "7d"

            def _candidate_horizons(self) -> List[str]:
                primary = self._resolve_horizon()
                fallbacks = [primary, "3d", "1d"]
                ordered: List[str] = []
                for horizon in fallbacks:
                    if horizon not in ordered:
                        ordered.append(horizon)
                return ordered

            def _get_prediction(self, ticker: str, date: str) -> Dict[str, Any] | None:
                cache_key = (ticker, date)
                if cache_key in self._prediction_cache:
                    cached = self._prediction_cache[cache_key]
                    if cached.get("_missing"):
                        return None
                    return cached

                try:
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT predicted_return, enter_prob, suggested_position_pct, exit_prob, model
                        FROM trading_model_predictions
                        WHERE ticker = ? AND dt <= ?
                        ORDER BY dt DESC, produced_at DESC
                        LIMIT 1
                        """,
                        (ticker, date),
                    )
                    row = cur.fetchone()
                    conn.close()
                    if row:
                        payload = {
                            "predicted_return": row[0],
                            "enter_prob": row[1],
                            "suggested_position_pct": row[2],
                            "exit_prob": row[3],
                            "model": row[4],
                        }
                        self._prediction_cache[cache_key] = payload
                        return payload

                    from backend.main import app_state

                    service = PredictionService(
                        database_path=app_state.get("database_path", self.db_path),
                        models_loaded=app_state.get("models_loaded", {}),
                    )
                    result = None
                    bar_as_of = datetime.strptime(date, "%Y-%m-%d")
                    for horizon in self._candidate_horizons():
                        try:
                            result = service.predict(ticker=ticker, horizon=horizon, as_of=bar_as_of)
                            break
                        except KeyError:
                            continue
                    if result is None:
                        self._prediction_cache[cache_key] = {"_missing": True}
                        log_key = (ticker, tuple(self._candidate_horizons()))
                        if log_key not in self._missing_model_log_keys:
                            logger.error(
                                "No recursive forecast model available for %s using horizons %s",
                                ticker,
                                self._candidate_horizons(),
                            )
                            self._missing_model_log_keys.add(log_key)
                        return None
                    targets = result.metadata.get("predicted_path_targets") or [result.predicted_return]
                    signal = self.signal_bridge.to_signal(type("ForecastProxy", (), {"predicted_targets": targets})())
                    payload = {
                        "predicted_return": signal.get("terminal_prediction", result.predicted_return),
                        "enter_prob": max(0.0, min(1.0, abs(signal.get("mean_prediction", 0.0)) * 10)),
                        "suggested_position_pct": signal.get("long_signal", 0.0) * self.max_position_pct,
                        "exit_prob": 0.0 if signal.get("long_signal", 0.0) > 0 else 0.5,
                        "model": result.model.model_name,
                    }
                    self._prediction_cache[cache_key] = payload
                    return payload
                except Exception as exc:
                    logger.warning("Error getting recursive forecast for %s on %s: %s", ticker, date, exc)
                    self._prediction_cache[cache_key] = {"_missing": True}
                    return None

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

        return RecursiveForecastBacktrader

    def generate_target_allocations(
        self,
        parameters: Dict[str, Any],
        symbols: List[str],
        as_of: datetime,
        current_prices: Dict[str, float],
        *,
        db_conn: Optional[sqlite3.Connection] = None,
    ) -> List[TargetAllocation]:
        params = parameters or {}
        threshold = max(float(params.get("prediction_threshold", 0.002)), 0.0)
        max_position_pct = min(max(float(params.get("max_position_pct", 0.1)), 0.0), 1.0)
        horizon_days = max(1, int(params.get("forecast_horizon_days", 5)))
        if horizon_days <= 1:
            candidate_horizons = ["1d"]
        elif horizon_days <= 3:
            candidate_horizons = ["3d", "1d"]
        else:
            candidate_horizons = ["7d", "3d", "1d"]

        from backend.main import app_state

        service = PredictionService(
            database_path=app_state.get("database_path", "data/backtest.db"),
            models_loaded=app_state.get("models_loaded", {}),
        )
        db_path = app_state.get("database_path", "data/backtest.db")

        allocations: List[TargetAllocation] = []
        own_conn = db_conn is None
        conn = sqlite3.connect(db_path) if own_conn else db_conn
        try:
            cur = conn.cursor()
            for symbol in symbols:
                if float(current_prices.get(symbol, 0.0) or 0.0) <= 0:
                    continue

                predicted_return = 0.0
                confidence = 0.0
                used_horizon = None
                source = "service"
                suggested_position_pct = 0.0

                try:
                    cur.execute(
                        """
                        SELECT predicted_return, enter_prob, suggested_position_pct
                        FROM trading_model_predictions
                        WHERE ticker = ? AND dt <= ?
                        ORDER BY dt DESC, produced_at DESC
                        LIMIT 1
                        """,
                        (symbol.upper(), as_of.date().isoformat()),
                    )
                except sqlite3.OperationalError:
                    cur.execute(
                        """
                        SELECT predicted_return, enter_prob, suggested_position_pct
                        FROM trading_model_predictions
                        WHERE ticker = ? AND dt <= ?
                        ORDER BY dt DESC
                        LIMIT 1
                        """,
                        (symbol.upper(), as_of.date().isoformat()),
                    )
                row = cur.fetchone()
                if row is not None:
                    source = "db"
                    predicted_return = float(row[0] or 0.0)
                    confidence = min(max(float(row[1] or 0.0), 0.0), 1.0)
                    suggested_position_pct = abs(float(row[2] or 0.0))
                else:
                    result = None
                    for horizon in candidate_horizons:
                        try:
                            result = service.predict(ticker=symbol, horizon=horizon, as_of=as_of)
                            used_horizon = horizon
                            break
                        except KeyError:
                            continue
                    if result is None:
                        allocations.append(
                            TargetAllocation(
                                ticker=symbol.upper(),
                                target_pct=0.0,
                                reason="no_model_available",
                                confidence=0.0,
                                timestamp=as_of,
                                metadata={"requested_horizons": candidate_horizons, "strategy": self.name},
                            )
                        )
                        continue
                    else:
                        predicted_return = float(result.predicted_return)
                        confidence = min(max(float(result.confidence), 0.0), 1.0)

                if abs(predicted_return) < threshold:
                    target_pct = 0.0
                    reason = "below_threshold"
                else:
                    direction = 1.0 if predicted_return > 0 else -1.0
                    strength_scale = min(
                        abs(predicted_return) / max(threshold if threshold > 0 else 0.001, 1e-9),
                        1.0,
                    )
                    confidence_scale = max(confidence, 0.25)
                    size = max_position_pct * strength_scale * confidence_scale
                    if suggested_position_pct > 1.0:
                        suggested_position_pct = suggested_position_pct / 100.0
                    if suggested_position_pct > 0.0:
                        size = max(size, max_position_pct * min(suggested_position_pct, 1.0))
                    target_pct = direction * min(max_position_pct, size)
                    reason = "forecast_signal"
                allocations.append(
                    TargetAllocation(
                        ticker=symbol.upper(),
                        target_pct=target_pct,
                        reason=reason,
                        confidence=confidence,
                        timestamp=as_of,
                        metadata={
                            "predicted_return": predicted_return,
                            "horizon": used_horizon,
                            "source": source,
                            "strategy": self.name,
                        },
                    )
                )
        finally:
            if own_conn:
                conn.close()
        return allocations
