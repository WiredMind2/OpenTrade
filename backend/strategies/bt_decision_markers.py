"""
Chart overlay markers for Backtrader strategy runs (``metrics.decision_markers``).

The backtest engine reads ``strategy.decision_markers`` after ``cerebro.run()``.
Strategies should inherit ``DecisionRecordingStrategy`` instead of ``bt.Strategy`` so
``buy``/``sell`` are instrumented once with no per-strategy marker boilerplate.
"""

from __future__ import annotations

from typing import Any

import backtrader as bt

_MAX_MARKERS: int | None = None


def _agent_log(message: str, data: dict[str, Any]) -> None:
    try:
        import json
        import time

        payload = {
            "sessionId": "2acb83",
            "runId": "post-fix",
            "hypothesisId": "H5",
            "location": "bt_decision_markers.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(
            r"c:\Users\willi\Documents\Python\Trading\backtesting\.cursor\debug-2acb83.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def record_bt_decision(strategy: Any, *, ticker: str, side: str, reason: str) -> None:
    """Append one buy/sell marker at the current bar (primary data feed date)."""
    if not hasattr(strategy, "decision_markers") or strategy.decision_markers is None:
        strategy.decision_markers = []
    markers = strategy.decision_markers
    if _MAX_MARKERS is not None and len(markers) >= _MAX_MARKERS:
        _agent_log("Marker cap hit; dropping marker", {"maxMarkers": _MAX_MARKERS, "currentMarkers": len(markers)})
        return
    s = (side or "").strip().lower()
    if s not in ("buy", "sell"):
        return
    tkr = (ticker or "").strip().upper() or "UNKNOWN"
    r = (reason or "").strip() or "signal"
    datas = getattr(strategy, "datas", None)
    if not datas:
        return
    day = datas[0].datetime.date(0).isoformat()[:10]
    markers.append({"date": day, "side": s, "ticker": tkr, "reason": r})
    if len(markers) in (1, 100, 500, 1000, 2000, 5000):
        _agent_log("Marker count milestone", {"markers": len(markers), "day": day, "side": s, "ticker": tkr})


class DecisionRecordingStrategy(bt.Strategy):
    """Same as ``bt.Strategy``, but appends to ``decision_markers`` on each ``buy``/``sell`` with size > 0."""

    def buy(self, *args: Any, **kwargs: Any) -> Any:
        out = super().buy(*args, **kwargs)
        self._capture_order_marker(kwargs, side="buy")
        return out

    def sell(self, *args: Any, **kwargs: Any) -> Any:
        out = super().sell(*args, **kwargs)
        self._capture_order_marker(kwargs, side="sell")
        return out

    def _capture_order_marker(self, kwargs: dict[str, Any], *, side: str) -> None:
        sz = kwargs.get("size")
        if sz is None:
            return
        try:
            n = int(sz)
        except (TypeError, ValueError):
            return
        if n <= 0:
            return
        data = kwargs.get("data")
        if data is None and getattr(self, "datas", None):
            data = self.datas[0] if len(self.datas) == 1 else None
        if data is None:
            return
        ticker = str(getattr(data, "_name", "") or "UNKNOWN").upper()
        reason = f"{self.__class__.__name__}_{side}"
        record_bt_decision(self, ticker=ticker, side=side, reason=reason)
