"""
Smoke-test Backtrader backtests for every registered catalog strategy (except sentiment_ml).
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.routes.backtest_engine import run_backtest_background
from backend.strategies import strategy_registry
from backend.tests.test_backtest_engine import _init_backtest_tables


def _param_defaults(strategy) -> dict:
    schema = getattr(strategy, "parameters_schema", None) or {}
    out: dict = {}
    for key, meta in schema.items():
        if isinstance(meta, dict) and "default" in meta:
            out[key] = meta["default"]
    return out


def _seed_prices(path: str) -> None:
    conn = sqlite3.connect(path)
    _init_backtest_tables(conn)
    start = datetime(2023, 1, 1)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    for i in range(700):  # ~2 years — covers EMA(200) warm-up before the 2024-04 test window
        day = (start + timedelta(days=i)).date().isoformat()
        for j, sym in enumerate(symbols):
            px = 40.0 + i * 0.04 + j * 2.5
            conn.execute(
                """
                INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sym, day, px, px * 1.01, px * 0.99, px, 1000),
            )
    conn.commit()
    conn.close()


class _StubForecast:
    def __init__(self):
        self.predicted_return = 0.001
        self.metadata = {"predicted_path_targets": [0.001, 0.002, 0.003]}
        self.model = type("M", (), {"model_name": "lightgbm_1d"})()


def _fake_predict(self, ticker, horizon, as_of=None, **kwargs):
    return _StubForecast()


@pytest.mark.parametrize(
    "strategy_name",
    [
        s["name"]
        for s in strategy_registry.list(catalog_only=False)
        if s["name"] not in {"sentiment_ml", "macd", "pairs_trading"}
    ],
)
def test_strategy_backtest_completes(tmp_path, strategy_name: str):
    """Each strategy completes signal-less Backtrader run with seeded OHLCV."""
    db_path = str(tmp_path / f"smoke_{strategy_name}.db")
    _seed_prices(db_path)
    strat = strategy_registry.get(strategy_name)
    assert strat is not None
    params = _param_defaults(strat)
    params["ticker"] = "AAPL"
    if strategy_name == "pairs_trading":
        params["pair_ticker"] = "MSFT"

    app_state = {"database_path": db_path, "strategy_registry": strategy_registry}
    backtest_id = f"bt_smoke_{strategy_name}"

    common = dict(
        backtest_id=backtest_id,
        strategy_name=strategy_name,
        start_date=datetime(2024, 4, 1),
        end_date=datetime(2024, 8, 31),
        initial_capital=100_000.0,
        parameters=params,
        app_state=app_state,
    )

    ws_patch = patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock())
    with ws_patch:
            asyncio.run(run_backtest_background(**common))

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT metrics FROM backtest_runs
        WHERE json_extract(metrics, '$.backtest_id') = ?
        ORDER BY id DESC LIMIT 1
        """,
        (backtest_id,),
    ).fetchone()
    conn.close()
    assert row is not None, f"No row for {strategy_name}"
    metrics = json.loads(row[0])
    assert metrics.get("status") == "completed", (strategy_name, metrics)
