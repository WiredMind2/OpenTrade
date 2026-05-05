import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _init_backtest_runs_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            started_at TEXT,
            completed_at TEXT,
            params JSON,
            initial_capital REAL,
            final_value REAL,
            total_return REAL,
            annualized_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_trades INTEGER,
            avg_trade_return REAL,
            volatility REAL,
            equity_curve TEXT,
            metrics JSON
        )
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def client():
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from backend.main import app

    return TestClient(app)


def test_trading_backtest_resolves_string_backtest_id_from_metrics(tmp_path, client):
    db_path = tmp_path / "backtests_route_lookup.db"
    _init_backtest_runs_table(str(db_path))
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO backtest_runs (
            name, started_at, completed_at, params, initial_capital, final_value,
            total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate,
            total_trades, avg_trade_return, volatility, equity_curve, metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "moving_average",
            "2026-04-01T00:00:00",
            "2026-05-01T00:00:00",
            "{}",
            100000.0,
            101000.0,
            0.01,
            0.12,
            0.8,
            0.05,
            0.55,
            10,
            25.0,
            0.2,
            "[]",
            json.dumps({"backtest_id": "bt_lookup_test", "status": "completed"}),
        ),
    )
    conn.commit()
    conn.close()

    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": object()}):
        response = client.get("/trading/backtest", params={"backtest_id": "bt_lookup_test"})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["strategy_name"] == "moving_average"
    assert body[0]["metrics"]["backtest_id"] == "bt_lookup_test"
    assert body[0]["annualized_return"] == 0.12
