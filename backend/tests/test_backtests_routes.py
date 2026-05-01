import sqlite3
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


def _init_backtest_runs_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id TEXT PRIMARY KEY,
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


def test_run_backtest_rejects_too_large_date_range(tmp_path):
    db_path = tmp_path / "backtests_route.db"
    _init_backtest_runs_table(str(db_path))

    client = TestClient(app)
    payload = {
        "strategy_name": "moving_average",
        "start_date": "2015-01-01T00:00:00",
        "end_date": "2023-01-01T00:00:00",
        "initial_capital": 100000.0,
        "parameters": {},
    }

    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": object()}):
        response = client.post("/backtest", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Date range too large (max 5 years)"


def test_get_backtest_result_resolves_string_backtest_id_from_metrics(tmp_path):
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

    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": object()}):
        response = client.get("/backtest/bt_lookup_test")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_name"] == "moving_average"
    assert body["metrics"]["backtest_id"] == "bt_lookup_test"
