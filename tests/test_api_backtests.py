"""
Unit tests for GET /trading/backtest (list and optional backtest_id lookup).
"""
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.utils.backtest_variants import compute_params_hash


def _insert_test_backtest_run(
    conn,
    client_backtest_id: str,
    name: str = "test_strategy",
    *,
    params=None,
    metrics=None,
    equity_curve: str = "[]",
    started_at=None,
    completed_at=None,
    **cols,
):
    """Insert a row compatible with production ``backtest_runs`` schema."""
    params = params or {}
    started_at = started_at or datetime.utcnow().isoformat()
    completed_at = completed_at or datetime.utcnow().isoformat()
    full_metrics = {"backtest_id": client_backtest_id, "status": "completed"}
    if metrics:
        full_metrics.update(metrics)
    defaults = {
        "initial_capital": 100000.0,
        "final_value": 105000.0,
        "total_return": 0.05,
        "annualized_return": 0.1,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.08,
        "win_rate": 0.65,
        "total_trades": 50,
        "avg_trade_return": 0.01,
        "volatility": 0.15,
    }
    defaults.update(cols)
    conn.execute(
        """
        INSERT INTO backtest_runs (
            name, params, params_hash, client_backtest_id, started_at, completed_at,
            initial_capital, final_value, total_return, annualized_return, sharpe_ratio,
            max_drawdown, win_rate, total_trades, avg_trade_return, volatility,
            equity_curve, metrics
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            name,
            json.dumps(params),
            compute_params_hash(params),
            client_backtest_id,
            started_at,
            completed_at,
            defaults["initial_capital"],
            defaults["final_value"],
            defaults["total_return"],
            defaults["annualized_return"],
            defaults["sharpe_ratio"],
            defaults["max_drawdown"],
            defaults["win_rate"],
            defaults["total_trades"],
            defaults["avg_trade_return"],
            defaults["volatility"],
            equity_curve,
            json.dumps(full_metrics),
        ),
    )


@pytest.mark.unit
class TestAPIBacktestEndpoints:
    """Tests for ``GET /trading/backtest``."""

    def setup_method(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
        from main import app, app_state

        import main as imported_main

        imported_path = getattr(imported_main, "__file__", "")
        assert imported_path.endswith("main.py"), imported_path

        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        conn = sqlite3.connect(self.temp_db.name)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                started_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT,
                params JSON,
                params_hash TEXT,
                variant_label TEXT,
                optimizer_mode TEXT,
                experiment_id TEXT,
                client_backtest_id TEXT,
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

        self.client = TestClient(app)
        app_state["database_path"] = self.temp_db.name
        app_state["models_loaded"] = {
            "lightgbm_1d": {"lgbm": Mock(), "embedder": "all-MiniLM-L6-v2"},
        }

    def teardown_method(self):
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_list_backtests_endpoint(self):
        conn = sqlite3.connect(self.temp_db.name)
        _insert_test_backtest_run(conn, "test_id", "test_strategy")
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["strategy_name"] == "test_strategy"
        assert "annualized_return" in data[0]
        assert "metrics" in data[0]

    def test_list_backtests_pagination(self):
        conn = sqlite3.connect(self.temp_db.name)
        for i in range(5):
            _insert_test_backtest_run(conn, f"test_id_{i}", f"test_strategy_{i}")
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_trading_backtest_by_id_not_found(self):
        response = self.client.get("/trading/backtest", params={"backtest_id": "nonexistent_id"})
        assert response.status_code == 404

    def test_trading_backtest_by_id_success(self):
        conn = sqlite3.connect(self.temp_db.name)
        _insert_test_backtest_run(conn, "test_id", "test_strategy")
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest", params={"backtest_id": "test_id"})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["strategy_name"] == "test_strategy"
        assert data[0]["initial_capital"] == 100000.0
        assert data[0]["final_value"] == 105000.0
        assert data[0]["metrics"]["backtest_id"] == "test_id"

    def test_trading_backtest_by_numeric_id(self):
        conn = sqlite3.connect(self.temp_db.name)
        _insert_test_backtest_run(conn, "client_xyz", "ma_run")
        conn.commit()
        conn.close()
        conn = sqlite3.connect(self.temp_db.name)
        rid = conn.execute("SELECT id FROM backtest_runs ORDER BY id DESC LIMIT 1").fetchone()[0]
        conn.close()

        response = self.client.get("/trading/backtest", params={"backtest_id": str(rid)})
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["strategy_name"] == "ma_run"

    def test_list_backtests_empty_database(self):
        response = self.client.get("/trading/backtest")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_backtests_filtering_pagination(self):
        conn = sqlite3.connect(self.temp_db.name)
        for i in range(5):
            _insert_test_backtest_run(
                conn,
                f"test_id_{i}",
                f"test_strategy_{i}",
                started_at=(datetime.utcnow() - timedelta(days=i)).isoformat(),
            )
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest?page=1&limit=2")
        assert response.status_code == 200
        assert len(response.json()) == 2

        response = self.client.get("/trading/backtest?page=2&limit=2")
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_trading_backtest_by_id_database_error(self):
        from main import app_state

        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        try:
            response = self.client.get("/trading/backtest", params={"backtest_id": "x"})
            assert response.status_code == 500
        finally:
            app_state["database_path"] = original_db_path

    def test_trading_backtest_data_integrity(self):
        conn = sqlite3.connect(self.temp_db.name)
        _insert_test_backtest_run(
            conn,
            "integrity_test",
            "integrity_strategy",
            params={"param1": "value1"},
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-31T00:00:00",
            initial_capital=100000.0,
            final_value=125000.0,
            total_return=0.25,
            annualized_return=0.3,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.75,
            total_trades=100,
            avg_trade_return=0.02,
            volatility=0.2,
            equity_curve='[{"date": "2024-01-01", "value": 100000.0}, {"date": "2024-01-31", "value": 125000.0}]',
            metrics={"additional_metric": 42},
        )
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest", params={"backtest_id": "integrity_test"})
        assert response.status_code == 200
        data = response.json()[0]
        assert data["strategy_name"] == "integrity_strategy"
        assert data["total_return"] == 0.25
        assert data["sharpe_ratio"] == 1.5
        assert data["win_rate"] == 0.75
        assert data["total_trades"] == 100
        assert len(data["equity_curve"]) == 2
        assert data["metrics"]["additional_metric"] == 42
        assert "decision_markers" not in data["metrics"] or isinstance(data["metrics"].get("decision_markers"), list)

    def test_list_backtests_database_error_returns_empty(self):
        from main import app_state

        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        try:
            response = self.client.get("/trading/backtest")
            assert response.status_code == 200
            assert response.json() == []
        finally:
            app_state["database_path"] = original_db_path
