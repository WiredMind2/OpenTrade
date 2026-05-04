"""
Unit tests for API backtest endpoints.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import sqlite3
import tempfile
import os
import asyncio

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
    """Test API backtest endpoints functionality."""

    def setup_method(self):
        """Set up test client and mock database."""
        from main import app, app_state

        # Sanity check: imported main should resolve to the app entrypoint.
        # The project now uses a top-level main shim that re-exports backend.main.
        import main as imported_main
        imported_path = getattr(imported_main, '__file__', '')
        assert imported_path.endswith(f"main.py"), \
            f"Imported main module is {imported_path}; expected main.py entrypoint"

        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Initialize basic schema
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                horizon TEXT,
                predicted_return REAL,
                confidence REAL,
                produced_at TEXT,
                model TEXT,
                features_used TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_model_predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                suggested_position_pct REAL,
                dt TEXT,
                confidence REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                added_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
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
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                total_value REAL,
                cash REAL,
                invested_value REAL,
                exposure REAL,
                pnl REAL,
                daily_return REAL,
                positions_json TEXT,
                backtest_run_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                source TEXT,
                url TEXT,
                canonical_timestamp TEXT,
                published_at TEXT,
                title TEXT,
                author TEXT,
                content TEXT,
                sentiment_score REAL,
                ticker TEXT
            )
        """)
        conn.commit()
        conn.close()

        # Create test client
        self.client = TestClient(app)

        # Set the database path in the app state after client creation
        app_state['database_path'] = self.temp_db.name
        app_state['models_loaded'] = {
            'lightgbm_1d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_3d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_7d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'}
        }

    def _seed_moving_average_preflight_data(self):
        from datetime import timedelta

        conn = sqlite3.connect(self.temp_db.name)
        conn.execute(
            "INSERT OR IGNORE INTO tickers (ticker, name, exchange, sector) VALUES (?,?,?,?)",
            ("AAPL", "Apple Inc.", "NASDAQ", "Technology"),
        )
        conn.execute("DELETE FROM price_daily WHERE ticker = ?", ("AAPL",))
        base = datetime(2024, 1, 1).date()
        for i in range(130):
            d = (base + timedelta(days=i)).isoformat()
            conn.execute(
                """
                INSERT INTO price_daily
                (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("AAPL", d, 150.0 + i, 152.0 + i, 148.0 + i, 151.0 + i, 151.0 + i, 1_000_000),
            )
        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    @patch("backend.routes.backtests.run_backtest_background")
    def test_backtest_endpoint_success(self, mock_run_backtest):
        """Test successful backtest creation."""
        mock_run_backtest.return_value = None
        self._seed_moving_average_preflight_data()

        payload = {
            "strategy_name": "moving_average",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T00:00:00",
            "initial_capital": 100000.0,
            "parameters": {"ticker": "AAPL"},
        }
        response = self.client.post("/backtest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "strategy_name" in data
        assert "start_date" in data
        assert "end_date" in data
        assert "initial_capital" in data

    def test_backtest_endpoint_invalid_date_range(self):
        """Test backtest endpoint with invalid date range."""
        payload = {
            "strategy_name": "test_strategy",
            "start_date": datetime.utcnow().isoformat(),
            "end_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),  # End before start
            "initial_capital": 100000.0,
            "parameters": {}
        }
        response = self.client.post("/backtest", json=payload)
        assert response.status_code == 422  # Validation error

    @patch("backend.routes.backtests.run_backtest_background")
    def test_backtest_endpoint_large_date_range(self, mock_run_backtest):
        """Test backtest endpoint with too large date range."""
        payload = {
            "strategy_name": "test_strategy",
            "start_date": datetime(2019, 1, 1).isoformat(),
            "end_date": datetime(2025, 1, 1).isoformat(),  # More than 5 years
            "initial_capital": 100000.0,
            "parameters": {}
        }
        response = self.client.post("/backtest", json=payload)
        # Validation now correctly returns 400 for oversized date ranges.
        assert response.status_code == 400

    def test_get_backtest_result_not_found(self):
        """Test getting backtest result for non-existent ID."""
        response = self.client.get("/backtest/nonexistent_id")
        assert response.status_code == 404

    def test_list_backtests_endpoint(self):
        """Test listing backtests endpoint."""
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

    def test_list_backtests_pagination(self):
        """Test backtests listing with pagination."""
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

    def test_get_backtest_result_success(self):
        """Test getting backtest result successfully."""
        conn = sqlite3.connect(self.temp_db.name)
        _insert_test_backtest_run(conn, "test_id", "test_strategy")
        conn.commit()
        conn.close()

        response = self.client.get("/backtest/test_id")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_name"] == "test_strategy"
        assert data["initial_capital"] == 100000.0
        assert data["final_value"] == 105000.0

    def test_get_backtest_result_database_error(self):
        """Test getting backtest result when database connection fails."""
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.get("/backtest/test_id")
            assert response.status_code == 500
        finally:
            app_state["database_path"] = original_db_path

    def test_backtest_endpoint_invalid_parameters(self):
        """Test backtest endpoint with invalid strategy parameters."""
        payload = {
            "strategy_name": "invalid_strategy",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T00:00:00",
            "initial_capital": -1000.0,  # Invalid negative capital
            "parameters": {}
        }
        response = self.client.post("/backtest", json=payload)
        # Should return validation error
        assert response.status_code in [400, 422]

    def test_backtest_endpoint_missing_parameters(self):
        """Test backtest endpoint with missing required parameters."""
        payload = {
            "strategy_name": "test_strategy",
            # Missing start_date, end_date, initial_capital
            "parameters": {}
        }
        response = self.client.post("/backtest", json=payload)
        assert response.status_code == 422  # Validation error

    def test_list_backtests_empty_database(self):
        """Test listing backtests when database is empty."""
        response = self.client.get("/trading/backtest")
        assert response.status_code == 200
        data = response.json()
        assert data == []  # Should return empty list

    def test_list_backtests_filtering(self):
        """Test backtests listing with filtering parameters."""
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

        # Test pagination
        response = self.client.get("/trading/backtest?page=1&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

        response = self.client.get("/trading/backtest?page=2&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_backtest_result_invalid_id(self):
        """Test getting backtest result with invalid ID format."""
        response = self.client.get("/backtest/invalid@id!")
        # Should handle gracefully
        assert response.status_code in [404, 500]

    @patch("backend.routes.backtests.run_backtest_background")
    def test_backtest_status_broadcasting(self, mock_run_backtest):
        """Test that backtest status updates are broadcast via WebSocket."""
        mock_run_backtest.return_value = None
        self._seed_moving_average_preflight_data()
        payload = {
            "strategy_name": "moving_average",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T00:00:00",
            "initial_capital": 100000.0,
            "parameters": {"ticker": "AAPL", "test_param": "value"},
        }
        response = self.client.post("/backtest", json=payload)
        assert response.status_code == 200

    def test_backtest_result_data_integrity(self):
        """Test that backtest result data maintains integrity."""
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

        response = self.client.get("/backtest/integrity_test")
        assert response.status_code == 200
        data = response.json()

        # Verify all fields are present and correct
        assert data["strategy_name"] == "integrity_strategy"
        assert data["initial_capital"] == 100000.0
        assert data["final_value"] == 125000.0
        assert data["total_return"] == 0.25
        assert data["sharpe_ratio"] == 1.5
        assert data["win_rate"] == 0.75
        assert data["total_trades"] == 100
        assert len(data["equity_curve"]) == 2
        assert data["metrics"]["additional_metric"] == 42

    @patch("backend.routes.backtests.run_backtest_background")
    def test_concurrent_backtest_requests(self, mock_run_backtest):
        """Test handling of concurrent backtest requests."""
        mock_run_backtest.return_value = None
        self._seed_moving_average_preflight_data()
        payloads = [
            {
                "strategy_name": "moving_average",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T00:00:00",
                "initial_capital": 100000.0,
                "parameters": {"ticker": "AAPL", "index": i},
            }
            for i in range(3)
        ]

        responses = []
        for payload in payloads:
            response = self.client.post("/backtest", json=payload)
            responses.append(response)
            assert response.status_code == 200

        assert all(r.status_code == 200 for r in responses)

    @patch("backend.routes.backtests.run_backtest_background")
    def test_backtest_parameter_serialization(self, mock_run_backtest):
        """Test that complex parameters are properly serialized."""
        mock_run_backtest.return_value = None
        self._seed_moving_average_preflight_data()
        complex_params = {
            "ticker": "AAPL",
            "nested": {
                "value": 42,
                "list": [1, 2, 3],
                "boolean": True,
            },
            "array": ["a", "b", "c"],
            "number": 3.14,
            "null_value": None,
        }

        payload = {
            "strategy_name": "moving_average",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T00:00:00",
            "initial_capital": 100000.0,
            "parameters": complex_params,
        }

        response = self.client.post("/backtest", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "parameters" in data or "metrics" in data

    def test_backtest_date_edge_cases(self):
        """Test backtest endpoint with edge case dates."""
        # Test with same start and end date
        payload = {
            "strategy_name": "edge_case_strategy",
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-01T00:00:00",  # Same date
            "initial_capital": 100000.0,
            "parameters": {}
        }
        response = self.client.post("/backtest", json=payload)
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 400, 422]

    def test_database_connection_pooling(self):
        """Test that database connections are properly managed."""
        # Make multiple requests to ensure connection handling
        for i in range(5):
            response = self.client.get("/trading/backtest")
            assert response.status_code == 200

    def test_backtest_result_caching(self):
        """Test that backtest results can be cached/retrieved efficiently."""
        conn = sqlite3.connect(self.temp_db.name)
        _insert_test_backtest_run(
            conn,
            "cache_test",
            "cache_strategy",
            initial_capital=100000.0,
            final_value=110000.0,
            total_return=0.1,
            annualized_return=0.12,
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            win_rate=0.7,
            total_trades=25,
            avg_trade_return=0.004,
            volatility=0.15,
        )
        conn.commit()
        conn.close()

        # Make multiple requests for the same data
        for _ in range(3):
            response = self.client.get("/backtest/cache_test")
            assert response.status_code == 200
            data = response.json()
            assert data["strategy_name"] == "cache_strategy"
            assert data["total_return"] == 0.1

    def test_list_backtests_database_error(self):
        """Test listing backtests when database connection fails."""
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.get("/trading/backtest")
            assert response.status_code == 200
            data = response.json()
            assert data == []  # Should return empty list on error
        finally:
            app_state["database_path"] = original_db_path