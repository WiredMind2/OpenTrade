"""
Integration tests for the trading backtesting system.
"""
import json
import pytest
import sqlite3
import tempfile
import os
import gc
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from main import app
from fastapi.testclient import TestClient


async def _integration_backtest_quick_finish(
    backtest_id: str,
    strategy_name: str,
    start_date,
    end_date,
    initial_capital: float,
    parameters: dict,
    app_state: dict,
):
    """Persist a minimal completed run so polling tests stay fast and deterministic."""
    from backend.utils.backtest_variants import compute_params_hash, variant_label_from_params

    params = parameters or {}
    params_hash = compute_params_hash(params)
    variant_label = params.get("variant_label") or variant_label_from_params(params)
    db_path = app_state["database_path"]
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(
        """
        INSERT INTO backtest_runs (
            name, params, params_hash, variant_label, optimizer_mode, experiment_id,
            client_backtest_id, started_at, completed_at, initial_capital,
            final_value, total_return, annualized_return, sharpe_ratio,
            max_drawdown, win_rate, total_trades, avg_trade_return,
            volatility, equity_curve, metrics
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            strategy_name,
            json.dumps(params),
            params_hash,
            variant_label,
            params.get("optimizer_mode"),
            params.get("experiment_id"),
            backtest_id,
            start_date.isoformat(),
            datetime.utcnow().isoformat(),
            initial_capital,
            float(initial_capital) * 1.01,
            0.01,
            0.02,
            0.5,
            0.05,
            0.55,
            2,
            0.001,
            0.12,
            "[]",
            json.dumps({"backtest_id": backtest_id, "status": "completed"}),
        ),
    )
    conn.commit()
    conn.close()


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Integration tests for complete trading workflow."""

    def setup_method(self):
        """Set up test environment."""
        import sys
        from pathlib import Path
        backend_path = str(Path(__file__).parent.parent / 'backend')
        # Ensure backend package is preferred on import path
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Initialize database with required tables
        conn = sqlite3.connect(self.temp_db.name)
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
            CREATE TABLE IF NOT EXISTS model_jobs (
                id TEXT PRIMARY KEY,
                model_name TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                config TEXT,
                result TEXT,
                error TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_model_predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                suggested_position_pct REAL,
                dt TEXT,
                confidence REAL,
                predicted_return REAL,
                enter_prob REAL,
                exit_prob REAL,
                model TEXT
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
        conn.commit()
        conn.close()

        # Set up app state
        from main import app_state
        from backend.strategies import strategy_registry
        app_state['database_path'] = self.temp_db.name
        # Re-discover strategies after sys.path is set
        from pathlib import Path
        strategies_pkg_dir = Path(__file__).parent.parent / 'backend' / 'strategies'
        strategy_registry.discover(strategies_pkg_dir)
        app_state['strategy_registry'] = strategy_registry
        self.registry = strategy_registry
        app_state.setdefault('start_time', datetime.utcnow())
        app_state.setdefault('active_websockets', set())
        app_state['models_loaded'] = {
            'lightgbm_1d': {'lgbm': MagicMock(), 'embedder': 'all-MiniLM-L6-v2'},
        }

        self.client = TestClient(app)
        self.auth_headers = {"Authorization": "Bearer test-token"}

    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, "client"):
            self.client.close()
        if os.path.exists(self.temp_db.name):
            for _ in range(10):
                try:
                    os.unlink(self.temp_db.name)
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.05)

    def _seed_moving_average_preflight_data(self):
        from datetime import timedelta

        conn = sqlite3.connect(self.temp_db.name)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                added_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
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

    def test_backtest_list_and_lookup_workflow(self):
        """Persisted runs are listed and retrievable via ``GET /trading/backtest``."""
        from backend.utils.backtest_variants import compute_params_hash

        conn = sqlite3.connect(self.temp_db.name)
        params = {"ticker": "AAPL"}
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
                "moving_average",
                json.dumps(params),
                compute_params_hash(params),
                "bt_integration_workflow",
                "2024-01-01T00:00:00",
                "2024-12-31T00:00:00",
                100000.0,
                101000.0,
                0.01,
                0.02,
                0.5,
                0.03,
                0.4,
                3,
                0.0,
                0.1,
                "[]",
                json.dumps(
                    {
                        "backtest_id": "bt_integration_workflow",
                        "status": "completed",
                        "decision_markers": [{"date": "2024-06-01", "side": "buy", "ticker": "AAPL"}],
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

        r = self.client.get("/trading/backtest", params={"limit": 20})
        assert r.status_code == 200
        rows = r.json()
        assert any(x.get("metrics", {}).get("backtest_id") == "bt_integration_workflow" for x in rows)

        one = self.client.get("/trading/backtest", params={"backtest_id": "bt_integration_workflow"})
        assert one.status_code == 200
        payload = one.json()
        assert len(payload) == 1
        assert payload[0]["metrics"]["decision_markers"][0]["side"] == "buy"

    def test_price_data_endpoint_integration(self):
        """Test price data endpoint with real database data."""
        # Insert test data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-01", 150.0, 152.0, 148.0, 151.0, 151.0, 1000000))
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-02", 151.0, 153.0, 149.0, 152.0, 152.0, 1000000))
        conn.commit()
        conn.close()

        # Test endpoint
        response = self.client.get("/data/prices/AAPL")
        assert response.status_code == 200

        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "data" in data
        assert len(data["data"]) >= 1
        # Data is ordered by date DESC, so first record is most recent
        assert data["data"][0]["close"] == 152.0

    def test_health_endpoint_integration(self):
        """Test health endpoint provides system status."""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data

    @pytest.mark.asyncio
    async def test_websocket_broadcast_integration(self):
        """Test WebSocket message broadcasting."""
        from routes.websocket import broadcast_websocket_message

        # Mock app_state with no connections (should not fail)
        with patch('main.app_state', {'active_websockets': set()}):
            message = {"type": "test", "message": "integration test"}
            await broadcast_websocket_message(message)
    @pytest.mark.asyncio
    async def test_websocket_end_to_end_message_flow(self):
        """Test end-to-end WebSocket message flow from backend broadcasting."""
        import asyncio
        from unittest.mock import AsyncMock
        from routes.websocket import broadcast_websocket_message, active_connections

        # Create a mock WebSocket client
        mock_ws_client = AsyncMock()
        received_messages = []

        # Mock the send_json method to capture messages
        async def capture_message(message):
            received_messages.append(message)

        mock_ws_client.send_json = capture_message

        # Add mock client to active connections
        active_connections.clear()
        active_connections.add(mock_ws_client)

        try:
            # Test 1: Broadcast a test message
            test_message = {
                "type": "test_message",
                "data": {"message": "Hello WebSocket"}
            }

            await broadcast_websocket_message(test_message)

            # Verify message was received
            assert len(received_messages) == 1
            assert received_messages[0] == test_message

            # Test 2: Simulate script execution status update
            script_status_message = {
                "type": "script_status",
                "data": {
                    "script_name": "backtest_runner",
                    "status": "completed",
                    "execution_id": "test_exec_123",
                    "start_time": datetime.utcnow().isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "output": "Script completed successfully",
                    "error": None,
                    "duration_seconds": 5.2
                }
            }

            await broadcast_websocket_message(script_status_message)

            # Verify script status message was received
            assert len(received_messages) == 2
            assert received_messages[1]["type"] == "script_status"
            assert received_messages[1]["data"]["script_name"] == "backtest_runner"
            assert received_messages[1]["data"]["status"] == "completed"

            # Test 3: Simulate pipeline status update
            pipeline_status_message = {
                "type": "pipeline_status",
                "data": {
                    "execution_id": "pipeline_test_456",
                    "current_step": "ingest_prices",
                    "completed_steps": ["apply_schema", "download_kaggle"],
                    "failed_steps": [],
                    "status": "running",
                    "start_time": datetime.utcnow().isoformat(),
                    "estimated_completion": None
                }
            }

            await broadcast_websocket_message(pipeline_status_message)

            # Verify pipeline status message was received
            assert len(received_messages) == 3
            assert received_messages[2]["type"] == "pipeline_status"
            assert received_messages[2]["data"]["current_step"] == "ingest_prices"
            assert "apply_schema" in received_messages[2]["data"]["completed_steps"]

            # Test 4: Simulate backtest status update
            backtest_status_message = {
                "type": "backtest_status",
                "data": {
                    "strategy_name": "sentiment_momentum",
                    "start_date": "2024-01-01",
                    "end_date": datetime.utcnow().isoformat(),
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
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": {
                        "backtest_id": "backtest_789",
                        "status": "completed"
                    },
                    "equity_curve": [
                        {"date": "2024-01-01", "value": 100000.0},
                        {"date": "2024-01-31", "value": 105000.0}
                    ]
                }
            }

            await broadcast_websocket_message(backtest_status_message)

            # Verify backtest status message was received
            assert len(received_messages) == 4
            assert received_messages[3]["type"] == "backtest_status"
            assert received_messages[3]["data"]["strategy_name"] == "sentiment_momentum"
            assert received_messages[3]["data"]["final_value"] == 105000.0
            assert received_messages[3]["data"]["metrics"]["status"] == "completed"

        finally:
            # Clean up
            active_connections.clear()
            received_messages.clear()
    @pytest.mark.asyncio
    async def test_script_execution_websocket_updates(self):
        """Test that script execution sends WebSocket status updates."""
        from unittest.mock import AsyncMock, patch
        from routes.websocket import broadcast_websocket_message, active_connections
        from routes.scripts import run_script_async, script_executions

        # Create a mock WebSocket client to capture messages
        mock_ws_client = AsyncMock()
        received_messages = []

        async def capture_message(message):
            received_messages.append(message)

        mock_ws_client.send_json = capture_message
        active_connections.clear()
        active_connections.add(mock_ws_client)

        try:
            # Mock the script execution to simulate completion
            with patch('routes.scripts.os.environ.get') as mock_env, \
                 patch('routes.scripts.asyncio.create_subprocess_exec') as mock_subprocess:

                # Allow execution (not in test mode)
                mock_env.return_value = None

                # Mock successful subprocess execution
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b"Script output", b"")
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process

                # Set up the execution record
                execution_id = "test_script_exec_123"
                script_executions[execution_id] = {
                    "script_name": "backtest_runner",
                    "status": "running",
                    "start_time": datetime.utcnow(),
                    "parameters": {},
                    "output": "",
                    "error": "",
                    "process": None
                }

                # Execute script
                await run_script_async(execution_id, "backtest_runner", {}, {})

                # Verify WebSocket message was sent
                assert len(received_messages) == 1
                message = received_messages[0]
                assert message["type"] == "script_status"
                assert message["data"]["script_name"] == "backtest_runner"
                assert message["data"]["status"] == "completed"
                assert message["data"]["execution_id"] == execution_id
                assert message["data"]["output"] == "Script output"
                assert message["data"]["error"] is None
                assert "duration_seconds" in message["data"]

        finally:
            active_connections.clear()
            received_messages.clear()

    def test_strategy_train_api_unsupported_strategy(self):
        """Moving-average training is parameter optimization and requires ticker and dates."""
        payload = {"config": {}}

        response = self.client.post("/api/strategies/moving_average/train", json=payload, headers=self.auth_headers)
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "ticker is required for strategy parameter training" in data["detail"]

    def _seed_multi_ticker_daily_prices(self, days: int = 420) -> None:
        """Seed enough calendar days to satisfy strategy preflight (e.g. 120+ bars in-range)."""
        conn = sqlite3.connect(self.temp_db.name)
        start = datetime(2024, 1, 1)
        for sym, bias in (("AAPL", 0.0), ("MSFT", 20.0), ("GOOGL", 40.0)):
            for i in range(days):
                day = (start + timedelta(days=i)).date().isoformat()
                c = 80.0 + i * 0.15 + bias
                conn.execute(
                    """
                    INSERT INTO price_daily
                    (ticker, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sym, day, c, c * 1.01, c * 0.99, c, c, 1000),
                )
        conn.commit()
        conn.close()

    def test_strategy_train_mean_reversion_signal_optimize_integration(self):
        """Mimics UI: POST /api/strategies/mean_reversion/train with dates and ticker."""
        self._seed_multi_ticker_daily_prices()
        response = self.client.post(
            "/api/strategies/mean_reversion/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "objective": "balanced",
                "max_evals": 6,
            },
            headers=self.auth_headers,
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["strategy"] == "mean_reversion"
        assert data["evaluations_run"] == 6
        assert "best_params" in data

    def test_strategy_train_pairs_trading_pair_ticker_validation_integration(self):
        """Pairs training without pair_ticker returns 400 with actionable detail."""
        self._seed_multi_ticker_daily_prices()
        response = self.client.post(
            "/api/strategies/pairs_trading/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "max_evals": 2,
            },
            headers=self.auth_headers,
        )
        assert response.status_code == 400
        assert "pair_ticker" in response.json()["detail"].lower()

    def test_strategy_train_pairs_trading_with_pair_integration(self):
        self._seed_multi_ticker_daily_prices()
        response = self.client.post(
            "/api/strategies/pairs_trading/train",
            json={
                "ticker": "AAPL",
                "pair_ticker": "MSFT",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "max_evals": 4,
            },
            headers=self.auth_headers,
        )
        assert response.status_code == 200, response.text
        assert response.json()["evaluations_run"] == 4

    def test_get_model_job_status(self):
        """Test GET /api/model_jobs/{job_id} endpoint."""
        # Insert a test job
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO model_jobs (id, model_name, status, created_at, config)
            VALUES (?, ?, ?, ?, ?)
        """, ("test_job_456", "sentiment_ml", "completed", "2024-01-01T10:00:00", '{"param": "value"}'))
        conn.commit()
        conn.close()

        # Test getting job status
        response = self.client.get("/api/model_jobs/test_job_456", headers=self.auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["job_id"] == "test_job_456"
        assert data["model_name"] == "sentiment_ml"
        assert data["status"] == "completed"
        assert data["config"] == '{"param": "value"}'

    def test_get_model_job_status_not_found(self):
        """Test GET /api/model_jobs/{job_id} for non-existent job."""
        response = self.client.get("/api/model_jobs/non_existent_job", headers=self.auth_headers)
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]

    def test_moving_average_backtest_end_to_end(self):
        """moving_average runs are exposed like other stored rows (unified GET /trading/backtest)."""
        from main import app_state
        from backend.utils.backtest_variants import compute_params_hash

        registry = app_state.get("strategy_registry")
        strategy = registry.get("moving_average")
        assert strategy is not None
        assert strategy.name == "moving_average"

        conn = sqlite3.connect(self.temp_db.name)
        params = {
            "ticker": "AAPL",
            "short_window": 10,
            "long_window": 30,
            "max_position_pct": 0.1,
        }
        bid = "bt_ma_e2e_integration"
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
                "moving_average",
                json.dumps(params),
                compute_params_hash(params),
                bid,
                "2024-01-01T00:00:00",
                "2024-12-31T00:00:00",
                100000.0,
                101000.0,
                0.01,
                0.02,
                0.5,
                0.03,
                0.4,
                3,
                0.0,
                0.1,
                "[]",
                json.dumps(
                    {
                        "backtest_id": bid,
                        "status": "completed",
                        "start_date": "2024-01-01T00:00:00",
                        "end_date": "2024-12-31T00:00:00",
                        "decision_markers": [{"date": "2024-06-01", "side": "buy", "ticker": "AAPL"}],
                    }
                ),
            ),
        )
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest", params={"backtest_id": bid})
        assert response.status_code == 200
        row = response.json()[0]
        assert row["strategy_name"] == "moving_average"
        assert row["metrics"]["backtest_id"] == bid
        assert row["metrics"]["decision_markers"][0]["side"] == "buy"

    @pytest.mark.asyncio
    async def test_pipeline_execution_websocket_updates(self):
        """Test that pipeline execution sends WebSocket status updates."""
        from unittest.mock import AsyncMock, patch
        from routes.websocket import broadcast_websocket_message, active_connections
        from routes.scripts import run_pipeline_async, script_executions

        # Create a mock WebSocket client to capture messages
        mock_ws_client = AsyncMock()
        received_messages = []

        async def capture_message(message):
            received_messages.append(message)

        mock_ws_client.send_json = capture_message
        active_connections.clear()
        active_connections.add(mock_ws_client)

        # Mock the pipeline execution
        with patch('routes.scripts.os.environ.get') as mock_env, \
             patch('routes.scripts.asyncio.create_subprocess_exec') as mock_subprocess:

            # Allow execution (not in test mode)
            mock_env.return_value = None

            # Mock successful subprocess execution for each step
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Step completed", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            # Set up the execution record
            execution_id = "test_pipeline_exec_456"
            script_executions[execution_id] = {
                "script_name": "run_pipeline",
                "status": "running",
                "start_time": datetime.utcnow(),
                "parameters": {"steps": ["ingest_prices"]},
                "current_step": None,
                "completed_steps": [],
                "failed_steps": [],
                "output": "",
                "error": "",
            }

            # Execute pipeline with single step
            await run_pipeline_async(execution_id, ["ingest_prices"], {})

            # Should receive at least one pipeline status update
            pipeline_messages = [msg for msg in received_messages if msg["type"] == "pipeline_status"]
            assert len(pipeline_messages) >= 1

            # Check the final status message
            final_message = pipeline_messages[-1]
            assert final_message["data"]["execution_id"] == execution_id
            assert final_message["data"]["status"] == "completed"
            assert "ingest_prices" in final_message["data"]["completed_steps"]
    @pytest.mark.asyncio
    async def test_backtest_execution_websocket_updates(self):
        """Test that backtest execution sends WebSocket status updates."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from routes.websocket import broadcast_websocket_message, active_connections
        from routes.backtest_engine import run_backtest_background

        # Create a mock WebSocket client to capture messages
        mock_ws_client = AsyncMock()
        received_messages = []

        async def capture_message(message):
            received_messages.append(message)

        mock_ws_client.send_json = capture_message
        active_connections.clear()
        active_connections.add(mock_ws_client)

        try:
            # Mock the backtest execution
            with patch('routes.backtest_engine.sqlite3') as mock_sqlite, \
                 patch('routes.backtest_engine.bt') as mock_bt:

                # Mock database connection and cursor
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.cursor.return_value = mock_cursor
                mock_sqlite.connect.return_value = mock_conn

                # Mock Backtrader components
                mock_cerebro = MagicMock()
                mock_strategy = MagicMock()
                mock_analyzer = MagicMock()
                mock_analyzer.get_analysis.return_value = {
                    'sharperatio': 1.2,
                    'max': {'drawdown': 0.08}
                }
                mock_strategy.analyzers.sharpe = mock_analyzer
                mock_strategy.analyzers.drawdown = mock_analyzer
                mock_strategy.analyzers.returns = mock_analyzer
                mock_strategy.analyzers.trades = mock_analyzer
                mock_strategy.equity_curve = [
                    {'date': '2024-01-01', 'value': 100000.0},
                    {'date': '2024-01-31', 'value': 105000.0}
                ]
                mock_strategy.trades = []

                mock_cerebro.addstrategy.return_value = None
                mock_cerebro.adddata.return_value = None
                mock_cerebro.broker.getvalue.return_value = 105000.0
                mock_cerebro.run.return_value = [mock_strategy]

                mock_bt.Cerebro.return_value = mock_cerebro
                mock_bt.feeds.PandasData.return_value = MagicMock()
                mock_bt.analyzers.SharpeRatio.return_value = mock_analyzer
                mock_bt.analyzers.DrawDown.return_value = mock_analyzer
                mock_bt.analyzers.Returns.return_value = mock_analyzer
                mock_bt.analyzers.TradeAnalyzer.return_value = mock_analyzer

                # Set up app state
                from main import app_state
                app_state['database_path'] = ':memory:'

                # Execute backtest
                backtest_id = "test_backtest_ws_789"
                await run_backtest_background(
                    backtest_id=backtest_id,
                    strategy_name="sentiment_momentum",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 12, 31),
                    initial_capital=100000.0,
                    parameters={},
                    app_state=app_state
                )

                # Should receive a backtest status update
                backtest_messages = [msg for msg in received_messages if msg["type"] == "backtest_status"]
                assert len(backtest_messages) == 1

                message = backtest_messages[0]
                assert message["data"]["strategy_name"] == "sentiment_momentum"
                assert message["data"]["metrics"]["status"] in ["completed", "failed"]
                assert message["data"]["metrics"]["backtest_id"] == backtest_id

        finally:
            active_connections.clear()
            received_messages.clear()