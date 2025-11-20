"""
Integration tests for the trading backtesting system.
"""
import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from main import app
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestIntegrationWorkflow:
    """Integration tests for complete trading workflow."""

    def setup_method(self):
        """Set up test environment."""
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
        conn.commit()
        conn.close()

        # Set up app state
        from main import app_state
        app_state['database_path'] = self.temp_db.name
        app_state['models_loaded'] = {
            'lightgbm_1d': {'lgbm': MagicMock(), 'embedder': 'all-MiniLM-L6-v2'},
        }

        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_data_ingestion_to_prediction_workflow(self):
        """Test complete workflow from data ingestion to prediction."""
        # Step 1: Insert test price data
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

        # Step 2: Insert test article data
        conn.execute("""
            INSERT INTO articles
            (source, url, canonical_timestamp, published_at, title, author, content, sentiment_score, ticker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("test_source", "http://test.com", "2024-01-01T10:00:00", "2024-01-01T10:00:00",
              "AAPL shows strong growth", "Test Author", "Apple stock is performing well", 0.8, "AAPL"))
        conn.commit()
        conn.close()

        # Step 3: Test prediction endpoint
        with patch('routes.predictions.make_prediction') as mock_predict:
            from schemas import PredictionResponse
            mock_predict.return_value = PredictionResponse(
                ticker="AAPL",
                horizon="1d",
                predicted_return=0.025,
                confidence=0.85,
                timestamp=datetime.utcnow(),
                model_version="1.0.0",
                features_used=["price_change", "volume"],
                metadata={}
            )

            payload = {
                "ticker": "AAPL",
                "horizon": "1d",
                "context": {}
            }
            response = self.client.post("/predict", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert data["ticker"] == "AAPL"
            assert data["horizon"] == "1d"
            assert "predicted_return" in data

    def test_backtest_creation_workflow(self):
        """Test backtest creation and status checking workflow."""
        # Step 1: Test backtest creation
        with patch('routes.backtests.run_backtest_background') as mock_run_backtest:
            mock_run_backtest.return_value = "test_backtest_123"

            payload = {
                "strategy_name": "test_strategy",
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-12-31T00:00:00",
                "initial_capital": 100000.0,
                "parameters": {}
            }
            response = self.client.post("/backtest", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert "strategy_name" in data
            assert "start_date" in data
            assert "initial_capital" in data

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
                    "script_name": "generate_trading_predictions",
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
            assert received_messages[1]["data"]["script_name"] == "generate_trading_predictions"
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
                    "script_name": "generate_trading_predictions",
                    "status": "running",
                    "start_time": datetime.utcnow(),
                    "parameters": {},
                    "output": "",
                    "error": "",
                    "process": None
                }

                # Execute script
                await run_script_async(execution_id, "generate_trading_predictions", {}, {})

                # Verify WebSocket message was sent
                assert len(received_messages) == 1
                message = received_messages[0]
                assert message["type"] == "script_status"
                assert message["data"]["script_name"] == "generate_trading_predictions"
                assert message["data"]["status"] == "completed"
                assert message["data"]["execution_id"] == execution_id
                assert message["data"]["output"] == "Script output"
                assert message["data"]["error"] is None
                assert "duration_seconds" in message["data"]

        finally:
            active_connections.clear()
            received_messages.clear()
            # Clean up script executions
            if execution_id in script_executions:
                del script_executions[execution_id]

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
                assert message["data"]["metrics"]["status"] == "completed"
                assert message["data"]["final_value"] == 105000.0
                assert message["data"]["metrics"]["backtest_id"] == backtest_id

        finally:
            active_connections.clear()
            received_messages.clear()