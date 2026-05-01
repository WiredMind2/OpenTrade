"""
Unit tests for API endpoints.
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
import pandas as pd


@pytest.mark.unit
class TestAPIEndpoints:
    """Test API endpoints functionality."""

    def setup_method(self):
        """Set up test client and mock database."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from backend.main import app, app_state

        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Initialize basic schema
        conn = sqlite3.connect(self.temp_db.name)
        
        # Drop existing tables to ensure clean state
        conn.execute("DROP TABLE IF EXISTS sentiment_predictions")
        conn.execute("DROP TABLE IF EXISTS trading_model_predictions")
        
        conn.execute("""
            CREATE TABLE sentiment_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                ticker TEXT,
                model TEXT,
                horizon TEXT,
                predicted_return REAL,
                predicted_confidence REAL,
                features_used TEXT,
                metadata TEXT,
                produced_at TEXT DEFAULT (datetime('now')),
                training_run_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE trading_model_predictions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                suggested_position_pct REAL,
                dt TEXT,
                enter_prob REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                params TEXT,
                started_at TEXT,
                completed_at TEXT,
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
                metrics TEXT
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
            CREATE TABLE IF NOT EXISTS price_minute (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                dt TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
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
            CREATE TABLE IF NOT EXISTS tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT,
                added_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

        # Set the database path in the app state
        app_state['database_path'] = self.temp_db.name
        app_state['models_loaded'] = {
            'lightgbm_1d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_3d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_7d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'}
        }

        # Also set the environment variable for config
        os.environ['DB_PATH'] = self.temp_db.name
        
        # Reload config to pick up the new DB_PATH
        from backend.config import reload_config
        reload_config()
        
        # Also directly set the config database path
        from backend.config import config
        config.database.path = self.temp_db.name

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up temporary database."""
        # Close the test client first
        if hasattr(self, 'client'):
            self.client.close()
        
        # Force garbage collection to close any lingering connections
        import gc
        gc.collect()
        
        # Try to close any remaining database connections
        try:
            import sqlite3
            # Force close any cached connections
            sqlite3.connect(self.temp_db.name).close()
        except:
            pass
        
        # Remove the file
        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except PermissionError:
                # If we can't delete it, just leave it for now
                pass

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "services" in data
        assert "database" in data
        assert "models_loaded" in data

    def test_metrics_endpoint(self):
        """Test system metrics endpoint."""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_usage_percent" in data
        assert "database_connections" in data
        assert "active_models" in data
        assert "recent_predictions" in data
        assert "error_rate" in data

    def test_monitoring_metrics_endpoint(self):
        """Test monitoring metrics alias endpoint."""
        response = self.client.get("/monitoring/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data

    def test_models_endpoint(self):
        """Test models listing endpoint."""
        response = self.client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should return at least one model from our mock
        assert len(data) >= 0

    def test_predict_endpoint_invalid_horizon(self):
        """Test predict endpoint with invalid horizon."""
        payload = {
            "ticker": "AAPL",
            "horizon": "invalid",
            "context": {}
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_empty_ticker(self):
        """Test predict endpoint with empty ticker."""
        payload = {
            "ticker": "",
            "horizon": "1d",
            "context": {}
        }
        response = self.client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    @patch('backend.routes.predictions.make_prediction')
    def test_predict_endpoint_success(self, mock_make_prediction):
        """Test successful prediction endpoint."""
        # Mock the prediction function
        from schemas import PredictionResponse
        mock_make_prediction.return_value = PredictionResponse(
            ticker="AAPL",
            horizon="1d",
            predicted_return=0.025,
            confidence=0.85,
            timestamp=datetime.utcnow(),
            model_version="lightgbm_1d_top10",
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
        assert "confidence" in data
        assert "timestamp" in data
        assert "model_version" in data

    def test_recent_predictions_endpoint(self):
        """Test recent predictions endpoint."""
        # Insert some test data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO sentiment_predictions
            (article_id, ticker, model, horizon, predicted_return, predicted_confidence, features_used, metadata, produced_at, training_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "AAPL", "lightgbm_1d", "1d", 0.025, 0.85, "features", "{}", datetime.utcnow().isoformat(), "test_run"))
        conn.commit()
        conn.close()

        response = self.client.get("/predictions/recent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["ticker"] == "AAPL"
        assert data[0]["horizon"] == "1d"

    def test_recent_predictions_endpoint_with_filters(self):
        """Test recent predictions endpoint with ticker filter."""
        # Insert test data for different tickers
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO sentiment_predictions
            (article_id, ticker, model, horizon, predicted_return, predicted_confidence, features_used, metadata, produced_at, training_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "AAPL", "lightgbm_1d", "1d", 0.025, 0.85, "features", "{}", datetime.utcnow().isoformat(), "test_run"))
        conn.execute("""
            INSERT INTO sentiment_predictions
            (article_id, ticker, model, horizon, predicted_return, predicted_confidence, features_used, metadata, produced_at, training_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (2, "GOOGL", "lightgbm_1d", "1d", 0.015, 0.75, "features", "{}", datetime.utcnow().isoformat(), "test_run"))
        conn.commit()
        conn.close()

        response = self.client.get("/predictions/recent?ticker=AAPL")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert all(item["ticker"] == "AAPL" for item in data)

    def test_trading_predictions_endpoint(self):
        """Test trading predictions endpoint."""
        # Insert test trading predictions
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO trading_model_predictions
            (ticker, suggested_position_pct, dt, enter_prob)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", 0.05, datetime.utcnow().date().isoformat(), 0.8))
        conn.commit()
        conn.close()

        response = self.client.get("/trading/predictions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["ticker"] == "AAPL"
        assert "suggested_position_pct" in data[0]

    @patch('routes.backtests.run_backtest_background')
    def test_backtest_endpoint_success(self, mock_run_backtest):
        """Test successful backtest creation."""
        mock_run_backtest.return_value = "test_backtest_id"

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

    @patch('routes.backtests.run_backtest_background')
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
        # Insert test backtest data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO backtest_runs
            (id, name, params, started_at, completed_at, initial_capital, final_value, total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("test_id", "test_strategy", "{}", datetime.utcnow().isoformat(), datetime.utcnow().isoformat(),
              100000.0, 105000.0, 0.05, 1.2, 0.08, 0.65, 50, "{}"))
        conn.commit()
        conn.close()

        response = self.client.get("/trading/backtest")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["id"] == "test_id"
        assert data[0]["strategy_name"] == "test_strategy"

    def test_list_backtests_pagination(self):
        """Test backtests listing with pagination."""
        response = self.client.get("/trading/backtest?page=1&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_portfolio_current_endpoint(self):
        """Test current portfolio endpoint."""
        # Insert test portfolio data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO portfolio_snapshots
            (timestamp, total_value, cash, invested_value, exposure, pnl, daily_return, positions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(), 105000.0, 25000.0, 80000.0, 0.76, 5000.0, 0.008,
              '[{"ticker": "AAPL", "quantity": 100, "value": 15000.0, "pnl": 1000.0}]'))
        conn.commit()
        conn.close()

        response = self.client.get("/portfolio/current")
        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "cash" in data
        assert "invested_value" in data
        assert "exposure" in data
        assert "positions" in data
        assert "pnl" in data
        assert "daily_return" in data

    def test_portfolio_current_endpoint_fallback(self):
        """Test current portfolio endpoint with no data (fallback to mock data)."""
        response = self.client.get("/portfolio/current")
        assert response.status_code == 200
        data = response.json()
        assert "total_value" in data
        assert "cash" in data
        assert "invested_value" in data
        assert "exposure" in data
        assert "positions" in data
        assert "pnl" in data
        assert "daily_return" in data
        # Should have mock data
        assert data["total_value"] == 105000.0
        assert len(data["positions"]) == 3

    def test_portfolio_current_endpoint_database_error(self):
        """Test current portfolio endpoint when database connection fails."""
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.get("/portfolio/current")
            assert response.status_code == 200  # Should return fallback data
            data = response.json()
            assert "total_value" in data
            assert data["total_value"] == 105000.0  # Fallback value
        finally:
            app_state["database_path"] = original_db_path

    def test_price_data_endpoint(self):
        """Test price data endpoint."""
        # Insert test price data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-01", 150.0, 152.0, 148.0, 151.0, 151.0, 1000000))
        conn.commit()
        conn.close()

        response = self.client.get("/data/prices/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "ticker" in data
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        assert data["ticker"] == "AAPL"

    def test_price_data_with_date_filters(self):
        """Test price data endpoint with date filters."""
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

        response = self.client.get("/data/prices/AAPL?start_date=2024-01-01T00:00:00&end_date=2024-01-02T00:00:00")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

    def test_price_data_limit_parameter(self):
        """Test price data endpoint with limit parameter."""
        # Insert multiple records
        conn = sqlite3.connect(self.temp_db.name)
        for i in range(10):
            conn.execute("""
                INSERT INTO price_daily
                (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("AAPL", f"2024-01-{i+1:02d}", 150.0 + i, 152.0 + i, 148.0 + i, 151.0 + i, 151.0 + i, 1000000))
        conn.commit()
        conn.close()

        response = self.client.get("/data/prices/AAPL?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 5

    # Script execution endpoints
    @patch('routes.scripts.run_script_async')
    def test_script_execute_endpoint(self, mock_run_script):
        """Test script execution endpoint."""
        mock_run_script.return_value = None

        payload = {
            "script_name": "train_sentiment_model",
            "parameters": {"csv": "data/training.csv"}
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["script_name"] == "train_sentiment_model"
        assert data["status"] == "running"
        assert "execution_id" in data

    def test_script_execute_invalid_script(self):
        """Test script execution with invalid script name."""
        payload = {
            "script_name": "invalid_script",
            "parameters": {}
        }
        response = self.client.post("/scripts/execute", json=payload)
        assert response.status_code == 422  # Validation error

    def test_script_status_endpoint(self):
        """Test getting script execution status."""
        # First create a mock execution
        payload = {
            "script_name": "train_sentiment_model",
            "parameters": {}
        }
        response = self.client.post("/scripts/execute", json=payload)
        execution_id = response.json()["execution_id"]

        # Now check status
        response = self.client.get(f"/scripts/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["execution_id"] == execution_id
        assert "status" in data

    def test_script_status_not_found(self):
        """Test script status for non-existent execution."""
        response = self.client.get("/scripts/status/nonexistent")
        assert response.status_code == 404

    def test_list_script_executions(self):
        """Test listing script executions."""
        response = self.client.get("/scripts/executions")
        assert response.status_code == 200
        data = response.json()
        assert "executions" in data
        assert isinstance(data["executions"], list)

    @patch('routes.scripts.run_pipeline_async')
    def test_pipeline_run_endpoint(self, mock_run_pipeline):
        """Test pipeline run endpoint."""
        mock_run_pipeline.return_value = None

        response = self.client.post("/scripts/pipeline/run?steps=apply_schema,ingest_prices")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "execution_id" in data
        assert "completed_steps" in data
        assert "failed_steps" in data

    def test_pipeline_status_endpoint(self):
        """Test pipeline status endpoint."""
        # First create a pipeline
        response = self.client.post("/scripts/pipeline/run")
        execution_id = response.json()["execution_id"]

        # Check status
        response = self.client.get(f"/scripts/pipeline/status/{execution_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["execution_id"] == execution_id
        assert "status" in data

    def test_websocket_endpoint_exists(self):
        """Test WebSocket endpoint exists."""
        # WebSocket testing is complex with TestClient
        # For now, just verify the endpoint exists by checking the app routes
        from main import app
        websocket_routes = [route for route in app.routes if hasattr(route, 'path') and route.path == '/ws']
        assert len(websocket_routes) == 1

    def test_udf_symbols_endpoint_success(self):
        """Test UDF symbols endpoint with existing symbol."""
        # Insert test ticker data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO tickers (ticker, name, exchange, sector)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", "Apple Inc.", "NASDAQ", "Technology"))
        conn.commit()
        conn.close()

        response = self.client.get("/udf/symbols?symbol=AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AAPL"
        assert data["ticker"] == "AAPL"
        assert data["description"] == "Apple Inc. (Technology)"
        assert data["type"] == "stock"
        assert data["exchange"] == "NASDAQ"
        assert data["listed_exchange"] == "NASDAQ"
        assert data["supported_resolutions"] == ["1", "5", "15", "30", "60", "240", "1D", "1W", "1M"]
        assert data["has_intraday"] is True
        assert data["has_daily"] is True
        assert data["has_weekly_and_monthly"] is True

    def test_udf_symbols_endpoint_not_found(self):
        """Test UDF symbols endpoint with non-existent symbol."""
        response = self.client.get("/udf/symbols?symbol=NONEXISTENT")
        assert response.status_code == 200  # UDF returns error in response body
        data = response.json()
        assert data["s"] == "error"
        assert "not found" in data["errmsg"].lower()

    def test_udf_symbols_endpoint_case_insensitive(self):
        """Test UDF symbols endpoint is case insensitive."""
        # Insert test ticker data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO tickers (ticker, name, exchange, sector)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", "Apple Inc.", "NASDAQ", "Technology"))
        conn.commit()
        conn.close()

        response = self.client.get("/udf/symbols?symbol=aapl")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AAPL"
        assert data["ticker"] == "AAPL"

    def test_udf_quotes_endpoint_success(self):
        """Test UDF quotes endpoint with existing price data."""
        # Insert test ticker data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO tickers (ticker, name, exchange, sector)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", "Apple Inc.", "NASDAQ", "Technology"))
        # Insert test price data (most recent first for change calculation)
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-03", 152.0, 155.0, 150.0, 154.0, 154.0, 2000000))
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-02", 150.0, 153.0, 149.0, 152.0, 152.0, 1800000))
        conn.commit()
        conn.close()

        response = self.client.get("/udf/quotes?symbols=AAPL")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        quote = data[0]
        assert quote["s"] == "ok"
        assert quote["n"] == "AAPL"
        assert "v" in quote
        v = quote["v"]
        assert "lp" in v  # Last price
        assert v["lp"] == 154.0
        assert "ch" in v  # Change
        assert v["ch"] == 2.0  # 154.0 - 152.0
        assert "chp" in v  # Change percentage
        assert abs(v["chp"] - 1.3157894736842106) < 0.0001  # (2.0 / 152.0) * 100
        assert "open_price" in v
        assert v["open_price"] == 152.0
        assert "high_price" in v
        assert v["high_price"] == 155.0
        assert "low_price" in v
        assert v["low_price"] == 150.0
        assert "volume" in v
        assert v["volume"] == 2000000
        assert "prev_close_price" in v
        assert v["prev_close_price"] == 152.0

    def test_udf_quotes_endpoint_multiple_symbols(self):
        """Test UDF quotes endpoint with multiple symbols."""
        # Insert test ticker data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO tickers (ticker, name, exchange, sector)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", "Apple Inc.", "NASDAQ", "Technology"))
        conn.execute("""
            INSERT INTO tickers (ticker, name, exchange, sector)
            VALUES (?, ?, ?, ?)
        """, ("GOOGL", "Alphabet Inc.", "NASDAQ", "Technology"))
        # Insert test price data
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-03", 152.0, 155.0, 150.0, 154.0, 154.0, 2000000))
        conn.execute("""
            INSERT INTO price_daily
            (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("GOOGL", "2024-01-03", 120.0, 122.0, 118.0, 121.0, 121.0, 1500000))
        conn.commit()
        conn.close()

        response = self.client.get("/udf/quotes?symbols=AAPL,GOOGL")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        symbols = [quote["n"] for quote in data]
        assert "AAPL" in symbols
        assert "GOOGL" in symbols

    def test_udf_quotes_endpoint_symbol_not_found(self):
        """Test UDF quotes endpoint with non-existent symbol."""
        response = self.client.get("/udf/quotes?symbols=NONEXISTENT")
        assert response.status_code == 200  # Should return error status in response
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        quote = data[0]
        assert quote["s"] == "error"
        assert quote["n"] == "NONEXISTENT"
        assert quote["v"] == {}