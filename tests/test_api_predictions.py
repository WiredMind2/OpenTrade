"""
Unit tests for API prediction endpoints.
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


@pytest.mark.unit
class TestAPIPredictionEndpoints:
    """Test API prediction endpoints functionality."""

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
        conn.execute("DROP TABLE IF EXISTS backtest_runs")
        conn.execute("DROP TABLE IF EXISTS portfolio_snapshots")
        conn.execute("DROP TABLE IF EXISTS price_daily")
        conn.execute("DROP TABLE IF EXISTS articles")
        
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
            CREATE TABLE backtest_runs (
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
            CREATE TABLE portfolio_snapshots (
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
            CREATE TABLE price_daily (
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
            CREATE TABLE articles (
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

        # Set the database path in the app state
        app_state['database_path'] = self.temp_db.name
        
        # Also set the environment variable for config
        os.environ['DB_PATH'] = self.temp_db.name
        
        # Reload config to pick up the new DB_PATH
        from backend.config import reload_config
        reload_config()

        app_state['models_loaded'] = {
            'lightgbm_1d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_3d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_7d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'}
        }

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

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
        # Insert some test data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO trading_model_predictions
            (ticker, suggested_position_pct, dt, enter_prob)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", 0.8, datetime.utcnow().isoformat(), 0.9))
        conn.commit()
        conn.close()

        response = self.client.get("/trading/predictions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["ticker"] == "AAPL"
        assert "suggested_position_pct" in data[0]

    def test_predict_endpoint_database_error(self):
        """Test prediction endpoint when database connection fails."""
        payload = {
            "ticker": "AAPL",
            "horizon": "1d",
            "context": {}
        }
        
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.post("/predict", json=payload)
            assert response.status_code == 500
        finally:
            app_state["database_path"] = original_db_path

    def test_recent_predictions_database_error(self):
        """Test recent predictions endpoint when database connection fails."""
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.get("/predictions/recent")
            assert response.status_code == 200
            data = response.json()
            assert data == []  # Should return empty list on error
        finally:
            app_state["database_path"] = original_db_path