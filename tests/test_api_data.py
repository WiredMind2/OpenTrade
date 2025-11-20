"""
Unit tests for API data endpoints.
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
class TestAPIDataEndpoints:
    """Test API data endpoints functionality."""

    def setup_method(self):
        """Set up test client and mock database."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
        from main import app, app_state

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

    def test_get_price_data_success(self):
        """Test getting price data successfully."""
        # Insert test price data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-01", 150.0, 155.0, 149.0, 154.0, 154.0, 1000000))
        conn.execute("""
            INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("AAPL", "2024-01-02", 154.0, 158.0, 153.0, 157.0, 157.0, 1200000))
        conn.commit()
        conn.close()

        response = self.client.get("/data/prices/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "ticker" in data
        assert "data" in data
        assert len(data["data"]) == 2
        # Data is ordered by date DESC, so first record is most recent
        assert data["data"][0]["close"] == 157.0
        assert data["data"][1]["close"] == 154.0

    def test_get_price_data_no_data(self):
        """Test getting price data for ticker with no data."""
        response = self.client.get("/data/prices/NONEXISTENT")
        assert response.status_code == 200
        data = response.json()
        assert "ticker" in data
        assert "data" in data
        assert len(data["data"]) == 0

    def test_get_price_data_invalid_date_range(self):
        """Test getting price data with invalid date range."""
        response = self.client.get("/data/prices/AAPL?start_date=2024-01-02&end_date=2024-01-01")
        assert response.status_code == 422  # Validation error

    def test_get_recent_predictions_success(self):
        """Test getting recent predictions successfully."""
        # Insert test prediction data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO sentiment_predictions (article_id, ticker, model, horizon, predicted_return, predicted_confidence, features_used, metadata, produced_at, training_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "AAPL", "lightgbm_1d", "1d", 0.02, 0.85, "{}", "{}", datetime.utcnow().isoformat(), "test_run"))
        conn.execute("""
            INSERT INTO trading_model_predictions (ticker, suggested_position_pct, dt, enter_prob)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", 0.1, datetime.utcnow().isoformat(), 0.75))
        conn.commit()
        conn.close()

        response = self.client.get("/predictions/recent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "ticker" in data[0]
        assert "predicted_return" in data[0]

    def test_get_recent_predictions_empty(self):
        """Test getting recent predictions when no data exists."""
        response = self.client.get("/predictions/recent")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_trading_predictions_success(self):
        """Test getting trading predictions successfully."""
        # Insert test trading prediction data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            INSERT INTO trading_model_predictions (ticker, suggested_position_pct, dt, enter_prob)
            VALUES (?, ?, ?, ?)
        """, ("AAPL", 0.1, datetime.utcnow().isoformat(), 0.75))
        conn.execute("""
            INSERT INTO trading_model_predictions (ticker, suggested_position_pct, dt, enter_prob)
            VALUES (?, ?, ?, ?)
        """, ("GOOGL", -0.05, datetime.utcnow().isoformat(), 0.82))
        conn.commit()
        conn.close()

        response = self.client.get("/trading/predictions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        assert data[0]["ticker"] in ["AAPL", "GOOGL"]
        assert "suggested_position_pct" in data[0]
        assert "confidence" in data[0]

    def test_get_price_data_database_error(self):
        """Test getting price data when database connection fails."""
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.get("/data/prices/AAPL")
            assert response.status_code == 500
        finally:
            app_state["database_path"] = original_db_path