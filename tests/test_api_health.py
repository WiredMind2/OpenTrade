"""
Unit tests for API health and monitoring endpoints.
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
from main import app, app_state


@pytest.mark.unit
class TestAPIHealthEndpoints:
    """Test API health and monitoring endpoints functionality."""

    def setup_method(self):
        """Set up test client and mock database."""
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
        app_state['models_loaded'] = {
            'lightgbm_1d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_3d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'},
            'lightgbm_7d': {'lgbm': Mock(), 'embedder': 'all-MiniLM-L6-v2'}
        }

        # Set up model registry with mock models
        from backend.models.registry import ModelRegistry
        registry = ModelRegistry()
        for model_name in ['lightgbm_1d', 'lightgbm_3d', 'lightgbm_7d']:
            mock_model = Mock()
            mock_model.name = model_name
            mock_model.type = 'lightgbm'
            mock_model.version = '1.0.0'
            mock_model.description = f'{model_name} model'
            mock_model.capabilities = ['predict']
            mock_model.get_config_schema.return_value.model_json_schema.return_value = {"type": "object"}
            registry.register(mock_model)
        app_state['model_registry'] = registry

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

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
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_usage_percent" in data

    def test_monitoring_metrics_endpoint(self):
        """Test monitoring metrics endpoint."""
        response = self.client.get("/monitoring/metrics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_models_endpoint(self):
        """Test models information endpoint."""
        response = self.client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have our mock models
        assert len(data) >= 3

    def test_health_endpoint_database_failure(self):
        """Test health check endpoint when database connection fails."""
        # Temporarily change database path to invalid path
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"
        
        try:
            response = self.client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"  # Overall status is still healthy
            assert data["services"]["database"] == "unhealthy"
        finally:
            app_state["database_path"] = original_db_path

    def test_monitoring_metrics_endpoint_database_failure(self):
        """Test monitoring metrics endpoint when database connection fails."""
        # Temporarily change database path to invalid path
        from backend.main import app_state
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"

        try:
            response = self.client.get("/monitoring/metrics")
            assert response.status_code == 200
            data = response.json()
            # Should still return metrics but with 0 database values
            assert "database_connections" in data
            assert data["database_connections"] == 0
        finally:
            app_state["database_path"] = original_db_path