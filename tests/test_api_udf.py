"""
Unit tests for UDF (Universal Data Feed) endpoints.
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
class TestAPIUDFEndpoints:
    """Test UDF endpoints functionality."""

    def setup_method(self):
        """Set up test client and mock database."""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Initialize basic schema with tickers table
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                name TEXT,
                exchange TEXT,
                sector TEXT
            )
        """)
        # Insert some test data
        conn.execute("""
            INSERT INTO tickers (ticker, name, exchange, sector) VALUES
            ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology'),
            ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology'),
            ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology'),
            ('TSLA', 'Tesla Inc.', 'NASDAQ', 'Consumer Cyclical')
        """)
        conn.commit()
        conn.close()

        # Set the database path in the app state
        app_state['database_path'] = self.temp_db.name

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_udf_config_endpoint(self):
        """Test UDF config endpoint returns 200 OK."""
        response = self.client.get("/udf/config")
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "exchanges" in data
        assert "symbols_types" in data
        assert "supported_resolutions" in data
        assert "supports_search" in data
        assert "supports_group_request" in data
        assert "supports_marks" in data
        assert "supports_timescale_marks" in data
        assert "supports_time" in data

        # Check exchanges (should include NASDAQ from our test data)
        exchanges = data["exchanges"]
        assert isinstance(exchanges, list)
        assert len(exchanges) > 0

        # Find NASDAQ exchange
        nasdaq_exchange = next((ex for ex in exchanges if ex["value"] == "NASDAQ"), None)
        assert nasdaq_exchange is not None
        assert nasdaq_exchange["name"] == "NASDAQ"
        assert nasdaq_exchange["desc"] == "NASDAQ Stock Exchange"

        # Check symbols types
        symbols_types = data["symbols_types"]
        assert isinstance(symbols_types, list)
        assert len(symbols_types) > 0
        assert any(st["value"] == "stock" for st in symbols_types)

        # Check supported resolutions
        supported_resolutions = data["supported_resolutions"]
        assert isinstance(supported_resolutions, list)
        assert "1D" in supported_resolutions
        assert "1W" in supported_resolutions
        assert "1M" in supported_resolutions

    def test_udf_symbols_endpoint_valid_symbol(self):
        """Test UDF symbols endpoint with valid symbol returns 200 OK."""
        response = self.client.get("/udf/symbols?symbol=AAPL")
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "name" in data
        assert "ticker" in data
        assert "description" in data
        assert "type" in data
        assert "session" in data
        assert "timezone" in data
        assert "exchange" in data
        assert "listed_exchange" in data
        assert "minmov" in data
        assert "pricescale" in data
        assert "has_intraday" in data
        assert "supported_resolutions" in data
        assert "has_daily" in data
        assert "has_weekly_and_monthly" in data
        assert "data_status" in data

        # Check values
        assert data["name"] == "AAPL"
        assert data["ticker"] == "AAPL"
        assert data["description"] == "Apple Inc. (Technology)"  # Enhanced with sector info
        assert data["type"] == "stock"
        assert data["exchange"] == "NASDAQ"
        assert data["has_intraday"] is True
        assert data["has_daily"] is True
        assert data["has_weekly_and_monthly"] is True

    def test_udf_symbols_endpoint_invalid_symbol(self):
        """Test UDF symbols endpoint with invalid symbol returns error."""
        response = self.client.get("/udf/symbols?symbol=INVALID")
        assert response.status_code == 200  # UDF returns 200 with error in JSON
        data = response.json()

        # Check error format
        assert "s" in data
        assert data["s"] == "error"
        assert "errmsg" in data
        assert "not found" in data["errmsg"].lower()

    def test_udf_symbols_endpoint_missing_symbol(self):
        """Test UDF symbols endpoint without symbol parameter."""
        response = self.client.get("/udf/symbols")
        assert response.status_code == 422  # FastAPI validation error

    def test_udf_config_endpoint_database_error(self):
        """Test UDF config endpoint handles database errors gracefully."""
        # Temporarily change database path to invalid path
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"

        try:
            response = self.client.get("/udf/config")
            assert response.status_code == 200  # Should return error in JSON format
            data = response.json()

            # Check error format
            assert "s" in data
            assert data["s"] == "error"
            assert "errmsg" in data
        finally:
            app_state["database_path"] = original_db_path

    def test_udf_symbols_endpoint_database_error(self):
        """Test UDF symbols endpoint handles database errors gracefully."""
        # Temporarily change database path to invalid path
        original_db_path = app_state["database_path"]
        app_state["database_path"] = "/invalid/path/db.sqlite"

        try:
            response = self.client.get("/udf/symbols?symbol=AAPL")
            assert response.status_code == 200  # Should return error in JSON format
            data = response.json()

            # Check error format
            assert "s" in data
            assert data["s"] == "error"
            assert "errmsg" in data
        finally:
            app_state["database_path"] = original_db_path

    def test_udf_history_endpoint_valid_request(self):
        """Test UDF history endpoint with valid request returns OHLC data."""
        # Create price_daily table with test data
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE price_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)

        # Insert test historical data for AAPL
        test_data = [
            ('AAPL', '2024-01-01', 150.0, 155.0, 149.0, 152.0, 1000000),
            ('AAPL', '2024-01-02', 152.0, 158.0, 151.0, 155.0, 1200000),
            ('AAPL', '2024-01-03', 155.0, 160.0, 154.0, 158.0, 1100000),
        ]

        conn.executemany("""
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, test_data)
        conn.commit()
        conn.close()

        # Test the history endpoint
        from_ts = int(datetime(2024, 1, 1).timestamp())
        to_ts = int(datetime(2024, 1, 3).timestamp())

        response = self.client.get(f"/udf/history?symbol=AAPL&resolution=1D&from_ts={from_ts}&to_ts={to_ts}")
        assert response.status_code == 200
        data = response.json()

        # Check UDF response format
        assert "s" in data
        assert data["s"] == "ok"
        assert "t" in data  # timestamps
        assert "o" in data  # opens
        assert "h" in data  # highs
        assert "l" in data  # lows
        assert "c" in data  # closes
        assert "v" in data  # volumes

        # Check data arrays have same length
        assert len(data["t"]) == len(data["o"]) == len(data["h"]) == len(data["l"]) == len(data["c"]) == len(data["v"])

        # Check we got 3 bars
        assert len(data["t"]) == 3

        # Check first bar data (should be in milliseconds)
        first_timestamp = data["t"][0]
        assert isinstance(first_timestamp, int)
        # Convert back to check date
        first_date = datetime.fromtimestamp(first_timestamp / 1000)
        assert first_date.date() == datetime(2024, 1, 1).date()

        # Check OHLC values
        assert data["o"][0] == 150.0
        assert data["h"][0] == 155.0
        assert data["l"][0] == 149.0
        assert data["c"][0] == 152.0
        assert data["v"][0] == 1000000

    def test_udf_history_endpoint_invalid_symbol(self):
        """Test UDF history endpoint with invalid symbol returns error."""
        from_ts = int(datetime(2024, 1, 1).timestamp())
        to_ts = int(datetime(2024, 1, 3).timestamp())

        response = self.client.get(f"/udf/history?symbol=INVALID&resolution=1D&from_ts={from_ts}&to_ts={to_ts}")
        assert response.status_code == 200  # UDF returns 200 with error in JSON
        data = response.json()

        # Check error format
        assert "s" in data
        assert data["s"] == "error"
        assert "errmsg" in data
        assert "not found" in data["errmsg"].lower()

    def test_udf_history_endpoint_automatic_fetch(self):
        """Test UDF history endpoint automatically fetches data when none is available."""
        # Create empty price_daily table
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE price_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)
        conn.commit()
        conn.close()

        from_ts = int(datetime(2024, 1, 1).timestamp())
        to_ts = int(datetime(2024, 1, 3).timestamp())

        response = self.client.get(f"/udf/history?symbol=AAPL&resolution=1D&from_ts={from_ts}&to_ts={to_ts}")
        assert response.status_code == 200
        data = response.json()

        # Should automatically fetch data and return ok status
        assert "s" in data
        assert data["s"] == "ok"
        assert "t" in data  # timestamps
        assert "o" in data  # opens
        assert "h" in data  # highs
        assert "l" in data  # lows
        assert "c" in data  # closes
        assert "v" in data  # volumes

        # Should have fetched at least some data
        assert len(data["t"]) > 0
        assert len(data["o"]) == len(data["h"]) == len(data["l"]) == len(data["c"]) == len(data["v"])