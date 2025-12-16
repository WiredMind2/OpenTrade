"""
Tests for script model adapters.
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.models.three_ma_adapter import ThreeMAAdapter


class TestScriptAdapter:
    """Test the script model adapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = ThreeMAAdapter()

    def test_adapter_initialization(self):
        """Test that the adapter initializes correctly."""
        assert self.adapter.name == "three_ma_crossover_v1"
        assert self.adapter.type == "script"
        assert self.adapter.version == "1.0.0"
        assert "predict" in self.adapter.capabilities
        assert "retrain" in self.adapter.capabilities

    def test_get_config_schema(self):
        """Test that the config schema is returned correctly."""
        schema = self.adapter.get_config_schema()
        assert schema is not None
        # Should be a Pydantic model
        assert hasattr(schema, 'model_json_schema')

    @patch('backend.models.three_ma_adapter.sqlite3')
    @patch('backend.models.three_ma_adapter.optimize_ma_periods')
    @patch('backend.models.three_ma_adapter.generate_predictions')
    def test_predict_with_skip_optimization(self, mock_generate, mock_optimize, mock_sqlite):
        """Test prediction with optimization skipped."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_sqlite.connect.return_value = mock_conn

        # Mock cursor and query results
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("AAPL", "2023-01-01", 0.05, 0.6, 0.05, 0.4)
        ]

        inputs = {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "tickers": ["AAPL"],
            "skip_optimization": True,
            "fixed_short": 5,
            "fixed_medium": 20,
            "fixed_long": 50
        }
        config = {}

        result = self.adapter.predict(inputs, config)

        # Verify the result structure
        assert "predictions" in result
        assert "meta" in result
        assert len(result["predictions"]) == 1

        prediction = result["predictions"][0]
        assert prediction["ticker"] == "AAPL"
        assert prediction["date"] == "2023-01-01"
        assert prediction["predicted_return"] == 0.05
        assert prediction["confidence"] == 0.6
        assert prediction["position_pct"] == 0.05
        assert prediction["model_version"] == "1.0.0"
        assert "features_used" in prediction
        assert "metadata" in prediction

        # Verify meta structure
        meta = result["meta"]
        assert meta["model_name"] == "three_ma_crossover_v1"
        assert meta["model_version"] == "1.0.0"
        assert "generated_at" in meta
        assert meta["date_range"]["start"] == "2023-01-01"
        assert meta["date_range"]["end"] == "2023-01-02"
        assert meta["tickers"] == ["AAPL"]
        assert meta["ma_periods"]["short"] == 5
        assert meta["ma_periods"]["medium"] == 20
        assert meta["ma_periods"]["long"] == 50
        assert meta["optimization_skipped"] is True

        # Verify functions were called correctly
        mock_generate.assert_called_once_with(mock_conn, "2023-01-01", "2023-01-02", 5, 20, 50)
        mock_optimize.assert_not_called()

    @patch('backend.models.three_ma_adapter.sqlite3')
    @patch('backend.models.three_ma_adapter.optimize_ma_periods')
    @patch('backend.models.three_ma_adapter.generate_predictions')
    def test_predict_with_optimization(self, mock_generate, mock_optimize, mock_sqlite):
        """Test prediction with optimization enabled."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_sqlite.connect.return_value = mock_conn

        # Mock optimization result
        mock_optimize.return_value = (7, 21, 63, 1.5)

        # Mock cursor and query results
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("AAPL", "2023-01-01", -0.03, 0.4, -0.03, 0.6)
        ]

        inputs = {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "tickers": ["AAPL"],
            "skip_optimization": False
        }
        config = {
            "short_range": [5, 7, 9],
            "medium_range": [15, 20, 25],
            "long_range": [40, 50, 60]
        }

        result = self.adapter.predict(inputs, config)

        # Verify optimization was called
        mock_optimize.assert_called_once_with(
            mock_conn, "2023-01-01", "2023-01-02",
            [5, 7, 9], [15, 20, 25], [40, 50, 60], ["AAPL"]
        )

        # Verify generation was called with optimized periods
        mock_generate.assert_called_once_with(mock_conn, "2023-01-01", "2023-01-02", 7, 21, 63)

        # Verify result
        assert len(result["predictions"]) == 1
        prediction = result["predictions"][0]
        assert prediction["position_pct"] == -0.03
        assert prediction["confidence"] == 0.4

        assert result["meta"]["optimization_skipped"] is False

    def test_predict_missing_required_inputs(self):
        """Test prediction with missing required inputs."""
        inputs = {"tickers": ["AAPL"]}
        config = {}

        with pytest.raises(ValueError, match="start and end dates are required"):
            self.adapter.predict(inputs, config)

    def test_predict_empty_tickers(self):
        """Test prediction with empty tickers list."""
        inputs = {
            "start": "2023-01-01",
            "end": "2023-01-02",
            "tickers": []
        }
        config = {}

        with pytest.raises(ValueError, match="tickers list cannot be empty"):
            self.adapter.predict(inputs, config)

    @patch('backend.models.three_ma_adapter.sqlite3')
    def test_retrain(self, mock_sqlite):
        """Test retraining the model."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_sqlite.connect.return_value = mock_conn

        # Mock optimization result
        with patch('backend.models.three_ma_adapter.optimize_ma_periods') as mock_optimize:
            mock_optimize.return_value = (8, 22, 65, 1.8)

            training_payload = {
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "tickers": ["AAPL", "GOOGL"]
            }
            config = {
                "short_range": [5, 8, 10],
                "medium_range": [18, 22, 26],
                "long_range": [50, 65, 80]
            }

            result = self.adapter.retrain(training_payload, config)

            # Verify optimization was called
            mock_optimize.assert_called_once_with(
                mock_conn, "2022-01-01", "2023-01-01",
                [5, 8, 10], [18, 22, 26], [50, 65, 80], ["AAPL", "GOOGL"]
            )

            # Verify result structure
            assert result["status"] == "completed"
            assert result["optimized_periods"]["short"] == 8
            assert result["optimized_periods"]["medium"] == 22
            assert result["optimized_periods"]["long"] == 65
            assert result["sharpe_ratio"] == 1.8
            assert result["training_date_range"]["start"] == "2022-01-01"
            assert result["training_date_range"]["end"] == "2023-01-01"

    def test_retrain_missing_dates(self):
        """Test retraining with missing dates."""
        training_payload = {"tickers": ["AAPL"]}
        config = {}

        with pytest.raises(ValueError, match="start_date and end_date are required"):
            self.adapter.retrain(training_payload, config)

    def test_save_not_supported(self):
        """Test that save raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Saving not supported"):
            self.adapter.save(Path("/tmp/test"))

    def test_load_not_supported(self):
        """Test that load raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Loading not supported"):
            self.adapter.load(Path("/tmp/test"))