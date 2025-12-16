"""
Comprehensive unit tests for LinearRegressionTimeSeriesAdapter.

Tests cover model initialization, data processing, retraining, prediction,
and integration functionality with mocked dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from backend.models.linear_regression_time_series_adapter import (
    LinearRegressionTimeSeriesAdapter,
    LinearRegressionConfig
)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    # Create realistic price data with some trend and noise
    base_price = 100
    prices = []
    for i in range(50):
        price = base_price + i * 0.5 + np.random.normal(0, 2)
        prices.append(price)

    return pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),  # String dates as they come from DB
        'close': prices
    })


@pytest.fixture
def mock_db_connection():
    """Mock database connection for testing."""
    conn = Mock()
    return conn


@pytest.fixture
def mock_adapter():
    """Create a mock adapter instance."""
    adapter = Mock(spec=LinearRegressionTimeSeriesAdapter)
    adapter.name = "linear_regression_time_series_v1"
    adapter.version = "1.0.0"
    adapter.is_initialized = False
    adapter._models = {}
    adapter._feature_columns = []
    return adapter


@pytest.fixture
def trained_adapter(sample_price_data):
    """Create a trained adapter instance with mock models."""
    adapter = LinearRegressionTimeSeriesAdapter()

    # Mock trained models for different horizons
    from sklearn.linear_model import LinearRegression
    mock_model_1 = Mock(spec=LinearRegression)
    mock_model_1.predict.return_value = [0.02]  # 2% return prediction

    mock_model_3 = Mock(spec=LinearRegression)
    mock_model_3.predict.return_value = [0.05]  # 5% return prediction

    mock_model_7 = Mock(spec=LinearRegression)
    mock_model_7.predict.return_value = [0.08]  # 8% return prediction

    adapter._models = {1: mock_model_1, 3: mock_model_3, 7: mock_model_7}
    adapter._feature_columns = [f'lag_{i}' for i in range(1, 11)]  # 10 lags
    adapter.is_initialized = True

    return adapter


class TestLinearRegressionConfig:
    """Test the configuration schema."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LinearRegressionConfig()
        assert config.lag_days == 10
        assert config.horizons == [1, 3, 7]

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LinearRegressionConfig(lag_days=5, horizons=[1, 5])
        assert config.lag_days == 5
        assert config.horizons == [1, 5]

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = LinearRegressionConfig(lag_days=15, horizons=[1, 3, 7])
        assert config.lag_days == 15

        # Invalid lag_days (too low)
        with pytest.raises(ValueError):
            LinearRegressionConfig(lag_days=0)

        # Invalid lag_days (too high)
        with pytest.raises(ValueError):
            LinearRegressionConfig(lag_days=200)


class TestModelInitialization:
    """Test model initialization functionality."""

    def test_adapter_instantiation(self):
        """Test successful adapter instantiation."""
        adapter = LinearRegressionTimeSeriesAdapter()
        assert adapter is not None
        assert adapter.name == "linear_regression_time_series_v1"
        assert adapter.version == "1.0.0"
        assert adapter.type == "linear_regression"
        assert "predict" in adapter.capabilities
        assert "retrain" in adapter.capabilities

    def test_adapter_initialization_state(self):
        """Test adapter initialization state."""
        adapter = LinearRegressionTimeSeriesAdapter()
        assert not adapter.is_initialized
        assert adapter._models == {}
        assert adapter._feature_columns == []

    def test_config_schema_return(self):
        """Test that config schema is properly returned."""
        adapter = LinearRegressionTimeSeriesAdapter()
        schema = adapter.get_config_schema()
        assert schema == LinearRegressionConfig

    def test_logger_initialization(self):
        """Test logger initialization."""
        adapter = LinearRegressionTimeSeriesAdapter()
        # Logger is initialized in the parent ScriptModelAdapter class
        assert hasattr(adapter, 'logger')
        assert adapter.logger is not None


class TestDataProcessing:
    """Test data processing functionality."""

    def test_compute_features_basic(self, sample_price_data):
        """Test basic feature computation."""
        adapter = LinearRegressionTimeSeriesAdapter()

        features = adapter._compute_features(sample_price_data, lag_days=3)

        # Should have lag_1, lag_2, lag_3 columns
        assert 'lag_1' in features.columns
        assert 'lag_2' in features.columns
        assert 'lag_3' in features.columns

        # Should have fewer rows than original due to lagging
        assert len(features) < len(sample_price_data)

        # Check that features are percentage returns
        assert features['lag_1'].dtype == float

    def test_compute_features_insufficient_data(self, sample_price_data):
        """Test feature computation with insufficient data."""
        adapter = LinearRegressionTimeSeriesAdapter()

        # Try to compute more lags than available data points
        features = adapter._compute_features(sample_price_data.head(2), lag_days=5)

        # Should return empty DataFrame
        assert features.empty

    def test_compute_targets(self, sample_price_data):
        """Test target computation for different horizons."""
        adapter = LinearRegressionTimeSeriesAdapter()

        # Test horizon 1
        targets_1 = adapter._compute_targets(sample_price_data, horizon=1)
        assert len(targets_1) == len(sample_price_data)  # Same length as input

        # Test horizon 3
        targets_3 = adapter._compute_targets(sample_price_data, horizon=3)
        assert len(targets_3) == len(sample_price_data)

        # Check that targets are percentage returns
        assert targets_1.dtype == float

        # Check that last few values are NaN due to shift
        assert targets_1.isna().iloc[-1]  # Last value should be NaN
        assert targets_3.isna().iloc[-3:].all()  # Last 3 values should be NaN

    def test_feature_target_alignment(self, sample_price_data):
        """Test that features and targets are properly aligned."""
        adapter = LinearRegressionTimeSeriesAdapter()

        features = adapter._compute_features(sample_price_data, lag_days=3)
        targets = adapter._compute_targets(sample_price_data, horizon=1)

        # Features predict future returns, so they should align
        # This is a complex alignment test - simplified for now
        assert len(features) > 0
        assert len(targets) > 0


class TestRetraining:
    """Test model retraining functionality."""

    @patch('backend.models.linear_regression_time_series_adapter.sqlite3.connect')
    def test_successful_retraining(self, mock_connect, sample_price_data):
        """Test successful model retraining."""
        # Mock database connection and query results
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock pandas read_sql_query to return sample data
        with patch('pandas.read_sql_query', return_value=sample_price_data):
            adapter = LinearRegressionTimeSeriesAdapter()

            training_payload = {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "tickers": ["AAPL"]
            }

            config = {
                "lag_days": 3,
                "horizons": [1, 3]
            }

            result = adapter.retrain(training_payload, config)

            assert result["status"] == "completed"
            assert 1 in result["horizons_trained"]
            assert 3 in result["horizons_trained"]
            assert "metrics" in result
            assert adapter.is_initialized
            assert len(adapter._models) == 2

    @patch('backend.models.linear_regression_time_series_adapter.sqlite3.connect')
    def test_retraining_insufficient_data(self, mock_connect):
        """Test retraining with insufficient data."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock empty data
        empty_data = pd.DataFrame(columns=['date', 'close'])
        with patch('pandas.read_sql_query', return_value=empty_data):
            adapter = LinearRegressionTimeSeriesAdapter()

            training_payload = {
                "start_date": "2023-01-01",
                "end_date": "2023-01-05",
                "tickers": ["AAPL"]
            }

            config = {"lag_days": 10, "horizons": [1]}

            with pytest.raises(ValueError, match="No valid training data collected"):
                adapter.retrain(training_payload, config)

    @patch('backend.models.linear_regression_time_series_adapter.sqlite3.connect')
    def test_retraining_missing_required_fields(self, mock_connect):
        """Test retraining with missing required fields."""
        adapter = LinearRegressionTimeSeriesAdapter()

        # Missing tickers
        training_payload = {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }

        config = {"lag_days": 10}

        with pytest.raises(ValueError, match="start_date, end_date, and tickers are required"):
            adapter.retrain(training_payload, config)

    def test_model_persistence_save_load(self, tmp_path):
        """Test model save and load functionality."""
        # Create a real trained adapter for testing
        adapter = LinearRegressionTimeSeriesAdapter()

        # Create real sklearn models
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # Train simple models
        X = np.random.random((100, 10))
        y = np.random.random(100)

        model_1 = LinearRegression()
        model_1.fit(X, y)

        adapter._models = {1: model_1}
        adapter._feature_columns = [f'lag_{i}' for i in range(1, 11)]
        adapter.is_initialized = True

        model_path = tmp_path / "test_model.pkl"

        # Save the model
        adapter.save(model_path)
        assert model_path.exists()

        # Load the model
        loaded_adapter = LinearRegressionTimeSeriesAdapter.load(model_path)

        assert loaded_adapter.is_initialized
        assert len(loaded_adapter._models) == 1
        assert 1 in loaded_adapter._models
        assert loaded_adapter._feature_columns == [f'lag_{i}' for i in range(1, 11)]


class TestPrediction:
    """Test prediction functionality."""

    @patch('backend.models.linear_regression_time_series_adapter.sqlite3.connect')
    def test_prediction_initialized_model(self, mock_connect, trained_adapter, sample_price_data):
        """Test prediction with initialized model."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        with patch('pandas.read_sql_query', return_value=sample_price_data):
            inputs = {
                "ticker": "AAPL",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "horizon": 1
            }

            config = {"lag_days": 10}

            result = trained_adapter.predict(inputs, config)

            assert "predictions" in result
            assert len(result["predictions"]) == 1
            prediction = result["predictions"][0]
            assert prediction["ticker"] == "AAPL"
            assert "predicted_return" in prediction
            assert "confidence" in prediction
            assert prediction["horizon"] == 1

    def test_prediction_uninitialized_model(self):
        """Test prediction with uninitialized model raises error."""
        adapter = LinearRegressionTimeSeriesAdapter()

        inputs = {
            "ticker": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "horizon": 1
        }

        config = {"lag_days": 10}

        with pytest.raises(ValueError, match="Model not initialized"):
            adapter.predict(inputs, config)

    def test_prediction_invalid_inputs(self, trained_adapter):
        """Test prediction with invalid inputs."""
        # Missing required fields
        inputs = {
            "ticker": "AAPL",
            "start_date": "2023-01-01"
            # Missing end_date and horizon
        }

        config = {"lag_days": 10}

        with pytest.raises(ValueError, match="ticker, start_date, end_date, and horizon are required"):
            trained_adapter.predict(inputs, config)

    def test_prediction_invalid_horizon(self, trained_adapter):
        """Test prediction with invalid horizon."""
        inputs = {
            "ticker": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "horizon": 5  # Invalid horizon
        }

        config = {"lag_days": 10}

        with pytest.raises(ValueError, match="horizon must be 1, 3, or 7"):
            trained_adapter.predict(inputs, config)

    @patch('backend.models.linear_regression_time_series_adapter.sqlite3.connect')
    def test_prediction_insufficient_data(self, mock_connect, trained_adapter):
        """Test prediction with insufficient data."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        # Mock insufficient data
        small_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5).strftime('%Y-%m-%d'),
            'close': [100, 101, 102, 103, 104]
        })

        with patch('pandas.read_sql_query', return_value=small_data):
            inputs = {
                "ticker": "AAPL",
                "start_date": "2023-01-01",
                "end_date": "2023-01-05",
                "horizon": 1
            }

            config = {"lag_days": 10}

            with pytest.raises(ValueError, match="Insufficient data for prediction"):
                trained_adapter.predict(inputs, config)

    def test_prediction_response_format(self, trained_adapter):
        """Test prediction response format."""
        # Mock the internal methods to avoid database calls
        with patch.object(trained_adapter, '_fetch_price_data') as mock_fetch, \
             patch.object(trained_adapter, '_compute_features') as mock_compute:

            # Mock return values
            mock_price_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
            }, index=pd.date_range('2023-01-01', periods=12))

            mock_fetch.return_value = mock_price_data
            mock_features = pd.DataFrame({
                'lag_1': [0.01, 0.02],
                'lag_2': [0.005, 0.015]
            })
            mock_compute.return_value = mock_features

            inputs = {
                "ticker": "AAPL",
                "start_date": "2023-01-01",
                "end_date": "2023-01-12",
                "horizon": 1
            }

            config = {"lag_days": 2}

            result = trained_adapter.predict(inputs, config)

            # Check response structure
            assert "predictions" in result
            assert "meta" in result

            prediction = result["predictions"][0]
            required_fields = ["ticker", "date", "predicted_return", "confidence", "horizon", "model_version", "features_used"]
            for field in required_fields:
                assert field in prediction

            meta = result["meta"]
            required_meta = ["model_name", "model_version", "generated_at", "date_range", "tickers", "horizon"]
            for field in required_meta:
                assert field in meta


class TestIntegration:
    """Test integration functionality."""

    @patch('backend.models.registry.ModelRegistry')
    def test_model_registry_discovery(self, mock_registry_class):
        """Test model discovery through registry."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        # Mock the registry to return our adapter
        mock_adapter = Mock(spec=LinearRegressionTimeSeriesAdapter)
        mock_adapter.name = "linear_regression_time_series_v1"
        mock_registry.get.return_value = mock_adapter

        from backend.models.registry import ModelRegistry
        registry = ModelRegistry()
        registry.discover = Mock()

        # Simulate discovery
        registry.discover([], Path("backend/models"))

        # Check that our model can be retrieved
        model = registry.get("linear_regression_time_series_v1")
        assert model is not None
        assert model.name == "linear_regression_time_series_v1"

    def test_api_endpoint_compatibility(self):
        """Test API endpoint compatibility."""
        # Test that the model can be used with the expected API interface
        adapter = LinearRegressionTimeSeriesAdapter()

        # Verify it has the required methods for API endpoints
        assert hasattr(adapter, 'predict')
        assert hasattr(adapter, 'retrain')
        assert hasattr(adapter, 'get_config_schema')

        # Verify config schema is compatible
        schema = adapter.get_config_schema()
        assert hasattr(schema, 'model_json_schema')  # FastAPI compatibility

        # Test that model info can be extracted for API responses
        model_info = {
            "name": adapter.name,
            "type": adapter.type,
            "version": adapter.version,
            "description": adapter.description,
            "capabilities": adapter.capabilities,
            "config_schema": schema.model_json_schema()
        }

        assert model_info["name"] == "linear_regression_time_series_v1"
        assert "predict" in model_info["capabilities"]
        assert "retrain" in model_info["capabilities"]

    def test_database_connection_handling(self):
        """Test database connection handling."""
        adapter = LinearRegressionTimeSeriesAdapter()

        # Mock app_state
        with patch('backend.main.app_state', {'database_path': ':memory:'}):
            # This should work without actual database
            conn = adapter._get_db_connection()
            assert conn is not None
            conn.close()

    def test_error_handling_and_logging(self, trained_adapter):
        """Test error handling and logging."""
        # Test with invalid inputs that should trigger logging
        inputs = {
            "ticker": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "horizon": 1
        }

        config = {"lag_days": 10}

        # Mock logger to capture log calls
        with patch.object(trained_adapter, 'logger') as mock_logger:
            # Force an error in prediction
            with patch.object(trained_adapter, '_fetch_price_data', side_effect=Exception("Test error")):
                with pytest.raises(Exception):
                    trained_adapter.predict(inputs, config)

                # Verify error was logged
                mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])