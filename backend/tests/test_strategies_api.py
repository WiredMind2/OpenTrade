"""
Unit and integration tests for strategies API endpoints and projection logic.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from backend.main import app
from backend.routes.strategies import ProjectionRequest


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestStrategiesAPI:
    """Test class for strategies API endpoints."""

    def test_list_strategies_success(self, client):
        """Test successful listing of strategies."""

        # Mock strategy registry
        mock_registry = MagicMock()
        mock_registry.list.return_value = [
            {
                'name': 'moving_average',
                'description': 'Moving Average Strategy',
                'type': 'rule',
                'parameters_schema': {
                    'short_window': {'type': 'int', 'default': 10},
                    'long_window': {'type': 'int', 'default': 30}
                },
                'can_train': False
            },
            {
                'name': 'sentiment_ml',
                'description': 'Sentiment ML Strategy',
                'type': 'ml',
                'parameters_schema': {
                    'prediction_threshold': {'type': 'float', 'default': 0.5}
                },
                'can_train': True
            }
        ]

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            response = client.get("/api/strategies")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]['name'] == 'moving_average'
        assert data[1]['name'] == 'sentiment_ml'

    def test_list_strategies_registry_not_found(self, client):
        """Test listing strategies when registry is not available."""

        with patch('backend.main.app_state', {}):
            response = client.get("/api/strategies")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_strategy_success(self, client):
        """Test successful retrieval of strategy metadata."""

        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy.name = 'moving_average'
        mock_strategy.description = 'Moving Average Strategy'
        mock_strategy.type = 'rule'
        mock_strategy.parameters_schema = {
            'short_window': {'type': 'int', 'default': 10},
            'long_window': {'type': 'int', 'default': 30}
        }
        mock_strategy.can_train = False

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_strategy

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            response = client.get("/api/strategies/moving_average")

        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'moving_average'
        assert data['type'] == 'rule'
        assert data['can_train'] is False
        assert 'parameters_schema' in data

    def test_get_strategy_not_found(self, client):
        """Test retrieval of non-existent strategy."""

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            response = client.get("/api/strategies/non_existent")

        assert response.status_code == 404
        assert "not found" in response.json()['detail'].lower()

    def test_project_strategy_success_moving_average(self, client):
        """Test successful projection for moving_average strategy."""

        # Mock strategy with project method
        mock_strategy = MagicMock()
        mock_strategy.name = 'moving_average'
        mock_strategy.parameters_schema = {
            'short_window': {'type': 'int', 'default': 10},
            'long_window': {'type': 'int', 'default': 30}
        }
        mock_strategy.project.return_value = {
            'projected_return': 0.15,
            'projected_volatility': 0.12,
            'confidence': 0.7,
            'projection_days': 30,
            'initial_capital': 100000.0,
            'projected_final_value': 115000.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_strategy

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            request_data = {
                'symbol': 'AAPL',
                'startTime': '2023-01-01T00:00:00Z',
                'startPrice': 150.0,
                'params': {'short_window': 10, 'long_window': 30},
                'horizon': 30
            }
            response = client.post(
                "/api/strategies/moving_average/project",
                json=request_data
            )

        assert response.status_code == 200
        data = response.json()
        assert 'projected_return' in data
        assert 'projected_volatility' in data
        assert 'confidence' in data
        assert data['projection_days'] == 30
        assert data['initial_capital'] == 100000.0
        mock_strategy.project.assert_called_once_with(
            parameters={'short_window': 10, 'long_window': 30},
            projection_days=30,
            initial_capital=150.0  # startPrice
        )

    def test_project_strategy_success_sentiment_ml(self, client):
        """Test successful projection for sentiment_ml strategy."""

        # Mock strategy with project method
        mock_strategy = MagicMock()
        mock_strategy.name = 'sentiment_ml'
        mock_strategy.parameters_schema = {
            'prediction_threshold': {'type': 'float', 'default': 0.5}
        }
        mock_strategy.project.return_value = {
            'projected_return': 0.25,
            'projected_volatility': 0.18,
            'confidence': 0.8,
            'projection_days': 60,
            'initial_capital': 50000.0,
            'projected_final_value': 62500.0,
            'avg_predicted_return': 0.02,
            'model_version': 'sentiment_model__20231201_120000__v1',
            'timestamp': datetime.utcnow().isoformat()
        }

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_strategy

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            request_data = {
                'symbol': 'TSLA',
                'startTime': '2023-01-01T00:00:00Z',
                'startPrice': 200.0,
                'params': {'prediction_threshold': 0.6},
                'horizon': 60
            }
            response = client.post(
                "/api/strategies/sentiment_ml/project",
                json=request_data
            )

        assert response.status_code == 200
        data = response.json()
        assert data['projected_return'] == 0.25
        assert data['confidence'] == 0.8
        assert 'model_version' in data
        mock_strategy.project.assert_called_once_with(
            parameters={'prediction_threshold': 0.6},
            projection_days=60,
            initial_capital=200.0  # startPrice
        )

    def test_project_strategy_invalid_params(self, client):
        """Test projection with invalid parameters."""

        # Mock strategy that raises exception
        mock_strategy = MagicMock()
        mock_strategy.name = 'moving_average'
        mock_strategy.project.side_effect = ValueError("Invalid parameters")

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_strategy

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            request_data = {
                'symbol': 'AAPL',
                'startTime': '2023-01-01T00:00:00Z',
                'startPrice': 150.0,
                'params': {'invalid_param': 'value'},
                'horizon': 30
            }
            response = client.post(
                "/api/strategies/moving_average/project",
                json=request_data
            )

        assert response.status_code == 400
        assert "Unknown parameter" in response.json()['detail']

    def test_project_strategy_not_found(self, client):
        """Test projection for non-existent strategy."""

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            request_data = {
                'symbol': 'AAPL',
                'startTime': '2023-01-01T00:00:00Z',
                'startPrice': 150.0,
                'params': {},
                'horizon': 30
            }
            response = client.post(
                "/api/strategies/non_existent/project",
                json=request_data
            )

        assert response.status_code == 404
        assert "not found" in response.json()['detail'].lower()

    def test_project_strategy_invalid_projection_days(self, client):
        """Test projection with invalid projection_days (too high)."""

        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()  # Strategy exists

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            request_data = {
                'symbol': 'AAPL',
                'startTime': '2023-01-01T00:00:00Z',
                'startPrice': 150.0,
                'params': {},
                'horizon': 400  # Exceeds max of 365
            }
            response = client.post(
                "/api/strategies/moving_average/project",
                json=request_data
            )

        # Should fail validation before reaching strategy.project()
        assert response.status_code == 422  # Validation error

    def test_project_strategy_invalid_capital(self, client):
        """Test projection with invalid initial_capital (negative)."""

        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock()  # Strategy exists

        with patch('backend.main.app_state', {'strategy_registry': mock_registry}):
            request_data = {
                'symbol': 'AAPL',
                'startTime': '2023-01-01T00:00:00Z',
                'startPrice': -1000.0,  # Invalid negative value
                'params': {},
                'horizon': 30
            }
            response = client.post(
                "/api/strategies/moving_average/project",
                json=request_data
            )

        # Should fail validation before reaching strategy.project()
        assert response.status_code == 422  # Validation error

    def test_public_strategies_endpoints_no_auth_required(self, client):
        """Test that public strategies endpoints don't require authentication."""
        endpoints = [
            "/api/strategies",
            "/api/strategies/moving_average"
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should not return auth errors
            assert response.status_code not in [401, 403]


class TestStrategyProjectionLogic:
    """Test class for strategy projection methods."""

    def test_moving_average_project_method(self):
        """Test the project method of MovingAverageStrategy."""
        from backend.strategies.moving_average import MovingAverageStrategy

        strategy = MovingAverageStrategy()

        # Test with default parameters
        result = strategy.project(
            parameters={},
            projection_days=30,
            initial_capital=100000.0
        )

        assert isinstance(result, dict)
        assert 'projected_return' in result
        assert 'projected_volatility' in result
        assert 'confidence' in result
        assert result['projection_days'] == 30
        assert result['initial_capital'] == 100000.0
        assert 'projected_final_value' in result
        assert 'timestamp' in result

        # Test with custom parameters
        result_custom = strategy.project(
            parameters={'short_window': 5, 'long_window': 20},
            projection_days=60,
            initial_capital=50000.0
        )

        assert result_custom['projection_days'] == 60
        assert result_custom['initial_capital'] == 50000.0

    def test_sentiment_ml_project_method(self):
        """Test the project method of SentimentMLStrategy."""
        pytest.skip("sentiment_ml strategy module removed from backend/strategies")

        # Mock database connection to simulate missing table (fallback behavior)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        # Simulate table not existing
        # predictions table was removed; this legacy query should no longer exist
        mock_cursor.execute.side_effect = Exception("no such table: trading_model_predictions")

        with patch('backend.main.app_state', {'database_path': 'data/backtest.db'}), \
             patch('sqlite3.connect', return_value=mock_conn):

            strategy = SentimentMLStrategy()

            # Test with default parameters
            result = strategy.project(
                parameters={},
                projection_days=30,
                initial_capital=100000.0
            )

            assert isinstance(result, dict)
            assert 'projected_return' in result
            assert 'projected_volatility' in result
            assert 'confidence' in result
            assert result['projection_days'] == 30
            assert result['initial_capital'] == 100000.0
            assert 'projected_final_value' in result
            assert 'timestamp' in result

            # When database table doesn't exist, it should return fallback values
            # The fallback doesn't include avg_predicted_return, predictions_used, etc.
            # So we check that it has the basic required fields

            # Test with custom parameters
            result_custom = strategy.project(
                parameters={'prediction_threshold': 0.7},
                projection_days=90,
                initial_capital=200000.0
            )

            assert result_custom['projection_days'] == 90
            assert result_custom['initial_capital'] == 200000.0

    def test_moving_average_project_with_mock_data(self):
        """Test moving average projection with mocked database data."""
        from backend.strategies.moving_average import MovingAverageStrategy

        # Mock database connection and data
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ('2023-01-01', 100.0, 1000),
            ('2023-01-02', 101.0, 1100),
            ('2023-01-03', 102.0, 1200),
        ]

        with patch('backend.main.app_state', {'database_path': 'data/backtest.db'}), \
             patch('sqlite3.connect', return_value=mock_conn):

            strategy = MovingAverageStrategy()
            result = strategy.project(
                parameters={},
                projection_days=30,
                initial_capital=100000.0
            )

        assert isinstance(result, dict)
        assert 'projected_return' in result
        assert result['projection_days'] == 30

    def test_sentiment_ml_project_with_mock_data(self):
        """Test sentiment ML projection with mocked database data."""
        pytest.skip("sentiment_ml strategy module removed from backend/strategies")

        # Mock database connection and data
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (0.1,),
            (0.15,),
            (0.08,),
        ]

        with patch('backend.main.app_state', {'database_path': 'data/backtest.db'}), \
             patch('sqlite3.connect', return_value=mock_conn):

            strategy = SentimentMLStrategy()
            result = strategy.project(
                parameters={},
                projection_days=30,
                initial_capital=100000.0
            )

        assert isinstance(result, dict)
        assert 'projected_return' in result
        assert 'avg_predicted_return' in result
        assert 'predictions_used' in result
        assert result['predictions_used'] == 3

    def test_projection_response_format(self):
        """Test that projection responses have consistent format."""
        from backend.strategies.moving_average import MovingAverageStrategy
        strategies = [MovingAverageStrategy()]

        for strategy in strategies:
            result = strategy.project(
                parameters={},
                projection_days=30,
                initial_capital=100000.0
            )

            # Required fields for all strategies
            required_fields = [
                'projected_return', 'projected_volatility', 'confidence',
                'projection_days', 'initial_capital', 'projected_final_value', 'timestamp'
            ]

            for field in required_fields:
                assert field in result, f"Missing required field: {field}"

            # Type checks
            assert isinstance(result['projected_return'], (int, float))
            assert isinstance(result['projected_volatility'], (int, float))
            assert isinstance(result['confidence'], (int, float))
            assert isinstance(result['projection_days'], int)
            assert isinstance(result['initial_capital'], (int, float))
            assert isinstance(result['projected_final_value'], (int, float))
            assert isinstance(result['timestamp'], str)

            # Value constraints
            assert result['projection_days'] > 0
            assert result['initial_capital'] > 0
            assert 0 <= result['confidence'] <= 1