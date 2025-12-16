"""
Tests for model training utilities.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.scripts.train_model import retrain_model


class TestRetrainModel:
    """Test the retrain_model function."""

    @patch('backend.main.app_state')
    def test_retraining_with_dates(self, mock_app_state):
        """Test retraining with valid dates."""
        # Mock the registry and model
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_model.capabilities = ["predict", "retrain"]
        mock_model.retrain.return_value = {"status": "completed"}

        mock_registry.get.return_value = mock_model
        mock_app_state.get.return_value = mock_registry

        start_date = '2023-01-01'
        end_date = '2023-12-31'
        model_name = 'test_model'

        result = retrain_model(model_name, start_date, end_date)

        assert result is not None
        assert result["status"] == "completed"
        mock_model.retrain.assert_called_once()

    @patch('backend.main.app_state')
    def test_retraining_missing_dates(self, mock_app_state):
        """Test retraining with missing dates."""
        model_name = 'test_model'

        with pytest.raises(ValueError, match="start_date and end_date required"):
            retrain_model(model_name, None, None)

    @patch('backend.main.app_state')
    def test_retraining_model_not_found(self, mock_app_state):
        """Test retraining with non-existent model."""
        # Mock the registry
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_app_state.get.return_value = mock_registry

        start_date = '2023-01-01'
        end_date = '2023-12-31'
        model_name = 'non_existent_model'

        with pytest.raises(ValueError, match="Model 'non_existent_model' not found"):
            retrain_model(model_name, start_date, end_date)

    @patch('backend.main.app_state')
    def test_retraining_model_not_supported(self, mock_app_state):
        """Test retraining with model that doesn't support retraining."""
        # Mock the registry and model
        mock_registry = MagicMock()
        mock_model = MagicMock()
        mock_model.capabilities = ["predict"]  # No "retrain"

        mock_registry.get.return_value = mock_model
        mock_app_state.get.return_value = mock_registry

        start_date = '2023-01-01'
        end_date = '2023-12-31'
        model_name = 'non_retrainable_model'

        with pytest.raises(ValueError, match="Model 'non_retrainable_model' does not support retraining"):
            retrain_model(model_name, start_date, end_date)

    @patch('backend.main.app_state')
    def test_retraining_registry_not_available(self, mock_app_state):
        """Test retraining when registry is not available."""
        mock_app_state.get.return_value = None

        start_date = '2023-01-01'
        end_date = '2023-12-31'
        model_name = 'test_model'

        with pytest.raises(ValueError, match="Model registry not available"):
            retrain_model(model_name, start_date, end_date)