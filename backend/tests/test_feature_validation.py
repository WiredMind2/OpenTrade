"""
Integration tests for feature validation in model predictions.

Tests ensure that input data features match the model's expected features
to prevent prediction failures due to feature mismatches.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from backend.models.adapters.joblib_adapter import JoblibModelAdapter


def test_feature_match():
    """Test successful prediction when features match."""
    # Create a mock model with expected features
    mock_model = MagicMock()
    mock_model.feature_names_ = ['feature1', 'feature2', 'feature3']
    mock_model.predict.return_value = [0.5]

    # Create adapter
    model_data = {'lgbm': mock_model, 'embedder': 'test'}
    adapter = JoblibModelAdapter('test_model', model_data)

    # Test data with matching features
    test_data = {
        'features': [1.0, 2.0, 3.0]  # 3 features
    }
    config = {}

    result = adapter.predict(test_data, config)
    assert result is not None
    assert 'prediction' in result
    assert 'confidence' in result


def test_feature_mismatch():
    """Test failure when features don't match."""
    # Create a mock model with expected features
    mock_model = MagicMock()
    mock_model.feature_names_ = ['feature1', 'feature2', 'feature3']

    # Create adapter
    model_data = {'lgbm': mock_model, 'embedder': 'test'}
    adapter = JoblibModelAdapter('test_model', model_data)

    # Test data with missing feature
    test_data = {
        'features': [1.0, 2.0]  # Only 2 features, expected 3
    }
    config = {}

    with pytest.raises(ValueError, match="Feature mismatch"):
        adapter.predict(test_data, config)


def test_feature_match_2d_array():
    """Test successful prediction with 2D feature array."""
    # Create a mock model with expected features
    mock_model = MagicMock()
    mock_model.feature_names_ = ['feature1', 'feature2']
    mock_model.predict.return_value = [0.5]

    # Create adapter
    model_data = {'lgbm': mock_model, 'embedder': 'test'}
    adapter = JoblibModelAdapter('test_model', model_data)

    # Test data with 2D features
    test_data = {
        'features': [[1.0, 2.0]]  # 1 sample, 2 features
    }
    config = {}

    result = adapter.predict(test_data, config)
    assert result is not None


def test_no_feature_validation_for_models_without_feature_names():
    """Test that validation is skipped for models without feature_names_."""
    # Create a mock model without feature_names_
    mock_model = MagicMock()
    del mock_model.feature_names_  # Ensure it doesn't have the attribute
    mock_model.predict.return_value = [0.5]

    # Create adapter
    model_data = {'lgbm': mock_model, 'embedder': 'test'}
    adapter = JoblibModelAdapter('test_model', model_data)

    # Test data with any number of features
    test_data = {
        'features': [1.0, 2.0, 3.0, 4.0]  # 4 features, no validation
    }
    config = {}

    result = adapter.predict(test_data, config)
    assert result is not None