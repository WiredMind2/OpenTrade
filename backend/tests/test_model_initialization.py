"""
Integration tests for model initialization and prediction.

Tests ensure models are properly initialized before prediction and handle
uninitialized states correctly.
"""

import pytest
import pandas as pd
import pydantic
from backend.models.adapters.base_adapter import BaseModelAdapter


class TestAdapterConfig(pydantic.BaseModel):
    """Config for test adapter."""
    pass


class TestAdapter(BaseModelAdapter):
    """Test adapter for model initialization testing."""

    def __init__(self):
        super().__init__("test_model", "test", "1.0.0", "Test model", ["predict"])

    def get_config_schema(self):
        return TestAdapterConfig

    def _predict_impl(self, inputs, config):
        # Simple prediction: return mean of data
        data = inputs.get('data', pd.Series([1, 2, 3]))
        return {'prediction': data.mean()}


def test_model_prediction_success():
    """Test successful prediction when model is initialized."""
    adapter = TestAdapter()
    adapter.is_initialized = True
    test_data = pd.Series([1, 2, 3, 4, 5])

    result = adapter.predict({'data': test_data}, {})

    assert result['prediction'] == 3.0


def test_model_prediction_uninitialized():
    """Test prediction failure when model is not initialized."""
    adapter = TestAdapter()
    adapter.is_initialized = False
    test_data = pd.Series([1, 2, 3])

    with pytest.raises(ValueError, match="Model not initialized"):
        adapter.predict({'data': test_data}, {})