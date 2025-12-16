"""
Unit tests for three_ma_adapter module.

Tests import success and initialization of the ThreeMAAdapter class.
"""

import pytest
from backend.models.three_ma_adapter import ThreeMAAdapter


def test_three_ma_adapter_import():
    """Test successful import of three_ma_adapter module."""
    try:
        from backend.models import three_ma_adapter
        assert three_ma_adapter is not None
    except ImportError as e:
        pytest.fail(f"Failed to import three_ma_adapter module: {e}")


def test_three_ma_adapter_initialization():
    """Test adapter initialization."""
    adapter = ThreeMAAdapter()
    assert adapter is not None
    assert hasattr(adapter, 'predict')
    assert hasattr(adapter, 'retrain')
    assert adapter.name == "three_ma_crossover_v1"
    assert adapter.version == "1.0.0"
    assert "predict" in adapter.capabilities
    assert "retrain" in adapter.capabilities


def test_three_ma_adapter_config_schema():
    """Test that the config schema is properly defined."""
    adapter = ThreeMAAdapter()
    schema = adapter.get_config_schema()
    assert schema is not None
    # Should be a Pydantic model
    assert hasattr(schema, '__fields__') or hasattr(schema, 'model_fields')


def test_three_ma_adapter_predict_method_exists():
    """Test that predict method exists and has correct signature."""
    adapter = ThreeMAAdapter()
    assert hasattr(adapter, 'predict')
    # Check method signature by calling with invalid inputs to ensure it raises expected errors
    with pytest.raises(ValueError, match="start and end dates are required"):
        adapter.predict({}, {})


def test_three_ma_adapter_retrain_method_exists():
    """Test that retrain method exists."""
    adapter = ThreeMAAdapter()
    assert hasattr(adapter, 'retrain')