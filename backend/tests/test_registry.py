"""
Unit tests for model registry module.
"""

import pytest
from backend.models.registry import ModelRegistry


def test_model_registry_import():
    """Test successful import of registry module."""
    try:
        from backend.models import registry
        assert registry is not None
    except ImportError as e:
        pytest.fail(f"Failed to import registry module: {e}")


def test_model_registry_initialization():
    """Test registry initialization."""
    registry = ModelRegistry()
    assert registry is not None
    assert hasattr(registry, 'register')  # Method to register models
    assert hasattr(registry, 'get')       # Method to get models
    assert hasattr(registry, 'list')      # Method to list models