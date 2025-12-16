"""
Unit tests for the model registry.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock

from backend.models.registry import ModelRegistry
from backend.models.adapters.joblib_adapter import JoblibModelAdapter


class TestModelRegistry:
    """Test cases for ModelRegistry."""

    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        registry = ModelRegistry()
        assert len(registry.list()) == 0

    def test_register_and_get_model(self):
        """Test registering and retrieving a model."""
        registry = ModelRegistry()
        mock_model = Mock()
        mock_model.name = "test_model"

        registry.register(mock_model)
        retrieved = registry.get("test_model")

        assert retrieved == mock_model

    def test_get_nonexistent_model(self):
        """Test getting a model that doesn't exist."""
        registry = ModelRegistry()
        assert registry.get("nonexistent") is None

    def test_list_models(self):
        """Test listing all registered models."""
        registry = ModelRegistry()

        mock_model1 = Mock()
        mock_model1.name = "model1"
        mock_model2 = Mock()
        mock_model2.name = "model2"

        registry.register(mock_model1)
        registry.register(mock_model2)

        models = registry.list()
        assert len(models) == 2
        assert mock_model1 in models
        assert mock_model2 in models

    def test_discover_joblib_models(self):
        """Test discovering joblib models from directory."""
        registry = ModelRegistry()

        # Create a temporary directory with a mock joblib file
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            # Create a mock joblib file
            joblib_file = models_dir / "test_model.joblib"

            # Mock the joblib.load function
            import backend.models.adapters.joblib_adapter as adapter_module
            original_load = adapter_module.joblib.load

            mock_model_data = {'lgbm': Mock(), 'embedder': 'test'}
            adapter_module.joblib.load = Mock(return_value=mock_model_data)

            try:
                # Create empty file to simulate joblib file
                joblib_file.touch()

                models = registry.discover_joblib_models(models_dir)
                assert len(models) == 1
                assert isinstance(models[0], JoblibModelAdapter)
                assert models[0].name == "test_model"

            finally:
                adapter_module.joblib.load = original_load

    def test_discover_joblib_models_no_directory(self):
        """Test discovering models when directory doesn't exist."""
        registry = ModelRegistry()
        nonexistent_dir = Path("/nonexistent/path")

        models = registry.discover_joblib_models(nonexistent_dir)
        assert len(models) == 0

    def test_discover_python_models(self):
        """Test discovering Python model modules."""
        registry = ModelRegistry()

        with tempfile.TemporaryDirectory() as temp_dir:
            models_pkg_dir = Path(temp_dir)

            # Create a mock Python model file
            model_file = models_pkg_dir / "test_model.py"

            # Write a simple model class
            model_file.write_text("""
from backend.models.base import BaseModel
import pydantic

class TestModel(BaseModel):
    def __init__(self):
        super().__init__("test_model", "test", "1.0.0", "Test model", ["predict"])

    def get_config_schema(self):
        return pydantic.BaseModel

    def predict(self, inputs, config):
        return {"result": "test"}

    def retrain(self, training_payload, config, background=False):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()
""")

            models = registry.discover_python_models(models_pkg_dir)
            assert len(models) == 1
            assert models[0].name == "test_model"

    def test_discover_python_models_no_directory(self):
        """Test discovering Python models when directory doesn't exist."""
        registry = ModelRegistry()
        nonexistent_dir = Path("/nonexistent/path")

        models = registry.discover_python_models(nonexistent_dir)
        assert len(models) == 0

    def test_thread_safety(self):
        """Test that registry operations are thread-safe."""
        import threading
        import time

        registry = ModelRegistry()
        results = []

        def add_models(thread_id):
            for i in range(10):
                mock_model = Mock()
                mock_model.name = f"model_{thread_id}_{i}"
                registry.register(mock_model)
                time.sleep(0.001)  # Small delay to encourage race conditions

        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_models, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        models = registry.list()
        assert len(models) == 50  # 5 threads * 10 models each