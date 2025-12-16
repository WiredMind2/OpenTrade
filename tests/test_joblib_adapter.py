"""
Unit tests for the joblib model adapter.
"""

import pytest
import numpy as np
import tempfile
import joblib
from pathlib import Path
from unittest.mock import Mock, patch

from backend.models.adapters.joblib_adapter import JoblibModelAdapter, JoblibModelConfig


class TestJoblibModelAdapter:
    """Test cases for JoblibModelAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        mock_model_data = {
            'lgbm': Mock(),
            'embedder': 'test_embedder'
        }

        adapter = JoblibModelAdapter("test_model", mock_model_data)

        assert adapter.name == "test_model"
        assert adapter.type == "lightgbm"
        assert adapter.version == "1.0.0"
        assert adapter.description == "Legacy lightgbm model for test_model"
        assert adapter.capabilities == ["predict"]
        assert adapter._model == mock_model_data['lgbm']
        assert adapter._embedder == 'test_embedder'

    def test_adapter_initialization_with_meta(self):
        """Test adapter with canonical format metadata."""
        mock_model_data = {
            'meta': {
                'type': 'custom_model',
                'version': '2.0.0',
                'description': 'Custom model description',
                'capabilities': ['predict', 'retrain']
            },
            'model': Mock(),
            'extras': {}
        }

        adapter = JoblibModelAdapter("test_model", mock_model_data)

        assert adapter.type == "custom_model"
        assert adapter.version == "2.0.0"
        assert adapter.description == "Custom model description"
        assert adapter.capabilities == ["predict", "retrain"]

    def test_get_config_schema(self):
        """Test getting configuration schema."""
        mock_model_data = {'lgbm': Mock(), 'embedder': 'test'}
        adapter = JoblibModelAdapter("test_model", mock_model_data)

        schema = adapter.get_config_schema()
        assert schema == JoblibModelConfig

    def test_predict_with_features(self):
        """Test prediction with direct features input."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.5])

        mock_model_data = {'lgbm': mock_model, 'embedder': 'test'}
        adapter = JoblibModelAdapter("test_model", mock_model_data)

        inputs = {'features': [0.1, 0.2, 1, 0.0, 0.0, 100.0, 0.1]}
        config = {}

        result = adapter.predict(inputs, config)

        assert 'prediction' in result
        assert 'confidence' in result
        assert 'model_name' in result
        assert result['prediction'] == 0.5
        assert result['model_name'] == "test_model"
        mock_model.predict.assert_called_once()

    def test_predict_without_features(self):
        """Test prediction without direct features (mock fallback)."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.3])

        mock_model_data = {'lgbm': mock_model, 'embedder': 'test'}
        adapter = JoblibModelAdapter("test_model", mock_model_data)

        inputs = {'ticker': 'AAPL', 'horizon': '1d'}
        config = {}

        result = adapter.predict(inputs, config)

        assert 'prediction' in result
        assert 'confidence' in result
        assert result['prediction'] == 0.3

    def test_predict_model_failure(self):
        """Test prediction when model fails."""
        mock_model = Mock()
        mock_model.predict.side_effect = RuntimeError("Model error")

        mock_model_data = {'lgbm': mock_model, 'embedder': 'test'}
        adapter = JoblibModelAdapter("test_model", mock_model_data)

        inputs = {'features': [0.1, 0.2, 1, 0.0, 0.0, 100.0, 0.1]}
        config = {}

        with pytest.raises(RuntimeError, match="Model error"):
            adapter.predict(inputs, config)

    def test_retrain_not_implemented(self):
        """Test that retrain raises NotImplementedError."""
        mock_model_data = {'lgbm': Mock(), 'embedder': 'test'}
        adapter = JoblibModelAdapter("test_model", mock_model_data)

        with pytest.raises(NotImplementedError):
            adapter.retrain({}, {}, False)

    def test_save_not_implemented(self):
        """Test that save raises NotImplementedError."""
        mock_model_data = {'lgbm': Mock(), 'embedder': 'test'}
        adapter = JoblibModelAdapter("test_model", mock_model_data)

        with pytest.raises(NotImplementedError):
            adapter.save(Path("/tmp/test"))

    @patch('backend.models.adapters.joblib_adapter.joblib.load')
    def test_load_from_file(self, mock_joblib_load):
        """Test loading adapter from file."""
        mock_model_data = {'lgbm': Mock(), 'embedder': 'test'}
        mock_joblib_load.return_value = mock_model_data

        with tempfile.NamedTemporaryFile(suffix='.joblib') as temp_file:
            temp_path = Path(temp_file.name)

            adapter = JoblibModelAdapter.load(temp_path)

            assert isinstance(adapter, JoblibModelAdapter)
            assert adapter.name == temp_path.stem
            mock_joblib_load.assert_called_once_with(temp_path)

    def test_load_from_dict(self):
        """Test loading adapter from dict."""
        mock_model_data = {'lgbm': Mock(), 'embedder': 'test'}

        adapter = JoblibModelAdapter.load(mock_model_data)

        assert isinstance(adapter, JoblibModelAdapter)
        assert adapter.name == "unknown"
        assert adapter._model_data == mock_model_data