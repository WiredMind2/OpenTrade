"""
Adapter for legacy joblib model bundles.

This module provides an adapter that wraps legacy joblib model files
to conform to the new BaseModel interface.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, Optional
import pydantic

from backend.logging_config import get_component_logger
from .base_adapter import BaseModelAdapter


class JoblibModelConfig(pydantic.BaseModel):
    """Configuration schema for joblib models."""
    horizon: Optional[str] = "1d"
    features: Optional[list] = ["article_sentiment", "price_momentum", "volume"]


class JoblibModelAdapter(BaseModelAdapter):
    """Adapter for legacy joblib model bundles."""

    def __init__(self, name: str, model_data: Dict[str, Any]):
        # Extract model info from the bundle
        self._model_data = model_data
        self._model = model_data.get('lgbm')
        self._embedder = model_data.get('embedder', 'all-MiniLM-L6-v2')

        # Extract metadata if available (canonical format)
        meta = model_data.get('meta', {})
        extras = model_data.get('extras', {})

        # Set model attributes
        model_type = meta.get('type', 'lightgbm')
        version = meta.get('version', '1.0.0')
        description = meta.get('description', f'Legacy {model_type} model for {name}')
        capabilities = meta.get('capabilities', ['predict'])

        super().__init__(name, model_type, version, description, capabilities)

        self.logger = get_component_logger("backend.models.adapters.joblib_adapter")
        self.is_initialized = self._model is not None
        self.expected_features = getattr(self._model, 'feature_names_', None) if self._model else None

    def get_config_schema(self) -> type[pydantic.BaseModel]:
        """Return the configuration schema for this model."""
        return JoblibModelConfig

    def _predict_impl(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the legacy model."""
        try:
            # For legacy compatibility, inputs can contain pre-computed features
            # or we need to compute them from ticker/horizon data
            if 'features' in inputs:
                # Direct feature input
                features = np.array(inputs['features'])
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                # Validate feature count
                if self.expected_features and features.shape[1] != len(self.expected_features):
                    raise ValueError(f"Feature mismatch: expected {len(self.expected_features)} features, got {features.shape[1]}")
            else:
                # Need to compute features from ticker/horizon
                # This is a simplified version - in practice would need full data access
                ticker = inputs.get('ticker', '')
                horizon = inputs.get('horizon', '1d')

                # For now, return a mock prediction based on legacy behavior
                # In a full implementation, this would query the database for articles/prices
                self.logger.warning(f"Computing features for {ticker} not implemented in adapter, using mock features")
                features = np.array([[0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0]])  # Mock features

            # Make prediction
            if self._model is None:
                raise ValueError("Model not loaded")

            prediction = float(self._model.predict(features)[0])

            # Calculate confidence (simplified)
            confidence = max(0.1, min(0.95, 1.0 - abs(prediction) * 2))

            return {
                'prediction': prediction,
                'confidence': confidence,
                'model_name': self.name,
                'features_used': config.get('features', ['article_sentiment', 'price_momentum', 'volume'])
            }

        except Exception as e:
            self.logger.error(f"Prediction failed for model {self.name}: {str(e)}")
            raise

    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        """Retrain the model (not supported for legacy models)."""
        raise NotImplementedError("Retraining not supported for legacy joblib models")

    def save(self, path: Path) -> None:
        """Save the model (not supported for legacy models)."""
        raise NotImplementedError("Saving not supported for legacy joblib models")

    @classmethod
    def load(cls, path: Union[Path, Dict[str, Any]]) -> 'JoblibModelAdapter':
        """Load a model from path or data dict."""
        if isinstance(path, dict):
            # Direct data dict
            model_data = path
            name = "unknown"
        else:
            # Load from file
            model_data = joblib.load(path)
            name = Path(path).stem

        return cls(name, model_data)