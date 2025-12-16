"""
Base model adapter with initialization checks.

This module provides a base adapter class that includes initialization state
tracking and validation for model prediction methods.
"""

from abc import ABC
from typing import Dict, Any
import pydantic

from backend.logging_config import get_component_logger
from ..base import BaseModel


class BaseModelAdapter(BaseModel):
    """Base adapter class with initialization state tracking."""

    def __init__(self, name: str, type: str, version: str, description: str, capabilities):
        super().__init__(name, type, version, description, capabilities)
        self.is_initialized = False
        self.logger = get_component_logger(f"backend.models.adapters.base_adapter.{name}")

    def predict(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction, checking initialization first."""
        if not self.is_initialized:
            raise ValueError("Model not initialized")
        return self._predict_impl(inputs, config)

    def _predict_impl(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation of prediction logic in subclasses."""
        raise NotImplementedError("Subclasses must implement _predict_impl method")

    def initialize(self) -> None:
        """Initialize the model."""
        self.is_initialized = True
        self.logger.info(f"Model {self.name} initialized")

    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        """Retrain the model."""
        # Default implementation - subclasses can override
        return {"status": "completed", "message": "Retraining not implemented"}

    def save(self, path) -> None:
        """Save the model."""
        raise NotImplementedError("Saving not implemented")

    @classmethod
    def load(cls, path):
        """Load the model."""
        raise NotImplementedError("Loading not implemented")