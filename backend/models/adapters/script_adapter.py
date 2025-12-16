"""
Base adapter for script-based models.

This module provides a base class for models that use external scripts
to perform predictions and retraining.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import pydantic

from backend.logging_config import get_component_logger
from .base_adapter import BaseModelAdapter


class ScriptModelConfig(pydantic.BaseModel):
    """Base configuration schema for script models."""
    pass


class ScriptModelAdapter(BaseModelAdapter):
    """Base adapter for script-based models."""

    def __init__(self, name: str, model_type: str, version: str, description: str, capabilities: List[str]):
        super().__init__(name, model_type, version, description, capabilities)
        self.logger = get_component_logger(f"backend.models.adapters.script_adapter.{name}")

    def get_config_schema(self) -> type[pydantic.BaseModel]:
        """Return the configuration schema for this model."""
        return ScriptModelConfig

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection from app state."""
        from backend.main import app_state  # Import here to avoid circular imports
        db_path = app_state.get("database_path", "data/backtest.db")
        return sqlite3.connect(db_path)

    def _predict_impl(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the script-based model."""
        raise NotImplementedError("Subclasses must implement _predict_impl method")

    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        """Retrain the model (may be a no-op for some models)."""
        self.logger.info(f"Retraining model {self.name}")
        # Default implementation - no-op
        return {"status": "completed", "message": "Retraining not required for this model"}

    def save(self, path: Path) -> None:
        """Save the model (not supported for script models)."""
        raise NotImplementedError("Saving not supported for script-based models")

    @classmethod
    def load(cls, path: Union[Path, Dict[str, Any]]) -> 'ScriptModelAdapter':
        """Load a model (not supported for script models)."""
        raise NotImplementedError("Loading not supported for script-based models")