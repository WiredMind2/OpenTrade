"""
Base model interface for the trading system.

This module defines the abstract base class that all models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Union
from pathlib import Path
import pydantic


class BaseModel(ABC):
    """Abstract base class for all models in the trading system."""

    def __init__(self, name: str, type: str, version: str, description: str, capabilities: List[str]):
        self.name = name
        self.type = type
        self.version = version
        self.description = description
        self.capabilities = capabilities

    @abstractmethod
    def get_config_schema(self) -> Type[pydantic.BaseModel]:
        """Return the Pydantic model for configuration validation."""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction with the given inputs and configuration."""
        pass

    @abstractmethod
    def retrain(self, training_payload: Dict[str, Any], config: Dict[str, Any], background: bool = False) -> Dict[str, Any]:
        """Retrain the model with new data."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model to the specified path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[Path, Dict[str, Any]]) -> 'BaseModel':
        """Load a model from the specified path or data dict."""
        pass