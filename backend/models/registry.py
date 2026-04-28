"""
Model registry for managing available models.

This module provides a centralized registry for discovering, loading,
and managing machine learning models.
"""

import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib.util
import inspect
import logging

# Import with error handling for missing dependencies
try:
    from backend.logging_config import get_component_logger
except ImportError as e:
    # Fallback to standard logging if component logger unavailable
    def get_component_logger(name: str):
        return logging.getLogger(name)

try:
    from .base import BaseModel
except ImportError as e:
    raise ImportError(f"Failed to import BaseModel from .base: {e}. Ensure the base model module is available.")

try:
    from .adapters.joblib_adapter import JoblibModelAdapter
except ImportError as e:
    # Joblib adapter is optional, log warning and set to None
    JoblibModelAdapter = None
    logging.warning(f"Failed to import JoblibModelAdapter: {e}. Joblib model discovery will be disabled.")


class ModelRegistry:
    """Thread-safe registry for managing models."""

    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._lock = threading.RLock()
        self.logger = get_component_logger("backend.models.registry")

    def discover_joblib_models(self, models_dir: Path) -> List[BaseModel]:
        """Discover and load joblib model files."""
        models = []

        if JoblibModelAdapter is None:
            self.logger.warning("JoblibModelAdapter not available, skipping joblib model discovery")
            return models

        if not models_dir.exists():
            self.logger.warning(f"Models directory not found: {models_dir}")
            return models

        for model_file in models_dir.glob("*.joblib"):
            try:
                model = JoblibModelAdapter.load(model_file)
                models.append(model)
                self.logger.info(f"Discovered joblib model: {model.name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {model_file}: {e}")

        return models

    def discover_python_models(self, models_pkg_dir: Path) -> List[BaseModel]:
        """Discover and load Python model modules."""
        models = []

        if not models_pkg_dir.exists():
            self.logger.warning(f"Models package directory not found: {models_pkg_dir}")
            return models

        # Add the models directory to Python path temporarily
        import sys
        if str(models_pkg_dir) not in sys.path:
            sys.path.insert(0, str(models_pkg_dir))

        try:
            for py_file in models_pkg_dir.glob("*.py"):
                if py_file.name.startswith('_'):
                    continue
                # Skip non-model modules that break discovery when imported out-of-package
                if py_file.name in {"registry.py", "base.py"}:
                    continue

                try:
                    # Import the module
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for BaseModel subclasses in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, BaseModel) and
                                obj != BaseModel and
                                # Ensure the class is defined in this module (not just imported)
                                getattr(obj, "__module__", None) == getattr(module, "__name__", None) and
                                not inspect.isabstract(obj)):
                                try:
                                    model = obj()
                                    models.append(model)
                                    self.logger.info(f"Discovered Python model: {model.name}")
                                except TypeError as e:
                                    # Class likely requires constructor args; treat as non-discoverable.
                                    self.logger.debug(f"Skipping model class {obj} (init args required): {e}")

                except Exception as e:
                    self.logger.error(f"Failed to load model module {py_file}: {e}")

        finally:
            # Clean up sys.path
            if str(models_pkg_dir) in sys.path:
                sys.path.remove(str(models_pkg_dir))

        return models

    def register(self, model: BaseModel) -> None:
        """Register a model in the registry."""
        with self._lock:
            if model.name in self._models:
                self.logger.warning(f"Model {model.name} already registered, overwriting")
            self._models[model.name] = model
            self.logger.info(f"Registered model: {model.name}")

    def get(self, name: str) -> Optional[BaseModel]:
        """Get a model by name."""
        with self._lock:
            return self._models.get(name)

    def list(self) -> List[BaseModel]:
        """List all registered models."""
        with self._lock:
            return list(self._models.values())

    def discover(self, models_dir: Path, models_pkg_dir: Optional[Path] = None) -> None:
        """Discover and register all models from directories."""
        # Discover joblib models
        joblib_models = self.discover_joblib_models(models_dir)
        for model in joblib_models:
            self.register(model)

        # Discover Python models if directory provided
        if models_pkg_dir:
            python_models = self.discover_python_models(models_pkg_dir)
            for model in python_models:
                self.register(model)

        self.logger.info(f"Discovered and registered {len(joblib_models)} joblib models and "
                        f"{len(python_models) if models_pkg_dir else 0} Python models")