"""
Strategy registry for managing available trading strategies.

This module provides a centralized registry for discovering, loading,
and managing trading strategies.
"""

import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib.util
import inspect

from backend.logging_config import get_component_logger
from .base import BaseStrategy


class StrategyRegistry:
    """Thread-safe registry for managing trading strategies."""

    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._lock = threading.RLock()
        self.logger = get_component_logger("backend.strategies.registry")

    def discover_python_strategies(self, strategies_pkg_dir: Path) -> List[BaseStrategy]:
        """Discover and load Python strategy modules."""
        strategies = []

        if not strategies_pkg_dir.exists():
            self.logger.warning(f"Strategies package directory not found: {strategies_pkg_dir}")
            return strategies

        # Add the strategies directory to Python path temporarily
        import sys
        if str(strategies_pkg_dir) not in sys.path:
            sys.path.insert(0, str(strategies_pkg_dir))

        try:
            for py_file in strategies_pkg_dir.glob("*.py"):
                if py_file.name in ('base.py', 'registry.py', '__init__.py'):
                    continue

                try:
                    # Import the module
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for BaseStrategy subclasses in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, BaseStrategy) and
                                obj != BaseStrategy and
                                not inspect.isabstract(obj)):
                                # Instantiate the strategy class
                                strategy = obj()
                                strategies.append(strategy)
                                self.logger.info(f"Discovered Python strategy: {strategy.name}")

                except Exception as e:
                    self.logger.error(f"Failed to load strategy module {py_file}: {e}")

        finally:
            # Clean up sys.path
            if str(strategies_pkg_dir) in sys.path:
                sys.path.remove(str(strategies_pkg_dir))

        return strategies

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy in the registry."""
        with self._lock:
            if strategy.name in self._strategies:
                self.logger.warning(f"Strategy {strategy.name} already registered, overwriting")
            self._strategies[strategy.name] = strategy
            self.logger.info(f"Registered strategy: {strategy.name}")

    def get(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        with self._lock:
            return self._strategies.get(name)

    def list(self) -> List[Dict[str, Any]]:
        """List all registered strategies as metadata dicts."""
        with self._lock:
            return [
                {
                    'name': strategy.name,
                    'description': strategy.description,
                    'type': strategy.type,
                    'parameters_schema': strategy.parameters_schema,
                    'can_train': strategy.can_train
                }
                for strategy in self._strategies.values()
            ]

    def discover(self, strategies_pkg_dir: Path) -> None:
        """Discover and register all strategies from the strategies package directory."""
        self.logger.info(f"Starting strategy discovery in {strategies_pkg_dir}")
        python_strategies = self.discover_python_strategies(strategies_pkg_dir)
        self.logger.info(f"Discovered {len(python_strategies)} Python strategy classes")
        for strategy in python_strategies:
            self.register(strategy)

        self.logger.info(f"Discovered and registered {len(python_strategies)} Python strategies")
        self.logger.info(f"Registered strategies: {[s.name for s in self._strategies.values()]}")