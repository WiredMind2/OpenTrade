"""
Trading strategies package.

This package contains trading strategy implementations and registry.
"""

from pathlib import Path
from .registry import StrategyRegistry

# Create and initialize the strategy registry
strategy_registry = StrategyRegistry()

# Get the strategies package directory
strategies_pkg_dir = Path(__file__).parent

# Discover and register all strategies
strategy_registry.discover(strategies_pkg_dir)