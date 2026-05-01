import sys
from pathlib import Path

from backend.strategies.registry import StrategyRegistry


def test_discovered_backtrader_strategy_module_is_registered():
    registry = StrategyRegistry()
    strategies_dir = Path(__file__).resolve().parents[1] / "strategies"
    discovered = registry.discover_python_strategies(strategies_dir)
    moving_average = next(s for s in discovered if s.name == "moving_average")

    strategy_class = moving_average.create_backtrader_strategy({})
    assert strategy_class.__module__ in sys.modules
