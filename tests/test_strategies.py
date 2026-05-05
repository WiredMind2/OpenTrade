"""
Unit tests for the strategy system components.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Import the strategy components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from backend.strategies.base import BaseStrategy
from backend.strategies.registry import StrategyRegistry


@pytest.mark.unit
class TestStrategyRegistry:
    """Test the StrategyRegistry class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.registry = StrategyRegistry()

    def test_registry_initialization(self):
        """Test registry initializes with empty strategies."""
        assert len(self.registry._strategies) == 0
        assert self.registry.list() == []

    def test_register_strategy(self):
        """Test registering a strategy."""
        # Create a mock strategy
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.name = "test_strategy"
        mock_strategy.description = "Test strategy"
        mock_strategy.type = "rule"
        mock_strategy.parameters_schema = {}
        mock_strategy.can_train = False

        self.registry.register(mock_strategy)

        assert "test_strategy" in self.registry._strategies
        assert self.registry._strategies["test_strategy"] == mock_strategy

    def test_register_duplicate_strategy(self):
        """Test registering a strategy with existing name overwrites."""
        # Create mock strategies
        mock_strategy1 = Mock(spec=BaseStrategy)
        mock_strategy1.name = "test_strategy"
        mock_strategy1.description = "Test strategy 1"

        mock_strategy2 = Mock(spec=BaseStrategy)
        mock_strategy2.name = "test_strategy"
        mock_strategy2.description = "Test strategy 2"

        self.registry.register(mock_strategy1)
        self.registry.register(mock_strategy2)  # Should overwrite

        assert self.registry._strategies["test_strategy"] == mock_strategy2

    def test_get_strategy(self):
        """Test getting a strategy by name."""
        # Create and register mock strategy
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.name = "test_strategy"
        self.registry.register(mock_strategy)

        retrieved = self.registry.get("test_strategy")
        assert retrieved == mock_strategy

        # Test getting non-existent strategy
        assert self.registry.get("non_existent") is None

    def test_list_strategies(self):
        """Test listing all registered strategies."""
        # Create and register mock strategies
        mock_strategy1 = Mock(spec=BaseStrategy)
        mock_strategy1.name = "strategy1"
        mock_strategy1.description = "Strategy 1"
        mock_strategy1.type = "rule"
        mock_strategy1.parameters_schema = {"param1": {"type": "number"}}
        mock_strategy1.can_train = False

        mock_strategy2 = Mock(spec=BaseStrategy)
        mock_strategy2.name = "strategy2"
        mock_strategy2.description = "Strategy 2"
        mock_strategy2.type = "ml"
        mock_strategy2.parameters_schema = {"param2": {"type": "string"}}
        mock_strategy2.can_train = True

        self.registry.register(mock_strategy1)
        self.registry.register(mock_strategy2)

        strategies = self.registry.list()

        assert len(strategies) == 2
        strategy_names = [s['name'] for s in strategies]
        assert "strategy1" in strategy_names
        assert "strategy2" in strategy_names

        # Check metadata structure
        for strategy in strategies:
            assert 'name' in strategy
            assert 'description' in strategy
            assert 'type' in strategy
            assert 'parameters_schema' in strategy
            assert 'can_train' in strategy

    def test_discover_python_strategies(self):
        """Test discovering Python strategy modules."""
        # Create a temporary directory with mock strategy files
        with tempfile.TemporaryDirectory() as temp_dir:
            strategies_dir = Path(temp_dir) / "strategies"
            strategies_dir.mkdir()

            # Create a mock strategy file
            strategy_file = strategies_dir / "mock_strategy.py"
            strategy_file.write_text("""
from backend.strategies.base import BaseStrategy

class MockStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(
            name="mock_strategy",
            description="Mock strategy for testing",
            type="rule",
            parameters_schema={},
            can_train=False
        )

    def create_backtrader_strategy(self, parameters):
        return None

    def project(self, parameters, projection_days=30, initial_capital=100000.0):
        return {}
""")

            # Discover strategies
            discovered = self.registry.discover_python_strategies(strategies_dir)

            assert len(discovered) == 1
            assert discovered[0].name == "mock_strategy"
            assert discovered[0].description == "Mock strategy for testing"

    def test_discover_with_invalid_module(self):
        """Test discovery handles invalid modules gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            strategies_dir = Path(temp_dir) / "strategies"
            strategies_dir.mkdir()

            # Create an invalid strategy file (missing required methods)
            strategy_file = strategies_dir / "invalid_strategy.py"
            strategy_file.write_text("""
from backend.strategies.base import BaseStrategy

class InvalidStrategy(BaseStrategy):
    pass  # Missing required methods
""")

            # Discovery should handle the error gracefully
            discovered = self.registry.discover_python_strategies(strategies_dir)
            # Should not crash, but may not discover invalid strategies
            assert isinstance(discovered, list)


@pytest.mark.unit
class TestBaseStrategy:
    """Test the BaseStrategy abstract base class."""

    def test_base_strategy_initialization(self):
        """Test BaseStrategy initializes correctly."""
        # Create a concrete implementation for testing
        class ConcreteStrategy(BaseStrategy):
            def create_backtrader_strategy(self, parameters):
                return Mock()

            def project(self, parameters, projection_days=30, initial_capital=100000.0):
                return {}

        strategy = ConcreteStrategy(
            name="test_strategy",
            description="Test description",
            type="rule",
            parameters_schema={"param1": {"type": "number"}},
            can_train=False
        )

        assert strategy.name == "test_strategy"
        assert strategy.description == "Test description"
        assert strategy.type == "rule"
        assert strategy.parameters_schema == {"param1": {"type": "number"}}
        assert strategy.can_train == False

    def test_train_not_implemented(self):
        """Test that train raises NotImplementedError for base class."""
        class ConcreteStrategy(BaseStrategy):
            def create_backtrader_strategy(self, parameters):
                return Mock()

            def project(self, parameters, projection_days=30, initial_capital=100000.0):
                return {}

        strategy = ConcreteStrategy(
            name="test_strategy",
            description="Test description",
            type="rule",
            parameters_schema={},
            can_train=False
        )

        with pytest.raises(NotImplementedError):
            strategy.train({})

    def test_create_backtrader_strategy_abstract(self):
        """Test that create_backtrader_strategy is abstract."""
        # This should raise TypeError when trying to instantiate BaseStrategy directly
        with pytest.raises(TypeError):
            BaseStrategy(
                name="test",
                description="test",
                type="rule",
                parameters_schema={},
                can_train=False
            )
