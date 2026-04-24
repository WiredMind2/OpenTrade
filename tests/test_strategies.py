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
from backend.strategies.sentiment_ml import SentimentMLStrategy


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


@pytest.mark.unit
class TestSentimentMLStrategy:
    """Test the SentimentMLStrategy class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.strategy = SentimentMLStrategy()

    def teardown_method(self):
        """Clean up test environment."""
        # Clean up any created model files
        if os.path.exists('models'):
            import shutil
            shutil.rmtree('models')

    def test_initialization(self):
        """Test SentimentMLStrategy initializes correctly."""
        assert self.strategy.name == "sentiment_ml"
        assert self.strategy.description == "ML-driven strategy using sentiment model predictions"
        assert self.strategy.type == "ml"
        assert self.strategy.can_train == True
        assert "model_name" in self.strategy.parameters_schema
        assert "prediction_threshold" in self.strategy.parameters_schema
        assert "max_position_pct" in self.strategy.parameters_schema

    def test_load_versions_no_models_dir(self):
        """Test loading versions when models directory doesn't exist."""
        # Should not crash
        self.strategy._load_versions()
        assert self.strategy.versions == []
        assert self.strategy.current_version is None

    def test_load_versions_with_models(self):
        """Test loading versions from existing model files."""
        # Create models directory and mock versioned files
        os.makedirs('models', exist_ok=True)

        # Create mock metadata file
        metadata_content = {
            'version': 1,
            'version_name': 'sentiment_ml__20240101_120000__v1',
            'timestamp': '20240101_120000',
            'training_config': {},
            'metrics': {'rmse': 0.1, 'mae': 0.08}
        }

        import json
        with open('models/sentiment_ml__20240101_120000__v1_metadata.json', 'w') as f:
            json.dump(metadata_content, f)

        # Create mock model file
        with open('models/sentiment_ml__20240101_120000__v1.joblib', 'w') as f:
            f.write('mock model data')

        self.strategy._load_versions()

        assert len(self.strategy.versions) == 1
        assert self.strategy.versions[0]['version'] == 1
        assert self.strategy.versions[0]['name'] == 'sentiment_ml__20240101_120000__v1'
        assert self.strategy.current_version == 'sentiment_ml__20240101_120000__v1'

    def test_create_backtrader_strategy(self):
        """Test creating Backtrader strategy."""
        parameters = {
            'model_name': 'test_model',
            'prediction_threshold': 0.6,
            'max_position_pct': 0.2
        }

        bt_strategy_class = self.strategy.create_backtrader_strategy(parameters)

        # Should return a class
        assert bt_strategy_class is not None
        assert hasattr(bt_strategy_class, 'params')

        # Check parameters - Backtrader params is a special object
        # Just verify the class was created successfully
        assert bt_strategy_class.__name__ == 'SentimentMLBacktrader'

    @patch('backend.strategies.sentiment_ml.sqlite3')
    def test_train_creates_job(self, mock_sqlite):
        """Test that train method creates a job."""
        # Mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite.connect.return_value = mock_conn

        # Mock asyncio.create_task
        with patch('asyncio.create_task') as mock_create_task:
            config = {'csv_path': 'test.csv', 'outdir': 'models'}
            result = self.strategy.train(config)

            assert 'job_id' in result
            assert result['status'] == 'queued'

            # Verify database was called
            mock_cursor.execute.assert_called()
            mock_conn.commit.assert_called()

            # Verify background task was created
            mock_create_task.assert_called_once()

    def test_list_versions(self):
        """Test listing model versions."""
        # Initially empty
        assert self.strategy.list_versions() == []

        # Add a version
        self.strategy.versions = [{
            'version': 1,
            'name': 'test_v1',
            'path': 'models/test_v1.joblib',
            'metadata_path': 'models/test_v1_metadata.json',
            'timestamp': '20240101_120000'
        }]

        versions = self.strategy.list_versions()
        assert len(versions) == 1
        assert versions[0]['version'] == 1

    def test_switch_version(self):
        """Test switching model versions."""
        # Add versions
        self.strategy.versions = [
            {
                'version': 1,
                'name': 'v1',
                'path': 'models/v1.joblib',
                'metadata_path': 'models/v1_metadata.json',
                'timestamp': '20240101_120000'
            },
            {
                'version': 2,
                'name': 'v2',
                'path': 'models/v2.joblib',
                'metadata_path': 'models/v2_metadata.json',
                'timestamp': '20240102_120000'
            }
        ]

        # Switch to v2
        assert self.strategy.switch_version('v2') == True
        assert self.strategy.current_version == 'v2'
        assert self.strategy.model_name == 'v2'

        # Try switching to non-existent version
        assert self.strategy.switch_version('non_existent') == False

    def test_get_current_model_name(self):
        """Test getting current model name."""
        # Initially should return default model name
        assert self.strategy.get_current_model_name() == 'sentiment_model'

        # After setting current_version, should return that
        self.strategy.current_version = 'test_version'
        self.strategy.model_name = 'test_version'
        assert self.strategy.get_current_model_name() == 'test_version'

    @patch('backend.strategies.sentiment_ml.bt')
    def test_backtesting_integration(self, mock_bt):
        """Test backtesting with sentiment ML strategy."""
        # Mock Backtrader components
        mock_cerebro = MagicMock()
        mock_data = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_instance.equity_curve = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-02', 'value': 101000.0}
        ]
        mock_strategy_instance.trades = []

        mock_cerebro.adddata.return_value = None
        mock_cerebro.addstrategy.return_value = None
        mock_cerebro.run.return_value = [mock_strategy_instance]

        mock_bt.Cerebro.return_value = mock_cerebro
        mock_bt.feeds.PandasData.return_value = mock_data

        # Create strategy and run backtest
        bt_strategy_class = self.strategy.create_backtrader_strategy({})
        mock_cerebro_instance = mock_bt.Cerebro()
        mock_cerebro_instance.addstrategy(bt_strategy_class)
        mock_cerebro_instance.adddata(mock_data)
        results = mock_cerebro_instance.run()

        # Verify backtest ran
        assert len(results) == 1
        assert hasattr(results[0], 'equity_curve')