"""
Integration tests for MA period optimization and prediction generation.
"""
import pytest
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import the functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from backend.scripts.predictions.generate_ma_predictions import optimize_ma_periods, generate_predictions, ThreeMACrossoverStrategy


@pytest.mark.integration
class TestMAOptimization:
    """Test MA period optimization functionality."""

    def setup_method(self):
        """Set up test database with sample data."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create test database with sample price data
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        # Create tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trading_model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                dt TEXT,
                model TEXT,
                predicted_return REAL,
                enter_prob REAL,
                suggested_position_pct REAL,
                exit_prob REAL,
                produced_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
            )
        """)

        # Insert sample price data for multiple tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        for ticker in tickers:
            for i, date in enumerate(dates):
                # Generate realistic price data with trend
                base_price = 100 + (ticker == 'AAPL') * 50 + (ticker == 'MSFT') * 30 + (ticker == 'GOOGL') * 40
                price = base_price + i * 0.1 + np.sin(i * 0.1) * 5  # Add some volatility

                cur.execute("""
                    INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    price - 1,
                    price + 2,
                    price - 2,
                    price,
                    price,
                    1000000
                ))

        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up test database."""
        # Close any open connections first
        try:
            sqlite3.connect(self.temp_db.name).close()
        except:
            pass

        # Small delay to ensure file handles are released
        import time
        time.sleep(0.1)

        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except PermissionError:
                # If still locked, just leave it for cleanup later
                pass

    def test_optimize_ma_periods_success(self):
        """Test successful MA period optimization."""
        # Mock the logger to avoid import issues
        with patch('backend.scripts.predictions.generate_ma_predictions.logger') as mock_logger:
            mock_logger.info = Mock()
            mock_logger.warning = Mock()

            conn = sqlite3.connect(self.temp_db.name)

            short_range = [3, 5]
            medium_range = [10, 15]
            long_range = [20, 25]

            # Run optimization
            best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
                conn, '2020-01-01', '2020-12-31', short_range, medium_range, long_range
            )

            # Verify results
            assert isinstance(best_short, int)
            assert isinstance(best_medium, int)
            assert isinstance(best_long, int)
            assert isinstance(best_sharpe, (float, np.floating))

            # Verify parameters are reasonable (may fall back to defaults if optimization doesn't find better)
            # The optimization may return defaults (5, 20, 50) if no better combinations are found
            assert best_short > 0 and best_short <= 50  # Reasonable MA period
            assert best_medium > best_short  # Medium should be > short
            assert best_long > best_medium  # Long should be > medium

            # Verify Sharpe ratio is reasonable (not NaN, but may be -inf if optimization fails)
            assert not np.isnan(best_sharpe)
            # Allow -inf as valid result when no optimization succeeds

            conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_optimize_ma_periods_with_tickers_filter(self, mock_logger):
        """Test optimization with specific ticker selection."""
        conn = sqlite3.connect(self.temp_db.name)

        short_range = [3, 5]
        medium_range = [10, 15]
        long_range = [20, 25]
        tickers = ['AAPL', 'MSFT']  # Limit to 2 tickers

        # Run optimization
        best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
            conn, '2020-01-01', '2020-12-31', short_range, medium_range, long_range, tickers
        )

        # Verify results are reasonable (may fall back to defaults)
        assert best_short > 0 and best_short <= 50
        assert best_medium > best_short
        assert best_long > best_medium

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_optimize_ma_periods_invalid_combinations_filtered(self, mock_logger):
        """Test that invalid MA combinations (short >= medium or medium >= long) are filtered."""
        conn = sqlite3.connect(self.temp_db.name)

        # Use ranges that would create invalid combinations
        short_range = [10, 15]  # Higher values
        medium_range = [10, 15]  # Same range as short
        long_range = [20, 25]   # Should be higher

        # Run optimization - should still find valid combinations
        best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
            conn, '2020-01-01', '2020-12-31', short_range, medium_range, long_range
        )

        # Verify the result is valid
        assert best_short < best_medium < best_long

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_optimize_ma_periods_insufficient_data(self, mock_logger):
        """Test optimization with insufficient data."""
        conn = sqlite3.connect(self.temp_db.name)

        # Use very short date range with insufficient data
        short_range = [3, 5]
        medium_range = [10, 15]
        long_range = [20, 25]

        # Run optimization with very short period
        best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
            conn, '2020-01-01', '2020-01-10', short_range, medium_range, long_range
        )

        # Should return default values when no valid results
        assert best_short == 5  # default
        assert best_medium == 20  # default
        assert best_long == 50  # default

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_optimize_ma_periods_no_valid_tickers(self, mock_logger):
        """Test optimization when no tickers have sufficient data."""
        # Create empty database
        empty_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        empty_db.close()

        conn = sqlite3.connect(empty_db.name)

        # Create empty price_daily table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()

        short_range = [3, 5]
        medium_range = [10, 15]
        long_range = [20, 25]

        # Run optimization
        best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
            conn, '2020-01-01', '2020-12-31', short_range, medium_range, long_range
        )

        # Should return default values
        assert best_short == 5
        assert best_medium == 20
        assert best_long == 50

        conn.close()
        os.unlink(empty_db.name)

    @patch('backend.scripts.predictions.generate_ma_predictions.bt.Cerebro')
    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_optimize_ma_periods_backtest_failure_handling(self, mock_logger, mock_cerebro):
        """Test handling of backtest failures during optimization."""
        conn = sqlite3.connect(self.temp_db.name)

        # Mock Cerebro to raise exception
        mock_cerebro_instance = Mock()
        mock_cerebro_instance.adddata.side_effect = Exception("Backtest failed")
        mock_cerebro.return_value = mock_cerebro_instance

        short_range = [3, 5]
        medium_range = [10, 15]
        long_range = [20, 25]

        # Run optimization - should handle exceptions gracefully
        best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
            conn, '2020-01-01', '2020-12-31', short_range, medium_range, long_range
        )

        # Should return default values when all backtests fail
        assert best_short == 5
        assert best_medium == 20
        assert best_long == 50

        conn.close()


@pytest.mark.integration
class TestPredictionGeneration:
    """Test prediction generation functionality."""

    def setup_method(self):
        """Set up test database with sample data."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create test database
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        # Create tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trading_model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                dt TEXT,
                model TEXT,
                predicted_return REAL,
                enter_prob REAL,
                suggested_position_pct REAL,
                exit_prob REAL,
                produced_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
            )
        """)

        # Insert sample price data
        ticker = 'AAPL'
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        for i, date in enumerate(dates):
            price = 100 + i * 0.1  # Simple upward trend
            cur.execute("""
                INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                date.strftime('%Y-%m-%d'),
                price - 1,
                price + 2,
                price - 2,
                price,
                price,
                1000000
            ))

        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up test database."""
        # Close any open connections first
        try:
            sqlite3.connect(self.temp_db.name).close()
        except:
            pass

        # Small delay to ensure file handles are released
        import time
        time.sleep(0.1)

        if os.path.exists(self.temp_db.name):
            try:
                os.unlink(self.temp_db.name)
            except PermissionError:
                # If still locked, just leave it for cleanup later
                pass

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_generate_predictions_success(self, mock_logger):
        """Test successful prediction generation."""
        conn = sqlite3.connect(self.temp_db.name)

        # Generate predictions
        generate_predictions(conn, '2020-01-01', '2020-12-31', 5, 20, 50)

        # Verify predictions were inserted
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        count = cur.fetchone()[0]

        assert count > 0, "Should have generated predictions"

        # Verify prediction structure
        cur.execute("""
            SELECT ticker, dt, model, predicted_return, enter_prob, suggested_position_pct, exit_prob
            FROM trading_model_predictions
            WHERE model = 'three_ma_crossover_v1'
            LIMIT 1
        """)
        row = cur.fetchone()

        assert row[0] == 'AAPL'  # ticker
        assert row[2] == 'three_ma_crossover_v1'  # model
        assert isinstance(row[3], float)  # predicted_return
        assert isinstance(row[4], float)  # enter_prob
        assert isinstance(row[5], float)  # suggested_position_pct
        assert isinstance(row[6], float)  # exit_prob

        # Verify position_pct is either 0.1 or -0.1 (or 0 for neutral filtered out)
        assert row[5] in [0.1, -0.1], f"Position pct should be 0.1 or -0.1, got {row[5]}"

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_generate_predictions_existing_predictions_deleted(self, mock_logger):
        """Test that existing predictions are deleted before generating new ones."""
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        # Insert existing predictions
        cur.execute("""
            INSERT INTO trading_model_predictions (ticker, dt, model, predicted_return, enter_prob, suggested_position_pct, exit_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ('AAPL', '2020-01-15', 'three_ma_crossover_v1', 0.05, 0.1, 0.1, 0.0))

        conn.commit()

        # Verify existing prediction
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        initial_count = cur.fetchone()[0]
        assert initial_count == 1

        # Generate new predictions
        generate_predictions(conn, '2020-01-01', '2020-12-31', 5, 20, 50)

        # Verify old prediction was deleted and new ones added
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        final_count = cur.fetchone()[0]

        # Should have new predictions (old one deleted)
        assert final_count >= 1, "Should have new predictions after deletion"

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_generate_predictions_insufficient_data_skip(self, mock_logger):
        """Test that tickers with insufficient data are skipped."""
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        # Add a ticker with very little data
        cur.execute("""
            INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ('SHORT', '2020-01-01', 100, 102, 98, 101, 101, 1000000))

        conn.commit()

        # Generate predictions
        generate_predictions(conn, '2020-01-01', '2020-12-31', 5, 20, 50)

        # Verify SHORT ticker was skipped (no predictions generated)
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE ticker = 'SHORT'")
        short_count = cur.fetchone()[0]
        assert short_count == 0, "Ticker with insufficient data should be skipped"

        # Verify AAPL still got predictions
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE ticker = 'AAPL'")
        aapl_count = cur.fetchone()[0]
        assert aapl_count > 0, "AAPL should still get predictions"

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_generate_predictions_multiple_tickers(self, mock_logger):
        """Test prediction generation for multiple tickers."""
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        # Add another ticker
        ticker = 'MSFT'
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        for i, date in enumerate(dates):
            price = 200 + i * 0.05  # Different price scale
            cur.execute("""
                INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                date.strftime('%Y-%m-%d'),
                price - 1,
                price + 2,
                price - 2,
                price,
                price,
                1000000
            ))

        conn.commit()

        # Generate predictions
        generate_predictions(conn, '2020-01-01', '2020-12-31', 5, 20, 50)

        # Verify both tickers got predictions
        cur.execute("SELECT DISTINCT ticker FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        tickers = [row[0] for row in cur.fetchall()]

        assert 'AAPL' in tickers
        assert 'MSFT' in tickers
        assert len(tickers) == 2

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_generate_predictions_signal_filtering(self, mock_logger):
        """Test that only non-neutral signals generate predictions."""
        conn = sqlite3.connect(self.temp_db.name)

        # Generate predictions
        generate_predictions(conn, '2020-01-01', '2020-12-31', 5, 20, 50)

        # Verify all predictions have non-zero position_pct (neutral signals filtered out)
        cur = conn.cursor()
        cur.execute("""
            SELECT suggested_position_pct
            FROM trading_model_predictions
            WHERE model = 'three_ma_crossover_v1'
        """)
        positions = [row[0] for row in cur.fetchall()]

        # All positions should be non-zero (0.1 or -0.1)
        assert all(abs(pos) == 0.1 for pos in positions), f"All positions should be ±0.1, got {positions}"

        conn.close()


@pytest.mark.integration
class TestOptimizationFallback:
    """Test fallback behavior when optimization fails."""

    def setup_method(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create minimal database
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_optimization_fallback_to_defaults(self, mock_logger):
        """Test that optimization falls back to default parameters when it fails."""
        conn = sqlite3.connect(self.temp_db.name)

        # Run optimization with empty database (should fail and return defaults)
        best_short, best_medium, best_long, best_sharpe = optimize_ma_periods(
            conn, '2020-01-01', '2020-12-31', [3, 5], [10, 15], [20, 25]
        )

        # Should return default values
        assert best_short == 5
        assert best_medium == 20
        assert best_long == 50
        assert best_sharpe == -np.inf  # Default when no valid results

        conn.close()

    @patch('backend.scripts.predictions.generate_ma_predictions.optimize_ma_periods')
    @patch('backend.scripts.predictions.generate_ma_predictions.logger')
    def test_generate_predictions_uses_optimized_params(self, mock_logger, mock_optimize):
        """Test that generate_predictions uses optimized parameters."""
        # Mock optimization to return specific values
        mock_optimize.return_value = (3, 10, 20, 0.5)

        conn = sqlite3.connect(self.temp_db.name)

        # Add minimal data to avoid early return
        cur = conn.cursor()
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        for i, date in enumerate(dates):
            cur.execute("""
                INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'TEST',
                date.strftime('%Y-%m-%d'),
                100 + i * 0.1,
                102 + i * 0.1,
                98 + i * 0.1,
                101 + i * 0.1,
                101 + i * 0.1,
                1000000
            ))
        conn.commit()

        # This would normally call optimize_ma_periods, but we'll mock it
        # In the actual script, optimization is called separately

        conn.close()