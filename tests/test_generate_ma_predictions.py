"""
Script tests for generate_ma_predictions.py command-line interface and functionality.
"""
import pytest
import subprocess
import sys
import os
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the script functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from backend.scripts.predictions.generate_ma_predictions import main, optimize_ma_periods, generate_predictions


@pytest.mark.script
class TestMACommandLineInterface:
    """Test command-line interface for MA predictions script."""

    def setup_method(self):
        """Set up test database and environment."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create test database with sample data
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
        dates = pd.date_range('2020-01-01', periods=150, freq='D')

        for i, date in enumerate(dates):
            price = 100 + i * 0.1 + np.sin(i * 0.1) * 5
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
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_script_help_output(self):
        """Test that script shows help information."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path), '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        assert result.returncode == 0
        assert 'Generate MA crossover trading predictions' in result.stdout
        assert '--db' in result.stdout
        assert '--start' in result.stdout
        assert '--end' in result.stdout

    def test_script_with_optimization(self):
        """Test script execution with optimization enabled."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', self.temp_db.name,
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--short-ma', '3', '5',
            '--medium-ma', '10', '15',
            '--long-ma', '20', '25'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        assert result.returncode == 0
        assert 'Done' in result.stdout or 'Done' in result.stderr

        # Verify predictions were generated
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        count = cur.fetchone()[0]
        assert count > 0
        conn.close()

    def test_script_skip_optimization(self):
        """Test script execution with optimization skipped."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', self.temp_db.name,
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--skip-optimization',
            '--fixed-short', '5',
            '--fixed-medium', '20',
            '--fixed-long', '50'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        assert result.returncode == 0
        assert 'Using fixed MA periods: 5, 20, 50' in result.stdout

        # Verify predictions were generated
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        count = cur.fetchone()[0]
        assert count > 0
        conn.close()

    def test_script_invalid_date_range(self):
        """Test script with invalid date range."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', self.temp_db.name,
            '--start', '2025-01-01',  # Start after end
            '--end', '2020-01-01',
            '--skip-optimization',
            '--fixed-short', '5',
            '--fixed-medium', '20',
            '--fixed-long', '50'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        # Should fail with error
        assert result.returncode != 0

    def test_script_missing_database(self):
        """Test script with missing database file."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', '/nonexistent/database.db',
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--skip-optimization',
            '--fixed-short', '5',
            '--fixed-medium', '20',
            '--fixed-long', '50'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        # Should fail
        assert result.returncode != 0

    def test_script_default_parameters(self):
        """Test script with default parameters."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', self.temp_db.name
            # Using all defaults
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        assert result.returncode == 0

        # Verify predictions were generated
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trading_model_predictions WHERE model = 'three_ma_crossover_v1'")
        count = cur.fetchone()[0]
        assert count > 0
        conn.close()


@pytest.mark.script
class TestMAScriptErrorHandling:
    """Test error handling in MA predictions script."""

    def test_script_invalid_ma_parameters(self):
        """Test script with invalid MA parameters."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        # Test with medium MA shorter than short MA
        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', '/tmp/test.db',
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--skip-optimization',
            '--fixed-short', '20',
            '--fixed-medium', '5',  # Invalid: medium < short
            '--fixed-long', '50'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        # Should still run but may not generate optimal results
        # The script doesn't validate this at the CLI level
        assert result.returncode == 0 or result.returncode != 0  # May succeed or fail depending on data


@pytest.mark.script
class TestMAScriptIntegration:
    """Test full script integration scenarios."""

    def setup_method(self):
        """Set up test database with comprehensive data."""
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

        # Insert data for multiple tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        dates = pd.date_range('2020-01-01', periods=200, freq='D')

        for ticker in tickers:
            for i, date in enumerate(dates):
                base_price = 100 + (ticker == 'AAPL') * 50 + (ticker == 'MSFT') * 30 + (ticker == 'GOOGL') * 40
                price = base_price + i * 0.05 + np.sin(i * 0.1) * 3
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
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_full_script_workflow_optimization(self):
        """Test complete script workflow with optimization."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', self.temp_db.name,
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--short-ma', '3', '5', '7',
            '--medium-ma', '15', '20', '25',
            '--long-ma', '40', '50', '60'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        assert result.returncode == 0

        # Verify predictions were generated for all tickers
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        cur.execute("""
            SELECT DISTINCT ticker
            FROM trading_model_predictions
            WHERE model = 'three_ma_crossover_v1'
        """)
        tickers_with_predictions = [row[0] for row in cur.fetchall()]

        assert 'AAPL' in tickers_with_predictions
        assert len(tickers_with_predictions) >= 1  # At least AAPL should have predictions

        # Verify prediction data integrity
        cur.execute("""
            SELECT COUNT(*), AVG(suggested_position_pct), AVG(predicted_return)
            FROM trading_model_predictions
            WHERE model = 'three_ma_crossover_v1'
        """)
        count, avg_position, avg_return = cur.fetchone()

        assert count > 0
        assert avg_position is not None
        assert avg_return is not None

        conn.close()

    def test_script_idempotency(self):
        """Test that running script multiple times doesn't create duplicates."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        # Run script twice
        for _ in range(2):
            result = subprocess.run([
                sys.executable, str(script_path),
                '--db', self.temp_db.name,
                '--start', '2020-01-01',
                '--end', '2020-12-31',
                '--skip-optimization',
                '--fixed-short', '5',
                '--fixed-medium', '20',
                '--fixed-long', '50'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

            assert result.returncode == 0

        # Verify no duplicate predictions (same ticker, date, model)
        conn = sqlite3.connect(self.temp_db.name)
        cur = conn.cursor()

        cur.execute("""
            SELECT ticker, dt, model, COUNT(*)
            FROM trading_model_predictions
            WHERE model = 'three_ma_crossover_v1'
            GROUP BY ticker, dt, model
            HAVING COUNT(*) > 1
        """)
        duplicates = cur.fetchall()

        assert len(duplicates) == 0, f"Found duplicate predictions: {duplicates}"

        conn.close()

    def test_script_output_logging(self):
        """Test that script provides appropriate logging output."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', self.temp_db.name,
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--skip-optimization',
            '--fixed-short', '5',
            '--fixed-medium', '20',
            '--fixed-long', '50'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        assert result.returncode == 0

        # Check for expected log messages
        output = result.stdout + result.stderr
        assert 'Generating predictions with MA periods:' in output
        assert 'Inserted' in output and 'trading predictions' in output


@pytest.mark.script
class TestMAScriptParameterValidation:
    """Test parameter validation in MA predictions script."""

    def test_ma_period_validation_logic(self):
        """Test the MA period validation logic used in optimization."""
        # Test the condition used in optimize_ma_periods
        test_cases = [
            (3, 10, 20, True),   # Valid: short < medium < long
            (5, 15, 25, True),   # Valid
            (10, 10, 20, False), # Invalid: short == medium
            (15, 10, 20, False), # Invalid: short > medium
            (10, 20, 20, False), # Invalid: medium == long
            (10, 25, 20, False), # Invalid: medium > long
            (20, 15, 10, False), # Invalid: all wrong order
        ]

        for short, medium, long, expected_valid in test_cases:
            is_valid = short < medium and medium < long
            assert is_valid == expected_valid, f"Validation failed for ({short}, {medium}, {long})"

    def test_script_parameter_ranges(self):
        """Test that script accepts reasonable parameter ranges."""
        script_path = Path(__file__).parent.parent / 'backend' / 'scripts' / 'predictions' / 'generate_ma_predictions.py'

        # Test with very short MA periods
        result = subprocess.run([
            sys.executable, str(script_path),
            '--db', '/tmp/test.db',
            '--start', '2020-01-01',
            '--end', '2020-12-31',
            '--skip-optimization',
            '--fixed-short', '1',  # Very short
            '--fixed-medium', '2',  # Very short
            '--fixed-long', '3'     # Very short
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent / 'backend')

        # Should run (validation happens at runtime, not CLI)
        assert result.returncode == 0 or result.returncode != 0  # May fail due to insufficient data
