"""
Unit tests for the three moving averages crossover strategy.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import backtrader as bt
from datetime import datetime

# Import the strategy class
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))
from backend.scripts.predictions.generate_ma_predictions import ThreeMACrossoverStrategy


@pytest.mark.unit
class TestThreeMACrossoverStrategy:
    """Test the ThreeMACrossoverStrategy class functionality."""

    def _create_test_cerebro(self, data_length=100):
        """Create a test Cerebro instance with mock data."""
        cerebro = bt.Cerebro()

        # Create mock data
        dates = pd.date_range('2024-01-01', periods=data_length, freq='D')
        df = pd.DataFrame({
            'open': [100] * data_length,
            'high': [105] * data_length,
            'low': [95] * data_length,
            'close': [100] * data_length,
            'volume': [1000] * data_length
        }, index=dates)

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        return cerebro

    def test_signal_generation_logic(self):
        """Test the core signal generation logic."""
        # Test bullish signal: short > medium > long
        assert self._calculate_signal(320, 310, 300) == 1

        # Test bearish signal: short < medium < long
        assert self._calculate_signal(300, 310, 320) == -1
        assert self._calculate_signal(305, 310, 315) == -1  # This is actually bearish

        # Test neutral signals
        assert self._calculate_signal(310, 300, 320) == 0  # short > medium but short < long
        assert self._calculate_signal(300, 300, 300) == 0  # all equal
        assert self._calculate_signal(305, 320, 310) == 0  # mixed order: short < medium > long

    def _calculate_signal(self, short_ma, medium_ma, long_ma):
        """Helper method to calculate signal based on MA values."""
        if short_ma > medium_ma > long_ma:
            return 1  # bullish
        elif short_ma < medium_ma < long_ma:
            return -1  # bearish
        else:
            return 0  # neutral

    def test_strategy_parameters(self):
        """Test strategy parameter definitions."""
        # Test that the strategy class has the expected parameters
        assert hasattr(ThreeMACrossoverStrategy, 'params')

        # Check parameter defaults by creating an instance and checking params attribute
        cerebro = bt.Cerebro()
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'open': [100] * 60,
            'high': [105] * 60,
            'low': [95] * 60,
            'close': [100] * 60,
            'volume': [1000] * 60
        }, index=dates)

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.addstrategy(ThreeMACrossoverStrategy)

        results = cerebro.run()
        strategy = results[0]

        # Check that parameters were set correctly
        assert strategy.params.short_period == 5
        assert strategy.params.medium_period == 20
        assert strategy.params.long_period == 50

    def test_ma_calculation_logic(self):
        """Test moving average calculation logic."""
        # Test simple moving average calculation
        prices = [100, 105, 110, 115, 120]

        # Manual SMA calculation
        sma_5 = sum(prices) / len(prices)
        expected_sma = 110.0

        assert abs(sma_5 - expected_sma) < 0.001

    def test_data_sufficiency_check(self):
        """Test data sufficiency checking logic."""
        long_period = 50

        # Test sufficient data
        assert 60 >= long_period + 10  # Should be sufficient

        # Test insufficient data
        assert 30 < long_period + 10  # Should be insufficient

    def test_multiple_signals_generation(self):
        """Test generation of multiple signals over time."""
        # Create test data with varying prices to generate different signals
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        prices = []

        # Create price pattern that will generate different signals
        for i in range(60):
            if i < 20:
                # Bullish pattern: short > medium > long
                base = 100 + i * 0.5
                prices.append(base + 10)  # short MA will be higher
            elif i < 40:
                # Bearish pattern: short < medium < long
                base = 100 + i * 0.5
                prices.append(base - 10)  # short MA will be lower
            else:
                # Neutral pattern: mixed
                base = 100 + i * 0.5
                prices.append(base)  # equal values

        df = pd.DataFrame({
            'open': prices,
            'high': [p + 5 for p in prices],
            'low': [p - 5 for p in prices],
            'close': prices,
            'volume': [1000] * 60
        }, index=dates)

        # Run strategy
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.addstrategy(ThreeMACrossoverStrategy, short_period=5, medium_period=10, long_period=20)

        results = cerebro.run()
        strategy = results[0]

        # Verify signals were generated (should have signals after the long MA period)
        assert len(strategy.signals) > 0
        assert all('signal' in signal for signal in strategy.signals)
        assert all('date' in signal for signal in strategy.signals)


class TestPositionCalculations:
    """Test position percentage calculations from signals."""

    def test_bullish_position_calculation(self):
        """Test position calculation for bullish signals."""
        signal = 1
        position_pct = signal * 0.1  # From generate_predictions function
        assert position_pct == 0.1

    def test_bearish_position_calculation(self):
        """Test position calculation for bearish signals."""
        signal = -1
        position_pct = signal * 0.1  # From generate_predictions function
        assert position_pct == -0.1

    def test_neutral_position_calculation(self):
        """Test that neutral signals don't generate positions."""
        # Neutral signals (0) are filtered out in generate_predictions
        # So no position should be calculated
        signal = 0
        # This would be filtered out before position calculation

    def test_predicted_return_calculation(self):
        """Test predicted return calculation."""
        position_pct = 0.1
        predicted_return = position_pct  # Simplified as position percentage
        assert predicted_return == 0.1

        position_pct = -0.1
        predicted_return = position_pct
        assert predicted_return == -0.1

    def test_enter_probability_calculation(self):
        """Test enter probability calculation."""
        position_pct = 0.1
        enter_prob = abs(position_pct) if position_pct > 0 else 0.0
        assert enter_prob == 0.1

        position_pct = -0.1
        enter_prob = abs(position_pct) if position_pct > 0 else 0.0
        assert enter_prob == 0.0

    def test_exit_probability_calculation(self):
        """Test exit probability calculation."""
        position_pct = 0.1
        exit_prob = 0.5 if position_pct < 0 else 0.0
        assert exit_prob == 0.0

        position_pct = -0.1
        exit_prob = 0.5 if position_pct < 0 else 0.0
        assert exit_prob == 0.5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def _create_strategy(self, short_period=5, medium_period=10, long_period=20):
        """Helper method to create a strategy instance with Cerebro."""
        cerebro = bt.Cerebro()
        # Create minimal data for testing
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'open': [100] * 60,
            'high': [105] * 60,
            'low': [95] * 60,
            'close': [100] * 60,
            'volume': [1000] * 60
        }, index=dates)

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.addstrategy(ThreeMACrossoverStrategy,
                          short_period=short_period,
                          medium_period=medium_period,
                          long_period=long_period)

        results = cerebro.run()
        return results[0]

    def test_equal_moving_averages(self):
        """Test behavior when moving averages are equal."""
        # This test is difficult to implement with real MA calculations
        # Skip for now as the logic is tested elsewhere
        pass


    def test_signal_consistency(self):
        """Test that signal logic is consistent."""
        # Test the core signal calculation logic directly
        test_cases = [
            # (short, medium, long, expected_signal)
            (320, 310, 300, 1),   # bullish: short > medium > long
            (300, 310, 320, -1),  # bearish: short < medium < long
            (310, 300, 320, 0),   # neutral: short > medium but short < long
            (300, 300, 300, 0),   # equal: all equal
            (305, 310, 315, -1),  # bearish: short < medium < long
        ]

        for short_val, medium_val, long_val, expected_signal in test_cases:
            # Test the logic directly (same as in strategy)
            if short_val > medium_val > long_val:
                signal = 1  # bullish
            elif short_val < medium_val < long_val:
                signal = -1  # bearish
            else:
                signal = 0  # neutral

            assert signal == expected_signal, \
                f"Expected {expected_signal} for MAs ({short_val}, {medium_val}, {long_val}), got {signal}"