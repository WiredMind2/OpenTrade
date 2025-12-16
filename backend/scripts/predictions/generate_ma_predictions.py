"""
Generate trading model predictions using three moving averages crossover strategy.

This script implements a three moving averages crossover strategy that:
- Uses short, medium, and long period SMAs
- Generates bullish signals when short > medium > long
- Generates bearish signals when short < medium < long
- Optimizes MA periods using Sharpe ratio
- Stores predictions in trading_model_predictions table with model name 'three_ma_crossover_v1'

Usage:
  python scripts/generate_ma_predictions.py --db data/backtest.db --start 2020-01-01 --end 2025-01-01
  python scripts/generate_ma_predictions.py --db data/backtest.db --short-ma 5 --medium-ma 20 --long-ma 50 --skip-optimization

Outputs:
  Rows in trading_model_predictions table with suggested_position_pct values.
"""
import sys
import os
# Add project root to path so we can import backend modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import sqlite3
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import logging
import sys

# Set up standalone logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create the logger instance
logger = logging.getLogger('generate_ma_predictions')
logger.setLevel(logging.INFO)


class ThreeMACrossoverStrategy(bt.Strategy):
    """Three Moving Averages Crossover Strategy."""

    params = (
        ('short_period', 5),
        ('medium_period', 20),
        ('long_period', 50),
    )

    def __init__(self):
        # Calculate moving averages
        self.short_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period
        )
        self.medium_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.medium_period
        )
        self.long_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period
        )

        # Track signals
        self.signals = []

    def next(self):
        """Generate signals based on MA crossover."""
        # Check for valid data
        if len(self.data) < self.params.long_period:
            return

        short_val = self.short_ma[0]
        medium_val = self.medium_ma[0]
        long_val = self.long_ma[0]

        # Generate signal
        signal = 0  # neutral
        if short_val > medium_val > long_val:
            signal = 1  # bullish
        elif short_val < medium_val < long_val:
            signal = -1  # bearish

        # Store signal with date
        current_date = self.data.datetime.date(0).isoformat()
        self.signals.append({
            'date': current_date,
            'signal': signal,
            'short_ma': short_val,
            'medium_ma': medium_val,
            'long_ma': long_val
        })


def optimize_ma_periods(conn, start_date, end_date, short_range, medium_range, long_range, tickers=None):
    """
    Optimize MA periods using Sharpe ratio.

    Args:
        conn: Database connection
        start_date: Start date for optimization
        end_date: End date for optimization
        short_range: Range of short MA periods to test
        medium_range: Range of medium MA periods to test
        long_range: Range of long MA periods to test
        tickers: List of tickers to use for optimization (default: all available)

    Returns:
        Tuple of (best_short, best_medium, best_long, best_sharpe)
    """
    logger.info("Starting MA period optimization...")

    if tickers is None:
        # Get all tickers with price data
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ticker FROM price_daily
            WHERE date >= ? AND date <= ?
            ORDER BY ticker
        """, (start_date, end_date))
        tickers = [row[0] for row in cur.fetchall()][:5]  # Limit to 5 tickers for speed

    logger.info(f"Optimizing using {len(tickers)} tickers: {tickers}")

    best_sharpe = -np.inf
    best_params = (5, 20, 50)  # default

    # Test all combinations
    param_combinations = list(product(short_range, medium_range, long_range))
    logger.info(f"Testing {len(param_combinations)} parameter combinations...")

    for short_p, medium_p, long_p in param_combinations:
        if short_p >= medium_p or medium_p >= long_p:
            continue  # Skip invalid combinations

        total_sharpe = 0
        valid_tickers = 0

        for ticker in tickers:
            try:
                # Get price data
                cur = conn.cursor()
                cur.execute("""
                    SELECT date, open, high, low, close, volume
                    FROM price_daily
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date ASC
                """, (ticker, start_date, end_date))

                price_data = cur.fetchall()
                if len(price_data) < long_p + 10:  # Need enough data
                    continue

                # Create DataFrame
                df = pd.DataFrame(price_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

                # Run backtest
                cerebro = bt.Cerebro()
                data = bt.feeds.PandasData(dataname=df)
                cerebro.adddata(data)
                cerebro.addstrategy(ThreeMACrossoverStrategy,
                                  short_period=short_p,
                                  medium_period=medium_p,
                                  long_period=long_p)

                # Add analyzers
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

                # Run backtest
                results = cerebro.run()
                strat = results[0]

                # Get Sharpe ratio
                sharpe_analysis = strat.analyzers.sharpe.get_analysis()
                sharpe = sharpe_analysis.get('sharperatio', None)
                if sharpe is not None and not np.isnan(float(sharpe)) and float(sharpe) != 0:
                    total_sharpe += float(sharpe)
                    valid_tickers += 1

            except Exception as e:
                logger.warning(f"Error testing {ticker} with params ({short_p}, {medium_p}, {long_p}): {e}")
                continue

        if valid_tickers > 0:
            avg_sharpe = total_sharpe / valid_tickers
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_params = (short_p, medium_p, long_p)
                logger.info(f"New best params: {best_params} with Sharpe: {best_sharpe:.3f}")

    logger.info(f"Optimization complete. Best params: {best_params} with Sharpe: {best_sharpe:.3f}")
    return best_params[0], best_params[1], best_params[2], best_sharpe


def generate_predictions(conn, start_date, end_date, short_period, medium_period, long_period):
    """
    Generate trading predictions using optimized MA strategy.

    Args:
        conn: Database connection
        start_date: Start date for predictions
        end_date: End date for predictions
        short_period: Short MA period
        medium_period: Medium MA period
        long_period: Long MA period
    """
    logger.info(f"Generating predictions with MA periods: {short_period}, {medium_period}, {long_period}")

    cur = conn.cursor()

    # Delete existing predictions for this model in the date range
    cur.execute('''
        DELETE FROM trading_model_predictions
        WHERE dt >= ? AND dt <= ? AND model = ?
    ''', (start_date, end_date, 'three_ma_crossover_v1'))
    logger.info(f'Deleted existing predictions for three_ma_crossover_v1 in range {start_date} to {end_date}')

    # Get all tickers with price data
    cur.execute("""
        SELECT DISTINCT ticker FROM price_daily
        WHERE date >= ? AND date <= ?
        ORDER BY ticker
    """, (start_date, end_date))

    tickers = [row[0] for row in cur.fetchall()]
    logger.info(f'Found {len(tickers)} tickers with price data')

    inserted_count = 0

    for ticker in tickers:
        try:
            # Get price data
            cur.execute("""
                SELECT date, open, high, low, close, volume
                FROM price_daily
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            """, (ticker, start_date, end_date))

            price_data = cur.fetchall()
            if len(price_data) < long_period + 10:
                continue

            # Create DataFrame
            df = pd.DataFrame(price_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Run strategy to get signals
            cerebro = bt.Cerebro()
            data = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data)
            cerebro.addstrategy(ThreeMACrossoverStrategy,
                              short_period=short_period,
                              medium_period=medium_period,
                              long_period=long_period)

            results = cerebro.run()
            strat = results[0]

            # Process signals
            for signal_data in strat.signals:
                signal = signal_data['signal']

                # Only generate predictions for clear signals (not neutral)
                if signal == 0:
                    continue

                # Convert signal to position percentage (-0.1 to 0.1 range)
                position_pct = signal * 0.1

                # Calculate predicted return (simplified as position percentage)
                predicted_return = position_pct

                # Insert prediction
                cur.execute('''
                    INSERT INTO trading_model_predictions
                    (ticker, dt, model, predicted_return, enter_prob, suggested_position_pct, exit_prob)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    signal_data['date'],
                    'three_ma_crossover_v1',
                    predicted_return,
                    abs(position_pct) if position_pct > 0 else 0.0,  # enter_prob
                    position_pct,
                    0.5 if position_pct < 0 else 0.0  # exit_prob
                ))

                if cur.rowcount > 0:
                    inserted_count += 1

        except Exception as e:
            logger.error(f"Error generating predictions for {ticker}: {e}")
            continue

    conn.commit()
    logger.info(f'Inserted {inserted_count} trading predictions')


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Generate MA crossover trading predictions')
    # Find project root by going up from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    default_db_path = os.path.join(project_root, 'data', 'backtest.db')
    parser.add_argument('--db', default=default_db_path)
    parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--short-ma', type=int, nargs='+', default=[3, 5, 7], help='Short MA periods to test')
    parser.add_argument('--medium-ma', type=int, nargs='+', default=[15, 20, 25], help='Medium MA periods to test')
    parser.add_argument('--long-ma', type=int, nargs='+', default=[40, 50, 60], help='Long MA periods to test')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip optimization and use fixed periods')
    parser.add_argument('--fixed-short', type=int, default=5, help='Fixed short MA period (when skipping optimization)')
    parser.add_argument('--fixed-medium', type=int, default=20, help='Fixed medium MA period (when skipping optimization)')
    parser.add_argument('--fixed-long', type=int, default=50, help='Fixed long MA period (when skipping optimization)')

    args = parser.parse_args()

    # Validate date range
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    if start_date >= end_date:
        logger.error(f"Start date ({args.start}) must be before end date ({args.end})")
        sys.exit(1)

    try:
        conn = sqlite3.connect(args.db)

        if args.skip_optimization:
            # Use fixed periods
            short_period = args.fixed_short
            medium_period = args.fixed_medium
            long_period = args.fixed_long
            logger.info(f"Using fixed MA periods: {short_period}, {medium_period}, {long_period}")
        else:
            # Optimize periods
            short_period, medium_period, long_period, best_sharpe = optimize_ma_periods(
                conn, args.start, args.end,
                args.short_ma, args.medium_ma, args.long_ma
            )

        # Generate predictions
        generate_predictions(conn, args.start, args.end, short_period, medium_period, long_period)

        conn.close()
        logger.info('Done')

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == '__main__':
    main()