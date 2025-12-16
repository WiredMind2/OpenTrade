"""
Generate trading model predictions from sentiment predictions.

This script converts sentiment scores from news articles into concrete trading decisions
(position sizes) for each ticker on each trading day. It aggregates all sentiment for a
ticker on a given day and converts to a suggested position percentage.

Usage:
   python scripts/generate_trading_predictions.py --db data/backtest.db --start 2020-01-01 --end 2025-01-01

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
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import sys

# Set up standalone logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create the logger instance
logger = logging.getLogger('generate_trading_predictions')
logger.setLevel(logging.INFO)


def get_next_trading_day(conn, date_str):
    """Get the next trading day after the given date."""
    cur = conn.cursor()
    cur.execute('SELECT date FROM price_daily WHERE date > ? ORDER BY date LIMIT 1', (date_str,))
    r = cur.fetchone()
    return r[0] if r else None


def aggregate_daily_sentiment(conn, date_str):
    """Aggregate sentiment predictions for all articles published on this date."""
    cur = conn.cursor()
    # Get all sentiment predictions for articles published on this date (from model, not ground truth)
    cur.execute('''
        SELECT sp.ticker, sp.horizon, sp.predicted_return, sp.article_id
        FROM sentiment_predictions sp
        JOIN articles a ON a.id = sp.article_id
        WHERE a.canonical_timestamp >= ? AND a.canonical_timestamp < ?
        AND sp.model LIKE 'lightgbm_%'
        ORDER BY sp.ticker, sp.horizon
    ''', (date_str, (datetime.fromisoformat(date_str) + timedelta(days=1)).isoformat()))
    
    rows = cur.fetchall()
    # Group by ticker
    ticker_sentiment = defaultdict(list)
    for ticker, horizon, pred_return, article_id in rows:
        ticker_sentiment[ticker].append({
            'horizon': horizon,
            'return': pred_return,
            'article_id': article_id
        })
    
    return ticker_sentiment


def sentiment_to_position(sentiment_scores, horizon_weights=None):
    """
    Convert sentiment scores to position percentage.
    
    Args:
        sentiment_scores: List of dicts with 'horizon' and 'return' keys
        horizon_weights: Dict mapping horizon (e.g., '1d') to weight
    
    Returns:
        Position percentage (float between -1 and 1, where 0 = no position)
    """
    if not sentiment_scores:
        return 0.0
    
    if horizon_weights is None:
        # Default: weight 1d sentiment most heavily, 3d moderately, 7d less
        horizon_weights = {
            '1d': 0.6,
            '3d': 0.3,
            '7d': 0.1
        }
    
    # Calculate weighted average sentiment
    weighted_sum = 0.0
    total_weight = 0.0
    
    for score in sentiment_scores:
        horizon = score['horizon']
        ret = score['return']
        weight = horizon_weights.get(horizon, 0.0)
        
        weighted_sum += ret * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    avg_sentiment = weighted_sum / total_weight
    
    # Convert sentiment to position size
    # Use a conservative mapping: sentiment of 0.02 (+2%) -> 2% position (not 20%)
    # Clamp to [-0.1, 0.1] to respect exposure cap
    position = max(-0.1, min(0.1, avg_sentiment * 1.0))
    
    return position


def generate_predictions(conn, start_date, end_date):
    """Generate trading predictions for all dates in the range."""
    cur = conn.cursor()
    
    # Delete existing predictions in the date range
    cur.execute('DELETE FROM trading_model_predictions WHERE dt >= ? AND dt <= ?',
                (start_date, end_date))
    logger.info('Deleted existing predictions in range %s to %s', start_date, end_date)
    
    # Get all unique dates from articles with model predictions
    cur.execute('''
        SELECT DISTINCT date(a.canonical_timestamp)
        FROM articles a
        JOIN sentiment_predictions sp ON sp.article_id = a.id
        WHERE a.canonical_timestamp >= ? AND a.canonical_timestamp < ?
        AND sp.model LIKE 'lightgbm_%'
        ORDER BY date(a.canonical_timestamp)
    ''', (start_date, end_date))
    
    article_dates = [r[0] for r in cur.fetchall()]
    logger.info('Found %d dates with articles', len(article_dates))
    
    inserted_count = 0
    
    for date_str in article_dates:
        ticker_sentiment = aggregate_daily_sentiment(conn, date_str)
        
        for ticker, scores in ticker_sentiment.items():
            position = sentiment_to_position(scores)
            
            # Only insert if position is meaningful
            if abs(position) < 0.005:  # Skip tiny positions (0.5%)
                continue
            
            # Get next trading day for this prediction
            next_day = get_next_trading_day(conn, date_str)
            if not next_day:
                continue
            
            # Calculate a simple predicted return as average of sentiment
            avg_return = sum(s['return'] for s in scores) / len(scores) if scores else 0.0
            
            cur.execute('''
                INSERT INTO trading_model_predictions 
                (ticker, dt, model, predicted_return, enter_prob, suggested_position_pct, exit_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                next_day,  # Execute on next trading day
                'sentiment_aggregator_v1',
                avg_return,
                abs(position) if position > 0 else 0.0,
                position,
                0.5 if position < 0 else 0.0  # Arbitrary exit prob
            ))
            
            if cur.rowcount > 0:
                inserted_count += 1
    
    conn.commit()
    logger.info('Inserted %d trading predictions', inserted_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Find project root by going up from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    default_db_path = os.path.join(project_root, 'data', 'backtest.db')
    parser.add_argument('--db', default=default_db_path)
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default='2025-01-01')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    generate_predictions(conn, args.start, args.end)
    conn.close()
    logger.info('Done')
