import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
"""
Labeling pipeline: map articles to tickers (via article_ticker table) and compute 1d/3d/7d labels using `price_daily`.

Labeling convention (accepted): open-of-T+1 -> close-of-T+N

Usage:
  python labeling.py --db data/backtest.db --horizons 1 3 7
"""
import argparse
import sqlite3
import os
import importlib.util
from datetime import datetime, timedelta
from backend.scripts.script_logger import logger


def load_label_debug_module():
    """
    Load the label_debug.py module with proper error handling for file existence.
    """
    debug_file = os.path.join(os.path.dirname(__file__), 'label_debug.py')
    if not os.path.exists(debug_file):
        raise FileNotFoundError(f"Debug file {debug_file} not found")
    spec = importlib.util.spec_from_file_location("label_debug", debug_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_next_trading_date(conn, ticker, date_str, offset_days):
    # very simple: assumes price_daily has continuous trading dates; finds date offset_days after date_str
    cur = conn.cursor()
    cur.execute('SELECT date FROM price_daily WHERE ticker = ? AND date > ? ORDER BY date ASC LIMIT ?', (ticker, date_str, offset_days))
    rows = cur.fetchall()
    if len(rows) < offset_days:
        return None
    return rows[offset_days - 1][0]


def label_articles(db_path: str, horizons):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # iterate articles and their tickers
    cur.execute('SELECT id, canonical_timestamp FROM articles')
    articles = cur.fetchall()
    # compute global max price date to quickly skip articles that are newer than our price data
    cur.execute('SELECT MAX(date) FROM price_daily')
    gmax = cur.fetchone()
    global_max_price_date = gmax[0] if gmax else None
    labeled = 0
    skipped_no_entry = 0
    skipped_no_exit = 0
    skipped_no_open = 0
    skipped_too_new = 0
    for aid, cts in articles:
        if not cts:
            continue
        # canonical timestamp -> date string
        pub_date = cts.split('T')[0]
        if global_max_price_date and pub_date >= global_max_price_date:
            skipped_too_new += 1
            continue
        cur.execute('SELECT ticker FROM article_ticker WHERE article_id = ?', (aid,))
        tickers = [r[0] for r in cur.fetchall()]
        if not tickers:
            continue
        for ticker in tickers:
            # skip if publication date is after the last available price for this ticker
            cur.execute('SELECT MAX(date) FROM price_daily WHERE ticker = ?', (ticker,))
            md = cur.fetchone()
            max_price_date = md[0] if md else None
            if not max_price_date or pub_date >= max_price_date:
                skipped_no_entry += 1
                continue
            # find next trading day (T+1)
            cur.execute('SELECT date FROM price_daily WHERE ticker = ? AND date > ? ORDER BY date ASC LIMIT 1', (ticker, pub_date))
            row = cur.fetchone()
            if not row:
                skipped_no_entry += 1
                continue
            entry_date = row[0]
            # get open price at entry_date
            cur.execute('SELECT open FROM price_daily WHERE ticker = ? AND date = ?', (ticker, entry_date))
            entry_open = cur.fetchone()
            if not entry_open or entry_open[0] is None:
                skipped_no_open += 1
                continue
            entry_open = entry_open[0]
            for h in horizons:
                # find close at T+h
                cur.execute('SELECT date FROM price_daily WHERE ticker = ? AND date >= ? ORDER BY date ASC LIMIT ?', (ticker, entry_date, h))
                rows = cur.fetchall()
                if len(rows) < h:
                    skipped_no_exit += 1
                    continue
                exit_date = rows[h-1][0]
                cur.execute('SELECT close FROM price_daily WHERE ticker = ? AND date = ?', (ticker, exit_date))
                exit_close = cur.fetchone()
                if not exit_close or exit_close[0] is None:
                    skipped_no_exit += 1
                    continue
                exit_close = exit_close[0]
                label = (exit_close - entry_open) / entry_open
                # insert into sentiment_predictions as a placeholder (model field left empty)
                cur.execute('INSERT INTO sentiment_predictions (article_id, ticker, model, horizon, predicted_return, predicted_confidence, produced_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (aid, ticker, 'label_groundtruth', f'{h}d', label, 1.0, datetime.utcnow().isoformat()))
                labeled += 1
    conn.commit()
    conn.close()
    logger.info('Inserted %d ground-truth labels (as sentiment_predictions rows)', labeled)
    logger.info('Skipped (no entry date): %d', skipped_no_entry)
    logger.info('Skipped (no open price): %d', skipped_no_open)
    logger.info('Skipped (no exit price): %d', skipped_no_exit)
    logger.info('Skipped (article newer than price data): %d', skipped_too_new)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--horizons', nargs='+', type=int, default=[1,3,7])
    args = parser.parse_args()
    label_articles(args.db, args.horizons)


if __name__ == '__main__':
    main()
