"""
Minimal backtest runner stub.

This runner demonstrates how to load trading_model_predictions for a given day and simulate market-open execution according to the backtester spec.

Usage:
  python backtest_runner.py --db data/backtest.db --start 2020-01-01 --end 2025-01-01

This is a stub: the real trading_model should be called to produce suggested_position_pct values.
"""
import argparse
import sqlite3
import os
from datetime import datetime, timedelta


def load_trading_predictions(conn, date_str):
    cur = conn.cursor()
    cur.execute('SELECT ticker, suggested_position_pct FROM trading_model_predictions WHERE dt = ?', (date_str,))
    return cur.fetchall()


def get_open_price(conn, ticker, date_str):
    cur = conn.cursor()
    cur.execute('SELECT open FROM price_daily WHERE ticker = ? AND date = ?', (ticker, date_str))
    r = cur.fetchone()
    return r[0] if r else None


def run_backtest(db_path: str, start_date: str, end_date: str, initial_capital: float = 100000.0, commission_per_share: float = 0.005, slippage_pct: float = 0.0002, exposure_cap: float = 0.5):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    capital = initial_capital
    positions = {}
    # iterate dates (naive daily loop using price_daily dates from the DB)
    cur.execute('SELECT DISTINCT date FROM price_daily WHERE date >= ? AND date <= ? ORDER BY date ASC', (start_date, end_date))
    dates = [r[0] for r in cur.fetchall()]
    for dt in dates:
        preds = load_trading_predictions(conn, dt)
        if not preds:
            # nothing to do for this day
            continue
        # compute desired allocations
        allocations = {}
        for ticker, pct in preds:
            allocations[ticker] = pct
        # enforce exposure cap
        total_requested = sum(abs(v) for v in allocations.values())
        if total_requested > exposure_cap:
            scale = exposure_cap / total_requested
            allocations = {t: v * scale for t, v in allocations.items()}
        # execute entries at open
        for ticker, pct in allocations.items():
            open_price = get_open_price(conn, ticker, dt)
            if open_price is None:
                continue
            dollars = pct * capital
            qty = int(dollars / open_price)
            if qty <= 0:
                continue
            exec_price = open_price * (1 + slippage_pct)
            cost = qty * exec_price + qty * commission_per_share
            if cost > capital:
                continue
            capital -= cost
            positions[ticker] = {'qty': qty, 'entry_price': exec_price}
            # record trade (omitted: insert into trades)
        # simplistic mark-to-market and end-of-day snapshot
        market_value = 0.0
        for ticker, pos in positions.items():
            cur.execute('SELECT close FROM price_daily WHERE ticker = ? AND date = ?', (ticker, dt))
            r = cur.fetchone()
            if r and r[0] is not None:
                market_value += pos['qty'] * r[0]
        total_value = capital + market_value
        # print daily snapshot
        print(f'{dt} total_value={total_value:.2f} cash={capital:.2f} market_value={market_value:.2f}')
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    args = parser.parse_args()
    run_backtest(args.db, args.start, args.end)


if __name__ == '__main__':
    main()
