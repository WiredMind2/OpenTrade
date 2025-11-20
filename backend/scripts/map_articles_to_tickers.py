"""
Map ingested articles to tickers using a simple heuristic:
- For each article, search title + content for occurrences of ticker symbols (word-boundary, uppercase) and ticker names (if provided in `tickers.name`).
- Insert rows into `article_ticker` with relevance_score=1.0 for matches.

This is a simple, fast mapper suitable for the MVP. For production you'll want a proper NER + alias table.

Usage:
  python scripts/map_articles_to_tickers.py --db data/backtest.db
"""
import sqlite3
import argparse
import re
import os


def load_tickers(conn):
    cur = conn.cursor()
    cur.execute('SELECT ticker, name FROM tickers')
    return cur.fetchall()


def map_articles(conn, tickers):
    cur = conn.cursor()
    cur.execute('SELECT id, title, content FROM articles')
    articles = cur.fetchall()
    mapped = 0
    for aid, title, content in articles:
        raw_text = ' '.join([t for t in [title or '', content or ''] if t])
        if not raw_text:
            continue
        found_any = False
        for ticker, name in tickers:
            ticker_sym = str(ticker).upper()
            # For single-letter tickers, require explicit $ prefix or parentheses to avoid matching common words
            if len(ticker_sym) == 1:
                # look for $A or (A)
                pattern = r'(\$' + re.escape(ticker_sym) + r'\b)|\(' + re.escape(ticker_sym) + r'\)'
                if re.search(pattern, raw_text):
                    try:
                        cur.execute('INSERT OR IGNORE INTO article_ticker (article_id, ticker, relevance_score) VALUES (?, ?, ?)', (aid, ticker_sym, 1.0))
                        mapped += cur.rowcount
                        found_any = True
                    except Exception as e:
                        print('Insert failed for', aid, ticker_sym, e)
            else:
                # For multi-letter tickers, match whole-word ticker (case-sensitive preferred) or $TICKER
                pattern = r'\b' + re.escape(ticker_sym) + r'\b'
                dollar = r'\$' + re.escape(ticker_sym) + r'\b'
                paren = r'\(' + re.escape(ticker_sym) + r'\)'
                if re.search(pattern, raw_text) or re.search(dollar, raw_text) or re.search(paren, raw_text):
                    try:
                        cur.execute('INSERT OR IGNORE INTO article_ticker (article_id, ticker, relevance_score) VALUES (?, ?, ?)', (aid, ticker_sym, 1.0))
                        mapped += cur.rowcount
                        found_any = True
                    except Exception as e:
                        print('Insert failed for', aid, ticker_sym, e)
            # Company name matching (if available)
            if name and not found_any:
                try:
                    if name.lower() in raw_text.lower():
                        cur.execute('INSERT OR IGNORE INTO article_ticker (article_id, ticker, relevance_score) VALUES (?, ?, ?)', (aid, ticker_sym, 0.8))
                        mapped += cur.rowcount
                        found_any = True
                except Exception as e:
                    print('Insert failed for', aid, ticker_sym, e)
        # optional: if no ticker found, leave unmapped for manual review
    conn.commit()
    return mapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    tickers = load_tickers(conn)
    if not tickers:
        print('No tickers found in DB. Run scripts/scan_csvs.py first to register tickers from your CSVs.')
    else:
        print(f'Loaded {len(tickers)} tickers from the DB')
        mapped = map_articles(conn, tickers)
        print(f'Inserted {mapped} article->ticker mappings')
    conn.close()


if __name__ == '__main__':
    main()
