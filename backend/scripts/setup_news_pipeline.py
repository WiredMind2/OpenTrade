import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.scripts.bootstrap_tickers import ensure_tickers_in_db, POPULAR_TICKERS
from backend.scripts import ingest_news
from backend.scripts import map_articles_to_tickers
from backend.logging_config import get_component_logger

logger = get_component_logger(__file__)


def main():
    parser = argparse.ArgumentParser(
        description='Bootstrap news support by creating tickers, ingesting news, and mapping articles to tickers.'
    )
    parser.add_argument(
        '--db',
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')),
        help='Path to the SQLite backtest database',
    )
    parser.add_argument(
        '--news-query',
        default='earnings OR acquisition OR merger OR revenue',
        help='Query string used to ingest news from NewsAPI',
    )
    parser.add_argument(
        '--news-from',
        default=None,
        help='Optional start date for news ingestion (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--news-to',
        default=None,
        help='Optional end date for news ingestion (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--newsapi-key',
        default=None,
        help='Optional NewsAPI key. If omitted, the script will use NEWSAPI_KEY from the environment or .env file.',
    )
    args = parser.parse_args()

    if args.newsapi_key:
        os.environ['NEWSAPI_KEY'] = args.newsapi_key

    if not ingest_news.NEWSAPI_KEY and not os.getenv('NEWSAPI_KEY'):
        raise SystemExit(
            'NEWSAPI_KEY is not set. Set it in your environment, add it to .env, or pass --newsapi-key.'
        )

    logger.info('Step 1/3: Ensuring tickers exist in the database...')
    ticker_results = ensure_tickers_in_db(args.db, POPULAR_TICKERS)
    logger.info(f"Added {ticker_results['added']} new tickers, {ticker_results['existing']} already existed.")

    logger.info('Step 2/3: Ingesting news articles from NewsAPI...')
    ingest_news.ingest_news_data(
        db_path=args.db,
        query=args.news_query,
        from_dt=args.news_from,
        to_dt=args.news_to,
        api_key=os.getenv('NEWSAPI_KEY'),
    )
    logger.info('News ingestion completed.')

    logger.info('Step 3/3: Mapping articles to tickers...')
    conn = sqlite3.connect(args.db)
    try:
        tickers = map_articles_to_tickers.load_tickers(conn)
        if not tickers:
            raise SystemExit(
                'No tickers were found in the database. Run a ticker bootstrap or supply ticker CSVs first.'
            )
        mapped = map_articles_to_tickers.map_articles(conn, tickers)
        logger.info(f'Inserted {mapped} article->ticker mappings.')
    finally:
        conn.close()

    logger.info('Setup complete. Your backend should now return ticker-specific news for supported symbols.')


if __name__ == '__main__':
    main()
