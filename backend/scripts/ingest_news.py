import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
"""
News ingestion skeleton using NewsAPI (modular connector). Stores articles into SQLite `articles` table.

Notes:
- Requires NEWSAPI_KEY in environment or .env file.
- This script is a skeleton: adapt query construction, rate limits, and parsing for production.
"""
import os
import sqlite3
import argparse
import time
from datetime import datetime
from dotenv import load_dotenv
from backend.scripts.script_logger import logger

load_dotenv()

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')


class NewsAPIConnector:
    def __init__(self, api_key: str):
        # Optional dependency: keep module importable in environments
        # where the third-party `newsapi` package is not installed (e.g. CI/tests).
        try:
            from newsapi import NewsApiClient  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Optional dependency missing: install `newsapi-python` to ingest news."
            ) from e

        self.client = NewsApiClient(api_key=api_key)

    def fetch_headlines(self, query: str, from_dt: str | None = None, to_dt: str | None = None, page=1, page_size=100, max_retries=3):
        # returns list of article dicts
        params = {
            'q': query,
            'language': 'en',
            'page': page,
            'page_size': page_size,
        }
        if from_dt:
            # NewsApiClient expects 'from_param' for the from date
            params['from_param'] = from_dt
        if to_dt:
            # NewsApiClient uses 'to' for the end date
            params['to'] = to_dt

        for attempt in range(max_retries):
            try:
                logger.info('Fetching headlines, attempt %d/%d', attempt + 1, max_retries)
                result = self.client.get_everything(**params)
                if not isinstance(result, dict):
                    raise ValueError('API response is not a valid dictionary')
                articles = result.get('articles', [])
                if not isinstance(articles, list):
                    raise ValueError('Articles in API response is not a list')
                logger.info('Successfully fetched %d articles', len(articles))
                return articles
            except Exception as e:
                logger.warning('Failed to fetch headlines on attempt %d: %s', attempt + 1, e)
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt  # exponential backoff
                    logger.info('Retrying in %d seconds', sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.error('Failed to fetch headlines after %d attempts', max_retries)
                    raise


def store_articles(db_path: str, articles: list):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        inserted = 0
        for a in articles:
            if not isinstance(a, dict):
                logger.warning('Skipping invalid article: not a dict')
                continue
            url = a.get('url')
            title = a.get('title')
            author = a.get('author')
            published_at = a.get('publishedAt')
            content = a.get('content') or a.get('description') or ''
            if not url or not title:
                logger.warning('Skipping article with missing url or title: %s', a)
                continue
            try:
                cur.execute(
                    'INSERT OR IGNORE INTO articles (source, url, canonical_timestamp, title, author, content, raw_html, fingerprint, lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    ('newsapi', url, published_at, title, author, content, None, None, 'en')
                )
                inserted += cur.rowcount
            except Exception as e:
                logger.error('Failed to insert %s: %s', url, e)
        conn.commit()
        logger.info('Inserted %d articles', inserted)
    except sqlite3.Error as e:
        logger.error('Database error while storing articles: %s', e)
        raise
    finally:
        if 'conn' in locals():
            conn.close()


def ingest_news_data(
    db_path: str | None = None,
    query: str = 'stock OR company OR earnings',
    from_dt: str | None = None,
    to_dt: str | None = None,
    api_key: str | None = None,
):
    """
    Ingest news data from NewsAPI and store in database.
    Returns True on success, raises exception on failure.
    """
    if db_path is None:
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db'))

    api_key = api_key or NEWSAPI_KEY
    if not api_key:
        raise ValueError('NEWSAPI_KEY not set in environment. Export it or add to .env file.')

    try:
        conn = NewsAPIConnector(api_key=api_key)
        articles = conn.fetch_headlines(query=query, from_dt=from_dt, to_dt=to_dt)
        store_articles(db_path, articles)
        return True
    except Exception as e:
        logger.error('Failed to ingest news data: %s', e)
        raise


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--query', default='stock OR company OR earnings')
    parser.add_argument('--from', dest='from_dt', default=None)
    parser.add_argument('--to', dest='to_dt', default=None)
    args = parser.parse_args()

    try:
        ingest_news_data(db_path=args.db, query=args.query, from_dt=args.from_dt, to_dt=args.to_dt)
    except Exception as e:
        logger.error('News ingestion failed: %s', e)
        exit(1)


if __name__ == '__main__':
    main()
