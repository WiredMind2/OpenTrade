"""
News ingestion skeleton using NewsAPI (modular connector). Stores articles into SQLite `articles` table.

Notes:
- Requires NEWSAPI_KEY in environment or .env file.
- This script is a skeleton: adapt query construction, rate limits, and parsing for production.
"""
import os
import sqlite3
import argparse
from datetime import datetime
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')


class NewsAPIConnector:
    def __init__(self, api_key: str):
        self.client = NewsApiClient(api_key=api_key)

    def fetch_headlines(self, query: str, from_dt: str | None = None, to_dt: str | None = None, page=1, page_size=100):
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
        # pass all params through to the underlying client
        result = self.client.get_everything(**params)
        return result.get('articles', [])


def store_articles(db_path: str, articles: list):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    inserted = 0
    for a in articles:
        url = a.get('url')
        title = a.get('title')
        author = a.get('author')
        published_at = a.get('publishedAt')
        content = a.get('content') or a.get('description') or ''
        try:
            cur.execute(
                'INSERT OR IGNORE INTO articles (source, url, canonical_timestamp, title, author, content, raw_html, fingerprint, lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                ('newsapi', url, published_at, title, author, content, None, None, 'en')
            )
            inserted += cur.rowcount
        except Exception as e:
            print('Failed to insert', url, e)
    conn.commit()
    conn.close()
    print(f'Inserted {inserted} articles')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--query', default='stock OR company OR earnings')
    parser.add_argument('--from', dest='from_dt', default=None)
    parser.add_argument('--to', dest='to_dt', default=None)
    args = parser.parse_args()

    if not NEWSAPI_KEY:
        print('NEWSAPI_KEY not set in environment. Export it or add to .env file.')
        return

    conn = NewsAPIConnector(api_key=NEWSAPI_KEY)
    articles = conn.fetch_headlines(query=args.query, from_dt=args.from_dt, to_dt=args.to_dt)
    store_articles(args.db, articles)


if __name__ == '__main__':
    main()
