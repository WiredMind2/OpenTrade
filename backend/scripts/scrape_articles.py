"""
Scrape full article content from stored URLs and update the `articles` table.

Behavior:
- Select articles where `content` is NULL or very short (heuristic) and `url` is not NULL.
- Fetch each URL, extract the main article HTML using readability-lxml, then extract plain text with BeautifulSoup.
- Update `articles.content` and `articles.raw_html`.

Usage:
  python scripts/scrape_articles.py --db data/backtest.db --limit 200

Be polite: the script waits 1 second between requests by default. Respect site TOS and robots.txt when scraping in production.
"""
import argparse
import sqlite3
import time
import requests
try:
    from readability import Document
except Exception:
    raise SystemExit("readability or its dependencies are missing. Please run: pip install 'lxml[html_clean]' readability-lxml or pip install lxml_html_clean readability-lxml")
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

HEADERS = {
    'User-Agent': os.getenv('SCRAPER_USER_AGENT', 'Mozilla/5.0 (compatible; backtesting-bot/1.0; +https://example.com)')
}


def fetch_html(url: str, timeout: int = 15):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f'Failed to fetch {url}: {e}')
        return None


def extract_main_text(html: str):
    try:
        doc = Document(html)
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, 'html.parser')
        text = soup.get_text(separator='\n').strip()
        return summary_html, text
    except Exception as e:
        print('Extraction failed:', e)
        return None, None


def scrape(db_path: str, limit: int = 100, pause: float = 1.0):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # heuristic: content is null, very short, or appears to be a cropped preview (e.g. "... [+3030 chars]")
    # NewsAPI and some providers truncate content into previews that include the pattern "[+<number> chars]".
    cur.execute(
        "SELECT id, url, content FROM articles WHERE url IS NOT NULL AND (content IS NULL OR length(content) < 200 OR content LIKE '%[+%') LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    print(f'Found {len(rows)} articles to scrape')
    updated = 0
    for aid, url, content in rows:
        html = fetch_html(url)
        if not html:
            continue
        summary_html, text = extract_main_text(html)
        if not text:
            continue
        try:
            cur.execute('UPDATE articles SET raw_html = ?, content = ? WHERE id = ?', (summary_html, text, aid))
            conn.commit()
            updated += 1
            print(f'Updated article {aid} from {url}')
        except Exception as e:
            print('DB update failed for', aid, e)
        time.sleep(pause)
    conn.close()
    print(f'Updated {updated} articles')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'backtest.db')))
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--pause', type=float, default=1.0)
    args = parser.parse_args()
    scrape(args.db, args.limit, args.pause)


if __name__ == '__main__':
    main()
