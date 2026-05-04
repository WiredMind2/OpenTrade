"""
News endpoints for the Trading Backtester API.
"""
import sqlite3
import sys
import os
import re

from fastapi import APIRouter, Query

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import get_config
from backend.logging_config import get_component_logger

router = APIRouter(prefix="/api/news", tags=["news"])

# Keywords for sentiment analysis
POSITIVE_KEYWORDS = [
    'surge', 'soar', 'rally', 'gain', 'profit', 'growth', 'up', 'bullish', 'beat',
    'exceed', 'record', 'high', 'strong', 'boost', 'rise', 'jump', 'positive',
    'upgrade', 'outperform', 'buy', 'recommend', 'success', 'breakthrough'
]
NEGATIVE_KEYWORDS = [
    'drop', 'fall', 'crash', 'plunge', 'loss', 'bearish', 'miss', 'down',
    'weak', 'cut', 'reduce', 'sell', 'downgrade', 'underperform', 'risk',
    'concern', 'warning', 'layoff', 'lawsuit', 'investigation', 'scandal'
]
IMPACT_KEYWORDS = {
    'high': ['earnings', 'revenue', 'profit', 'guidance', 'acquisition', 'merger',
             'lawsuit', 'investigation', 'sec', 'fda', 'bankruptcy', 'recall',
             'layoff', 'restructure', 'scandal', 'breakthrough', 'record'],
    'medium': ['forecast', 'outlook', 'partnership', 'launch', 'product', 'expansion',
               'hiring', 'contract', 'deal', 'funding', 'investment'],
    'low': ['update', 'announcement', 'meeting', 'conference', 'report', 'study']
}


def _resolve_database_path() -> str:
    """Same DB file as the rest of the API (see ``main.app_state``)."""
    from backend.main import app_state

    db_path = app_state.get("database_path")
    if db_path:
        return str(db_path)
    config = get_config()
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", config.database.path)
    )


def _ticker_text_pattern(ticker_sym: str) -> re.Pattern:
    """Word-boundary / $ / parens rules aligned with ``map_articles_to_tickers``."""
    t = str(ticker_sym).upper()
    if len(t) == 1:
        return re.compile(r"(\$" + re.escape(t) + r"\b)|\(" + re.escape(t) + r"\)")
    return re.compile(
        r"\b"
        + re.escape(t)
        + r"\b|\$"
        + re.escape(t)
        + r"\b|\("
        + re.escape(t)
        + r"\)"
    )


def _article_matches_ticker(
    title: str | None,
    content: str | None,
    ticker_upper: str,
    pattern: re.Pattern,
    company_name: str | None,
) -> bool:
    raw = " ".join([x for x in [title or "", content or ""] if x])
    if not raw:
        return False
    if pattern.search(raw):
        return True
    if company_name and company_name.lower() in raw.lower():
        return True
    return False


def _fetch_recent_articles_for_ticker_fallback(
    conn: sqlite3.Connection,
    ticker_upper: str,
    limit: int,
    *,
    company_name: str | None,
) -> list:
    """When ``article_ticker`` is empty, match recent articles by title/content heuristics."""
    pattern = _ticker_text_pattern(ticker_upper)
    cur = conn.execute(
        """
        SELECT id, title, content, source, url, canonical_timestamp
        FROM articles
        ORDER BY datetime(canonical_timestamp) DESC
        LIMIT 800
        """
    )
    rows = cur.fetchall()
    out: list = []
    for row in rows:
        if _article_matches_ticker(
            row["title"], row["content"], ticker_upper, pattern, company_name
        ):
            out.append(row)
        if len(out) >= limit:
            break
    return out


def analyze_sentiment(text: str) -> str:
    """Analyze text and return sentiment: positive, negative, or neutral."""
    if not text:
        return 'neutral'
    text_lower = text.lower()
    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    return 'neutral'


def analyze_impact(text: str) -> str:
    """Analyze text and return impact level: low, medium, or high."""
    if not text:
        return 'low'
    text_lower = text.lower()
    for kw in IMPACT_KEYWORDS['high']:
        if kw in text_lower:
            return 'high'
    for kw in IMPACT_KEYWORDS['medium']:
        if kw in text_lower:
            return 'medium'
    return 'low'


@router.get("/")
async def get_news(
    ticker: str | None = Query(None, description="Filter by ticker symbol"),
    limit: int = Query(50, description="Maximum number of results to return"),
):
    """
    Get news articles.

    - If ticker provided: prefer ``article_ticker`` mappings; if none exist (common when
      ingest ran without ``map_articles_to_tickers``), fall back to matching title/content
      with the same heuristics as the mapper script.
    - If no ticker: return latest global articles
    """
    logger = get_component_logger(__file__)

    db_path = _resolve_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        t = (ticker or "").strip()
        rows: list = []
        if t:
            tu = t.upper()
            query = """
                SELECT
                    at.ticker,
                    a.title,
                    a.content as summary,
                    a.source,
                    a.url,
                    a.canonical_timestamp as published_at,
                    at.relevance_score
                FROM article_ticker at
                JOIN articles a ON a.id = at.article_id
                WHERE at.ticker = ?
                ORDER BY a.canonical_timestamp DESC
                LIMIT ?
            """
            rows = list(conn.execute(query, (tu, limit)).fetchall())
            if not rows:
                name_row = conn.execute(
                    "SELECT name FROM tickers WHERE ticker = ?",
                    (tu,),
                ).fetchone()
                company_name = str(name_row["name"]) if name_row and name_row["name"] else None
                fb = _fetch_recent_articles_for_ticker_fallback(
                    conn, tu, limit, company_name=company_name
                )
                rows = [
                    {
                        "ticker": tu,
                        "title": r["title"],
                        "summary": r["content"],
                        "source": r["source"],
                        "url": r["url"],
                        "published_at": r["canonical_timestamp"],
                        "relevance_score": 0.5,
                    }
                    for r in fb
                ]
        else:
            query = """
                SELECT
                    at.ticker,
                    a.title,
                    a.content as summary,
                    a.source,
                    a.url,
                    a.canonical_timestamp as published_at,
                    at.relevance_score
                FROM articles a
                LEFT JOIN article_ticker at ON a.id = at.article_id
                ORDER BY a.canonical_timestamp DESC
                LIMIT ?
            """
            rows = list(conn.execute(query, (limit,)).fetchall())

        results = []
        for row in rows:
            article = dict(row) if isinstance(row, sqlite3.Row) else row
            if article.get("relevance_score") is None:
                article["relevance_score"] = 0.0
            full_text = f"{article.get('title', '')} {article.get('summary', '')}"
            article["sentiment"] = analyze_sentiment(full_text)
            article["impact"] = analyze_impact(full_text)
            results.append(article)

        return results

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise
    finally:
        conn.close()