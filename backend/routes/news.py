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


def get_db_connection():
    """Get a database connection."""
    config = get_config()
    db_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", config.database.path)
    )
    return sqlite3.connect(db_path)


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
    ticker: str = Query(None, description="Filter by ticker symbol"),
    limit: int = Query(50, description="Maximum number of results to return")
):
    """
    Get news articles.
    
    - If ticker provided: return articles filtered by ticker
    - If no ticker: return latest global articles
    """
    logger = get_component_logger(__file__)
    
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    
    try:
        if ticker:
            # Get articles for specific ticker
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
            cursor = conn.execute(query, (ticker.upper(), limit))
        else:
            # Get latest global articles
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
            cursor = conn.execute(query, (limit,))
        
        rows = cursor.fetchall()
        
        # Convert Row objects to dicts with sentiment analysis
        results = []
        for row in rows:
            article = dict(row)
            # Combine title and summary for sentiment analysis
            full_text = f"{article.get('title', '')} {article.get('summary', '')}"
            article['sentiment'] = analyze_sentiment(full_text)
            article['impact'] = analyze_impact(full_text)
            results.append(article)
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise
    finally:
        conn.close()