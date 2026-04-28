import pytest
from unittest.mock import patch, MagicMock
from backend.scripts.ingest_news import ingest_news_data


def test_ingest_news_success():
    # Mock successful API response without requiring external `newsapi` dependency.
    mock_conn = MagicMock()
    mock_conn.fetch_headlines.return_value = [
        {'title': 'Test News', 'content': 'Test content', 'url': 'http://test.com', 'publishedAt': '2023-01-01'}
    ]

    with patch('backend.scripts.ingest_news.NewsAPIConnector', return_value=mock_conn):
        with patch('backend.scripts.ingest_news.store_articles'):
            result = ingest_news_data(api_key='test-key')
            assert result is True


def test_ingest_news_api_failure():
    mock_conn = MagicMock()
    mock_conn.fetch_headlines.side_effect = Exception('API Error')

    with patch('backend.scripts.ingest_news.NewsAPIConnector', return_value=mock_conn):
        with pytest.raises(Exception):
            ingest_news_data(api_key='test-key')