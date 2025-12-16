import pytest
from unittest.mock import patch, MagicMock
from backend.scripts.ingest_news import ingest_news_data


def test_ingest_news_success():
    # Mock successful API response
    with patch('newsapi.NewsApiClient.get_everything') as mock_get:
        mock_get.return_value = {'articles': [{'title': 'Test News', 'content': 'Test content', 'url': 'http://test.com', 'publishedAt': '2023-01-01'}]}

        with patch('backend.scripts.ingest_news.store_articles'):
            result = ingest_news_data()
            assert result is True


def test_ingest_news_api_failure():
    # Mock API failure
    with patch('newsapi.NewsApiClient.get_everything', side_effect=Exception('API Error')):
        with pytest.raises(Exception):
            ingest_news_data()