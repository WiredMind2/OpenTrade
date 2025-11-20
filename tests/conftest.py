"""
Shared test fixtures and configuration for trading backtesting tests.
"""
import sys
import os
# Ensure project root is on sys.path so tests can import top-level modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import tempfile
import sqlite3
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_db():
    """Temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        yield db_path
    finally:
        # On Windows the DB file can be briefly locked; retry deletion a few times
        if os.path.exists(db_path):
            import time, gc
            for attempt in range(5):
                try:
                    gc.collect()
                    os.unlink(db_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
            else:
                # Last resort: ignore if still locked
                try:
                    os.remove(db_path)
                except Exception:
                    pass


@pytest.fixture
def mock_price_data():
    """Mock price data for testing."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'ticker': 'AAPL',
            'date': date.strftime('%Y-%m-%d'),
            'open': 150.0 + i,
            'high': 152.0 + i,
            'low': 148.0 + i,
            'close': 151.0 + i,
            'adjusted_close': 151.0 + i,
            'volume': 1000000
        })
        data.append({
            'ticker': 'MSFT',
            'date': date.strftime('%Y-%m-%d'),
            'open': 300.0 + i,
            'high': 302.0 + i,
            'low': 298.0 + i,
            'close': 301.0 + i,
            'adjusted_close': 301.0 + i,
            'volume': 800000
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_articles():
    """Mock news articles for testing."""
    return [
        {
            'id': 1,
            'source': 'newsapi',
            'url': 'https://example.com/article1',
            'canonical_timestamp': '2024-01-15T10:00:00',
            'title': 'Apple Reports Strong Q4 Earnings',
            'author': 'John Doe',
            'content': 'Apple Inc. reported better than expected Q4 earnings...',
            'raw_html': '<html>Apple earnings article</html>',
            'fingerprint': 'abc123',
            'lang': 'en'
        },
        {
            'id': 2,
            'source': 'newsapi',
            'url': 'https://example.com/article2',
            'canonical_timestamp': '2024-01-16T11:00:00',
            'title': 'Microsoft Azure Growth Continues',
            'author': 'Jane Smith',
            'content': 'Microsoft reported strong growth in Azure cloud services...',
            'raw_html': '<html>Microsoft Azure article</html>',
            'fingerprint': 'def456',
            'lang': 'en'
        }
    ]


@pytest.fixture
def mock_sentiment_predictions():
    """Mock sentiment predictions for testing."""
    return [
        {
            'article_id': 1,
            'ticker': 'AAPL',
            'model': 'lightgbm_1d',
            'horizon': '1d',
            'predicted_return': 0.025,
            'predicted_confidence': 0.85,
            'produced_at': '2024-01-15T12:00:00'
        },
        {
            'article_id': 1,
            'ticker': 'AAPL',
            'model': 'lightgbm_3d',
            'horizon': '3d',
            'predicted_return': 0.050,
            'predicted_confidence': 0.78,
            'produced_at': '2024-01-15T12:00:00'
        },
        {
            'article_id': 2,
            'ticker': 'MSFT',
            'model': 'lightgbm_1d',
            'horizon': '1d',
            'predicted_return': -0.015,
            'predicted_confidence': 0.82,
            'produced_at': '2024-01-16T12:00:00'
        }
    ]


@pytest.fixture
def schema_sql():
    """Database schema SQL for testing."""
    return """
-- Simplified schema for testing
CREATE TABLE IF NOT EXISTS tickers (
  ticker TEXT PRIMARY KEY,
  name TEXT,
  exchange TEXT,
  sector TEXT,
  added_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT,
  url TEXT UNIQUE,
  canonical_timestamp TEXT,
  fetched_at TEXT DEFAULT (datetime('now')),
  title TEXT,
  author TEXT,
  raw_html TEXT,
  content TEXT,
  fingerprint TEXT,
  lang TEXT,
  metadata JSON
);

CREATE TABLE IF NOT EXISTS article_ticker (
  article_id INTEGER,
  ticker TEXT,
  relevance_score REAL,
  PRIMARY KEY(article_id, ticker),
  FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS price_daily (
  ticker TEXT,
  date TEXT,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  adjusted_close REAL,
  volume INTEGER,
  PRIMARY KEY (ticker, date),
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sentiment_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id INTEGER,
  ticker TEXT,
  model TEXT,
  horizon TEXT,
  predicted_return REAL,
  predicted_confidence REAL,
  produced_at TEXT DEFAULT (datetime('now')),
  training_run_id TEXT,
  FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE SET NULL,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS trading_model_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT,
  dt TEXT,
  model TEXT,
  predicted_return REAL,
  enter_prob REAL,
  suggested_position_pct REAL,
  exit_prob REAL,
  produced_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT,
  started_at TEXT DEFAULT (datetime('now')),
  completed_at TEXT,
  params JSON,
  metrics JSON
);

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  backtest_run_id INTEGER,
  ticker TEXT,
  entry_dt TEXT,
  entry_price REAL,
  exit_dt TEXT,
  exit_price REAL,
  quantity INTEGER,
  position_pct REAL,
  fees REAL,
  slippage REAL,
  pnl REAL,
  FOREIGN KEY(backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(ticker) REFERENCES tickers(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  backtest_run_id INTEGER,
  dt TEXT,
  cash REAL,
  market_value REAL,
  total_value REAL,
  exposure REAL,
  positions_json JSON,
  FOREIGN KEY(backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);
"""


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'NEWSAPI_KEY': 'test_key',
        'DB_PATH': 'test.db',
        'INITIAL_CAPITAL': 100000,
        'COMMISSION_PER_SHARE': 0.005,
        'SLIPPAGE_PCT': 0.0002
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv('NEWSAPI_KEY', 'test_newsapi_key')
    monkeypatch.setenv('DB_PATH', 'test.db')
    monkeypatch.setenv('INITIAL_CAPITAL', '100000')
    monkeypatch.setenv('COMMISSION_PER_SHARE', '0.005')
    monkeypatch.setenv('SLIPPAGE_PCT', '0.0002')


def init_test_db(db_path, schema_sql):
    """Helper function to initialize test database with schema."""
    conn = sqlite3.connect(db_path)
    
    # Execute schema creation
    conn.executescript(schema_sql)
    
    # Insert test tickers
    tickers = [
        ('AAPL', 'Apple Inc.', 'NASDAQ', 'Technology'),
        ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology'),
        ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'Technology')
    ]
    
    conn.executemany(
        'INSERT OR IGNORE INTO tickers (ticker, name, exchange, sector) VALUES (?, ?, ?, ?)',
        tickers
    )
    
    conn.commit()
    conn.close()


@pytest.fixture
def populated_test_db(temp_db, schema_sql, mock_articles, mock_sentiment_predictions, mock_price_data):
    """Populated test database with sample data."""
    init_test_db(temp_db, schema_sql)
    
    conn = sqlite3.connect(temp_db)
    cur = conn.cursor()
    
    # Insert articles
    for article in mock_articles:
        cur.execute('''
            INSERT INTO articles (source, url, canonical_timestamp, title, author, content, raw_html, fingerprint, lang)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['source'], article['url'], article['canonical_timestamp'],
            article['title'], article['author'], article['content'],
            article['raw_html'], article['fingerprint'], article['lang']
        ))
    
    # Insert article-ticker mappings
    mappings = [
        (1, 'AAPL', 1.0),
        (2, 'MSFT', 1.0)
    ]
    cur.executemany('INSERT INTO article_ticker (article_id, ticker, relevance_score) VALUES (?, ?, ?)', mappings)
    
    # Insert price data
    for _, row in mock_price_data.iterrows():
        cur.execute('''
            INSERT OR IGNORE INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['ticker'], row['date'], row['open'], row['high'],
            row['low'], row['close'], row['adjusted_close'], row['volume']
        ))
    
    # Insert sentiment predictions
    for pred in mock_sentiment_predictions:
        cur.execute('''
            INSERT INTO sentiment_predictions (article_id, ticker, model, horizon, predicted_return, predicted_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            pred['article_id'], pred['ticker'], pred['model'],
            pred['horizon'], pred['predicted_return'], pred['predicted_confidence']
        ))
    
    conn.commit()
    conn.close()
    
    return temp_db