import tempfile
import os
import sqlite3
import time
import gc
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from backend.main import app, app_state


def _safe_unlink(path, retries=5, delay=0.1):
    """Windows-friendly temp file cleanup with brief retries."""
    for _ in range(retries):
        try:
            os.unlink(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(delay)
    os.unlink(path)


def init_min_schema(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS price_daily (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adjusted_close REAL,
        volume INTEGER,
        PRIMARY KEY(ticker,date)
    );

    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        title TEXT,
        content TEXT,
        published_at TEXT,
        sentiment_score REAL
    );

    CREATE TABLE IF NOT EXISTS sentiment_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER,
        ticker TEXT,
        model TEXT,
        horizon TEXT,
        predicted_return REAL,
        predicted_confidence REAL,
        produced_at TEXT,
        features_used TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS trading_model_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        suggested_position_pct REAL,
        dt TEXT,
        enter_prob REAL
    );
    ''')
    conn.commit()
    conn.close()


def test_predict_with_model_loaded_but_no_data_returns_zero():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        app_state['database_path'] = db

        class DummyModel:
            def predict(self, X):
                return [0.5]

        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': DummyModel(), 'embedder': 'x'}}
        with TestClient(app) as client:
            payload = {"ticker": "ZZZ", "horizon": "1d", "context": {}}
            r = client.post('/predict', json=payload)
            assert r.status_code == 200
            data = r.json()
            # With no articles and no price_data the code falls back to 0.0
            assert data['predicted_return'] == 0.0
    finally:
        _safe_unlink(db)


def test_predict_model_none_returns_500():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        app_state['database_path'] = db
        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': None}}
        with TestClient(app) as client:
            payload = {"ticker": "ABC", "horizon": "1d", "context": {}}
            r = client.post('/predict', json=payload)
            assert r.status_code == 500
    finally:
        _safe_unlink(db)


def test_predict_with_articles_only_uses_model():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        conn = sqlite3.connect(db)
        pub = (datetime.utcnow() - timedelta(days=1)).isoformat()
        conn.execute('INSERT INTO articles (ticker,title,content,published_at,sentiment_score) VALUES (?,?,?,?,?)',
                     ('ABC', 't', 'c', pub, 0.3))
        conn.commit()
        conn.close()

        class MyModel:
            def predict(self, X):
                return [0.123]

        app_state['database_path'] = db
        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': MyModel(), 'embedder': 'x'}}
        with TestClient(app) as client:
            payload = {"ticker": "ABC", "horizon": "1d", "context": {}}
            r = client.post('/predict', json=payload)
            assert r.status_code == 200
            data = r.json()
            assert abs(data['predicted_return'] - 0.123) < 1e-6
    finally:
        _safe_unlink(db)


def test_trading_predictions_returns_rows_when_present():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        conn = sqlite3.connect(db)
        conn.execute('INSERT INTO trading_model_predictions (ticker,suggested_position_pct,dt,enter_prob) VALUES (?,?,?,?)',
                     ('ABC', 0.2, '2024-01-01', 0.7))
        conn.commit()
        conn.close()

        os.environ['DB_PATH'] = db
        from backend.config import reload_config
        reload_config()
        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/trading/predictions')
            assert r.status_code == 200
            data = r.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]['ticker'] == 'ABC'
    finally:
        _safe_unlink(db)


def test_chart_data_with_history_but_no_predictions_returns_hist_only():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        conn = sqlite3.connect(db)
        d = datetime.utcnow().date().isoformat()
        conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                     ('HST', d, 1,2,0.5,1.5,1.5,100))
        conn.commit()
        conn.close()

        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/predictions/chart-data/HST')
            assert r.status_code == 200
            body = r.json()
            assert len(body['historical_data']) == 1
            assert body['predictions'] == []
    finally:
        _safe_unlink(db)
