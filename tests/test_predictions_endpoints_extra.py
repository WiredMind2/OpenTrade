import tempfile
import os
import sqlite3
import json
from datetime import datetime
from fastapi.testclient import TestClient

from backend.main import app, app_state


def init_min_schema(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS sentiment_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            ticker TEXT,
            model TEXT,
            horizon TEXT,
            predicted_return REAL,
            predicted_confidence REAL,
            produced_at TEXT DEFAULT (datetime('now')),
            features_used TEXT,
            metadata TEXT,
            training_run_id TEXT
        );

        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            title TEXT,
            content TEXT,
            published_at TEXT,
            sentiment_score REAL
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
            PRIMARY KEY(ticker,date)
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


def test_predictions_recent_empty_and_limit():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        # Ensure app config uses our temp DB
        os.environ['DB_PATH'] = db
        from backend.config import reload_config
        reload_config()
        app_state['database_path'] = db
        client = TestClient(app)

        # No rows -> empty list
        r = client.get('/predictions/recent')
        assert r.status_code == 200
        assert r.json() == []

        # Insert 3 rows and request limit=2
        conn = sqlite3.connect(db)
        for i in range(3):
            conn.execute('''INSERT INTO sentiment_predictions (article_id, ticker, model, horizon, predicted_return, predicted_confidence, produced_at, features_used, metadata, training_run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (i+1, 'AAPL', 'lightgbm_1d', '1d', 0.01 * (i+1), 0.5 + i*0.1, datetime.utcnow().isoformat(), 'features', '{}', 'test_run'))
        conn.commit()
        conn.close()

        r = client.get('/predictions/recent?limit=2')
        assert r.status_code == 200
        arr = r.json()
        assert isinstance(arr, list) and len(arr) == 2

    finally:
        try:
            os.unlink(db)
        except Exception:
            pass


def test_recent_metadata_parsing_and_filter():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        conn = sqlite3.connect(db)
        conn.execute('''INSERT INTO sentiment_predictions (ticker, horizon, predicted_return, predicted_confidence, produced_at, model, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     ('AAPL', '1d', 0.02, 0.8, datetime.utcnow().isoformat(), 'lightgbm_1d', json.dumps({'foo':'bar'})))
        conn.commit()
        conn.close()

        os.environ['DB_PATH'] = db
        from backend.config import reload_config
        reload_config()
        app_state['database_path'] = db
        client = TestClient(app)
        r = client.get('/predictions/recent?ticker=AAPL')
        assert r.status_code == 200
        arr = r.json()
        assert len(arr) == 1
        assert arr[0]['ticker'] == 'AAPL'
        assert 'metadata' in arr[0]
        assert isinstance(arr[0]['metadata'], dict)

    finally:
        try:
            os.unlink(db)
        except Exception:
            pass


def test_post_predict_model_not_loaded_returns_404():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        # Ensure app uses our temp DB and models dir
        os.environ['DB_PATH'] = db
        app_state['database_path'] = db
        # Ensure no models loaded
        app_state['models_loaded'] = {}
        client = TestClient(app)

        payload = {"ticker": "AAPL", "horizon": "1d", "context": {}}
        r = client.post('/predict', json=payload)
        assert r.status_code == 404
    finally:
        try:
            os.unlink(db)
        except Exception:
            pass


def test_post_predict_model_predict_exception_fallback(monkeypatch):
    # Create temp DB and a fake model that raises
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        app_state['database_path'] = db

        class BrokenModel:
            def predict(self, X):
                raise RuntimeError('model failed')

        # load a model into app_state
        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': BrokenModel(), 'embedder': 'all'}}

        client = TestClient(app)
        payload = {"ticker": "AAPL", "horizon": "1d", "context": {}}
        r = client.post('/predict', json=payload)
        assert r.status_code == 200
        data = r.json()
        assert 'predicted_return' in data
        assert 'confidence' in data

        # Also check the prediction was stored in the DB
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM sentiment_predictions')
        cnt = cur.fetchone()[0]
        assert cnt == 1
        conn.close()

    finally:
        try:
            os.unlink(db)
        except Exception:
            pass


def test_trading_predictions_returns_empty_list_when_none():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_min_schema(db)
        app_state['database_path'] = db
        client = TestClient(app)
        r = client.get('/trading/predictions')
        assert r.status_code == 200
        assert isinstance(r.json(), list)
    finally:
        try:
            os.unlink(db)
        except Exception:
            pass
