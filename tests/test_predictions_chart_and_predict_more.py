import tempfile
import os
import sqlite3
import json
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


def init_schema(db_path: str):
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

    CREATE TABLE IF NOT EXISTS sentiment_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER,
        ticker TEXT,
        model TEXT,
        horizon TEXT,
        predicted_return REAL,
        confidence REAL,
        produced_at TEXT,
        features_used TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        title TEXT,
        content TEXT,
        published_at TEXT,
        sentiment_score REAL
    );
    ''')
    conn.commit()
    conn.close()


def test_chart_data_empty_hist_and_preds():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_schema(db)
        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/predictions/chart-data/FOO')
            assert r.status_code == 200
            data = r.json()
            assert data['ticker'] == 'FOO'
            assert data['historical_data'] == []
            assert data['predictions'] == []
    finally:
        _safe_unlink(db)


def test_chart_data_with_hist_and_raw_predictions_and_include_raw():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_schema(db)
        conn = sqlite3.connect(db)
        # Insert two days of prices
        today = datetime.utcnow().date()
        d0 = (today - timedelta(days=2)).isoformat()
        d1 = (today - timedelta(days=1)).isoformat()
        conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                     ('ABC', d0, 10, 11, 9, 10.5, 10.5, 1000))
        conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                     ('ABC', d1, 10.5, 12, 10, 11.0, 11.0, 1500))

        # Insert a prediction produced on d1 for horizon 1d
        produced_at = d1 + 'T12:00:00'
        conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,confidence,produced_at,model,metadata) VALUES (?,?,?,?,?,?,?)',
                     ('ABC', '1d', 0.1, 0.8, produced_at, 'lightgbm_1d', json.dumps({'src':'test'})))
        conn.commit()
        conn.close()

        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/predictions/chart-data/ABC?include_raw=true')
            assert r.status_code == 200
            data = r.json()
            assert len(data['historical_data']) == 2
            # Predictions present
            assert 'predictions' in data
            assert len(data['predictions']) == 1
            assert 'raw_predictions' in data
            assert len(data['raw_predictions']) == 1
            p = data['predictions'][0]
            # predicted_price computed from base price (close on produced_at) * (1 + predicted_return)
            assert p['predicted_price'] is not None
    finally:
        _safe_unlink(db)


def test_chart_data_aggregation_modes_and_malformed_produced_at():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_schema(db)
        conn = sqlite3.connect(db)
        # Insert price for base
        base_date = (datetime.utcnow().date() - timedelta(days=3)).isoformat()
        conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                     ('XYZ', base_date, 20, 21, 19, 20.0, 20.0, 500))

        # Two predictions that target same date (produced_at same day) but different confidences and produced_at timestamps
        produced_a = base_date + 'T08:00:00'
        produced_b = base_date + 'T16:00:00'
        conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,confidence,produced_at,model) VALUES (?,?,?,?,?,?)',
                     ('XYZ', '1d', 0.05, 0.4, produced_a, 'lightgbm_1d'))
        conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,confidence,produced_at,model) VALUES (?,?,?,?,?,?)',
                     ('XYZ', '1d', 0.15, 0.9, produced_b, 'lightgbm_1d'))

        # Malformed produced_at entry
        conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,confidence,produced_at,model) VALUES (?,?,?,?,?,?)',
                     ('XYZ', '1d', 0.2, 0.2, 'not-a-date', 'lightgbm_1d'))

        conn.commit()
        conn.close()

        app_state['database_path'] = db
        with TestClient(app) as client:
            # avg
            r = client.get('/predictions/chart-data/xyz?aggregate=avg')
            assert r.status_code == 200
            data = r.json()
            # There should be at least one aggregated result for the target date
            assert any('count' in p for p in data['predictions'])

            # latest
            r = client.get('/predictions/chart-data/xyz?aggregate=latest')
            assert r.status_code == 200
            data2 = r.json()
            assert any('predicted_price' in p for p in data2['predictions'])

            # max_conf
            r = client.get('/predictions/chart-data/xyz?aggregate=max_conf')
            assert r.status_code == 200
            data3 = r.json()
            assert any('predicted_price' in p for p in data3['predictions'])

    finally:
        _safe_unlink(db)


def test_predict_success_with_model_and_data():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_schema(db)
        conn = sqlite3.connect(db)
        # insert some article rows
        pub = (datetime.utcnow() - timedelta(days=1)).isoformat()
        conn.execute('INSERT INTO articles (ticker,title,content,published_at,sentiment_score) VALUES (?,?,?,?,?)',
                     ('MSFT', 't', 'c', pub, 0.2))
        # insert price data
        for i in range(5):
            d = (datetime.utcnow().date() - timedelta(days=i)).isoformat()
            conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                         ('MSFT', d, 100+i, 101+i, 99+i, 100+i, 100+i, 1000+i*10))
        conn.commit()
        conn.close()

        # Put a fake model that returns [0.02]
        class FakeModel:
            def predict(self, X):
                return [0.02]

        app_state['database_path'] = db
        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': FakeModel(), 'embedder': 'x'}}

        with TestClient(app) as client:
            payload = {"ticker": "MSFT", "horizon": "1d", "context": {}}
            r = client.post('/predict', json=payload)
            assert r.status_code == 200
            js = r.json()
            assert abs(js['predicted_return'] - 0.02) < 1e-6
            assert 'confidence' in js

        # ensure stored
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*) FROM sentiment_predictions')
        cnt = cur.fetchone()[0]
        assert cnt == 1
        conn.close()

    finally:
        _safe_unlink(db)
