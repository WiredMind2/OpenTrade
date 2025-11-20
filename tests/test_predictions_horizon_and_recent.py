import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from backend.main import app, app_state


def init_db(db):
    conn = sqlite3.connect(db)
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS price_daily (
        ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adjusted_close REAL, volume INTEGER, PRIMARY KEY(ticker,date)
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
        id INTEGER PRIMARY KEY AUTOINCREMENT, article_id INTEGER, ticker TEXT, model TEXT, horizon TEXT,
        predicted_return REAL, predicted_confidence REAL, produced_at TEXT, features_used TEXT, metadata TEXT, training_run_id TEXT
    );
    ''')
    conn.commit()
    conn.close()


def test_chart_data_horizon_3d_and_7d():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_db(db)
        conn = sqlite3.connect(db)
        base = (datetime.utcnow().date() - timedelta(days=10))
        # Insert price rows for several days
        for i in range(12):
            d = (base + timedelta(days=i)).isoformat()
            conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                         ('MUL', d, 10+i, 11+i, 9+i, 10+i, 10+i, 1000))

        # Insert a 3d prediction produced on day 2
        produced = (base + timedelta(days=2)).isoformat() + 'T09:00:00'
        conn.execute('INSERT INTO sentiment_predictions (article_id, ticker, model, horizon, predicted_return, predicted_confidence, produced_at, features_used, metadata, training_run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (1, 'MUL', 'lightgbm_3d', '3d', 0.05, 0.6, produced, 'features', '{}', 'test_run'))

        # Insert a 7d prediction
        produced7 = (base + timedelta(days=1)).isoformat() + 'T09:00:00'
        conn.execute('INSERT INTO sentiment_predictions (article_id, ticker, model, horizon, predicted_return, predicted_confidence, produced_at, features_used, metadata, training_run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (2, 'MUL', 'lightgbm_7d', '7d', 0.02, 0.7, produced7, 'features', '{}', 'test_run'))

        conn.commit()
        conn.close()

        os.environ['DB_PATH'] = db
        from backend.config import reload_config
        reload_config()
        app_state['database_path'] = db
        client = TestClient(app)
        r3 = client.get('/predictions/chart-data/MUL?horizon=3d')
        assert r3.status_code == 200
        body3 = r3.json()
        assert 'predictions' in body3

        r7 = client.get('/predictions/chart-data/MUL?horizon=7d')
        assert r7.status_code == 200
        body7 = r7.json()
        assert 'predictions' in body7
    finally:
        os.unlink(db)


def test_get_recent_predictions_parses_features_and_metadata():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_db(db)
        conn = sqlite3.connect(db)
        conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,predicted_confidence,produced_at,model,features_used,metadata) VALUES (?,?,?,?,?,?,?,?)',
                     ('PR', '1d', 0.01, 0.5, datetime.utcnow().isoformat(), 'lightgbm_1d', 'a,b,c', '{"x":1}'))
        conn.commit()
        conn.close()

        os.environ['DB_PATH'] = db
        from backend.config import reload_config
        reload_config()
        app_state['database_path'] = db
        client = TestClient(app)
        r = client.get('/predictions/recent')
        assert r.status_code == 200
        body = r.json()
        assert len(body) >= 1
        p = body[0]
        assert isinstance(p.get('features_used'), list)
        assert isinstance(p.get('metadata'), dict)
    finally:
        os.unlink(db)


def test_confidence_clamp_bounds():
    # model returns 0.0 -> 1 - 0 = 1 -> min(...)=0.95
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        init_db(db)
        conn = sqlite3.connect(db)
        # create multiple price rows so returns is a numpy array
        for i in range(3):
            d = (datetime.utcnow().date() - timedelta(days=i)).isoformat()
            conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                         ('CL', d, 1+i,2+i,1+i,1.5+i,1.5+i,100+i))
        conn.commit()
        conn.close()

        class M0:
            def predict(self, X):
                return [0.0]

        app_state['database_path'] = db
        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': M0(), 'embedder': 'x'}}
        client = TestClient(app)
        r = client.post('/predict', json={'ticker':'CL','horizon':'1d','context':{}})
        assert r.status_code == 200
        js = r.json()
        assert abs(js['confidence'] - 0.95) < 1e-6

        # model returns large value to force lower clamp
        class M1:
            def predict(self, X):
                return [1.0]

        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': M1(), 'embedder': 'x'}}
        r2 = client.post('/predict', json={'ticker':'CL','horizon':'1d','context':{}})
        assert r2.status_code == 200
        js2 = r2.json()
        assert abs(js2['confidence'] - 0.1) < 1e-6
    finally:
        os.unlink(db)
