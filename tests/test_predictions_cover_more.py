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
    # Final attempt to surface any persistent issue
    os.unlink(path)


def make_db_with_price_and_predicted_conf(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS price_daily (
        ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adjusted_close REAL, volume INTEGER, PRIMARY KEY(ticker,date)
    );
    CREATE TABLE IF NOT EXISTS sentiment_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, article_id INTEGER, ticker TEXT, model TEXT, horizon TEXT,
        predicted_return REAL, predicted_confidence REAL, produced_at TEXT
    );
    ''')
    # insert price
    d = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
    conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                 ('TST', d, 5,6,4,5.5,5.5,100))
    # insert prediction using predicted_confidence column
    conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,predicted_confidence,produced_at,model) VALUES (?,?,?,?,?,?)',
                 ('TST', '1d', 0.2, 0.77, d + 'T12:00:00', 'lightgbm_1d'))
    conn.commit()
    conn.close()


def test_chart_data_reads_predicted_confidence():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        make_db_with_price_and_predicted_conf(db)
        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/predictions/chart-data/TST?include_raw=true')
            assert r.status_code == 200
            body = r.json()
            assert 'raw_predictions' in body
            rp = body['raw_predictions'][0]
            assert abs(rp['confidence'] - 0.77) < 1e-6
    finally:
        _safe_unlink(db)


def test_predict_with_price_data_only_uses_model_and_volume_features():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        conn = sqlite3.connect(db)
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS price_daily (
            ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, adjusted_close REAL, volume INTEGER, PRIMARY KEY(ticker,date)
        );
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, title TEXT, content TEXT, published_at TEXT, sentiment_score REAL
        );
        CREATE TABLE IF NOT EXISTS sentiment_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, article_id INTEGER, ticker TEXT, model TEXT, horizon TEXT, predicted_return REAL, confidence REAL, produced_at TEXT, features_used TEXT, metadata TEXT
        );
        ''')
        # insert 3 price rows
        for i in range(3):
            d = (datetime.utcnow().date() - timedelta(days=i)).isoformat()
            conn.execute('INSERT INTO price_daily (ticker,date,open,high,low,close,adjusted_close,volume) VALUES (?,?,?,?,?,?,?,?)',
                         ('PRC', d, 10+i, 11+i, 9+i, 10.5+i, 10.5+i, 1000+i*10))
        conn.commit()
        conn.close()

        class M:
            def predict(self, X):
                return [0.123]

        app_state['database_path'] = db
        app_state['models_loaded'] = {'lightgbm_1d': {'lgbm': M(), 'embedder': 'x'}}
        with TestClient(app) as client:
            r = client.post('/predict', json={'ticker':'PRC','horizon':'1d','context':{}})
            assert r.status_code == 200
            js = r.json()
            assert abs(js['predicted_return'] - 0.123) < 1e-6
    finally:
        _safe_unlink(db)


def test_trading_predictions_table_missing_returns_empty_list(monkeypatch):
    # Create DB without trading_model_predictions table
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        conn = sqlite3.connect(db)
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS price_daily (ticker TEXT, date TEXT);
        ''')
        conn.commit()
        conn.close()

        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/trading/predictions')
            assert r.status_code == 200
            assert r.json() == []
    finally:
        _safe_unlink(db)
