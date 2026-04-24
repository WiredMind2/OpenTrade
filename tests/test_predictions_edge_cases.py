import tempfile
import os
import sqlite3
import time
import gc
import pandas as pd
from datetime import datetime
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


def test_chart_data_handles_read_sql_query_error(monkeypatch):
    # Create a DB with price table only
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        conn = sqlite3.connect(db)
        conn.execute('CREATE TABLE IF NOT EXISTS price_daily (ticker TEXT, date TEXT, close REAL)')
        conn.commit()
        conn.close()

        app_state['database_path'] = db
        # Monkeypatch pandas.read_sql_query to raise when called the first time (simulate failure)
        orig = pd.read_sql_query

        def raising_read_sql(*args, **kwargs):
            raise sqlite3.OperationalError("simulated failure")

        monkeypatch.setattr(pd, 'read_sql_query', raising_read_sql)

        with TestClient(app) as client:
            r = client.get('/predictions/chart-data/ZZZ')
            assert r.status_code == 200
            body = r.json()
            assert body['historical_data'] == []

        # restore
        monkeypatch.setattr(pd, 'read_sql_query', orig)
    finally:
        _safe_unlink(db)


def test_aggregation_preserves_null_date_entries():
    f = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    f.close()
    db = f.name
    try:
        conn = sqlite3.connect(db)
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS price_daily (ticker TEXT, date TEXT, close REAL);
        CREATE TABLE IF NOT EXISTS sentiment_predictions (id INTEGER PRIMARY KEY, ticker TEXT, horizon TEXT, predicted_return REAL, confidence REAL, produced_at TEXT, model TEXT);
        ''')
        # insert a prediction with malformed produced_at so produced_date becomes None
        conn.execute('INSERT INTO sentiment_predictions (ticker,horizon,predicted_return,confidence,produced_at,model) VALUES (?,?,?,?,?,?)',
                     ('NNN','1d', 0.1, 0.5, 'not-a-date', 'lightgbm_1d'))
        conn.commit()
        conn.close()

        app_state['database_path'] = db
        with TestClient(app) as client:
            r = client.get('/predictions/chart-data/NNN?aggregate=avg')
            assert r.status_code == 200
            body = r.json()
            # The aggregated list should retain the entry without target date (date==None)
            assert any(p.get('date') is None or p.get('date') == None for p in body.get('predictions', []))
    finally:
        _safe_unlink(db)
