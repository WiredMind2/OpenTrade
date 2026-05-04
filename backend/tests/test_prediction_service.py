import sqlite3
from datetime import datetime, timedelta

import numpy as np

from backend.ml.feature_pipeline import FeatureInput, FeaturePipeline
from backend.ml.prediction_service import PredictionService


class DummyModel:
    def predict(self, features):
        return np.array([float(np.mean(features) * 0.01)])


def _seed_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            canonical_timestamp TEXT,
            sentiment_score REAL
        );
        CREATE TABLE article_ticker (
            article_id INTEGER,
            ticker TEXT
        );
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            close REAL,
            volume INTEGER
        );
        CREATE TABLE sentiment_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            horizon TEXT,
            predicted_return REAL,
            predicted_confidence REAL,
            produced_at TEXT,
            model TEXT,
            features_used TEXT,
            metadata TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO articles (id, title, content, canonical_timestamp, sentiment_score) VALUES (1, 't', 'c', ?, ?)",
        (datetime.utcnow().isoformat(), 0.2),
    )
    conn.execute("INSERT INTO article_ticker (article_id, ticker) VALUES (1, 'AAPL')")
    conn.execute(
        "INSERT INTO price_daily (ticker, date, close, volume) VALUES ('AAPL', ?, 100.0, 1000)",
        (datetime.utcnow().date().isoformat(),),
    )
    conn.commit()


def test_feature_pipeline_is_deterministic(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    _seed_db(conn)
    pipeline = FeaturePipeline()
    v1 = pipeline.build_vector(conn, FeatureInput(ticker="AAPL", as_of=datetime.utcnow()))
    v2 = pipeline.build_vector(conn, FeatureInput(ticker="AAPL", as_of=datetime.utcnow()))
    assert v1.shape == v2.shape
    assert np.allclose(v1, v2)
    conn.close()


def test_prediction_service_persists_result(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    _seed_db(conn)
    conn.close()
    service = PredictionService(
        database_path=str(db_path),
        models_loaded={"lightgbm_1d": {"lgbm": DummyModel()}},
    )
    result = service.predict("AAPL", "1d")
    assert result.ticker == "AAPL"
    conn2 = sqlite3.connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM sentiment_predictions").fetchone()[0]
    conn2.close()
    assert count == 1


def _seed_multi_day_prices(conn: sqlite3.Connection) -> None:
    base = datetime(2024, 1, 1)
    for i in range(10):
        day = (base + timedelta(days=i)).date().isoformat()
        close = 100.0 + float(i)
        conn.execute(
            "INSERT INTO price_daily (ticker, date, close, volume) VALUES ('AAPL', ?, ?, 1000)",
            (day, close),
        )
    conn.commit()


def test_prediction_service_historical_as_of_skips_persist_by_default(tmp_path):
    db_path = tmp_path / "hist.db"
    conn = sqlite3.connect(db_path)
    _seed_db(conn)
    _seed_multi_day_prices(conn)
    conn.close()
    service = PredictionService(
        database_path=str(db_path),
        models_loaded={"lightgbm_1d": {"lgbm": DummyModel()}},
    )
    result = service.predict("AAPL", "1d", as_of=datetime(2024, 1, 5))
    assert result.timestamp.date().isoformat() == "2024-01-05"
    assert result.metadata.get("simulation_as_of", "").startswith("2024-01-05")
    conn2 = sqlite3.connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM sentiment_predictions").fetchone()[0]
    conn2.close()
    assert count == 0


def test_prediction_service_forward_actuals_when_requested(tmp_path):
    db_path = tmp_path / "fwd.db"
    conn = sqlite3.connect(db_path)
    _seed_db(conn)
    _seed_multi_day_prices(conn)
    conn.close()
    service = PredictionService(
        database_path=str(db_path),
        models_loaded={"lightgbm_3d": {"lgbm": DummyModel()}},
    )
    result = service.predict(
        "AAPL",
        "3d",
        as_of=datetime(2024, 1, 2),
        include_forward_actuals=True,
    )
    fwd = result.metadata.get("forward_actual_closes") or []
    assert len(fwd) == 3
    assert [row["close"] for row in fwd] == [102.0, 103.0, 104.0]
