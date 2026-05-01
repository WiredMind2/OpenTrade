import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.strategies.moving_average import MovingAverageStrategy
from backend.strategies.recursive_forecast_strategy import RecursiveForecastStandaloneStrategy


class _Registry:
    def get(self, name):
        if name == "moving_average":
            return MovingAverageStrategy()
        if name == "recursive_forecast":
            return RecursiveForecastStandaloneStrategy()
        return None


def _seed_db(path, include_predictions=False):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE price_daily (
          ticker TEXT,
          date TEXT,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          volume INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE trading_model_predictions (
          ticker TEXT,
          dt TEXT,
          predicted_return REAL,
          enter_prob REAL,
          suggested_position_pct REAL
        )
        """
    )
    start = datetime(2024, 1, 1)
    for i in range(220):
        day = (start + timedelta(days=i)).date().isoformat()
        close = 100 + i * 0.1
        conn.execute(
            "INSERT INTO price_daily (ticker, date, open, high, low, close, volume) VALUES ('AAPL', ?, ?, ?, ?, ?, 1000)",
            (day, close, close * 1.01, close * 0.99, close),
        )
        if include_predictions:
            conn.execute(
                "INSERT INTO trading_model_predictions (ticker, dt, predicted_return, enter_prob, suggested_position_pct) VALUES ('AAPL', ?, 0.01, 0.6, 0.1)",
                (day,),
            )
    conn.commit()
    conn.close()


def test_preflight_recursive_forecast_blocks_without_predictions(tmp_path):
    db_path = tmp_path / "preflight_no_preds.db"
    _seed_db(db_path, include_predictions=False)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _Registry()}):
        res = client.post(
            "/api/strategies/recursive_forecast/preflight",
            json={
                "ticker": "AAPL",
                "start_date": "2024-02-01T00:00:00",
                "end_date": "2024-06-01T00:00:00",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["ready"] is False
    assert any(i["code"] == "PREDICTION_GAP" for i in body["issues"])


def test_preflight_moving_average_ready_with_history(tmp_path):
    db_path = tmp_path / "preflight_ma.db"
    _seed_db(db_path, include_predictions=False)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _Registry()}):
        res = client.post(
            "/api/strategies/moving_average/preflight",
            json={
                "ticker": "AAPL",
                "start_date": "2024-02-01T00:00:00",
                "end_date": "2024-06-01T00:00:00",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["ready"] is True
