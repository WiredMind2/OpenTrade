import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.strategies.moving_average import MovingAverageStrategy
from backend.strategies.recursive_forecast_strategy import RecursiveForecastStandaloneStrategy


class _Registry:
    def __init__(self):
        self._strategies = {
            "moving_average": MovingAverageStrategy(),
            "recursive_forecast": RecursiveForecastStandaloneStrategy(),
        }

    def get(self, name):
        return self._strategies.get(name)

    def list(self):
        return []


def _init_prices_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS price_daily (
          ticker TEXT,
          date TEXT,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          adjusted_close REAL,
          volume INTEGER
        );
        """
    )
    conn.execute(
        """
        INSERT INTO price_daily (ticker, date, open, high, low, close, adjusted_close, volume)
        VALUES ('AAPL', '2026-05-01', 100.0, 101.0, 99.0, 100.5, 100.5, 1000)
        """
    )
    conn.commit()
    conn.close()


def test_strategy_forecast_endpoint_returns_structured_forecast(tmp_path):
    db_path = tmp_path / "strategy_forecast_api.db"
    _init_prices_db(str(db_path))
    client = TestClient(app)
    payload = {"symbol": "AAPL", "params": {}, "horizon_days": 5}
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _Registry()}):
        response = client.post("/api/strategies/moving_average/forecast", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AAPL"
    assert "predicted_return" in body
    assert "predicted_path" in body


def test_strategy_signals_endpoint_returns_target_allocations(tmp_path):
    db_path = tmp_path / "strategy_signals_api.db"
    _init_prices_db(str(db_path))
    client = TestClient(app)
    payload = {
        "symbols": ["AAPL"],
        "current_prices": {"AAPL": 100.5},
        "params": {"prediction_threshold": 0.001, "max_position_pct": 0.1},
    }
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _Registry()}):
        response = client.post("/api/strategies/moving_average/signals", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "moving_average"
    assert len(body["signals"]) == 1
    assert body["signals"][0]["ticker"] == "AAPL"
    assert "target_pct" in body["signals"][0]
