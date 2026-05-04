import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.strategies.moving_average import MovingAverageStrategy
from backend.strategies.recursive_forecast_strategy import RecursiveForecastStandaloneStrategy


class _TrainRegistry:
    def get(self, name):
        if name == "moving_average":
            return MovingAverageStrategy()
        if name == "recursive_forecast":
            return RecursiveForecastStandaloneStrategy()
        return None


def _seed_optimize_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT,
          started_at TEXT,
          completed_at TEXT,
          params JSON,
          params_hash TEXT,
          variant_label TEXT,
          optimizer_mode TEXT,
          experiment_id TEXT,
          client_backtest_id TEXT,
          initial_capital REAL,
          final_value REAL,
          total_return REAL,
          annualized_return REAL,
          sharpe_ratio REAL,
          max_drawdown REAL,
          win_rate REAL,
          total_trades INTEGER,
          avg_trade_return REAL,
          volatility REAL,
          equity_curve TEXT,
          metrics JSON
        )
        """
    )
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
    for i in range(420):
        day = (start + timedelta(days=i)).date().isoformat()
        close = 100.0 + (i * 0.08) + (2.0 if i % 13 == 0 else 0.0)
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('AAPL', ?, ?, ?, ?, ?, 1000)
            """,
            (day, close, close * 1.01, close * 0.99, close),
        )
    conn.commit()
    conn.close()


def test_train_strategy_optimizes_moving_average(tmp_path):
    db_path = tmp_path / "optimize_ma.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/moving_average/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "objective": "balanced",
                "max_evals": 6,
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["strategy"] == "moving_average"
    assert body["ticker"] == "AAPL"
    assert body["evaluations_run"] == 6
    assert "best_params" in body
    assert "best_metrics" in body


def test_train_strategy_recursive_forecast_fails_without_loaded_models(tmp_path):
    db_path = tmp_path / "optimize_recursive.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry(), "models_loaded": {}}):
        res = client.post(
            "/api/strategies/recursive_forecast/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2024-07-31T00:00:00",
                "objective": "balanced",
                "max_evals": 1,
            },
        )
    assert res.status_code == 400
    body = res.json()
    assert "No recursive forecast models available" in body["detail"]
