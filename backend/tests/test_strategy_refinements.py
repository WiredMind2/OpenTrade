import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from backend.strategies.moving_average import MovingAverageStrategy
from backend.strategies.recursive_forecast_strategy import RecursiveForecastStandaloneStrategy


def _seed_price_series(db_path, ticker, closes, start_date=datetime(2026, 1, 1)):
    conn = sqlite3.connect(db_path)
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
    for idx, close in enumerate(closes):
        day = (start_date + timedelta(days=idx)).date().isoformat()
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, 1000)
            """,
            (ticker, day, close, close, close, close),
        )
    conn.commit()
    conn.close()


def test_moving_average_emits_bullish_crossover_signal(tmp_path):
    db_path = tmp_path / "ma_crossover.db"
    _seed_price_series(db_path, "AAPL", [10.0, 9.0, 8.0, 9.0, 10.0])

    strategy = MovingAverageStrategy()
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        allocations = strategy.generate_target_allocations(
            parameters={"short_window": 2, "long_window": 3, "max_position_pct": 0.1},
            symbols=["AAPL"],
            as_of=datetime(2026, 1, 5),
            current_prices={"AAPL": 10.0},
        )

    assert len(allocations) == 1
    assert allocations[0].reason == "ma_bullish_cross"
    assert allocations[0].target_pct == 0.1


def test_moving_average_holds_without_new_crossover(tmp_path):
    db_path = tmp_path / "ma_hold.db"
    _seed_price_series(db_path, "AAPL", [10.0, 11.0, 12.0, 13.0, 14.0])

    strategy = MovingAverageStrategy()
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        allocations = strategy.generate_target_allocations(
            parameters={"short_window": 2, "long_window": 3, "max_position_pct": 0.1},
            symbols=["AAPL"],
            as_of=datetime(2026, 1, 5),
            current_prices={"AAPL": 14.0},
        )

    assert allocations == []


def test_recursive_forecast_uses_db_predictions_without_models(tmp_path):
    db_path = tmp_path / "recursive_db_signal.db"
    conn = sqlite3.connect(db_path)
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
    conn.execute(
        """
        INSERT INTO trading_model_predictions (ticker, dt, predicted_return, enter_prob, suggested_position_pct)
        VALUES ('AAPL', '2026-01-05', 0.02, 0.8, 1.0)
        """
    )
    conn.commit()
    conn.close()

    strategy = RecursiveForecastStandaloneStrategy()
    with patch("backend.main.app_state", {"database_path": str(db_path), "models_loaded": {}}):
        allocations = strategy.generate_target_allocations(
            parameters={"prediction_threshold": 0.001, "max_position_pct": 0.1},
            symbols=["AAPL"],
            as_of=datetime(2026, 1, 6),
            current_prices={"AAPL": 100.0},
        )

    assert len(allocations) == 1
    assert allocations[0].target_pct > 0
    assert allocations[0].metadata["source"] == "db"


def test_recursive_forecast_returns_no_model_available_when_models_missing(tmp_path):
    db_path = tmp_path / "recursive_no_model.db"
    conn = sqlite3.connect(db_path)
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
    conn.commit()
    conn.close()

    strategy = RecursiveForecastStandaloneStrategy()
    with patch("backend.main.app_state", {"database_path": str(db_path), "models_loaded": {}}):
        allocations = strategy.generate_target_allocations(
            parameters={"prediction_threshold": 0.0005, "max_position_pct": 0.1, "forecast_horizon_days": 5},
            symbols=["AAPL"],
            as_of=datetime(2026, 1, 6),
            current_prices={"AAPL": 100.0},
        )

    assert len(allocations) == 1
    assert allocations[0].reason == "no_model_available"
