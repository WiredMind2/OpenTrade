import asyncio
import json
import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, patch

from backend.routes.backtest_engine import run_backtest_background


def _init_backtest_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE backtest_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT,
          started_at TEXT,
          completed_at TEXT,
          params JSON,
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
        );

        CREATE TABLE trading_model_predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ticker TEXT,
          dt TEXT
        );

        CREATE TABLE price_daily (
          ticker TEXT,
          date TEXT,
          open REAL,
          high REAL,
          low REAL,
          close REAL,
          volume INTEGER
        );
        """
    )
    conn.commit()


class _DummyStrategy:
    def create_backtrader_strategy(self, parameters):
        return object


class _DummyRegistry:
    def get(self, strategy_name):
        return _DummyStrategy()


class _EmptyRegistry:
    def get(self, strategy_name):
        return None


class _KeyErrorStrategy:
    def create_backtrader_strategy(self, parameters):
        raise KeyError("moving_average")


class _KeyErrorRegistry:
    def get(self, strategy_name):
        return _KeyErrorStrategy()


class _FakeBroker:
    def setcash(self, value):
        return None

    def setcommission(self, commission):
        return None

    def getvalue(self):
        return 100000.0


class _FakeCerebroNoResults:
    def __init__(self):
        self.broker = _FakeBroker()

    def addstrategy(self, strategy_class, **parameters):
        return None

    def adddata(self, data):
        return None

    def addanalyzer(self, analyzer, _name):
        return None

    def run(self):
        return []


class _LiveRegistry:
    def get(self, strategy_name):
        if strategy_name == "moving_average":
            from backend.strategies.moving_average import MovingAverageStrategy

            return MovingAverageStrategy()
        return None


class _LiveRecursiveRegistry:
    def get(self, strategy_name):
        if strategy_name == "recursive_forecast":
            from backend.strategies.recursive_forecast_strategy import RecursiveForecastStandaloneStrategy

            return RecursiveForecastStandaloneStrategy()
        return None


def test_run_backtest_background_persists_failure_when_no_data(tmp_path):
    db_path = tmp_path / "backtest_engine_no_data.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _DummyRegistry(),
    }

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_no_data",
                strategy_name="moving_average",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id, name, metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    assert isinstance(row[0], int)
    assert row[1] == "moving_average"
    metrics = json.loads(row[2])
    assert metrics["status"] == "failed"
    assert metrics["backtest_id"] == "bt_test_no_data"
    assert "No market data available" in metrics["error"]


def test_run_backtest_background_handles_empty_backtrader_results(tmp_path):
    db_path = tmp_path / "backtest_engine_empty_results.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    conn.execute(
        "INSERT INTO trading_model_predictions (ticker, dt) VALUES ('AAPL', '2023-01-10')"
    )
    for day in range(1, 36):
        date = f"2023-01-{day:02d}" if day <= 31 else f"2023-02-{day - 31:02d}"
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('AAPL', ?, 100.0, 101.0, 99.0, 100.5, 1000)
            """,
            (date,),
        )
    conn.commit()
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _DummyRegistry(),
    }

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()), patch(
        "backend.routes.backtest_engine.bt.Cerebro", new=_FakeCerebroNoResults
    ), patch("backend.routes.backtest_engine.bt.feeds.PandasData", new=lambda **kwargs: object()):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_empty_results",
                strategy_name="moving_average",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name, metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "moving_average"
    metrics = json.loads(row[1])
    assert metrics["status"] == "failed"
    assert metrics["backtest_id"] == "bt_test_empty_results"
    assert "no strategy results" in metrics["error"].lower()


def test_run_backtest_background_falls_back_to_price_daily_tickers(tmp_path):
    db_path = tmp_path / "backtest_engine_price_daily_fallback.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    # No trading_model_predictions on purpose; fallback should still find ticker from price_daily.
    for day in range(1, 36):
        date = f"2023-01-{day:02d}" if day <= 31 else f"2023-02-{day - 31:02d}"
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('MSFT', ?, 250.0, 252.0, 249.0, 251.0, 1200)
            """,
            (date,),
        )
    conn.commit()
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _DummyRegistry(),
    }

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()), patch(
        "backend.routes.backtest_engine.bt.Cerebro", new=_FakeCerebroNoResults
    ), patch("backend.routes.backtest_engine.bt.feeds.PandasData", new=lambda **kwargs: object()):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_price_daily_fallback",
                strategy_name="moving_average",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    metrics = json.loads(row[0])
    # Fallback should reach Backtrader run path; failure then comes from fake empty results.
    assert "no strategy results" in metrics["error"].lower()


def test_run_backtest_background_handles_unknown_strategy(tmp_path):
    db_path = tmp_path / "backtest_engine_unknown_strategy.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _EmptyRegistry(),
    }

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_unknown_strategy",
                strategy_name="sentiment_momentum",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name, metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "sentiment_momentum"
    metrics = json.loads(row[1])
    assert metrics["status"] == "failed"
    assert metrics["backtest_id"] == "bt_test_unknown_strategy"
    assert "not registered" in metrics["error"].lower()


def test_run_backtest_background_persists_exception_type_for_keyerror(tmp_path):
    db_path = tmp_path / "backtest_engine_keyerror.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _KeyErrorRegistry(),
    }

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_keyerror",
                strategy_name="moving_average",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name, metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "moving_average"
    metrics = json.loads(row[1])
    assert metrics["status"] == "failed"
    assert metrics["backtest_id"] == "bt_test_keyerror"
    assert metrics["error"] == "KeyError: 'moving_average'"


def test_run_backtest_background_skips_insufficient_price_history_for_indicators(tmp_path):
    db_path = tmp_path / "backtest_engine_min_bars.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)

    # Both tickers are eligible from predictions; only MSFT has enough rows for long_window=30.
    conn.execute(
        "INSERT INTO trading_model_predictions (ticker, dt) VALUES ('AAPL', '2023-01-10')"
    )
    conn.execute(
        "INSERT INTO trading_model_predictions (ticker, dt) VALUES ('MSFT', '2023-01-10')"
    )
    conn.execute(
        """
        INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
        VALUES ('AAPL', '2023-01-10', 100.0, 101.0, 99.0, 100.5, 1000)
        """
    )
    for day in range(1, 36):
        date = f"2023-01-{day:02d}" if day <= 31 else f"2023-02-{day - 31:02d}"
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('MSFT', ?, 250.0, 252.0, 249.0, 251.0, 1200)
            """,
            (date,),
        )
    conn.commit()
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _DummyRegistry(),
    }

    created_feed_names = []

    def _fake_pandas_data(**kwargs):
        created_feed_names.append(kwargs.get("name"))
        return object()

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()), patch(
        "backend.routes.backtest_engine.bt.Cerebro", new=_FakeCerebroNoResults
    ), patch("backend.routes.backtest_engine.bt.feeds.PandasData", new=_fake_pandas_data), patch(
        "backend.routes.backtest_engine._refresh_daily_prices_for_backtest", return_value=False
    ):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_min_bars",
                strategy_name="moving_average",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    # Regression: short history ticker should be skipped to avoid Backtrader SMA index errors.
    assert created_feed_names == ["MSFT"]


def test_run_backtest_background_auto_refreshes_missing_daily_bars(tmp_path):
    db_path = tmp_path / "backtest_engine_auto_refresh.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    conn.execute(
        "INSERT INTO trading_model_predictions (ticker, dt) VALUES ('AAPL', '2023-01-10')"
    )
    for day in range(1, 6):
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('AAPL', ?, 100.0, 101.0, 99.0, 100.5, 1000)
            """,
            (f"2023-01-0{day}",),
        )
    conn.commit()
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _DummyRegistry(),
    }

    created_feed_names = []
    refresh_calls = []

    def _fake_pandas_data(**kwargs):
        created_feed_names.append(kwargs.get("name"))
        return object()

    def _fake_refresh(db_path_arg, ticker, fetch_start, fetch_end):
        refresh_calls.append((ticker, fetch_start, fetch_end))
        conn_local = sqlite3.connect(db_path_arg)
        for day in range(6, 36):
            date = f"2023-01-{day:02d}" if day <= 31 else f"2023-02-{day - 31:02d}"
            conn_local.execute(
                """
                INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, 100.0, 101.0, 99.0, 100.5, 1000)
                """,
                (ticker, date),
            )
        conn_local.commit()
        conn_local.close()
        return True

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()), patch(
        "backend.routes.backtest_engine.bt.Cerebro", new=_FakeCerebroNoResults
    ), patch("backend.routes.backtest_engine.bt.feeds.PandasData", new=_fake_pandas_data), patch(
        "backend.routes.backtest_engine._refresh_daily_prices_for_backtest",
        side_effect=_fake_refresh,
    ):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_test_auto_refresh",
                strategy_name="moving_average",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    assert refresh_calls, "Expected auto-refresh to be attempted for insufficient bars"
    assert created_feed_names == ["AAPL"]


def test_run_backtest_background_live_moving_average_minimal_patching(tmp_path):
    db_path = tmp_path / "backtest_engine_live_ma.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    conn.execute(
        "INSERT INTO trading_model_predictions (ticker, dt) VALUES ('AAPL', '2026-04-01')"
    )
    for day in range(1, 46):
        date = f"2026-03-{day:02d}" if day <= 31 else f"2026-04-{day - 31:02d}"
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('AAPL', ?, ?, ?, ?, ?, ?)
            """,
            (date, 100.0 + day, 101.0 + day, 99.0 + day, 100.5 + day, 1000 + day),
        )
    conn.commit()
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _LiveRegistry(),
    }

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()), patch(
        "backend.routes.backtest_engine._refresh_daily_prices_for_backtest", return_value=False
    ):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_live_minimal_patch",
                strategy_name="moving_average",
                start_date=datetime(2026, 4, 1),
                end_date=datetime(2026, 5, 1),
                initial_capital=100000.0,
                parameters={},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    metrics = json.loads(row[0])
    assert metrics["backtest_id"] == "bt_live_minimal_patch"
    assert metrics["status"] == "completed"


def test_run_backtest_background_live_recursive_forecast_horizon_fallback(tmp_path):
    db_path = tmp_path / "backtest_engine_live_recursive.db"
    conn = sqlite3.connect(db_path)
    _init_backtest_tables(conn)
    for day in range(1, 46):
        date = f"2026-03-{day:02d}" if day <= 31 else f"2026-04-{day - 31:02d}"
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('AAPL', ?, ?, ?, ?, ?, ?)
            """,
            (date, 100.0 + day, 101.0 + day, 99.0 + day, 100.5 + day, 1000 + day),
        )
    conn.commit()
    conn.close()

    app_state = {
        "database_path": str(db_path),
        "strategy_registry": _LiveRecursiveRegistry(),
    }

    class _Prediction:
        def __init__(self, horizon):
            self.predicted_return = 0.01
            self.metadata = {"predicted_path_targets": [0.002, 0.004, 0.01]}
            self.model = type("M", (), {"model_name": f"lightgbm_{horizon}"})()

    call_horizons = []

    def _predict_side_effect(self, ticker, horizon, **kwargs):
        call_horizons.append(horizon)
        if horizon == "7d":
            raise KeyError("No model available for horizon 7d")
        return _Prediction(horizon)

    with patch("backend.routes.backtest_engine.broadcast_websocket_message", new=AsyncMock()), patch(
        "backend.routes.backtest_engine._refresh_daily_prices_for_backtest", return_value=False
    ), patch("backend.strategies.recursive_forecast.PredictionService.predict", new=_predict_side_effect), patch(
        "backend.main.app_state", {"database_path": str(db_path), "models_loaded": {}}
    ):
        asyncio.run(
            run_backtest_background(
                backtest_id="bt_live_recursive_fallback",
                strategy_name="recursive_forecast",
                start_date=datetime(2026, 4, 1),
                end_date=datetime(2026, 5, 1),
                initial_capital=100000.0,
                parameters={"forecast_horizon_days": 5},
                app_state=app_state,
            )
        )

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT metrics FROM backtest_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    metrics = json.loads(row[0])
    assert metrics["status"] == "completed"
    assert metrics["backtest_id"] == "bt_live_recursive_fallback"
    assert "7d" in call_horizons
    assert "3d" in call_horizons or "1d" in call_horizons
