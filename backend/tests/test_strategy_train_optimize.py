import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.strategy_framework import strategy_supports_signal_parameter_training
from backend.strategies.cross_sectional_ls import CrossSectionalLSStrategy
from backend.strategies.mean_reversion import MeanReversionStrategy
from backend.strategies.moving_average import MovingAverageStrategy
from backend.strategies.pairs_trading import PairsTradingStrategy
from backend.strategies.recursive_forecast_strategy import RecursiveForecastStandaloneStrategy
from backend.strategies.rl_portfolio_allocator import RLPortfolioAllocatorStrategy
from backend.strategies.ts_momentum import TsMomentumStrategy
from backend.strategies.volatility_targeting import VolatilityTargetingStrategy


class _TrainRegistry:
    def get(self, name):
        if name == "moving_average":
            return MovingAverageStrategy()
        if name == "recursive_forecast":
            return RecursiveForecastStandaloneStrategy()
        if name == "mean_reversion":
            return MeanReversionStrategy()
        if name == "ts_momentum":
            return TsMomentumStrategy()
        if name == "pairs_trading":
            return PairsTradingStrategy()
        if name == "cross_sectional_ls":
            return CrossSectionalLSStrategy()
        if name == "rl_portfolio_allocator":
            return RLPortfolioAllocatorStrategy()
        return None


class _VolOnlyRegistry:
    def get(self, name):
        if name == "volatility_targeting":
            return VolatilityTargetingStrategy()
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
        close_ms = 200.0 + (i * 0.05) + (1.0 if i % 17 == 0 else 0.0)
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('MSFT', ?, ?, ?, ?, ?, 1100)
            """,
            (day, close_ms, close_ms * 1.01, close_ms * 0.99, close_ms),
        )
        close_go = 150.0 + (i * 0.06)
        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES ('GOOGL', ?, ?, ?, ?, ?, 900)
            """,
            (day, close_go, close_go * 1.01, close_go * 0.99, close_go),
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


def test_signal_parameter_training_flags():
    assert strategy_supports_signal_parameter_training("mean_reversion")
    assert strategy_supports_signal_parameter_training("pairs_trading")
    assert not strategy_supports_signal_parameter_training("volatility_targeting")


def test_train_mean_reversion_signal_optimize(tmp_path):
    db_path = tmp_path / "optimize_mr.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/mean_reversion/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "objective": "balanced",
                "max_evals": 8,
            },
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["strategy"] == "mean_reversion"
    assert body["evaluations_run"] == 8
    assert "best_params" in body


def test_train_ts_momentum_signal_optimize(tmp_path):
    db_path = tmp_path / "optimize_tsm.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/ts_momentum/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "objective": "sharpe",
                "max_evals": 9,
            },
        )
    assert res.status_code == 200, res.text
    assert res.json()["evaluations_run"] == 9


def test_train_pairs_trading_requires_pair_ticker(tmp_path):
    db_path = tmp_path / "optimize_pairs_noleg.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/pairs_trading/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "max_evals": 2,
            },
        )
    assert res.status_code == 400
    assert "pair_ticker" in res.json()["detail"].lower()


def test_train_pairs_trading_with_pair_ticker(tmp_path):
    db_path = tmp_path / "optimize_pairs.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/pairs_trading/train",
            json={
                "ticker": "AAPL",
                "pair_ticker": "MSFT",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "max_evals": 6,
            },
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["evaluations_run"] == 6
    assert body["ticker"] == "AAPL"


def test_train_cross_sectional_ls_multi_universe(tmp_path):
    db_path = tmp_path / "optimize_cs.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/cross_sectional_ls/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "universe_limit": 5,
                "max_evals": 10,
            },
        )
    assert res.status_code == 200, res.text
    assert res.json()["strategy"] == "cross_sectional_ls"


def test_train_rl_portfolio_allocator(tmp_path):
    db_path = tmp_path / "optimize_rlalloc.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _TrainRegistry()}):
        res = client.post(
            "/api/strategies/rl_portfolio_allocator/train",
            json={
                "ticker": "MSFT",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
                "universe_limit": 4,
                "max_evals": 8,
            },
        )
    assert res.status_code == 200, res.text


def test_train_volatility_targeting_not_supported(tmp_path):
    db_path = tmp_path / "optimize_vol.db"
    _seed_optimize_db(db_path)
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path), "strategy_registry": _VolOnlyRegistry()}):
        res = client.post(
            "/api/strategies/volatility_targeting/train",
            json={
                "ticker": "AAPL",
                "start_date": "2024-06-01T00:00:00",
                "end_date": "2025-03-01T00:00:00",
            },
        )
    assert res.status_code == 400
    assert "does not support training" in res.json()["detail"]
