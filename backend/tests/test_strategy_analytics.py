import sqlite3
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app


def _seed_analytics_db(conn: sqlite3.Connection) -> None:
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

        CREATE TABLE trades (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          backtest_run_id INTEGER,
          ticker TEXT,
          entry_dt TEXT,
          entry_price REAL,
          exit_dt TEXT,
          exit_price REAL,
          quantity INTEGER,
          position_pct REAL,
          fees REAL,
          slippage REAL,
          pnl REAL
        );

        CREATE TABLE portfolio_snapshots (
          backtest_run_id INTEGER,
          dt TEXT,
          cash REAL,
          market_value REAL,
          total_value REAL,
          exposure REAL,
          positions_json JSON
        );

        CREATE TABLE price_daily (
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

    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
        [
            ("SPY", "2024-01-01", 100.0),
            ("SPY", "2024-01-02", 101.0),
            ("SPY", "2024-01-03", 100.5),
        ],
    )

    conn.execute(
        """
        INSERT INTO backtest_runs (name, initial_capital, equity_curve, metrics, completed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "moving_average",
            100000.0,
            '[{"date":"2024-01-01","value":100000},{"date":"2024-01-02","value":101000},{"date":"2024-01-03","value":100500}]',
            '{"status":"completed"}',
            "2024-01-03",
        ),
    )
    run_id = conn.execute("SELECT id FROM backtest_runs WHERE name = 'moving_average'").fetchone()[0]

    conn.executemany(
        "INSERT INTO trades (backtest_run_id, ticker, entry_dt, exit_dt, pnl) VALUES (?, ?, ?, ?, ?)",
        [
            (run_id, "AAPL", "2024-01-01", "2024-01-02", 120.0),
            (run_id, "MSFT", "2024-01-02", "2024-01-03", -50.0),
        ],
    )

    # Portfolio snapshots should be usable as a fallback for equity curves if needed
    conn.executemany(
        "INSERT INTO portfolio_snapshots (backtest_run_id, dt, total_value) VALUES (?, ?, ?)",
        [
            (run_id, "2024-01-01", 100000.0),
            (run_id, "2024-01-02", 101000.0),
            (run_id, "2024-01-03", 100500.0),
        ],
    )
    conn.commit()


def test_strategy_analytics_filters(tmp_path):
    db_path = tmp_path / "analytics.db"
    conn = sqlite3.connect(db_path)
    _seed_analytics_db(conn)
    conn.close()

    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        res = client.get("/api/strategy-analytics/filters")
    assert res.status_code == 200
    body = res.json()
    assert "moving_average" in body["strategies"]
    assert "SPY" in body["benchmarks"]
    assert "MAX" in body["available_presets"]


def test_strategy_analytics_summary_and_timeseries(tmp_path):
    db_path = tmp_path / "analytics.db"
    conn = sqlite3.connect(db_path)
    _seed_analytics_db(conn)
    conn.close()

    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        res = client.get(
            "/api/strategy-analytics/summary",
            params={"strategies": ["moving_average"], "benchmark_ticker": "SPY", "preset": "MAX", "granularity": "daily"},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["benchmark_ticker"] == "SPY"
        assert len(body["metrics"]) == 1
        assert body["metrics"][0]["strategy"] == "moving_average"
        assert body["metrics"][0]["total_trades"] == 2

        ts = client.get(
            "/api/strategy-analytics/timeseries/moving_average",
            params={"benchmark_ticker": "SPY", "preset": "MAX", "granularity": "daily", "rolling_window": 5},
        )
        assert ts.status_code == 200
        ts_body = ts.json()
        assert ts_body["strategy"] == "moving_average"
        assert len(ts_body["points"]) >= 2
        assert len(ts_body["benchmark_points"]) >= 2


def test_strategy_analytics_distributions(tmp_path):
    db_path = tmp_path / "analytics.db"
    conn = sqlite3.connect(db_path)
    _seed_analytics_db(conn)
    conn.close()

    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        res = client.get("/api/strategy-analytics/distributions/moving_average")
    assert res.status_code == 200
    body = res.json()
    assert body["strategy"] == "moving_average"
    assert isinstance(body["returns_histogram"], list)
    assert isinstance(body["trade_pnl_histogram"], list)
    assert isinstance(body["holding_period_histogram"], list)
