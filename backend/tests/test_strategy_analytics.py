import json
import sqlite3
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app
from backend.utils.backtest_variants import compute_params_hash


def _seed_analytics_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE backtest_runs (
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

    params_a = {"short_window": 5, "long_window": 30, "max_position_pct": 0.1}
    hash_a = compute_params_hash(params_a)
    conn.execute(
        """
        INSERT INTO backtest_runs (
            name, params, params_hash, initial_capital, final_value, equity_curve, metrics, completed_at,
            total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
            avg_trade_return, volatility
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "moving_average",
            json.dumps(params_a),
            hash_a,
            100000.0,
            100500.0,
            '[{"date":"2024-01-01","value":100000},{"date":"2024-01-02","value":101000},{"date":"2024-01-03","value":100500}]',
            '{"status":"completed"}',
            "2024-01-03",
            0.005,
            0.01,
            0.5,
            0.02,
            0.5,
            2,
            35.0,
            0.15,
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


def test_strategy_variant_summary_and_timeseries(tmp_path):
    db_path = tmp_path / "variants.db"
    conn = sqlite3.connect(db_path)
    _seed_analytics_db(conn)
    params_b = {"short_window": 10, "long_window": 50, "max_position_pct": 0.08}
    hash_b = compute_params_hash(params_b)
    conn.execute(
        """
        INSERT INTO backtest_runs (
            name, params, params_hash, initial_capital, final_value, equity_curve, metrics, completed_at,
            total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate, total_trades,
            avg_trade_return, volatility
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "moving_average",
            json.dumps(params_b),
            hash_b,
            100000.0,
            110000.0,
            '[{"date":"2024-01-01","value":100000},{"date":"2024-01-02","value":105000},{"date":"2024-01-03","value":110000}]',
            '{"status":"completed"}',
            "2024-01-03",
            0.10,
            0.12,
            1.2,
            0.05,
            0.6,
            3,
            40.0,
            0.12,
        ),
    )
    conn.commit()
    conn.close()

    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        res = client.get(
            "/api/strategy-analytics/variants/summary",
            params={"strategy": "moving_average", "objective": "return", "top_n": 5},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["strategy"] == "moving_average"
        assert len(body["variants"]) == 2
        # Higher return variant first
        assert body["variants"][0]["total_return"] >= body["variants"][1]["total_return"]
        hashes = ",".join(v["params_hash"] for v in body["variants"])
        ts = client.get(
            "/api/strategy-analytics/variants/timeseries",
            params={
                "strategy": "moving_average",
                "params_hashes": hashes,
                "preset": "MAX",
                "granularity": "daily",
                "objective": "return",
            },
        )
        assert ts.status_code == 200
        tsb = ts.json()
        assert len(tsb["variant_series"]) == 2


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


def test_strategy_variant_distribution(tmp_path):
    db_path = tmp_path / "analytics.db"
    conn = sqlite3.connect(db_path)
    _seed_analytics_db(conn)
    conn.close()

    ph = compute_params_hash({"short_window": 5, "long_window": 30, "max_position_pct": 0.1})
    client = TestClient(app)
    with patch("backend.main.app_state", {"database_path": str(db_path)}):
        res = client.get(
            "/api/strategy-analytics/variants/distributions/moving_average",
            params={"params_hash": ph},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["strategy"].startswith("moving_average:")
    assert isinstance(body["returns_histogram"], list)
    assert len(body["returns_histogram"]) >= 1
