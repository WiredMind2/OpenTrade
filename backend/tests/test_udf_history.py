import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


def _seed_udf_tables(db_path: str) -> datetime:
    base_dt = datetime(2026, 4, 26, 10, 0, 0)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE tickers (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                sector TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE price_minute (
                ticker TEXT NOT NULL,
                dt TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, dt)
            )
            """
        )

        cur.execute(
            "INSERT INTO tickers (ticker, name, exchange, sector) VALUES (?, ?, ?, ?)",
            ("AAPL", "Apple Inc.", "NASDAQ", "Technology"),
        )

        rows = []
        for i in range(30):
            ts = base_dt + timedelta(minutes=i)
            px = 100.0 + (i * 0.1)
            rows.append(
                (
                    "AAPL",
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    px,
                    px + 0.2,
                    px - 0.2,
                    px + 0.05,
                    1000 + i,
                )
            )

        cur.executemany(
            """
            INSERT INTO price_minute (ticker, dt, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    return base_dt


def test_udf_history_countback_uses_prior_bars_for_5m_resolution(tmp_path):
    db_path = tmp_path / "udf_history_countback.db"
    base_dt = _seed_udf_tables(str(db_path))
    client = TestClient(app)

    # Request a future window that has no rows; countback should still return prior bars.
    from_dt = base_dt + timedelta(hours=1)
    to_dt = from_dt + timedelta(minutes=10)

    with patch("backend.main.app_state", {"database_path": str(db_path)}), patch(
        "backend.routes.udf._maybe_refresh_latest_data", return_value=False
    ) as mock_refresh, patch("backend.routes.udf.fetch_external_data", return_value=False) as mock_fetch:
        response = client.get(
            "/udf/history",
            params={
                "symbol": "AAPL",
                "resolution": "5",
                "from_ts": int(from_dt.timestamp()),
                "to_ts": int(to_dt.timestamp()),
                "countback": 3,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["s"] == "ok"
    assert len(payload["t"]) == 3
    assert len(payload["o"]) == 3
    assert len(payload["h"]) == 3
    assert len(payload["l"]) == 3
    assert len(payload["c"]) == 3
    assert len(payload["v"]) == 3
    assert payload["t"] == sorted(payload["t"])
    mock_refresh.assert_called_once()
    mock_fetch.assert_not_called()

