"""
Storage helpers for ML metadata and operational logging.
"""

import sqlite3


def ensure_ml_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS ml_model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            horizon TEXT NOT NULL,
            feature_schema_version TEXT,
            metrics JSON,
            artifact_path TEXT,
            is_active INTEGER DEFAULT 0,
            trained_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ml_registry_horizon_active ON ml_model_registry(horizon, is_active);

        CREATE TABLE IF NOT EXISTS ml_run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_type TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata JSON,
            started_at TEXT DEFAULT (datetime('now')),
            finished_at TEXT
        );
        """
    )

    cur.execute("PRAGMA table_info('sentiment_predictions')")
    cols = [r[1] for r in cur.fetchall()]
    if "model_version" not in cols:
        cur.execute("ALTER TABLE sentiment_predictions ADD COLUMN model_version TEXT")
    if "feature_schema_version" not in cols:
        cur.execute("ALTER TABLE sentiment_predictions ADD COLUMN feature_schema_version TEXT")
    if "prediction_latency_ms" not in cols:
        cur.execute("ALTER TABLE sentiment_predictions ADD COLUMN prediction_latency_ms REAL")
    conn.commit()
