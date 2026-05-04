"""
Background news ingestion: at most once per UTC day while the API process runs.

Uses ``app_kv`` to remember the last successful run so rapid restarts do not
hammer NewsAPI. Actual fetch uses ``backend.scripts.ingest_news``.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.logging_config import get_component_logger

logger = get_component_logger(__file__)

APP_KV_LAST_SUCCESS_KEY = "news_auto_ingest_last_success_utc"
DEFAULT_INTERVAL_SEC = 86400
DEFAULT_QUERY = "stock OR company OR earnings"


def _ensure_app_kv(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_kv (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
    )


def _get_last_success_utc(conn: sqlite3.Connection) -> Optional[str]:
    _ensure_app_kv(conn)
    row = conn.execute(
        "SELECT value FROM app_kv WHERE key = ?", (APP_KV_LAST_SUCCESS_KEY,)
    ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _set_last_success_utc(conn: sqlite3.Connection, iso_utc: str) -> None:
    _ensure_app_kv(conn)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    conn.execute(
        """
        INSERT INTO app_kv (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (APP_KV_LAST_SUCCESS_KEY, iso_utc, now),
    )
    conn.commit()


def _seconds_until_next_run(last_success_iso: Optional[str], interval_sec: int) -> float:
    if not last_success_iso or not last_success_iso.strip():
        return 0.0
    raw = last_success_iso.strip().replace("Z", "+00:00")
    try:
        last = datetime.fromisoformat(raw)
    except ValueError:
        return 0.0
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - last).total_seconds()
    return max(0.0, float(interval_sec) - elapsed)


def _run_ingest_sync(database_path: str, *, query: str, api_key: str, lookback_days: int = 2) -> None:
    from backend.scripts import ingest_news

    from_dt = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    ingest_news.ingest_news_data(
        db_path=database_path,
        query=query,
        from_dt=from_dt,
        api_key=api_key,
    )


async def daily_news_auto_ingest_worker(
    database_path: str,
    *,
    api_key: str,
    interval_sec: int = DEFAULT_INTERVAL_SEC,
    query: str = DEFAULT_QUERY,
) -> None:
    """
    Loop: wait until ``interval_sec`` since last successful ingest, then fetch recent
    headlines and store them. Blocks the event loop only via ``asyncio.to_thread``
    around the synchronous NewsAPI + SQLite path.
    """
    logger.info(
        "News auto-ingest worker started (interval=%ss, query=%r)",
        interval_sec,
        query,
    )
    while True:
        try:
            conn = sqlite3.connect(database_path)
            try:
                conn.execute("PRAGMA foreign_keys = ON;")
                last = _get_last_success_utc(conn)
            finally:
                conn.close()

            wait_sec = _seconds_until_next_run(last, interval_sec)
            if wait_sec > 0:
                logger.info("News auto-ingest: next run in %.0f seconds", wait_sec)
                await asyncio.sleep(wait_sec)

            await asyncio.to_thread(_run_ingest_sync, database_path, query=query, api_key=api_key)

            conn2 = sqlite3.connect(database_path)
            try:
                conn2.execute("PRAGMA foreign_keys = ON;")
                _set_last_success_utc(conn2, datetime.now(timezone.utc).replace(microsecond=0).isoformat())
            finally:
                conn2.close()

            logger.info("News auto-ingest: completed successfully")

        except asyncio.CancelledError:
            logger.info("News auto-ingest worker cancelled")
            raise
        except ModuleNotFoundError as e:
            logger.warning(
                "News auto-ingest skipped (optional dependency): %s. "
                "Install newsapi-python to enable ingestion.",
                e,
            )
            await asyncio.sleep(interval_sec)
        except Exception:
            logger.exception("News auto-ingest failed; will retry after interval")
            await asyncio.sleep(min(3600, interval_sec))


def parse_interval_sec() -> int:
    raw = os.getenv("NEWS_AUTO_INGEST_INTERVAL_SEC", str(DEFAULT_INTERVAL_SEC))
    try:
        v = int(raw)
        return max(300, v)
    except ValueError:
        return DEFAULT_INTERVAL_SEC


def parse_query() -> str:
    return os.getenv("NEWS_INGEST_QUERY", DEFAULT_QUERY).strip() or DEFAULT_QUERY


def news_auto_ingest_disabled() -> bool:
    return os.getenv("NEWS_AUTO_INGEST", "1").strip().lower() in ("0", "false", "no", "off")
