"""
Persistence helpers for signal-driven backtest artifacts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from backend.domain.trading import OrderIntent, TargetAllocation


class BacktestRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path

    def persist_signals(
        self,
        backtest_id: str,
        signals: Iterable[TargetAllocation],
        *,
        backtest_run_id: Optional[int] = None,
    ) -> None:
        rows = [
            (
                backtest_id,
                backtest_run_id,
                s.timestamp.isoformat(),
                s.ticker,
                float(s.target_pct),
                s.reason,
                float(s.confidence),
                json.dumps(s.metadata or {}),
            )
            for s in signals
        ]
        if not rows:
            return
        conn = sqlite3.connect(self.database_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executemany(
                """
                INSERT INTO strategy_signals
                (backtest_id, backtest_run_id, signal_time, ticker, target_pct, reason, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def persist_order_intents(
        self,
        backtest_id: str,
        intents: Iterable[OrderIntent],
        *,
        backtest_run_id: Optional[int] = None,
    ) -> None:
        rows = [
            (
                backtest_id,
                backtest_run_id,
                i.timestamp.isoformat(),
                i.ticker,
                i.side,
                float(i.notional_delta),
                i.reason,
                json.dumps(i.metadata or {}),
            )
            for i in intents
        ]
        if not rows:
            return
        conn = sqlite3.connect(self.database_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executemany(
                """
                INSERT INTO order_intents
                (backtest_id, backtest_run_id, intent_time, ticker, side, notional_delta, reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def persist_order_fills(
        self,
        backtest_id: str,
        fills: List[Dict[str, Any]],
        *,
        backtest_run_id: Optional[int] = None,
    ) -> None:
        if not fills:
            return
        conn = sqlite3.connect(self.database_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executemany(
                """
                INSERT INTO order_fills
                (backtest_id, backtest_run_id, fill_time, ticker, side, quantity, fill_price, fees, slippage, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        backtest_id,
                        backtest_run_id,
                        f.get("fill_time", datetime.utcnow().isoformat()),
                        f.get("ticker"),
                        f.get("side"),
                        int(f.get("quantity", 0)),
                        float(f.get("fill_price", 0.0)),
                        float(f.get("fees", 0.0)),
                        float(f.get("slippage", 0.0)),
                        json.dumps(f.get("metadata", {})),
                    )
                    for f in fills
                ],
            )
            conn.commit()
        finally:
            conn.close()
