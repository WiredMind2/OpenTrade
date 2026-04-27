"""
Chronological data source for forecasting.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DataSource:
    database_path: str

    def load_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        query = """
            SELECT date, open, high, low, close, volume
            FROM price_daily
            WHERE ticker = ?
        """
        params = [ticker.upper()]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"
        with sqlite3.connect(self.database_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
