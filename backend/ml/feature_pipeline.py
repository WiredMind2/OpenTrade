"""
Unified feature pipeline for model training and realtime inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import sqlite3


DEFAULT_FEATURE_NAMES: List[str] = [
    "avg_sentiment",
    "sentiment_volatility",
    "article_count",
    "avg_return_5d",
    "avg_return_20d",
    "return_volatility_20d",
    "volume_mean_20d",
    "volume_trend_20d",
]


@dataclass
class FeatureInput:
    ticker: str
    as_of: datetime


class FeaturePipeline:
    """Builds deterministic numeric feature vectors for a ticker."""

    schema_version = "ml_features_v1"

    def __init__(self, feature_names: Sequence[str] | None = None):
        self.feature_names = list(feature_names or DEFAULT_FEATURE_NAMES)

    def build_vector(self, conn: sqlite3.Connection, payload: FeatureInput) -> np.ndarray:
        articles = self._load_articles(conn, payload.ticker, payload.as_of)
        price_data = self._load_prices(conn, payload.ticker, payload.as_of)
        feature_map = self._compute_feature_map(articles, price_data)
        self._validate_features(feature_map)
        return np.array([feature_map[name] for name in self.feature_names], dtype=float).reshape(1, -1)

    def _load_articles(
        self,
        conn: sqlite3.Connection,
        ticker: str,
        as_of: datetime,
    ) -> List[Tuple[str, str, str, float]]:
        cur = conn.cursor()
        since = (as_of - timedelta(days=7)).date().isoformat()
        columns = {row[1] for row in cur.execute("PRAGMA table_info(articles)").fetchall()}
        timestamp_col = "canonical_timestamp" if "canonical_timestamp" in columns else "published_at"
        try:
            cur.execute(
                """
                SELECT a.title, a.content, a.{timestamp_col}, a.sentiment_score
                FROM articles a
                JOIN article_ticker at ON at.article_id = a.id
                WHERE at.ticker = ? AND a.{timestamp_col} >= ?
                ORDER BY a.{timestamp_col} DESC
                LIMIT 50
                """.format(timestamp_col=timestamp_col),
                (ticker.upper(), since),
            )
            return cur.fetchall()
        except sqlite3.OperationalError:
            cur.execute(
                """
                SELECT title, content, {timestamp_col}, 0.0 as sentiment_score
                FROM articles
                WHERE {timestamp_col} >= ?
                ORDER BY {timestamp_col} DESC
                LIMIT 50
                """.format(timestamp_col=timestamp_col),
                (since,),
            )
            return cur.fetchall()

    def _load_prices(
        self,
        conn: sqlite3.Connection,
        ticker: str,
        as_of: datetime,
    ) -> List[Tuple[float, int]]:
        cur = conn.cursor()
        since = (as_of - timedelta(days=45)).date().isoformat()
        cur.execute(
            """
            SELECT close, volume
            FROM price_daily
            WHERE ticker = ? AND date >= ?
            ORDER BY date DESC
            LIMIT 45
            """,
            (ticker.upper(), since),
        )
        return cur.fetchall()

    def _compute_feature_map(
        self,
        articles: List[Tuple[str, str, str, float]],
        price_data: List[Tuple[float, int]],
    ) -> dict:
        sentiments = np.array([float(r[3]) for r in articles if r[3] is not None], dtype=float)
        article_count = float(len(articles))
        avg_sentiment = float(np.mean(sentiments)) if sentiments.size else 0.0
        sentiment_volatility = float(np.std(sentiments)) if sentiments.size > 1 else 0.0

        closes = np.array([float(r[0]) for r in price_data if r[0] is not None], dtype=float)
        volumes = np.array([float(r[1]) for r in price_data if r[1] is not None], dtype=float)
        if closes.size > 1:
            returns = np.diff(closes) / closes[:-1]
        else:
            returns = np.array([0.0], dtype=float)
        avg_return_5d = float(np.mean(returns[:5])) if returns.size else 0.0
        avg_return_20d = float(np.mean(returns[:20])) if returns.size else 0.0
        return_volatility_20d = float(np.std(returns[:20])) if returns.size > 1 else 0.0

        volume_mean_20d = float(np.mean(volumes[:20])) if volumes.size else 0.0
        volume_trend_20d = (
            float(np.polyfit(range(len(volumes[:20])), volumes[:20], 1)[0]) if volumes.size > 1 else 0.0
        )

        return {
            "avg_sentiment": avg_sentiment,
            "sentiment_volatility": sentiment_volatility,
            "article_count": article_count,
            "avg_return_5d": avg_return_5d,
            "avg_return_20d": avg_return_20d,
            "return_volatility_20d": return_volatility_20d,
            "volume_mean_20d": volume_mean_20d,
            "volume_trend_20d": volume_trend_20d,
        }

    def _validate_features(self, feature_map: dict) -> None:
        missing = [name for name in self.feature_names if name not in feature_map]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        for name in self.feature_names:
            val = feature_map[name]
            if np.isnan(val) or np.isinf(val):
                feature_map[name] = 0.0
