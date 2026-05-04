"""
Strategy analytics API: filters + per-strategy parameter variants (Performance tab).
"""
import asyncio
import json
import math
import sqlite3
from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.logging_config import get_component_logger
from backend.schemas import (
    DistributionBucket,
    StrategyDistributionResponse,
    StrategyFilterMetadataResponse,
    StrategyTimeseriesPoint,
    TickerStrategyLeaderboard,
    TickerStrategyLeaderboardResponse,
    TickerStrategyRow,
    StrategyVariantRow,
    StrategyVariantSummaryResponse,
    StrategyVariantTimeseriesResponse,
    VariantSeriesPayload,
)
from backend.services.strategy_framework import StrategyOptimizerEngine

logger = get_component_logger(__file__)
router = APIRouter(prefix="/api/strategy-analytics", tags=["Strategy Analytics"])

PRESETS = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "YTD": -1, "MAX": None}
ANNUALIZATION = {"daily": 252, "weekly": 52, "monthly": 12}


def _parse_equity_curve(raw_curve: str | None, initial_capital: float) -> pd.DataFrame:
    if not raw_curve:
        return pd.DataFrame(columns=["date", "equity"])
    try:
        parsed = json.loads(raw_curve)
    except (TypeError, json.JSONDecodeError):
        return pd.DataFrame(columns=["date", "equity"])
    rows: List[Dict[str, Any]] = []
    for point in parsed:
        date_raw = point.get("date") or point.get("dt") or point.get("time")
        value_raw = point.get("value") or point.get("equity") or point.get("total_value")
        if date_raw is None or value_raw is None:
            continue
        try:
            dt = pd.to_datetime(date_raw, utc=True).tz_convert(None)
            rows.append({"date": dt, "equity": float(value_raw)})
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date").drop_duplicates("date")
    if df["equity"].iloc[0] == 0:
        df.loc[df.index[0], "equity"] = initial_capital
    return df


def _apply_preset(series_df: pd.DataFrame, preset: str) -> pd.DataFrame:
    if series_df.empty or preset not in PRESETS:
        return series_df
    now = pd.Timestamp.utcnow().tz_convert(None)
    if preset == "YTD":
        start = pd.Timestamp(year=now.year, month=1, day=1)
        return series_df.loc[series_df["date"] >= start]
    days = PRESETS[preset]
    if days is None:
        return series_df
    start = now - timedelta(days=days)
    return series_df.loc[series_df["date"] >= start]


def _resample_equity(series_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if series_df.empty or granularity == "daily":
        return series_df
    rule = "W-FRI" if granularity == "weekly" else "M"
    resampled = (
        series_df.set_index("date")["equity"].resample(rule).last().dropna().reset_index()
    )
    return resampled


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak) - 1.0


def _compute_rolling_ratio(returns: pd.Series, window: int, annual_factor: int, downside: bool) -> pd.Series:
    mean = returns.rolling(window).mean()
    if downside:
        std = returns.mask(returns > 0, other=0.0).rolling(window).std().replace(0, np.nan)
    else:
        std = returns.rolling(window).std().replace(0, np.nan)
    return (mean / std) * math.sqrt(annual_factor)


def _build_benchmark_series(conn: sqlite3.Connection, ticker: str, granularity: str, preset: str) -> pd.DataFrame:
    benchmark = pd.read_sql_query(
        """
        SELECT date, close
        FROM price_daily
        WHERE ticker = ?
        ORDER BY date ASC
        """,
        conn,
        params=[ticker.upper()],
    )
    if benchmark.empty:
        return pd.DataFrame(columns=["date", "equity"])
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=True).dt.tz_convert(None)
    benchmark["equity"] = benchmark["close"].astype(float)
    benchmark = benchmark[["date", "equity"]]
    benchmark = _apply_preset(benchmark, preset)
    benchmark = _resample_equity(benchmark, granularity)
    if benchmark.empty:
        return benchmark
    benchmark["equity"] = 100.0 * (benchmark["equity"] / benchmark["equity"].iloc[0])
    return benchmark


def _histogram(values: pd.Series, bins: int) -> List[DistributionBucket]:
    if values.empty:
        return []
    counts, edges = np.histogram(values, bins=bins)
    buckets: List[DistributionBucket] = []
    for i, count in enumerate(counts):
        label = f"{edges[i]:.4f}..{edges[i + 1]:.4f}"
        midpoint = float((edges[i] + edges[i + 1]) / 2.0)
        buckets.append(DistributionBucket(bucket=label, count=int(count), value=midpoint))
    return buckets


def _timeseries_to_points(df: pd.DataFrame) -> List[StrategyTimeseriesPoint]:
    points: List[StrategyTimeseriesPoint] = []
    for _, row in df.iterrows():
        points.append(
            StrategyTimeseriesPoint(
                date=row["date"].date().isoformat(),
                normalized_equity=float(row["equity"]),
                drawdown=float(row.get("drawdown", 0.0) or 0.0),
                rolling_sharpe=float(row["rolling_sharpe"]) if pd.notna(row.get("rolling_sharpe")) else None,
                rolling_sortino=float(row["rolling_sortino"]) if pd.notna(row.get("rolling_sortino")) else None,
                rolling_volatility=float(row["rolling_volatility"]) if pd.notna(row.get("rolling_volatility")) else None,
                period_return=float(row["period_return"]) if pd.notna(row.get("period_return")) else None,
            )
        )
    return points


def _backtest_runs_columns(conn: sqlite3.Connection) -> set:
    cur = conn.cursor()
    return {row[1] for row in cur.execute("PRAGMA table_info(backtest_runs)").fetchall()}


def _variant_row_score(engine: StrategyOptimizerEngine, objective: str, row: pd.Series) -> float:
    return engine.score(
        {
            "sharpe_ratio": float(row.get("sharpe_ratio") or 0.0),
            "total_return": float(row.get("total_return") or 0.0),
            "max_drawdown": float(row.get("max_drawdown") or 0.0),
        },
        objective,
    )


def _load_variant_runs_df(conn: sqlite3.Connection, strategy: str) -> pd.DataFrame:
    cols = _backtest_runs_columns(conn)
    if "params_hash" not in cols:
        return pd.DataFrame()
    df = pd.read_sql_query(
        """
        SELECT rowid AS id, name, params_hash, variant_label, params, sharpe_ratio,
               total_return, max_drawdown, win_rate, total_trades, volatility,
               annualized_return, completed_at, initial_capital, equity_curve, final_value
        FROM backtest_runs
        WHERE name = ?
          AND params_hash IS NOT NULL AND TRIM(params_hash) != ''
          AND final_value IS NOT NULL
        """,
        conn,
        params=[strategy],
    )
    return df


def _load_all_variant_runs_df(conn: sqlite3.Connection) -> pd.DataFrame:
    cols = _backtest_runs_columns(conn)
    if "params_hash" not in cols:
        return pd.DataFrame()
    return pd.read_sql_query(
        """
        SELECT rowid AS id, name, params_hash, variant_label, params, sharpe_ratio,
               total_return, max_drawdown, win_rate, total_trades, volatility,
               annualized_return, completed_at, final_value
        FROM backtest_runs
        WHERE params_hash IS NOT NULL
          AND TRIM(params_hash) != ''
          AND final_value IS NOT NULL
          AND name IS NOT NULL
          AND TRIM(name) != ''
        """,
        conn,
    )


def _params_dict(raw_params: Any) -> Dict[str, Any]:
    if isinstance(raw_params, dict):
        return raw_params
    if isinstance(raw_params, str):
        try:
            parsed = json.loads(raw_params)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _run_tickers_map(conn: sqlite3.Connection) -> Dict[int, List[str]]:
    try:
        tickers_df = pd.read_sql_query(
            """
            SELECT backtest_run_id, ticker
            FROM trades
            WHERE ticker IS NOT NULL
              AND TRIM(ticker) != ''
            """,
            conn,
        )
    except Exception:
        return {}
    out: Dict[int, List[str]] = {}
    if tickers_df.empty:
        return out
    tickers_df["backtest_run_id"] = tickers_df["backtest_run_id"].astype(int)
    tickers_df["ticker"] = tickers_df["ticker"].astype(str).str.upper()
    for run_id, g in tickers_df.groupby("backtest_run_id"):
        unique = sorted({t for t in g["ticker"].tolist() if t})
        if unique:
            out[int(run_id)] = unique
    return out


def _sync_ticker_strategy_leaderboard(
    database_path: str,
    objective: str,
    top_n: int,
    ticker: str | None,
) -> TickerStrategyLeaderboardResponse:
    obj = (objective or "balanced").lower()
    if obj not in {"sharpe", "return", "drawdown", "balanced"}:
        raise HTTPException(status_code=400, detail="objective must be sharpe, return, drawdown, or balanced")
    top_n = max(1, min(int(top_n or 5), 25))
    ticker_filter = ticker.strip().upper() if isinstance(ticker, str) and ticker.strip() else None

    conn = sqlite3.connect(database_path)
    try:
        df = _load_all_variant_runs_df(conn)
        if df.empty:
            return TickerStrategyLeaderboardResponse(objective=obj, top_n=top_n, tickers=[])

        engine = StrategyOptimizerEngine(database_path)
        working = df.copy()
        working["_score"] = working.apply(lambda r: _variant_row_score(engine, obj, r), axis=1)
        run_tickers = _run_tickers_map(conn)

        exploded: List[Dict[str, Any]] = []
        for _, row in working.iterrows():
            run_id = int(row["id"])
            row_tickers = run_tickers.get(run_id, [])
            if not row_tickers:
                fallback = _params_dict(row.get("params")).get("ticker")
                if isinstance(fallback, str) and fallback.strip():
                    row_tickers = [fallback.strip().upper()]
            for t in row_tickers:
                if ticker_filter and t != ticker_filter:
                    continue
                payload = row.to_dict()
                payload["ticker"] = t
                exploded.append(payload)

        if not exploded:
            return TickerStrategyLeaderboardResponse(objective=obj, top_n=top_n, tickers=[])

        expanded = pd.DataFrame(exploded)
        expanded["strategy"] = expanded["name"].astype(str)
        run_counts = expanded.groupby(["ticker", "strategy"]).size().rename("run_count")

        # Keep one representative row per (ticker, strategy): best objective score.
        best_rows = (
            expanded.sort_values(["_score", "id"], ascending=[False, False])
            .groupby(["ticker", "strategy"], as_index=False)
            .head(1)
            .copy()
        )
        best_rows = best_rows.sort_values(["ticker", "_score", "id"], ascending=[True, False, False])

        leaderboard: List[TickerStrategyLeaderboard] = []
        for ticker_value, g in best_rows.groupby("ticker", sort=True):
            top_rows = g.head(top_n)
            strategy_rows: List[TickerStrategyRow] = []
            for _, row in top_rows.iterrows():
                params_obj = _params_dict(row.get("params"))
                strategy = str(row.get("strategy") or row.get("name") or "")
                strategy_rows.append(
                    TickerStrategyRow(
                        ticker=str(ticker_value),
                        strategy=strategy,
                        params_hash=str(row.get("params_hash") or ""),
                        variant_label=row.get("variant_label"),
                        representative_run_id=int(row["id"]),
                        run_count=int(run_counts.get((str(ticker_value), strategy), 1)),
                        total_return=float(row.get("total_return") or 0.0),
                        annualized_return=float(row.get("annualized_return") or 0.0),
                        sharpe_ratio=float(row.get("sharpe_ratio") or 0.0),
                        max_drawdown=float(row.get("max_drawdown") or 0.0),
                        win_rate=float(row.get("win_rate") or 0.0),
                        total_trades=int(row.get("total_trades") or 0),
                        volatility=float(row.get("volatility") or 0.0),
                        params=params_obj,
                        last_completed_at=str(row["completed_at"]) if row.get("completed_at") is not None else None,
                    )
                )
            leaderboard.append(TickerStrategyLeaderboard(ticker=str(ticker_value), strategies=strategy_rows))

        return TickerStrategyLeaderboardResponse(objective=obj, top_n=top_n, tickers=leaderboard)
    finally:
        conn.close()


def _representative_run_ids_by_hash(
    df: pd.DataFrame, objective: str, engine: StrategyOptimizerEngine
) -> Dict[str, int]:
    """For each params_hash, pick the run rowid with best objective score."""
    out: Dict[str, int] = {}
    if df.empty:
        return out
    for ph, g in df.groupby("params_hash"):
        g2 = g.copy()
        g2["_score"] = g2.apply(lambda r: _variant_row_score(engine, objective, r), axis=1)
        best = g2.sort_values(["_score", "id"], ascending=[False, False]).iloc[0]
        out[str(ph)] = int(best["id"])
    return out


def _sync_variant_summary(
    database_path: str,
    strategy: str,
    objective: str,
    top_n: int,
) -> StrategyVariantSummaryResponse:
    obj = (objective or "balanced").lower()
    if obj not in {"sharpe", "return", "drawdown", "balanced"}:
        raise HTTPException(status_code=400, detail="objective must be sharpe, return, drawdown, or balanced")
    top_n = max(1, min(int(top_n or 10), 50))

    conn = sqlite3.connect(database_path)
    try:
        df = _load_variant_runs_df(conn, strategy)
        if df.empty:
            return StrategyVariantSummaryResponse(strategy=strategy, objective=obj, top_n=top_n, variants=[])

        engine = StrategyOptimizerEngine(database_path)
        id_by_hash = _representative_run_ids_by_hash(df, obj, engine)
        rep_ids = list(id_by_hash.values())
        best_rows = df[df["id"].isin(rep_ids)].copy()
        best_rows["_score"] = best_rows.apply(lambda r: _variant_row_score(engine, obj, r), axis=1)
        best_rows = best_rows.sort_values(["_score", "id"], ascending=[False, False]).head(top_n)

        counts = df.groupby("params_hash").size()

        variants: List[StrategyVariantRow] = []
        for _, row in best_rows.iterrows():
            ph = str(row["params_hash"])
            params_obj: Dict[str, Any] = {}
            raw_p = row.get("params")
            if raw_p and isinstance(raw_p, str):
                try:
                    params_obj = json.loads(raw_p)
                except (json.JSONDecodeError, TypeError):
                    params_obj = {}
            elif isinstance(raw_p, dict):
                params_obj = raw_p

            variants.append(
                StrategyVariantRow(
                    params_hash=ph,
                    variant_label=row.get("variant_label"),
                    strategy=strategy,
                    representative_run_id=int(row["id"]),
                    run_count=int(counts.get(ph, 1)),
                    total_return=float(row.get("total_return") or 0.0),
                    annualized_return=float(row.get("annualized_return") or 0.0),
                    sharpe_ratio=float(row.get("sharpe_ratio") or 0.0),
                    max_drawdown=float(row.get("max_drawdown") or 0.0),
                    win_rate=float(row.get("win_rate") or 0.0),
                    total_trades=int(row.get("total_trades") or 0),
                    volatility=float(row.get("volatility") or 0.0),
                    params=params_obj,
                    last_completed_at=str(row["completed_at"]) if row.get("completed_at") is not None else None,
                )
            )
        return StrategyVariantSummaryResponse(strategy=strategy, objective=obj, top_n=top_n, variants=variants)
    finally:
        conn.close()


def _normalized_equity_df_from_run(
    conn: sqlite3.Connection,
    run_id: int,
    preset: str,
    granularity: str,
) -> pd.DataFrame:
    row = pd.read_sql_query(
        "SELECT initial_capital, equity_curve FROM backtest_runs WHERE rowid = ?",
        conn,
        params=[run_id],
    )
    if row.empty:
        return pd.DataFrame(columns=["date", "equity"])
    ic = float(row.iloc[0]["initial_capital"] or 100000.0)
    curve_df = _parse_equity_curve(row.iloc[0].get("equity_curve"), ic)
    if curve_df.empty:
        return curve_df
    curve_df = curve_df.copy()
    curve_df["equity"] = 100.0 * (curve_df["equity"] / curve_df["equity"].iloc[0])
    curve_df = _apply_preset(curve_df, preset)
    curve_df = _resample_equity(curve_df, granularity)
    return curve_df


def _sync_variant_timeseries(
    database_path: str,
    strategy: str,
    params_hashes: List[str],
    benchmark_ticker: str,
    preset: str,
    granularity: str,
    rolling_window: int,
    objective: str,
) -> StrategyVariantTimeseriesResponse:
    obj = (objective or "balanced").lower()
    if obj not in {"sharpe", "return", "drawdown", "balanced"}:
        raise HTTPException(status_code=400, detail="objective must be sharpe, return, drawdown, or balanced")
    if granularity not in ANNUALIZATION:
        raise HTTPException(status_code=400, detail="Unsupported granularity")
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail="Unsupported preset")

    conn = sqlite3.connect(database_path)
    try:
        df = _load_variant_runs_df(conn, strategy)
        if df.empty or not params_hashes:
            benchmark_series = _build_benchmark_series(conn, benchmark_ticker, granularity, preset)
            bp = _timeseries_to_points(benchmark_series) if not benchmark_series.empty else []
            return StrategyVariantTimeseriesResponse(
                strategy=strategy,
                benchmark_ticker=benchmark_ticker.upper(),
                granularity=granularity,
                benchmark_points=bp,
                variant_series=[],
            )

        engine = StrategyOptimizerEngine(database_path)
        id_by_hash = _representative_run_ids_by_hash(df, obj, engine)
        annual_factor = ANNUALIZATION[granularity]

        variant_series: List[VariantSeriesPayload] = []
        for ph in params_hashes:
            phs = str(ph).strip()
            rid = id_by_hash.get(phs)
            if rid is None:
                continue
            meta_rows = df[df["id"] == rid]
            if meta_rows.empty:
                continue
            row_meta = meta_rows.iloc[0]
            vdf = _normalized_equity_df_from_run(conn, rid, preset, granularity)
            if vdf.empty or len(vdf) < 2:
                continue
            vdf = vdf.copy()
            vdf["period_return"] = vdf["equity"].pct_change()
            vdf["drawdown"] = _compute_drawdown(vdf["equity"])
            vdf["rolling_sharpe"] = _compute_rolling_ratio(
                vdf["period_return"].fillna(0.0), rolling_window, annual_factor, downside=False
            )
            vdf["rolling_sortino"] = _compute_rolling_ratio(
                vdf["period_return"].fillna(0.0), rolling_window, annual_factor, downside=True
            )
            vdf["rolling_volatility"] = (
                vdf["period_return"].rolling(rolling_window).std() * math.sqrt(annual_factor)
            )
            variant_series.append(
                VariantSeriesPayload(
                    params_hash=phs,
                    variant_label=row_meta.get("variant_label"),
                    representative_run_id=rid,
                    points=_timeseries_to_points(vdf),
                )
            )

        benchmark_series = _build_benchmark_series(conn, benchmark_ticker, granularity, preset)
        benchmark_points = _timeseries_to_points(benchmark_series) if not benchmark_series.empty else []

        return StrategyVariantTimeseriesResponse(
            strategy=strategy,
            benchmark_ticker=benchmark_ticker.upper(),
            granularity=granularity,
            benchmark_points=benchmark_points,
            variant_series=variant_series,
        )
    finally:
        conn.close()


def _sync_variant_distribution_for_hash(
    database_path: str,
    strategy: str,
    params_hash: str,
    objective: str,
) -> StrategyDistributionResponse:
    obj = (objective or "balanced").lower()
    conn = sqlite3.connect(database_path)
    try:
        df = _load_variant_runs_df(conn, strategy)
        if df.empty:
            raise HTTPException(status_code=404, detail="No variant runs for strategy")
        subset = df[df["params_hash"].astype(str) == str(params_hash)]
        if subset.empty:
            raise HTTPException(status_code=404, detail="Unknown params_hash for this strategy")
        engine = StrategyOptimizerEngine(database_path)
        subset = subset.copy()
        subset["_score"] = subset.apply(lambda r: _variant_row_score(engine, obj, r), axis=1)
        best = subset.sort_values(["_score", "id"], ascending=[False, False]).iloc[0]
        curve_df = _parse_equity_curve(best.get("equity_curve"), float(best.get("initial_capital") or 100000.0))
        returns = curve_df["equity"].pct_change().dropna() if not curve_df.empty else pd.Series(dtype=float)
        label = f"{strategy}:{params_hash[:12]}"
        return StrategyDistributionResponse(
            strategy=label,
            returns_histogram=_histogram(returns, 12),
            trade_pnl_histogram=[],
            holding_period_histogram=[],
            pnl_by_symbol=[],
        )
    finally:
        conn.close()


def _sync_strategy_analytics_filters(database_path: str) -> StrategyFilterMetadataResponse:
    conn = sqlite3.connect(database_path)
    try:
        strategies = pd.read_sql_query("SELECT DISTINCT name FROM backtest_runs ORDER BY name", conn)["name"].dropna().astype(str).tolist()
        benchmark_df = pd.read_sql_query("SELECT DISTINCT ticker FROM price_daily ORDER BY ticker LIMIT 50", conn)
        range_df = pd.read_sql_query("SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM price_daily", conn)
        return StrategyFilterMetadataResponse(
            strategies=strategies,
            benchmarks=benchmark_df["ticker"].dropna().astype(str).tolist(),
            available_presets=list(PRESETS.keys()),
            available_granularities=["daily", "weekly", "monthly"],
            rolling_windows=[30, 90, 252],
            min_date=range_df.iloc[0]["min_date"],
            max_date=range_df.iloc[0]["max_date"],
        )
    finally:
        conn.close()


@router.get("/filters", response_model=StrategyFilterMetadataResponse)
async def get_strategy_analytics_filters():
    from backend.main import app_state

    return await asyncio.to_thread(_sync_strategy_analytics_filters, app_state["database_path"])


@router.get("/variants/summary", response_model=StrategyVariantSummaryResponse)
async def get_strategy_variant_summary(
    strategy: str = Query(..., description="Single strategy name (backtest_runs.name)"),
    objective: str = Query("balanced", description="sharpe|return|drawdown|balanced"),
    top_n: int = Query(10, ge=1, le=50),
):
    from backend.main import app_state

    return await asyncio.to_thread(
        _sync_variant_summary,
        app_state["database_path"],
        strategy,
        objective,
        top_n,
    )


@router.get("/variants/timeseries", response_model=StrategyVariantTimeseriesResponse)
async def get_strategy_variant_timeseries(
    strategy: str = Query(...),
    params_hashes: str = Query(..., description="Comma-separated params_hash values"),
    benchmark_ticker: str = Query(default="SPY"),
    preset: str = Query(default="MAX"),
    granularity: str = Query(default="daily"),
    rolling_window: int = Query(default=30, ge=5, le=252),
    objective: str = Query(default="balanced"),
):
    from backend.main import app_state

    hashes = [h.strip() for h in params_hashes.split(",") if h.strip()]
    if not hashes:
        raise HTTPException(status_code=400, detail="params_hashes is required")
    return await asyncio.to_thread(
        _sync_variant_timeseries,
        app_state["database_path"],
        strategy,
        hashes,
        benchmark_ticker,
        preset,
        granularity,
        rolling_window,
        objective,
    )


@router.get("/variants/distributions/{strategy}", response_model=StrategyDistributionResponse)
async def get_strategy_variant_distribution(
    strategy: str,
    params_hash: str = Query(..., min_length=8),
    objective: str = Query(default="balanced"),
):
    from backend.main import app_state

    return await asyncio.to_thread(
        _sync_variant_distribution_for_hash,
        app_state["database_path"],
        strategy,
        params_hash,
        objective,
    )


@router.get("/tickers/leaderboard", response_model=TickerStrategyLeaderboardResponse)
async def get_ticker_strategy_leaderboard(
    objective: str = Query("balanced", description="sharpe|return|drawdown|balanced"),
    top_n: int = Query(5, ge=1, le=25),
    ticker: str | None = Query(default=None, description="Optional ticker filter"),
):
    from backend.main import app_state

    return await asyncio.to_thread(
        _sync_ticker_strategy_leaderboard,
        app_state["database_path"],
        objective,
        top_n,
        ticker,
    )
