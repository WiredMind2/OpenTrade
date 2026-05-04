"""
Advanced strategy analytics endpoints for performance comparison dashboards.
"""
import asyncio
import json
import math
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.logging_config import get_component_logger
from backend.schemas import (
    DistributionBucket,
    StrategyComparisonSummaryResponse,
    StrategyDistributionResponse,
    StrategyFilterMetadataResponse,
    StrategyMetricPoint,
    StrategyTimeseriesPoint,
    StrategyTimeseriesResponse,
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


def _parse_equity_curve(raw_curve: Optional[str], initial_capital: float) -> pd.DataFrame:
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


def _returns_to_monthly_map(series_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if series_df.empty:
        return {}
    temp = series_df.copy()
    temp["month"] = temp["date"].dt.to_period("M")
    month_last = temp.groupby("month")["equity"].last()
    monthly_returns = month_last.pct_change().dropna()
    heatmap: Dict[str, Dict[str, float]] = {}
    for period, value in monthly_returns.items():
        year = str(period.year)
        month = f"{period.month:02d}"
        if year not in heatmap:
            heatmap[year] = {}
        heatmap[year][month] = float(value)
    return heatmap


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


def _aggregate_series_for_strategy(
    conn: sqlite3.Connection,
    strategy: str,
    granularity: str,
    preset: str,
) -> pd.DataFrame:
    run_series = []
    run_ids = pd.read_sql_query(
        "SELECT id, initial_capital, equity_curve FROM backtest_runs WHERE name = ?",
        conn,
        params=[strategy],
    )
    if run_ids.empty:
        return pd.DataFrame(columns=["date", "equity"])
    for _, row in run_ids.iterrows():
        curve_df = _parse_equity_curve(row["equity_curve"], float(row["initial_capital"] or 100000.0))
        if curve_df.empty:
            snapshots = pd.read_sql_query(
                """
                SELECT dt, total_value
                FROM portfolio_snapshots
                WHERE CAST(backtest_run_id AS TEXT) = CAST(? AS TEXT)
                ORDER BY dt ASC
                """,
                conn,
                params=[row["id"]],
            )
            if not snapshots.empty:
                curve_df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(snapshots["dt"], utc=True).dt.tz_convert(None),
                        "equity": snapshots["total_value"].astype(float),
                    }
                )
        if not curve_df.empty:
            curve_df["norm_equity"] = 100.0 * (curve_df["equity"] / curve_df["equity"].iloc[0])
            run_series.append(curve_df[["date", "norm_equity"]])
    if not run_series:
        return pd.DataFrame(columns=["date", "equity"])
    merged = pd.concat(run_series, axis=0)
    avg = merged.groupby("date")["norm_equity"].mean().reset_index()
    avg = avg.rename(columns={"norm_equity": "equity"})
    avg = _apply_preset(avg, preset)
    avg = _resample_equity(avg, granularity)
    return avg


def _compute_summary_metrics(
    strategy: str,
    series_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    annual_factor: int,
    run_count: int,
    trade_stats: Dict[str, float],
) -> StrategyMetricPoint:
    if series_df.empty or len(series_df) < 2:
        return StrategyMetricPoint(
            strategy=strategy,
            run_count=run_count,
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
            information_ratio=0.0,
            alpha=0.0,
            beta=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            expectancy=trade_stats["expectancy"],
            total_trades=int(trade_stats["total_trades"]),
        )
    returns = series_df["equity"].pct_change().dropna()
    total_return = float((series_df["equity"].iloc[-1] / series_df["equity"].iloc[0]) - 1.0)
    years = max(len(series_df) / annual_factor, 1 / annual_factor)
    cagr = float((series_df["equity"].iloc[-1] / series_df["equity"].iloc[0]) ** (1 / years) - 1)
    volatility = float(returns.std() * math.sqrt(annual_factor)) if not returns.empty else 0.0
    sharpe = float((returns.mean() / returns.std()) * math.sqrt(annual_factor)) if returns.std() not in (0, np.nan) else 0.0
    downside_std = returns.mask(returns > 0, other=0.0).std()
    sortino = float((returns.mean() / downside_std) * math.sqrt(annual_factor)) if downside_std not in (0, np.nan) else 0.0
    drawdown = _compute_drawdown(series_df["equity"])
    max_dd = float(abs(drawdown.min())) if not drawdown.empty else 0.0
    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

    common_len = min(len(returns), len(benchmark_returns))
    strat_r = returns.tail(common_len).reset_index(drop=True)
    bench_r = benchmark_returns.tail(common_len).reset_index(drop=True)
    active = strat_r - bench_r if common_len > 0 else pd.Series(dtype=float)
    info_ratio = float((active.mean() / active.std()) * math.sqrt(annual_factor)) if common_len > 1 and active.std() != 0 else 0.0
    beta = float(np.cov(strat_r, bench_r)[0][1] / np.var(bench_r)) if common_len > 1 and np.var(bench_r) != 0 else 0.0
    alpha = float((strat_r.mean() - beta * bench_r.mean()) * annual_factor) if common_len > 1 else 0.0

    return StrategyMetricPoint(
        strategy=strategy,
        run_count=run_count,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        information_ratio=info_ratio,
        alpha=alpha,
        beta=beta,
        volatility=volatility,
        max_drawdown=max_dd,
        win_rate=trade_stats["win_rate"],
        profit_factor=trade_stats["profit_factor"],
        avg_win=trade_stats["avg_win"],
        avg_loss=trade_stats["avg_loss"],
        expectancy=trade_stats["expectancy"],
        total_trades=int(trade_stats["total_trades"]),
    )


def _get_trade_stats(conn: sqlite3.Connection, strategy: str) -> Dict[str, float]:
    query = """
    SELECT t.pnl
    FROM trades t
    JOIN backtest_runs b ON CAST(t.backtest_run_id AS TEXT) = CAST(b.id AS TEXT)
    WHERE b.name = ?
    """
    trades = pd.read_sql_query(query, conn, params=[strategy])
    if trades.empty:
        return {"win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "expectancy": 0.0, "total_trades": 0.0}
    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    return {
        "win_rate": float((len(wins) / len(pnl)) if len(pnl) else 0.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
        "total_trades": float(len(pnl)),
    }


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


def _resolve_strategy_names(conn: sqlite3.Connection, selected: List[str]) -> List[str]:
    found = pd.read_sql_query("SELECT DISTINCT name FROM backtest_runs ORDER BY name", conn)
    strategies = [row for row in found["name"].dropna().astype(str).tolist() if row]
    if selected:
        return [name for name in selected if name in strategies]
    return strategies[:10]


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
        SELECT id, name, params_hash, variant_label, params, sharpe_ratio, total_return,
               max_drawdown, win_rate, total_trades, volatility, annualized_return,
               completed_at, initial_capital, equity_curve, final_value
        FROM backtest_runs
        WHERE name = ?
          AND params_hash IS NOT NULL AND TRIM(params_hash) != ''
          AND final_value IS NOT NULL
        """,
        conn,
        params=[strategy],
    )
    return df


def _representative_run_ids_by_hash(
    df: pd.DataFrame, objective: str, engine: StrategyOptimizerEngine
) -> Dict[str, int]:
    """For each params_hash, pick the run id with best objective score (tie-break: higher id)."""
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
        "SELECT initial_capital, equity_curve FROM backtest_runs WHERE id = ?",
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
        rid = int(best["id"])
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


def _sync_strategy_analytics_summary(
    database_path: str,
    strategies: List[str],
    benchmark_ticker: str,
    preset: str,
    granularity: str,
    rolling_window: int,
) -> StrategyComparisonSummaryResponse:
    conn = sqlite3.connect(database_path)
    try:
        selected = _resolve_strategy_names(conn, strategies)
        benchmark_series = _build_benchmark_series(conn, benchmark_ticker, granularity, preset)
        benchmark_returns = benchmark_series["equity"].pct_change().dropna() if not benchmark_series.empty else pd.Series(dtype=float)

        metrics: List[StrategyMetricPoint] = []
        all_dates: List[pd.Timestamp] = []
        for strategy in selected:
            run_count_df = pd.read_sql_query("SELECT COUNT(*) AS c FROM backtest_runs WHERE name = ?", conn, params=[strategy])
            run_count = int(run_count_df.iloc[0]["c"]) if not run_count_df.empty else 0
            strategy_series = _aggregate_series_for_strategy(conn, strategy, granularity, preset)
            if not strategy_series.empty:
                all_dates.extend(strategy_series["date"].tolist())
            trade_stats = _get_trade_stats(conn, strategy)
            metrics.append(
                _compute_summary_metrics(
                    strategy=strategy,
                    series_df=strategy_series,
                    benchmark_returns=benchmark_returns,
                    annual_factor=ANNUALIZATION[granularity],
                    run_count=run_count,
                    trade_stats=trade_stats,
                )
            )

        start_date = min(all_dates).date().isoformat() if all_dates else None
        end_date = max(all_dates).date().isoformat() if all_dates else None
        return StrategyComparisonSummaryResponse(
            benchmark_ticker=benchmark_ticker.upper(),
            granularity=granularity,
            rolling_window=rolling_window,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
        )
    finally:
        conn.close()


def _sync_strategy_timeseries(
    database_path: str,
    strategy: str,
    benchmark_ticker: str,
    preset: str,
    granularity: str,
    rolling_window: int,
) -> StrategyTimeseriesResponse:
    conn = sqlite3.connect(database_path)
    try:
        strategy_series = _aggregate_series_for_strategy(conn, strategy, granularity, preset)
        if strategy_series.empty:
            raise HTTPException(status_code=404, detail=f"No timeseries data found for strategy '{strategy}'")
        benchmark_series = _build_benchmark_series(conn, benchmark_ticker, granularity, preset)
        annual_factor = ANNUALIZATION[granularity]

        strategy_series["period_return"] = strategy_series["equity"].pct_change()
        strategy_series["drawdown"] = _compute_drawdown(strategy_series["equity"])
        strategy_series["rolling_sharpe"] = _compute_rolling_ratio(
            strategy_series["period_return"].fillna(0.0), rolling_window, annual_factor, downside=False
        )
        strategy_series["rolling_sortino"] = _compute_rolling_ratio(
            strategy_series["period_return"].fillna(0.0), rolling_window, annual_factor, downside=True
        )
        strategy_series["rolling_volatility"] = (
            strategy_series["period_return"].rolling(rolling_window).std() * math.sqrt(annual_factor)
        )

        benchmark_df = benchmark_series.copy()
        if not benchmark_df.empty:
            benchmark_df["drawdown"] = _compute_drawdown(benchmark_df["equity"])
        monthly_returns = _returns_to_monthly_map(strategy_series.rename(columns={"equity": "equity"}))

        return StrategyTimeseriesResponse(
            strategy=strategy,
            benchmark_ticker=benchmark_ticker.upper(),
            granularity=granularity,
            points=_timeseries_to_points(strategy_series),
            benchmark_points=_timeseries_to_points(benchmark_df) if not benchmark_df.empty else [],
            monthly_returns=monthly_returns,
        )
    finally:
        conn.close()


def _sync_strategy_distributions(database_path: str, strategy: str) -> StrategyDistributionResponse:
    conn = sqlite3.connect(database_path)
    try:
        returns_df = _aggregate_series_for_strategy(conn, strategy, "daily", "MAX")
        returns = returns_df["equity"].pct_change().dropna() if not returns_df.empty else pd.Series(dtype=float)

        trades = pd.read_sql_query(
            """
            SELECT t.pnl, t.ticker, t.entry_dt, t.exit_dt
            FROM trades t
            JOIN backtest_runs b ON CAST(t.backtest_run_id AS TEXT) = CAST(b.id AS TEXT)
            WHERE b.name = ?
            """,
            conn,
            params=[strategy],
        )
        trade_pnl = trades["pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
        holding_periods = pd.Series(dtype=float)
        if not trades.empty:
            entry = pd.to_datetime(trades["entry_dt"], errors="coerce")
            exit_ = pd.to_datetime(trades["exit_dt"], errors="coerce")
            holding_periods = (exit_ - entry).dt.total_seconds().div(86400).dropna()

        symbol_buckets: List[DistributionBucket] = []
        if not trades.empty:
            symbol_pnl = trades.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
            symbol_buckets = [
                DistributionBucket(bucket=str(ticker), count=int(abs(value) > 0), value=float(value))
                for ticker, value in symbol_pnl.items()
            ]

        return StrategyDistributionResponse(
            strategy=strategy,
            returns_histogram=_histogram(returns, 12),
            trade_pnl_histogram=_histogram(trade_pnl, 12),
            holding_period_histogram=_histogram(holding_periods, 10),
            pnl_by_symbol=symbol_buckets,
        )
    finally:
        conn.close()


@router.get("/filters", response_model=StrategyFilterMetadataResponse)
async def get_strategy_analytics_filters():
    from backend.main import app_state

    return await asyncio.to_thread(_sync_strategy_analytics_filters, app_state["database_path"])


@router.get("/summary", response_model=StrategyComparisonSummaryResponse)
async def get_strategy_analytics_summary(
    strategies: List[str] = Query(default=[]),
    benchmark_ticker: str = Query(default="SPY"),
    preset: str = Query(default="MAX"),
    granularity: str = Query(default="daily"),
    rolling_window: int = Query(default=30, ge=5, le=252),
):
    from backend.main import app_state

    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail="Unsupported preset")
    if granularity not in ANNUALIZATION:
        raise HTTPException(status_code=400, detail="Unsupported granularity")

    return await asyncio.to_thread(
        _sync_strategy_analytics_summary,
        app_state["database_path"],
        strategies,
        benchmark_ticker,
        preset,
        granularity,
        rolling_window,
    )


@router.get("/timeseries/{strategy}", response_model=StrategyTimeseriesResponse)
async def get_strategy_timeseries(
    strategy: str,
    benchmark_ticker: str = Query(default="SPY"),
    preset: str = Query(default="MAX"),
    granularity: str = Query(default="daily"),
    rolling_window: int = Query(default=30, ge=5, le=252),
):
    from backend.main import app_state

    if preset not in PRESETS or granularity not in ANNUALIZATION:
        raise HTTPException(status_code=400, detail="Unsupported filter option")

    return await asyncio.to_thread(
        _sync_strategy_timeseries,
        app_state["database_path"],
        strategy,
        benchmark_ticker,
        preset,
        granularity,
        rolling_window,
    )


@router.get("/distributions/{strategy}", response_model=StrategyDistributionResponse)
async def get_strategy_distributions(strategy: str):
    from backend.main import app_state

    return await asyncio.to_thread(_sync_strategy_distributions, app_state["database_path"], strategy)


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
