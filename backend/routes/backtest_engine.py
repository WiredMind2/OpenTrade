"""
Backtest execution engine for the Trading Backtester API.
"""
import json
import sqlite3
from collections import Counter
import numpy as np
import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from backend.domain.trading import ExecutionConfig, OrderIntent, TargetAllocation
from backend.logging_config import get_component_logger
from backend.storage.backtest_repository import BacktestRepository
from backend.utils.backtest_variants import compute_params_hash, variant_label_from_params
from .websocket import broadcast_websocket_message


logger = get_component_logger(__file__)


def _backtrader_strategy_kwargs(
    strategy_class: Type[Any], parameters: Dict[str, Any] | None
) -> Dict[str, Any]:
    """Pass only keys declared on Strategy.params; extras (e.g. ticker) break bt.Strategy.__init__."""
    params_meta = getattr(strategy_class, "params", ()) or ()
    if not isinstance(params_meta, (tuple, list)):
        return {}
    names = {
        p[0]
        for p in params_meta
        if isinstance(p, (tuple, list)) and len(p) >= 1 and isinstance(p[0], str)
    }
    if not names:
        return {}
    return {k: v for k, v in (parameters or {}).items() if k in names}


def _build_execution_config(parameters: Dict[str, Any]) -> ExecutionConfig:
    return ExecutionConfig(
        min_trade_notional=float(parameters.get("min_trade_notional", 100.0)),
        commission_per_share=float(parameters.get("commission_per_share", 0.005)),
        slippage_bps=float(parameters.get("slippage_bps", 0.0)),
        max_gross_exposure=float(parameters.get("max_gross_exposure", 1.0)),
        rebalance_frequency=str(parameters.get("rebalance_frequency", "daily")),
    )


def _run_signal_execution_backtest(
    strategy: Any,
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    parameters: Dict[str, Any],
    price_frames: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    exec_config = _build_execution_config(parameters)
    all_dates = sorted(
        {
            pd.to_datetime(d).date().isoformat()
            for df in price_frames.values()
            for d in df["date"].tolist()
        }
    )
    cash = float(initial_capital)
    positions: Dict[str, float] = {}
    equity_curve: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    signal_records: List[TargetAllocation] = []
    order_intents: List[OrderIntent] = []
    order_fills: List[Dict[str, Any]] = []

    for date_str in all_dates:
        current_prices: Dict[str, float] = {}
        symbols: List[str] = []
        for ticker, df in price_frames.items():
            row = df[df["date"] == date_str]
            if row.empty:
                continue
            close_px = float(row.iloc[0]["close"])
            current_prices[ticker] = close_px
            symbols.append(ticker)
        if not symbols:
            continue
        as_of = datetime.fromisoformat(date_str)
        allocations = strategy.generate_target_allocations(
            parameters=parameters,
            symbols=symbols,
            as_of=as_of,
            current_prices=current_prices,
        )
        signal_records.extend(allocations)

        total_value = cash + sum((positions.get(t, 0.0) * current_prices.get(t, 0.0)) for t in current_prices)
        for alloc in allocations:
            ticker = alloc.ticker
            px = current_prices.get(ticker)
            if not px or px <= 0:
                continue
            target_notional = alloc.target_pct * total_value
            current_shares = positions.get(ticker, 0.0)
            current_notional = current_shares * px
            delta_notional = target_notional - current_notional
            if abs(delta_notional) < exec_config.min_trade_notional:
                continue
            side = "buy" if delta_notional > 0 else "sell"
            order_intents.append(
                OrderIntent(
                    ticker=ticker,
                    side=side,
                    notional_delta=delta_notional,
                    reason=alloc.reason,
                    timestamp=as_of,
                    metadata=alloc.metadata,
                )
            )
            qty = int(abs(delta_notional) / px)
            if qty <= 0:
                continue
            trade_notional = qty * px
            slippage = trade_notional * (exec_config.slippage_bps / 10000.0)
            fees = qty * exec_config.commission_per_share
            if side == "buy":
                total_cost = trade_notional + fees + slippage
                if total_cost > cash:
                    qty = int(max((cash - fees - slippage), 0) / px)
                    if qty <= 0:
                        continue
                    trade_notional = qty * px
                    total_cost = trade_notional + fees + slippage
                cash -= total_cost
                positions[ticker] = positions.get(ticker, 0.0) + qty
            else:
                qty = min(qty, int(max(current_shares, 0)))
                if qty <= 0:
                    continue
                trade_notional = qty * px
                cash += trade_notional - fees - slippage
                positions[ticker] = positions.get(ticker, 0.0) - qty
                if positions[ticker] <= 0:
                    positions.pop(ticker, None)
            trades.append(
                {
                    "date": date_str,
                    "ticker": ticker,
                    "side": side,
                    "size": qty,
                    "price": px,
                    "pnl": 0.0,
                    "pnlcomm": -(fees + slippage),
                }
            )
            order_fills.append(
                {
                    "fill_time": as_of.isoformat(),
                    "ticker": ticker,
                    "side": side,
                    "quantity": qty,
                    "fill_price": px,
                    "fees": fees,
                    "slippage": slippage,
                    "metadata": {"reason": alloc.reason},
                }
            )
        mtm_value = cash + sum((positions.get(t, 0.0) * current_prices.get(t, 0.0)) for t in current_prices)
        equity_curve.append({"date": date_str, "value": mtm_value})

    final_value = equity_curve[-1]["value"] if equity_curve else float(initial_capital)
    total_return = (final_value - initial_capital) / initial_capital if initial_capital else 0.0
    equity_values = [float(point["value"]) for point in equity_curve]
    if len(equity_values) > 1:
        returns = np.diff(equity_values) / np.maximum(equity_values[:-1], 1e-9)
        volatility = float(np.std(returns) * np.sqrt(252))
        sharpe_ratio = float((np.mean(returns) / np.std(returns)) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
    else:
        volatility = 0.0
        sharpe_ratio = 0.0
    running_peak = -np.inf
    max_drawdown = 0.0
    for v in equity_values:
        running_peak = max(running_peak, v)
        if running_peak > 0:
            max_drawdown = max(max_drawdown, (running_peak - v) / running_peak)
    pnl_comm = [float(t.get("pnlcomm", 0.0)) for t in trades]
    total_trades = len(trades)
    win_trades = len([v for v in pnl_comm if v > 0])
    win_rate = (win_trades / total_trades) if total_trades else 0.0
    avg_trade_return = float(np.mean(pnl_comm)) if pnl_comm else 0.0
    days = max((end_date.date() - start_date.date()).days, 1)
    annualized_return = float(((1 + total_return) ** (365 / days)) - 1) if total_return > -1 else -1.0
    signal_reason_counts = dict(
        Counter(
            alloc.reason
            for alloc in signal_records
            if getattr(alloc, "reason", None)
        )
    )
    fill_reason_counts = dict(
        Counter(
            str((fill.get("metadata") or {}).get("reason"))
            for fill in order_fills
            if (fill.get("metadata") or {}).get("reason") is not None
        )
    )
    avg_abs_target_pct = float(
        np.mean(
            [abs(float(getattr(alloc, "target_pct", 0.0) or 0.0)) for alloc in signal_records]
        )
    ) if signal_records else 0.0
    intent_notional_abs = float(
        np.sum([abs(float(getattr(intent, "notional_delta", 0.0) or 0.0)) for intent in order_intents])
    ) if order_intents else 0.0
    fill_notional_abs = float(
        np.sum(
            [
                abs(float(fill.get("quantity", 0.0) or 0.0) * float(fill.get("fill_price", 0.0) or 0.0))
                for fill in order_fills
            ]
        )
    ) if order_fills else 0.0
    turnover_vs_initial = (fill_notional_abs / float(initial_capital)) if initial_capital else 0.0
    return {
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "avg_trade_return": avg_trade_return,
        "volatility": volatility,
        "equity_curve": equity_curve,
        "trades": trades,
        "signal_records": signal_records,
        "order_intents": order_intents,
        "order_fills": order_fills,
        "execution_summary": {
            "engine": "signal",
            "signals_emitted": len(signal_records),
            "order_intents": len(order_intents),
            "order_fills": len(order_fills),
            "signal_reason_counts": signal_reason_counts,
            "fill_reason_counts": fill_reason_counts,
            "avg_abs_target_pct": avg_abs_target_pct,
            "intent_notional_abs": intent_notional_abs,
            "fill_notional_abs": fill_notional_abs,
            "turnover_vs_initial": turnover_vs_initial,
        },
    }


def _refresh_daily_prices_for_backtest(
    db_path: str,
    ticker: str,
    fetch_start: datetime,
    fetch_end: datetime,
) -> bool:
    """Best-effort refresh of daily OHLCV data for one ticker."""
    try:
        import yfinance as yf
    except Exception:
        logger.warning("yfinance not available; cannot refresh %s", ticker)
        return False

    tkr = (ticker or "").upper().strip()
    if not tkr:
        return False

    try:
        # Yahoo end date is exclusive.
        history = yf.Ticker(tkr).history(
            start=fetch_start.date().isoformat(),
            end=(fetch_end.date() + timedelta(days=1)).isoformat(),
            interval="1d",
        )
        if history is None or history.empty:
            return False

        df = history.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df[df["date"].notna()]
        if df.empty:
            return False

        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            columns = {row[1] for row in cur.execute("PRAGMA table_info(price_daily)").fetchall()}
            has_adjusted_close = "adjusted_close" in columns
            has_tickers_table = bool(
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tickers'"
                ).fetchone()
            )

            if has_tickers_table:
                cur.execute(
                    "INSERT OR IGNORE INTO tickers (ticker, name, exchange) VALUES (?, ?, ?)",
                    (tkr, None, None),
                )

            if has_adjusted_close:
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO price_daily
                    (ticker, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            tkr,
                            r["date"],
                            float(r["open"]) if pd.notna(r["open"]) else None,
                            float(r["high"]) if pd.notna(r["high"]) else None,
                            float(r["low"]) if pd.notna(r["low"]) else None,
                            float(r["close"]) if pd.notna(r["close"]) else None,
                            float(r["adjusted_close"])
                            if ("adjusted_close" in df.columns and pd.notna(r.get("adjusted_close")))
                            else (float(r["close"]) if pd.notna(r["close"]) else None),
                            int(r["volume"]) if pd.notna(r["volume"]) else None,
                        )
                        for _, r in df.iterrows()
                    ],
                )
            else:
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO price_daily
                    (ticker, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            tkr,
                            r["date"],
                            float(r["open"]) if pd.notna(r["open"]) else None,
                            float(r["high"]) if pd.notna(r["high"]) else None,
                            float(r["low"]) if pd.notna(r["low"]) else None,
                            float(r["close"]) if pd.notna(r["close"]) else None,
                            int(r["volume"]) if pd.notna(r["volume"]) else None,
                        )
                        for _, r in df.iterrows()
                    ],
                )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Failed to refresh daily prices for %s: %s", tkr, e)
        return False


def persist_optimizer_evaluation_run(
    database_path: str,
    *,
    strategy_name: str,
    parameters: Dict[str, Any],
    client_backtest_id: str,
    experiment_id: str | None,
    optimizer_mode: str | None,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    execution: Dict[str, Any],
    objective: str,
    evaluation_score: float,
) -> int:
    """Insert one completed backtest row for an optimizer evaluation (signal engine)."""
    params_hash = compute_params_hash(parameters)
    variant_label = (parameters or {}).get("variant_label") or variant_label_from_params(parameters)
    total_return = float(execution.get("total_return", 0.0) or 0.0)
    final_value = float(execution.get("final_value", initial_capital) or initial_capital)
    equity_curve = execution.get("equity_curve") or []
    sharpe_ratio = float(execution.get("sharpe_ratio", 0.0) or 0.0)
    max_drawdown = float(execution.get("max_drawdown", 0.0) or 0.0)
    win_rate = float(execution.get("win_rate", 0.0) or 0.0)
    total_trades = int(execution.get("total_trades", 0) or 0)
    avg_trade_return = float(execution.get("avg_trade_return", 0.0) or 0.0)
    volatility = float(execution.get("volatility", 0.0) or 0.0)
    annualized_return = float(execution.get("annualized_return", 0.0) or 0.0)
    execution_summary = execution.get("execution_summary") or {}

    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute(
            """
            INSERT INTO backtest_runs (
                name, params, params_hash, variant_label, optimizer_mode, experiment_id,
                client_backtest_id, started_at, completed_at, initial_capital,
                final_value, total_return, annualized_return, sharpe_ratio,
                max_drawdown, win_rate, total_trades, avg_trade_return,
                volatility, equity_curve, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_name,
                json.dumps(parameters),
                params_hash,
                variant_label,
                optimizer_mode,
                experiment_id,
                client_backtest_id,
                start_date.isoformat(),
                datetime.utcnow().isoformat(),
                initial_capital,
                final_value,
                total_return,
                annualized_return,
                sharpe_ratio,
                max_drawdown,
                win_rate,
                total_trades,
                avg_trade_return,
                volatility,
                json.dumps(equity_curve),
                json.dumps(
                    {
                        "backtest_id": client_backtest_id,
                        "status": "completed",
                        "optimization_candidate": True,
                        "experiment_id": experiment_id,
                        "optimizer_mode": optimizer_mode,
                        "objective": objective,
                        "evaluation_score": evaluation_score,
                        "params_hash": params_hash,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "win_rate": win_rate,
                        "total_trades": total_trades,
                        "avg_trade_return": avg_trade_return,
                        "volatility": volatility,
                        "execution_summary": execution_summary,
                    }
                ),
            ),
        )
        rid = int(cur.lastrowid)
        conn.commit()
        return rid
    finally:
        conn.close()


def insert_pending_backtest_run(
    database_path: str,
    *,
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    parameters: Dict[str, Any],
    backtest_id: str,
) -> int:
    """Insert a queued row so clients can poll ``GET /backtest/{id}`` before work finishes."""
    parameters = parameters or {}
    ph = compute_params_hash(parameters)
    vl = (parameters or {}).get("variant_label") or variant_label_from_params(parameters)
    om = (parameters or {}).get("optimizer_mode")
    eid = (parameters or {}).get("experiment_id")
    metrics = {
        "backtest_id": backtest_id,
        "status": "running",
        "phase": "queued",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute(
            """
            INSERT INTO backtest_runs (
                name, params, params_hash, variant_label, optimizer_mode, experiment_id,
                client_backtest_id, started_at, completed_at, initial_capital,
                final_value, total_return, annualized_return, sharpe_ratio,
                max_drawdown, win_rate, total_trades, avg_trade_return,
                volatility, equity_curve, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                strategy_name,
                json.dumps(parameters),
                ph,
                vl,
                om,
                eid,
                backtest_id,
                start_date.isoformat(),
                None,
                initial_capital,
                initial_capital,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                json.dumps([]),
                json.dumps(metrics),
            ),
        )
        rid = int(cur.lastrowid)
        conn.commit()
        return rid
    finally:
        conn.close()


def _merge_backtest_run_metrics(database_path: str, run_row_id: int, patch: Dict[str, Any]) -> None:
    conn = sqlite3.connect(database_path)
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT metrics FROM backtest_runs WHERE id = ?", (run_row_id,)).fetchone()
        base: Dict[str, Any] = {}
        if row and row[0]:
            raw = row[0]
            try:
                base = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
            except (json.JSONDecodeError, TypeError):
                base = {}
        if not isinstance(base, dict):
            base = {}
        base.update(patch)
        cur.execute(
            "UPDATE backtest_runs SET metrics = ? WHERE id = ?",
            (json.dumps(base), run_row_id),
        )
        conn.commit()
    finally:
        conn.close()


async def run_backtest_background(
    backtest_id: str,
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    parameters: Dict[str, Any],
    app_state: Dict[str, Any],
    pending_run_id: Optional[int] = None,
):
    """Run backtest in background using Backtrader."""
    parameters = parameters or {}
    db_path = app_state["database_path"]
    try:
        logger.info(f"Running background backtest: {strategy_name} (ID: {backtest_id})")

        # Get strategy from registry
        registry = app_state["strategy_registry"]
        strategy = registry.get(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' is not registered")

        ticker = str(parameters.get("ticker", "AAPL")).upper()

        if pending_run_id is not None:
            from backend.services.strategy_framework import StrategyPreflightService

            _merge_backtest_run_metrics(db_path, pending_run_id, phase="preflight")
            preflight = StrategyPreflightService(db_path).evaluate(
                strategy_name=strategy_name,
                strategy=strategy,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if not preflight.ready:
                issue_messages = "; ".join(issue.message for issue in preflight.issues)
                fail_metrics = {
                    "backtest_id": backtest_id,
                    "status": "failed",
                    "phase": "failed",
                    "error": f"Preflight failed: {issue_messages}",
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                }
                conn_pf = sqlite3.connect(db_path)
                try:
                    cur_pf = conn_pf.cursor()
                    cur_pf.execute(
                        "UPDATE backtest_runs SET completed_at = ?, metrics = ? WHERE id = ?",
                        (datetime.utcnow().isoformat(), json.dumps(fail_metrics), pending_run_id),
                    )
                    conn_pf.commit()
                finally:
                    conn_pf.close()
                await broadcast_websocket_message(
                    {
                        "type": "backtest_status",
                        "data": {
                            "strategy_name": strategy_name,
                            "start_date": start_date,
                            "end_date": end_date,
                            "initial_capital": initial_capital,
                            "final_value": initial_capital,
                            "total_return": 0.0,
                            "annualized_return": 0.0,
                            "sharpe_ratio": 0.0,
                            "max_drawdown": 0.0,
                            "win_rate": 0.0,
                            "total_trades": 0,
                            "avg_trade_return": 0.0,
                            "volatility": 0.0,
                            "timestamp": datetime.utcnow(),
                            "metrics": fail_metrics,
                            "equity_curve": [],
                        },
                    }
                )
                return

            _merge_backtest_run_metrics(db_path, pending_run_id, phase="loading_data")

        strategy_class = strategy.create_backtrader_strategy(parameters)
        execution_mode = str(parameters.get("execution_mode", "backtrader")).lower()

        # Add data feeds for tickers that have predictions
        min_bars_required = max(
            int(parameters.get("short_window", 10)),
            int(parameters.get("long_window", 30)),
            2,
        )
        lookback_days = max(min_bars_required * 3, 60)
        feed_start_date = (start_date - timedelta(days=lookback_days)).date().isoformat()
        added_feeds = 0
        price_frames: Dict[str, pd.DataFrame] = {}
        conn = sqlite3.connect(app_state["database_path"])
        try:
            cur = conn.cursor()
            if pending_run_id is not None:
                _merge_backtest_run_metrics(db_path, pending_run_id, phase="loading_prices")

            # Prefer tickers with predictions in-range.
            cur.execute("""
                SELECT DISTINCT ticker
                FROM trading_model_predictions
                WHERE dt >= ? AND dt <= ?
            """, (start_date.date().isoformat(), end_date.date().isoformat()))

            tickers = [row[0] for row in cur.fetchall()]
            if not tickers:
                # Fallback for non-ML strategies: use available price history directly.
                cur.execute("""
                    SELECT DISTINCT ticker
                    FROM price_daily
                    WHERE date >= ? AND date <= ?
                    ORDER BY ticker ASC
                    LIMIT 10
                """, (start_date.date().isoformat(), end_date.date().isoformat()))
                tickers = [row[0] for row in cur.fetchall()]

            for ticker in tickers[:10]:  # Limit to 10 tickers for performance
                # Get price data with a lookback window so indicators can warm up.
                cur.execute(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM price_daily
                    WHERE ticker = ? AND date >= ? AND date <= ?
                    ORDER BY date ASC
                    """,
                    (ticker, feed_start_date, end_date.date().isoformat()),
                )
                price_data = cur.fetchall()
                if not price_data:
                    continue
                if len(price_data) < min_bars_required:
                    logger.info(
                        f"Ticker {ticker} has {len(price_data)} bars, attempting auto-refresh "
                        f"for at least {min_bars_required}"
                    )
                    _refresh_daily_prices_for_backtest(
                        app_state["database_path"],
                        ticker,
                        start_date - timedelta(days=lookback_days),
                        end_date,
                    )
                    cur.execute(
                        """
                        SELECT date, open, high, low, close, volume
                        FROM price_daily
                        WHERE ticker = ? AND date >= ? AND date <= ?
                        ORDER BY date ASC
                        """,
                        (ticker, feed_start_date, end_date.date().isoformat()),
                    )
                    price_data = cur.fetchall()
                    if len(price_data) < min_bars_required:
                        logger.info(
                            f"Skipping ticker {ticker}: {len(price_data)} bars available, "
                            f"{min_bars_required} required"
                        )
                        continue

                # Create pandas DataFrame
                df = pd.DataFrame(price_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                price_frames[ticker] = df
                added_feeds += 1
        finally:
            conn.close()

        if added_feeds == 0:
            raise ValueError(
                f"No market data available for {strategy_name} in range "
                f"{start_date.date().isoformat()} to {end_date.date().isoformat()}"
            )

        if pending_run_id is not None:
            _merge_backtest_run_metrics(db_path, pending_run_id, phase="executing")

        if execution_mode == "signal":
            logger.info(f"Starting signal execution for {strategy_name}")
            execution = _run_signal_execution_backtest(
                strategy=strategy,
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                parameters=parameters,
                price_frames=price_frames,
            )
            final_value = execution["final_value"]
            total_return = execution["total_return"]
            annualized_return = execution["annualized_return"]
            sharpe_ratio = execution["sharpe_ratio"]
            max_drawdown = execution["max_drawdown"]
            win_rate = execution["win_rate"]
            total_trades = execution["total_trades"]
            avg_trade_return = execution["avg_trade_return"]
            volatility = execution["volatility"]
            equity_curve = execution["equity_curve"]
            trades = execution["trades"]
            execution_summary = execution["execution_summary"]
            signal_records = execution["signal_records"]
            order_intents = execution["order_intents"]
            order_fills = execution["order_fills"]
            if strategy_name == "recursive_forecast":
                reason_counts = execution_summary.get("signal_reason_counts", {}) if isinstance(execution_summary, dict) else {}
                no_model_count = int(reason_counts.get("no_model_available", 0) or 0)
                non_missing_count = int(
                    sum(v for k, v in reason_counts.items() if k != "no_model_available")
                ) if isinstance(reason_counts, dict) else 0
                if no_model_count > 0 and non_missing_count == 0 and len(order_fills) == 0:
                    raise ValueError(
                        "No recursive forecast models/predictions available for requested period. "
                        "Run the ML pipeline to generate model predictions before backtesting."
                    )
        else:
            # Set up Backtrader
            cerebro = bt.Cerebro()
            cerebro.addstrategy(
                strategy_class, **_backtrader_strategy_kwargs(strategy_class, parameters)
            )
            for ticker, df in price_frames.items():
                data = bt.feeds.PandasData(
                    dataname=df,
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=5,
                    name=ticker,
                )
                cerebro.adddata(data)
            # Set broker parameters
            cerebro.broker.setcash(initial_capital)
            cerebro.broker.setcommission(commission=parameters.get('commission_per_share', 0.005))
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            logger.info(f"Starting Backtrader execution for {strategy_name}")
            results = cerebro.run()
            if not results:
                raise RuntimeError("Backtest engine returned no strategy results")
            strat = results[0]
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - initial_capital) / initial_capital
            sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
            annualized_return = strat.analyzers.returns.get_analysis().get('rnorm100', 0)
            trade_analysis = strat.analyzers.trades.get_analysis()
            total_trades = trade_analysis.get('total', {}).get('total', 0)
            win_trades = trade_analysis.get('won', {}).get('total', 0)
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            trades = getattr(strat, "trades", []) or []
            pnl_comm = [t.get('pnlcomm', 0.0) for t in trades if isinstance(t, dict)]
            avg_trade_return = np.mean(pnl_comm) if pnl_comm else 0
            equity_curve = getattr(strat, "equity_curve", []) or []
            equity_values = [point.get('value') for point in equity_curve if isinstance(point, dict) and point.get('value') is not None]
            if len(equity_values) > 1:
                returns = np.diff(equity_values) / equity_values[:-1]
                volatility = np.std(returns) * np.sqrt(252)
            else:
                volatility = 0
            execution_summary = {
                "engine": "backtrader",
                "signals_emitted": 0,
                "order_intents": 0,
                "order_fills": 0,
            }
            signal_records = []
            order_intents = []
            order_fills = []

        # Store results in database
        params_hash = compute_params_hash(parameters)
        variant_label = (parameters or {}).get("variant_label") or variant_label_from_params(parameters)
        optimizer_mode = (parameters or {}).get("optimizer_mode")
        experiment_id = (parameters or {}).get("experiment_id")

        conn = sqlite3.connect(app_state["database_path"])
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        completed_metrics = {
            "backtest_id": backtest_id,
            "status": "completed",
            "phase": "completed",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "avg_trade_return": avg_trade_return,
            "volatility": volatility,
            "execution_summary": execution_summary,
            "params_hash": params_hash,
        }

        if pending_run_id is not None:
            cur.execute(
                """
                UPDATE backtest_runs SET
                    completed_at = ?,
                    final_value = ?, total_return = ?, annualized_return = ?,
                    sharpe_ratio = ?, max_drawdown = ?, win_rate = ?, total_trades = ?,
                    avg_trade_return = ?, volatility = ?, equity_curve = ?, metrics = ?
                WHERE id = ?
                """,
                (
                    datetime.utcnow().isoformat(),
                    final_value,
                    total_return,
                    annualized_return,
                    sharpe_ratio,
                    max_drawdown,
                    win_rate,
                    total_trades,
                    avg_trade_return,
                    volatility,
                    json.dumps(equity_curve),
                    json.dumps(completed_metrics),
                    pending_run_id,
                ),
            )
            run_row_id = pending_run_id
        else:
            cur.execute(
                """
                INSERT INTO backtest_runs (
                    name, params, params_hash, variant_label, optimizer_mode, experiment_id,
                    client_backtest_id, started_at, completed_at, initial_capital,
                    final_value, total_return, annualized_return, sharpe_ratio,
                    max_drawdown, win_rate, total_trades, avg_trade_return,
                    volatility, equity_curve, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    json.dumps(parameters),
                    params_hash,
                    variant_label,
                    optimizer_mode,
                    experiment_id,
                    backtest_id,
                    start_date.isoformat(),
                    datetime.utcnow().isoformat(),
                    initial_capital,
                    final_value,
                    total_return,
                    annualized_return,
                    sharpe_ratio,
                    max_drawdown,
                    win_rate,
                    total_trades,
                    avg_trade_return,
                    volatility,
                    json.dumps(equity_curve),
                    json.dumps(completed_metrics),
                ),
            )
            run_row_id = int(cur.lastrowid)

        conn.commit()
        conn.close()
        repo = BacktestRepository(app_state["database_path"])
        repo.persist_signals(backtest_id, signal_records, backtest_run_id=run_row_id)
        repo.persist_order_intents(backtest_id, order_intents, backtest_run_id=run_row_id)
        repo.persist_order_fills(backtest_id, order_fills, backtest_run_id=run_row_id)

        logger.info(f"Background backtest completed: {strategy_name} (ID: {backtest_id})")
        logger.info(f"Results - Final Value: ${final_value:.2f}, Total Return: {total_return:.2%}")

        # Broadcast backtest status update
        await broadcast_websocket_message({
            "type": "backtest_status",
            "data": {
                "strategy_name": strategy_name,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "annualized_return": annualized_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "avg_trade_return": avg_trade_return,
                "volatility": volatility,
                "timestamp": datetime.utcnow(),
                "metrics": {
                    "backtest_id": backtest_id,
                    "status": "completed",
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "total_trades": total_trades,
                    "avg_trade_return": avg_trade_return,
                        "volatility": volatility,
                        "execution_summary": execution_summary,
                },
                "equity_curve": equity_curve
            }
        })

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        logger.exception(f"Background backtest failed: {error_message}")

        # Store failure in database
        try:
            ph = compute_params_hash(parameters)
            vl = (parameters or {}).get("variant_label") or variant_label_from_params(parameters)
            fail_row_metrics = {
                "backtest_id": backtest_id,
                "status": "failed",
                "phase": "failed",
                "error": error_message,
                "params_hash": ph,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }
            conn = sqlite3.connect(app_state["database_path"])
            try:
                cur = conn.cursor()
                if pending_run_id is not None:
                    cur.execute(
                        """
                        UPDATE backtest_runs SET completed_at = ?, metrics = ? WHERE id = ?
                        """,
                        (datetime.utcnow().isoformat(), json.dumps(fail_row_metrics), pending_run_id),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO backtest_runs (
                            name, params, params_hash, variant_label, optimizer_mode, experiment_id,
                            client_backtest_id, started_at, completed_at, metrics
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            strategy_name,
                            json.dumps(parameters),
                            ph,
                            vl,
                            (parameters or {}).get("optimizer_mode"),
                            (parameters or {}).get("experiment_id"),
                            backtest_id,
                            start_date.isoformat(),
                            datetime.utcnow().isoformat(),
                            json.dumps(fail_row_metrics),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

            # Broadcast backtest failure status update
            await broadcast_websocket_message({
                "type": "backtest_status",
                "data": {
                    "strategy_name": strategy_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_capital": initial_capital,
                    "final_value": initial_capital,  # No change on failure
                    "total_return": 0.0,
                    "annualized_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "total_trades": 0,
                    "avg_trade_return": 0.0,
                    "volatility": 0.0,
                    "timestamp": datetime.utcnow(),
                    "metrics": fail_row_metrics,
                    "equity_curve": []
                }
            })
        except Exception as db_e:
            logger.error(f"Failed to store backtest failure: {str(db_e)}")