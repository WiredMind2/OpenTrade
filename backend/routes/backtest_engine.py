"""
Backtest execution engine for the Trading Backtester API.
"""
import json
import sqlite3
from collections import Counter
import numpy as np
import pandas as pd
import backtrader as bt
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from backend.domain.trading import ExecutionConfig, OrderIntent, TargetAllocation
from backend.logging_config import get_component_logger
from backend.services.backtrader_engine import build_run_config, run_backtrader_once
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


def _decision_markers_from_signal_execution(execution: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per executed buy/sell (aligned with signal-engine ``trades`` / ``order_fills``)."""
    trades_list = execution.get("trades") or []
    fills_list = execution.get("order_fills") or []
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(trades_list):
        if not isinstance(t, dict):
            continue
        raw_date = t.get("date")
        if raw_date is None:
            continue
        date = str(raw_date)[:10]
        side_raw = t.get("side")
        side = str(side_raw).lower() if side_raw is not None else ""
        if side not in ("buy", "sell"):
            continue
        ticker = t.get("ticker")
        tic = str(ticker).upper() if ticker else None
        reason: Optional[str] = None
        if i < len(fills_list):
            f = fills_list[i]
            if isinstance(f, dict):
                meta = f.get("metadata") or {}
                if isinstance(meta, dict):
                    r = meta.get("reason")
                    if r is not None:
                        reason = str(r)
        row: Dict[str, Any] = {"date": date, "side": side, "ticker": tic}
        if reason:
            row["reason"] = reason
        out.append(row)
    # Return all markers; frontend controls any thinning if desired.
    out.sort(key=lambda r: str(r.get("date", ""))[:10])
    return out


def _normalize_bt_decision_markers(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return []
    # Return all markers; frontend controls any thinning if desired.
    raw_sorted = list(raw)
    raw_sorted.sort(key=lambda r: str(r.get("date", ""))[:10] if isinstance(r, dict) else "")
    for m in raw_sorted:
        if not isinstance(m, dict):
            continue
        d = m.get("date")
        if d is None:
            continue
        date = str(d)[:10]
        side_raw = m.get("side")
        side = str(side_raw).lower() if side_raw is not None else ""
        if side not in ("buy", "sell"):
            continue
        row: Dict[str, Any] = {"date": date, "side": side}
        tic = m.get("ticker")
        if tic:
            row["ticker"] = str(tic).upper()
        r = m.get("reason")
        if r is not None:
            row["reason"] = str(r)
        out.append(row)
    return out


def _build_execution_config(parameters: Dict[str, Any]) -> ExecutionConfig:
    return ExecutionConfig(
        min_trade_notional=float(parameters.get("min_trade_notional", 100.0)),
        commission_per_share=float(parameters.get("commission_per_share", 0.005)),
        slippage_bps=float(parameters.get("slippage_bps", 0.0)),
        max_gross_exposure=float(parameters.get("max_gross_exposure", 1.0)),
        rebalance_frequency=str(parameters.get("rebalance_frequency", "daily")),
    )


def _should_rebalance_signal(
    current_day: date,
    last_rebalance_day: Optional[date],
    frequency: str,
) -> bool:
    if last_rebalance_day is None:
        return True
    freq = (frequency or "daily").strip().lower()
    if freq == "daily":
        return True
    if freq == "weekly":
        return current_day.isocalendar()[:2] != last_rebalance_day.isocalendar()[:2]
    if freq == "monthly":
        return (current_day.year, current_day.month) != (
            last_rebalance_day.year,
            last_rebalance_day.month,
        )
    return True


def _run_signal_execution_backtest(
    strategy: Any,
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    parameters: Dict[str, Any],
    price_frames: Dict[str, pd.DataFrame],
    signal_db_conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    exec_config = _build_execution_config(parameters)
    # O(1) close lookup per (ticker, calendar day); first row wins on duplicate dates (matches iloc[0]).
    close_by_ticker_date: Dict[str, Dict[str, float]] = {}
    for ticker, df in price_frames.items():
        day_close: Dict[str, float] = {}
        for d, c in zip(df["date"].tolist(), df["close"].tolist()):
            dk = pd.to_datetime(d).date().isoformat()
            if dk not in day_close:
                day_close[dk] = float(c)
        close_by_ticker_date[ticker] = day_close
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
    last_rebalance_day: Optional[date] = None

    for date_str in all_dates:
        current_day = datetime.fromisoformat(date_str).date()
        current_prices: Dict[str, float] = {}
        symbols: List[str] = []
        for ticker in price_frames.keys():
            close_px = close_by_ticker_date.get(ticker, {}).get(date_str)
            if close_px is None:
                continue
            current_prices[ticker] = close_px
            symbols.append(ticker)
        if not symbols:
            continue
        as_of = datetime.fromisoformat(date_str)
        if not _should_rebalance_signal(
            current_day,
            last_rebalance_day,
            exec_config.rebalance_frequency,
        ):
            mtm_value = cash + sum(
                (positions.get(t, 0.0) * current_prices.get(t, 0.0)) for t in current_prices
            )
            equity_curve.append({"date": date_str, "value": mtm_value})
            continue
        allocations = strategy.generate_target_allocations(
            parameters=parameters,
            symbols=symbols,
            as_of=as_of,
            current_prices=current_prices,
            db_conn=signal_db_conn,
        )
        gross_target = float(
            np.sum([abs(float(getattr(alloc, "target_pct", 0.0) or 0.0)) for alloc in allocations])
        )
        if gross_target > 0 and exec_config.max_gross_exposure > 0 and gross_target > exec_config.max_gross_exposure:
            scale = exec_config.max_gross_exposure / gross_target
            for alloc in allocations:
                alloc.target_pct = float(alloc.target_pct) * scale
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
        last_rebalance_day = current_day

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
    raw_markers = execution.get("decision_markers")
    if isinstance(raw_markers, list):
        decision_markers = _normalize_bt_decision_markers(raw_markers)
    else:
        decision_markers = _decision_markers_from_signal_execution(execution)
    # #region agent log
    try:
        import json as _json, time as _time
        with open(r"c:\Users\willi\Documents\Python\Trading\backtesting\.cursor\debug-2acb83.log","a",encoding="utf-8") as _f:
            _f.write(_json.dumps({"sessionId":"2acb83","runId":"post-fix","hypothesisId":"H6","location":"backtest_engine.py:persist_optimizer_evaluation_run","message":"Persisting optimizer decision_markers length","data":{"strategy":strategy_name,"markers":len(decision_markers) if isinstance(decision_markers,list) else None,"first":decision_markers[0]["date"][:10] if isinstance(decision_markers,list) and decision_markers else None,"last":decision_markers[-1]["date"][:10] if isinstance(decision_markers,list) and decision_markers else None}, "timestamp":int(_time.time()*1000)})+"\n")
    except Exception:
        pass
    # #endregion

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
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "win_rate": win_rate,
                        "total_trades": total_trades,
                        "avg_trade_return": avg_trade_return,
                        "volatility": volatility,
                        "execution_summary": execution_summary,
                        "decision_markers": decision_markers,
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
    """Insert a queued row for a background run (used when ``pending_run_id`` is passed to ``run_backtest_background``)."""
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


def _merge_backtest_run_metrics(
    database_path: str,
    run_row_id: int,
    patch: Dict[str, Any],
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    own_conn = conn is None
    if conn is None:
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
        if own_conn:
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
    metrics_conn: Optional[sqlite3.Connection] = None
    if pending_run_id is not None:
        metrics_conn = sqlite3.connect(db_path)
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

            _merge_backtest_run_metrics(
                db_path, pending_run_id, {"phase": "preflight"}, conn=metrics_conn
            )
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
                cur_pf = metrics_conn.cursor()
                cur_pf.execute(
                    "UPDATE backtest_runs SET completed_at = ?, metrics = ? WHERE id = ?",
                    (datetime.utcnow().isoformat(), json.dumps(fail_metrics), pending_run_id),
                )
                metrics_conn.commit()
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

            _merge_backtest_run_metrics(
                db_path, pending_run_id, {"phase": "loading_data"}, conn=metrics_conn
            )

        strategy_parameters = {
            **parameters,
            # Preserve indicator warmup history while blocking pre-window trading in strategies
            # that declare this optional parameter.
            "backtest_start_date": start_date.date().isoformat(),
        }
        strategy_class = strategy.create_backtrader_strategy(strategy_parameters)
        requested_execution_mode = str(parameters.get("execution_mode", "backtrader")).lower()
        execution_mode = "backtrader"
        if requested_execution_mode == "signal":
            logger.warning(
                "Signal execution mode requested for %s but is disabled; forcing backtrader",
                strategy_name,
            )

        # Add data feeds for tickers that have predictions
        min_bars_required = max(
            int(parameters.get("short_window", 10)),
            int(parameters.get("long_window", 30)),
            int(parameters.get("vol_lookback", 0) or 0),
            int(parameters.get("momentum_lookback", 0) or 0),
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
                _merge_backtest_run_metrics(
                    db_path, pending_run_id, {"phase": "loading_prices"}, conn=metrics_conn
                )

            # Universe selection (predictions-free): use available price history directly.
            cur.execute(
                """
                SELECT DISTINCT ticker
                FROM price_daily
                WHERE date >= ? AND date <= ?
                ORDER BY ticker ASC
                LIMIT 10
                """,
                (start_date.date().isoformat(), end_date.date().isoformat()),
            )
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
            _merge_backtest_run_metrics(
                db_path, pending_run_id, {"phase": "executing"}, conn=metrics_conn
            )

        logger.info(f"Starting Backtrader execution for {strategy_name}")
        execution = run_backtrader_once(
            strategy_class=strategy_class,
            strategy_kwargs=_backtrader_strategy_kwargs(strategy_class, strategy_parameters),
            price_frames=price_frames,
            config=build_run_config(initial_capital=initial_capital, parameters=strategy_parameters),
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
        signal_records = []
        order_intents = []
        order_fills = []
        decision_markers = execution.get("decision_markers") or []

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
            "decision_markers": decision_markers,
        }
        # #region agent log
        try:
            import json as _json, time as _time
            with open(r"c:\Users\willi\Documents\Python\Trading\backtesting\.cursor\debug-2acb83.log","a",encoding="utf-8") as _f:
                _f.write(_json.dumps({"sessionId":"2acb83","runId":"post-fix","hypothesisId":"H6","location":"backtest_engine.py:run_backtest_background","message":"Completed run decision_markers length","data":{"strategy":strategy_name,"markers":len(decision_markers) if isinstance(decision_markers,list) else None,"first":decision_markers[0]["date"][:10] if isinstance(decision_markers,list) and decision_markers else None,"last":decision_markers[-1]["date"][:10] if isinstance(decision_markers,list) and decision_markers else None}, "timestamp":int(_time.time()*1000)})+"\n")
        except Exception:
            pass
        # #endregion

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
                "metrics": completed_metrics,
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
    finally:
        if metrics_conn is not None:
            try:
                metrics_conn.close()
            except Exception:
                pass


async def evaluate_strategy_runtime_once(
    *,
    strategy_name: str,
    ticker: str,
    parameters: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    app_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute one fresh strategy run and return metrics/curve.

    This reuses run_backtest_background() for execution parity, then removes the
    temporary persisted run to avoid relying on stale historical backtest rows.
    """
    evaluation_id = f"runtime_eval_{compute_params_hash({**parameters, 'ticker': ticker})[:12]}_{int(datetime.utcnow().timestamp())}"
    runtime_params = {**(parameters or {}), "ticker": ticker.upper()}
    await run_backtest_background(
        backtest_id=evaluation_id,
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        parameters=runtime_params,
        app_state=app_state,
        pending_run_id=None,
    )

    db_path = app_state["database_path"]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, params_hash, metrics, equity_curve, final_value, total_return, annualized_return,
                   sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return, volatility
            FROM backtest_runs
            WHERE client_backtest_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (evaluation_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {
                "status": "failed",
                "error": "Runtime evaluation did not produce a backtest row",
                "strategy_name": strategy_name,
                "ticker": ticker.upper(),
                "params_hash": compute_params_hash(runtime_params),
                "metrics": {},
                "equity_curve": [],
            }

        metrics = json.loads(row["metrics"] or "{}") if row["metrics"] else {}
        equity_curve = json.loads(row["equity_curve"] or "[]") if row["equity_curve"] else []
        result = {
            "status": str(metrics.get("status", "completed")),
            "error": metrics.get("error"),
            "strategy_name": row["name"],
            "ticker": ticker.upper(),
            "params_hash": row["params_hash"] or compute_params_hash(runtime_params),
            "metrics": {
                "final_value": row["final_value"],
                "total_return": row["total_return"],
                "annualized_return": row["annualized_return"],
                "sharpe_ratio": row["sharpe_ratio"],
                "max_drawdown": row["max_drawdown"],
                "win_rate": row["win_rate"],
                "total_trades": row["total_trades"],
                "avg_trade_return": row["avg_trade_return"],
                "volatility": row["volatility"],
                **metrics,
            },
            "equity_curve": equity_curve if isinstance(equity_curve, list) else [],
        }

        cur.execute("DELETE FROM backtest_runs WHERE id = ?", (row["id"],))
        conn.commit()
        return result
    finally:
        conn.close()
