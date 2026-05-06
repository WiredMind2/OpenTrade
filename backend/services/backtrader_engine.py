from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import backtrader as bt
import numpy as np
import pandas as pd


@dataclass
class BacktraderRunConfig:
    initial_capital: float
    commission_rate: float
    slippage_bps: float
    maxcpus: int = 1
    preload: bool = True
    runonce: bool = True
    optdatas: bool = True
    optreturn: bool = False


def build_run_config(initial_capital: float, parameters: Dict[str, Any] | None) -> BacktraderRunConfig:
    params = parameters or {}
    commission_rate = float(
        params.get("commission_rate", params.get("commission_per_share", 0.005))
    )
    return BacktraderRunConfig(
        initial_capital=float(initial_capital),
        commission_rate=max(0.0, commission_rate),
        slippage_bps=max(0.0, float(params.get("slippage_bps", 0.0) or 0.0)),
        maxcpus=max(1, int(params.get("maxcpus", 1) or 1)),
        preload=bool(params.get("bt_preload", True)),
        runonce=bool(params.get("bt_runonce", True)),
        optdatas=bool(params.get("bt_optdatas", True)),
        optreturn=bool(params.get("bt_optreturn", False)),
    )


def add_price_feeds(cerebro: bt.Cerebro, price_frames: Dict[str, pd.DataFrame]) -> None:
    for ticker, frame in price_frames.items():
        data = bt.feeds.PandasData(
            dataname=frame,
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            name=ticker,
        )
        cerebro.adddata(data)


def attach_default_analyzers(cerebro: bt.Cerebro) -> None:
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")


def _extract_markers(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for marker in raw:
        if not isinstance(marker, dict):
            continue
        side = str(marker.get("side", "")).lower()
        date = marker.get("date")
        if side not in {"buy", "sell"} or date is None:
            continue
        row: Dict[str, Any] = {"date": str(date)[:10], "side": side}
        ticker = marker.get("ticker")
        if ticker:
            row["ticker"] = str(ticker).upper()
        reason = marker.get("reason")
        if reason is not None:
            row["reason"] = str(reason)
        out.append(row)
    return out


def _virtual_close_open_positions(strategy_result: Any) -> List[Dict[str, Any]]:
    """
    Treat any end-of-run open positions as if closed immediately at the last close.

    This is intentionally a "stats-only" view: we do not submit Backtrader orders
    (the run is already complete). We compute mark-to-market PnL based on each data's
    last close and the position average price.
    """
    out: List[Dict[str, Any]] = []
    datas = getattr(strategy_result, "datas", None) or []
    if not isinstance(datas, (list, tuple)) or not datas:
        return out

    for d in datas:
        try:
            pos = strategy_result.getposition(d)
        except Exception:
            continue
        if pos is None:
            continue
        size = float(getattr(pos, "size", 0.0) or 0.0)
        if abs(size) <= 0:
            continue
        avg_px = float(getattr(pos, "price", 0.0) or 0.0)
        try:
            last_close = float(d.close[0])
        except Exception:
            continue
        if last_close <= 0 or avg_px <= 0:
            continue

        # Long: pnl = (last - avg) * size. Short: size < 0 so formula still holds.
        pnl = (last_close - avg_px) * size
        ticker = str(getattr(d, "_name", "") or "UNKNOWN").upper()
        out.append(
            {
                "ticker": ticker,
                "side": "virtual_close",
                "size": size,
                "price": last_close,
                "entry_price": avg_px,
                "pnl": pnl,
                # Commissions/slippage for the "virtual close" are assumed zero; commissions
                # already paid during the run are reflected in cash/value.
                "pnlcomm": pnl,
            }
        )
    return out


def extract_run_metrics(
    strategy_result: Any, initial_capital: float, final_value: float
) -> Dict[str, Any]:
    sharpe_ratio = float(
        ((strategy_result.analyzers.sharpe.get_analysis() or {}).get("sharperatio", 0.0) or 0.0)
    )
    # Backtrader DrawDown analyzer reports percentage points (e.g. 18.7), while
    # API consumers expect ratio units (0.187).
    max_drawdown = float(
        (
            (strategy_result.analyzers.drawdown.get_analysis() or {})
            .get("max", {})
            .get("drawdown", 0.0)
        )
        or 0.0
    ) / 100.0
    # Returns analyzer exposes rnorm100 in percent units; normalize to ratio.
    annualized_return = float(
        ((strategy_result.analyzers.returns.get_analysis() or {}).get("rnorm100", 0.0) or 0.0)
    ) / 100.0
    trade_analysis = strategy_result.analyzers.trades.get_analysis() or {}
    total_node = trade_analysis.get("total", {}) or {}
    won_node = trade_analysis.get("won", {}) or {}
    lost_node = trade_analysis.get("lost", {}) or {}

    won_trades = int(won_node.get("total", 0) or 0) if isinstance(won_node, dict) else 0
    lost_trades = int(lost_node.get("total", 0) or 0) if isinstance(lost_node, dict) else 0

    # Backtrader's TradeAnalyzer reports open trades separately. Using total["total"]
    # (open + closed) in the denominator can produce an artificial 0% win rate for
    # strategies that hold positions open through the end of the run.
    closed_trades = 0
    if isinstance(total_node, dict):
        closed_raw = total_node.get("closed")
        if closed_raw is not None:
            try:
                closed_trades = int(closed_raw or 0)
            except (TypeError, ValueError):
                closed_trades = 0
        else:
            closed_trades = won_trades + lost_trades
            if closed_trades <= 0:
                # Some TradeAnalyzer outputs omit "closed" and only provide "total" and "open".
                # In that case, "total" represents (open + closed). If open == total, there
                # were no closed trades and we must NOT treat "total" as closed.
                open_raw = total_node.get("open")
                total_raw = total_node.get("total")
                open_tr = None
                tot_tr = None
                try:
                    open_tr = int(open_raw) if open_raw is not None else None
                except (TypeError, ValueError):
                    open_tr = None
                try:
                    tot_tr = int(total_raw) if total_raw is not None else None
                except (TypeError, ValueError):
                    tot_tr = None

                if tot_tr is None:
                    closed_trades = 0
                elif open_tr is not None and open_tr >= tot_tr and tot_tr > 0:
                    closed_trades = 0
                elif open_tr is not None and open_tr == 0 and tot_tr > 0:
                    closed_trades = tot_tr
                elif open_tr is None and tot_tr > 0:
                    # If "open" is missing, treat "total" as closed for older analyzer shapes.
                    closed_trades = tot_tr
                else:
                    closed_trades = 0
    else:
        closed_trades = won_trades + lost_trades

    total_trades = max(int(closed_trades), 0)
    win_trades = max(int(won_trades), 0)

    trades = getattr(strategy_result, "trades", []) or []
    if not isinstance(trades, list):
        trades = []

    # User-facing assumption: all trades are "closed" at end-of-run for stats.
    virtual_closes = _virtual_close_open_positions(strategy_result)
    augmented_trades: List[Dict[str, Any]] = [
        t for t in trades if isinstance(t, dict)
    ] + virtual_closes

    # If we had open positions but no closed trades (common for multi-asset allocators),
    # count each virtual close as a closed trade for win-rate/trade stats.
    if virtual_closes:
        total_trades = int(total_trades) + len(virtual_closes)
        win_trades = int(win_trades) + sum(1 for t in virtual_closes if float(t.get("pnlcomm", 0.0) or 0.0) > 0.0)

    win_rate = (float(win_trades) / float(total_trades)) if total_trades > 0 else 0.0

    pnl_comm = [float(t.get("pnlcomm", 0.0) or 0.0) for t in augmented_trades if isinstance(t, dict)]
    avg_trade_return = float(np.mean(pnl_comm)) if pnl_comm else 0.0

    equity_curve = getattr(strategy_result, "equity_curve", []) or []
    equity_values = [
        float(point.get("value"))
        for point in equity_curve
        if isinstance(point, dict) and point.get("value") is not None
    ]
    if len(equity_values) > 1:
        arr = np.asarray(equity_values, dtype=float)
        returns = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
        volatility = float(np.std(returns) * np.sqrt(252))
    else:
        volatility = 0.0

    return {
        "final_value": float(final_value),
        "total_return": (float(final_value) - float(initial_capital)) / float(initial_capital),
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "avg_trade_return": avg_trade_return,
        "volatility": volatility,
        "equity_curve": equity_curve,
        "trades": augmented_trades,
        "decision_markers": _extract_markers(getattr(strategy_result, "decision_markers", None)),
        "execution_summary": {
            "engine": "backtrader",
            "signals_emitted": 0,
            "order_intents": 0,
            "order_fills": 0,
        },
    }


def run_backtrader_once(
    strategy_class: Type[Any],
    strategy_kwargs: Dict[str, Any],
    price_frames: Dict[str, pd.DataFrame],
    config: BacktraderRunConfig,
) -> Dict[str, Any]:
    cerebro = bt.Cerebro(
        preload=config.preload,
        runonce=config.runonce,
        maxcpus=config.maxcpus,
        optdatas=config.optdatas,
        optreturn=config.optreturn,
        stdstats=False,
    )
    cerebro.addstrategy(strategy_class, **(strategy_kwargs or {}))
    add_price_feeds(cerebro, price_frames)
    cerebro.broker.setcash(config.initial_capital)
    cerebro.broker.setcommission(commission=config.commission_rate)
    if config.slippage_bps > 0:
        cerebro.broker.set_slippage_perc(config.slippage_bps / 10000.0)
    attach_default_analyzers(cerebro)

    results = cerebro.run(maxcpus=config.maxcpus)
    if not results:
        raise RuntimeError("Backtest engine returned no strategy results")
    strategy_result = results[0]
    final_value = float(cerebro.broker.getvalue())
    return extract_run_metrics(strategy_result, config.initial_capital, final_value)
