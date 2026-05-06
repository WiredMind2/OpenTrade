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
    for marker in raw[:500]:
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


def extract_run_metrics(
    strategy_result: Any, initial_capital: float, final_value: float
) -> Dict[str, Any]:
    sharpe_ratio = float(
        ((strategy_result.analyzers.sharpe.get_analysis() or {}).get("sharperatio", 0.0) or 0.0)
    )
    # Backtrader returns these analyzer values in percent units:
    # drawdown=17.98 means 17.98%, rnorm100=12.3 means 12.3%.
    # The rest of the API stores percentages as decimals, so normalize here.
    max_drawdown_pct = float(
        (
            (strategy_result.analyzers.drawdown.get_analysis() or {})
            .get("max", {})
            .get("drawdown", 0.0)
        )
        or 0.0
    )
    annualized_return_pct = float(
        ((strategy_result.analyzers.returns.get_analysis() or {}).get("rnorm100", 0.0) or 0.0)
    )
    max_drawdown = abs(max_drawdown_pct) / 100.0
    annualized_return = annualized_return_pct / 100.0
    trade_analysis = strategy_result.analyzers.trades.get_analysis() or {}
    total_trades = int((trade_analysis.get("total", {}) or {}).get("total", 0) or 0)
    win_trades = int((trade_analysis.get("won", {}) or {}).get("total", 0) or 0)
    win_rate = (float(win_trades) / float(total_trades)) if total_trades > 0 else 0.0

    trades = getattr(strategy_result, "trades", []) or []
    pnl_comm = [float(t.get("pnlcomm", 0.0) or 0.0) for t in trades if isinstance(t, dict)]
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
        "trades": trades,
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
