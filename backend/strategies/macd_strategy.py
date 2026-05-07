"""
MACD Strategy with Trend Filter and Position Sizing.

This module implements a dynamic MACD strategy using an EMA trend filter,
Zero-line pullbacks, and 1% risk position sizing.

It also includes a standalone execution helper so the strategy can be run
outside of the main application for troubleshooting and visualization.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Type

# Ensure standalone execution from the repository root works when the module
# is executed directly (python backend/strategies/macd_strategy.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import backtrader as bt
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_STOCK_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "V",
    "MA",
    "BAC",
    "WMT",
    "PG",
    "KO",
    "DIS",
    "NKE",
    "MCD",
    "SBUX",
    "HD",
    "CVX",
    "XOM",
    "INTC",
    "AMD",
    "ORCL",
    "IBM",
    "BA",
    "CAT",
    "LMT",
    "ABNB",
    "SHOP",
    "SPOT",
]
DEFAULT_FOREX_TICKERS = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURJPY=X",
    "GBPJPY=X",
    "EURGBP=X",
    "AUDJPY=X",
    "EURAUD=X",
    "GBPAUD=X",
    "EURNZD=X",
]
DEFAULT_COMMODITY_TICKERS = [
    "GC=F",
    "CL=F",
    "SI=F",
    "NG=F",
    "ZW=F",
    "ZC=F",
    "ZS=F",
    "ZL=F",
    "HG=F",
    "PA=F",
    "C=F",
    "S=F",
    "KC=F",
    "CT=F",
    "LE=F",
    "HE=F",
]
DEFAULT_FUND_TICKERS = [
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "VOO",
    "VTI",
    "XLK",
    "XLF",
    "XLY",
    "XLP",
    "XLE",
    "XLU",
    "XLB",
    "XLRE",
    "XBI",
    "XHB",
    "XLC",
    "XLI",
    "GLD",
    "SLV",
    "USO",
    "TLT",
    "IEF",
    "EEM",
    "EFA",
    "AGG",
    "LQD",
    "HYG",
]
DEFAULT_MARKET_TICKER_CATEGORIES = {
    "stocks": DEFAULT_STOCK_TICKERS,
    "forex": DEFAULT_FOREX_TICKERS,
    "commodities": DEFAULT_COMMODITY_TICKERS,
    "funds": DEFAULT_FUND_TICKERS,
}
DEFAULT_MARKET_TICKERS = [
    *DEFAULT_STOCK_TICKERS,
    *DEFAULT_FOREX_TICKERS,
    *DEFAULT_COMMODITY_TICKERS,
    *DEFAULT_FUND_TICKERS,
]

from backend.strategies.base import BaseStrategy
from backend.strategies.support import capability_profile, param_float, param_int


class MACDStrategy(BaseStrategy):
    """Improved MACD strategy with Trend Filter & 1% Position Sizing."""

    def __init__(self):
        parameters_schema = {
            "macd_fast": param_int(12, "MACD Fast Period"),
            "macd_slow": param_int(26, "MACD Slow Period"),
            "macd_signal": param_int(9, "MACD Signal Period"),
            "ema_period": param_int(200, "EMA Trend Filter Period"),
            "lowest_period": param_int(10, "Period for structural Stop Loss (Lowest Low)"),
            "risk_pct": param_float(0.01, "Account risk per trade (e.g. 0.01 for 1%)"),
            "reward_ratio": param_float(1.5, "Risk-to-Reward Ratio (e.g. 1.5)"),
        }

        super().__init__(
            name="macd",
            description="MACD with 200 EMA trend filter and zero-line pullback logic",
            type="rule",
            parameters_schema=parameters_schema,
            can_train=False
        )

    @staticmethod
    def _normalize_parameters(parameters: Dict[str, Any]) -> Dict[str, float]:
        params = parameters or {}
        return {
            "macd_fast": max(2, int(params.get("macd_fast", 12))),
            "macd_slow": max(3, int(params.get("macd_slow", 26))),
            "macd_signal": max(2, int(params.get("macd_signal", 9))),
            "ema_period": max(10, int(params.get("ema_period", 200))),
            "lowest_period": max(2, int(params.get("lowest_period", 10))),
            "risk_pct": min(max(float(params.get("risk_pct", 0.01)), 0.001), 1.0),
            "reward_ratio": max(0.1, float(params.get("reward_ratio", 1.5))),
        }

    def get_capability_profile(self) -> Dict[str, Any]:
        return capability_profile(min_history_bars=250)

    def create_backtrader_strategy(self, parameters: Dict[str, Any]) -> Type[bt.Strategy]:
        """Create and return a Backtrader strategy class with MACD logic."""
        normalized = self._normalize_parameters(parameters)

        class UpgradedMACDCrossover(bt.Strategy):
            params = (
                ("macd_fast", normalized["macd_fast"]),
                ("macd_slow", normalized["macd_slow"]),
                ("macd_signal", normalized["macd_signal"]),
                ("ema_period", normalized["ema_period"]),
                ("lowest_period", normalized["lowest_period"]),
                ("risk_pct", normalized["risk_pct"]),
                ("reward_ratio", normalized["reward_ratio"]),
            )

            def __init__(self):
                self.trades = []
                self.equity_curve = []
                self.entry_points = []
                self.exit_points = []
                
                # Multi-data support (assuming strategy might run on multiple feeds)
                self.inds = dict()
                for i, d in enumerate(self.datas):
                    inds = {}
                    
                    # Core indicators
                    inds['macd'] = bt.indicators.MACD(
                        d.close,
                        period_me1=self.p.macd_fast,
                        period_me2=self.p.macd_slow,
                        period_signal=self.p.macd_signal
                    )
                    inds['ema_trend'] = bt.indicators.EMA(d.close, period=self.p.ema_period)
                    inds['lowest_low'] = bt.indicators.Lowest(d.low, period=self.p.lowest_period)
                    inds['highest_high'] = bt.indicators.Highest(d.high, period=self.p.lowest_period)
                    
                    # CrossOver signal (1 if MACD crosses above Signal, -1 if below)
                    inds['crossover'] = bt.indicators.CrossOver(inds['macd'].macd, inds['macd'].signal)
                    
                    self.inds[d] = inds

            def next(self):
                # Record equity
                self.equity_curve.append({
                    "date": self.datetime.date(0).isoformat(),
                    "value": self.broker.get_value(),
                    "cash": self.broker.get_cash()
                })

                for d in self.datas:
                    pos = self.getposition(d)
                    if pos.size:
                        # Already in a position, bracket order handles exit
                        continue
                        
                    inds = self.inds[d]
                    
                    price = d.close[0]
                    ema = inds['ema_trend'][0]
                    macd_val = inds['macd'].macd[0]
                    signal_val = inds['macd'].signal[0]
                    cross = inds['crossover'][0]
                    
                    # BUY LOGIC
                    # 1. Price above 200 EMA (Uptrend)
                    # 2. Both MACD and Signal are below 0 (Slingshot pulled back)
                    # 3. MACD crosses ABOVE signal
                    if price > ema and macd_val < 0 and signal_val < 0 and cross == 1:
                        # Determine Stop Loss
                        stop_loss_price = inds['lowest_low'][0]
                        risk_per_share = price - stop_loss_price
                        
                        if risk_per_share > 0:
                            # Position Sizing
                            account_value = self.broker.get_value()
                            risk_amount = account_value * self.p.risk_pct
                            shares_to_buy = int(risk_amount / risk_per_share)
                            
                            if shares_to_buy > 0:
                                # Determine Take Profit
                                take_profit_price = price + (risk_per_share * self.p.reward_ratio)
                                
                                # Send Bracket Order
                                self.buy_bracket(
                                    data=d,
                                    size=shares_to_buy,
                                    stopprice=stop_loss_price,
                                    limitprice=take_profit_price,
                                    exectype=bt.Order.Market  # Entry
                                )
                    
                    # SELL LOGIC
                    # 1. Price below 200 EMA (Downtrend)
                    # 2. Both MACD and Signal are above 0
                    # 3. MACD crosses BELOW signal
                    elif price < ema and macd_val > 0 and signal_val > 0 and cross == -1:
                        stop_loss_price = inds['highest_high'][0]
                        risk_per_share = stop_loss_price - price
                        
                        if risk_per_share > 0:
                            account_value = self.broker.get_value()
                            risk_amount = account_value * self.p.risk_pct
                            shares_to_short = int(risk_amount / risk_per_share)
                            
                            if shares_to_short > 0:
                                take_profit_price = price - (risk_per_share * self.p.reward_ratio)
                                
                                self.sell_bracket(
                                    data=d,
                                    size=shares_to_short,
                                    stopprice=stop_loss_price,
                                    limitprice=take_profit_price,
                                    exectype=bt.Order.Market
                                )

            def notify_order(self, order):
                if order.status != order.Completed:
                    return

                executed_date = self.data.datetime.date(0)
                executed_price = getattr(order.executed, "price", None)
                if executed_price is None:
                    return

                if order.parent is None:
                    # Entry execution for a traded signal.
                    if order.isbuy():
                        self.entry_points.append({
                            "date": executed_date,
                            "price": executed_price,
                            "side": "buy",
                        })
                    elif order.issell():
                        self.entry_points.append({
                            "date": executed_date,
                            "price": executed_price,
                            "side": "sell",
                        })
                else:
                    # Exit execution for a bracket order.
                    if order.isbuy():
                        self.exit_points.append({
                            "date": executed_date,
                            "price": executed_price,
                            "side": "buy",
                        })
                    elif order.issell():
                        self.exit_points.append({
                            "date": executed_date,
                            "price": executed_price,
                            "side": "sell",
                        })

            def notify_trade(self, trade):
                if trade.isclosed:
                    self.trades.append({
                        "date": self.datetime.datetime(0).isoformat(),
                        "ref": trade.ref,
                        "pnl": trade.pnl,
                        "pnlcomm": trade.pnlcomm,
                    })

        return UpgradedMACDCrossover

    def project(self, parameters: Dict[str, Any], projection_days: int = 30, initial_capital: float = 100000.0) -> Dict[str, Any]:
        return {
            "projected_return": 0.0,
            "projected_volatility": 0.0,
            "confidence_interval": [0.0, 0.0],
            "metrics": {}
        }


def _download_yfinance_intraday_data(
    ticker: str,
    interval: str = "1h",
    lookback_days: int = 60,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download intraday bars from yfinance for the requested interval and range."""
    if start_date is not None:
        datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is not None:
        datetime.strptime(end_date, "%Y-%m-%d")

    if start_date or end_date:
        if lookback_days is None:
            lookback_days = 60
        lookback_days = min(max(1, lookback_days), 60)
        now_date = datetime.utcnow().date()
        if end_date is None:
            end_date_dt = now_date
        else:
            end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

        if end_date_dt < now_date - timedelta(days=60):
            raise ValueError(
                "Intraday data is only available for approximately the last 60 days. "
                "Please choose an end_date within that window."
            )

        start_date_dt = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if start_date is not None
            else end_date_dt - timedelta(days=lookback_days - 1)
        )

        if (end_date_dt - start_date_dt).days + 1 > 60:
            start_date_dt = end_date_dt - timedelta(days=59)
            print(
                "Warning: intraday range is limited to the most recent 60 days. "
                f"Using start_date={start_date_dt.isoformat()} and end_date={end_date_dt.isoformat()}."
            )

        start_date = start_date_dt.isoformat()
        end_date = end_date_dt.isoformat()

        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    else:
        lookback_days = min(max(1, lookback_days), 60)
        raw = yf.download(
            ticker,
            period=f"{lookback_days}d",
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

    if raw.empty:
        raise ValueError(
            f"No intraday data downloaded for ticker {ticker} with interval {interval}. "
            f"Verify the ticker, date range, interval, or network connectivity."
        )

    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.dropna()
    if raw.empty:
        raise ValueError(
            f"Downloaded intraday data for ticker {ticker} contained no valid bars after cleaning. "
            f"Verify the ticker and try again."
        )

    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.rename(columns={"Adj Close": "adjclose"})

    return raw[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )


def _download_yfinance_daily_data(
    ticker: str,
    periods: int = 1200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily bars from yfinance and trim to the requested period count or range."""
    if start_date is not None:
        datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is not None:
        datetime.strptime(end_date, "%Y-%m-%d")

    # Use daily data so the 200 EMA is based on a daily trend.
    if start_date or end_date:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    else:
        period_days = max(periods + 30, 365)
        period_string = f"{period_days}d"
        raw = yf.download(
            ticker,
            period=period_string,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )

    if raw.empty:
        raise ValueError(
            f"No daily data downloaded for ticker {ticker}. "
            f"Verify the ticker symbol, date range, or network connectivity."
        )

    # Flatten MultiIndex columns returned by yfinance for a single ticker.
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.dropna()
    if raw.empty:
        raise ValueError(
            f"Downloaded data for ticker {ticker} contained no valid daily bars after cleaning. "
            f"Verify the ticker and try again."
        )

    if not (start_date or end_date):
        if len(raw) < periods:
            print(
                f"Warning: only {len(raw)} daily bars available for {ticker}; "
                f"requested {periods}. Using the available range."
            )
            trimmed = raw.copy()
        else:
            trimmed = raw.tail(periods).copy()
    else:
        trimmed = raw.copy()

    trimmed.index = pd.DatetimeIndex(trimmed.index).tz_localize(None)
    trimmed = trimmed.rename(columns={"Adj Close": "adjclose"})

    return trimmed[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )


def _load_db_daily_data(
    ticker: str,
    periods: int = 300,
    db_path: str = "data/backtest.db",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load daily bars from the local app database and trim to the requested period count."""
    if start_date is not None:
        datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is not None:
        datetime.strptime(end_date, "%Y-%m-%d")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if start_date and end_date:
            query = (
                "SELECT date, open, high, low, close, volume "
                "FROM price_daily "
                "WHERE ticker = ? AND date >= ? AND date <= ? "
                "ORDER BY date ASC"
            )
            params = (ticker.upper(), start_date, end_date)
        elif start_date:
            query = (
                "SELECT date, open, high, low, close, volume "
                "FROM price_daily "
                "WHERE ticker = ? AND date >= ? "
                "ORDER BY date ASC"
            )
            params = (ticker.upper(), start_date)
        elif end_date:
            query = (
                "SELECT date, open, high, low, close, volume "
                "FROM price_daily "
                "WHERE ticker = ? AND date <= ? "
                "ORDER BY date ASC"
            )
            params = (ticker.upper(), end_date)
        else:
            query = (
                "SELECT date, open, high, low, close, volume "
                "FROM price_daily "
                "WHERE ticker = ? "
                "ORDER BY date DESC "
                "LIMIT ?"
            )
            params = (ticker.upper(), periods)

        rows = cur.execute(query, params).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No daily data available for ticker {ticker} in database {db_path}. "
            f"Verify the ticker and date range."
        )

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if len(df) > periods and not (start_date or end_date):
        df = df.tail(periods).copy()

    df = df.set_index("date")
    return df


def run_ticker_batch(
    parameters: Dict[str, Any] | None,
    tickers: list[str],
    interval: str,
    lookback_days: int,
    periods: int,
    initial_capital: float,
    commission_rate: float,
    slippage_bps: float,
    use_db_data: bool,
    db_path: str,
    start_date: str | None,
    end_date: str | None,
) -> Dict[str, Any]:
    results = []
    aggregate_pnl = 0.0
    aggregate_trades = 0
    total_final_value = 0.0
    skipped = []

    for ticker in tickers:
        try:
            result = run_standalone(
                parameters=parameters,
                ticker=ticker,
                periods=periods,
                interval=interval,
                lookback_days=lookback_days,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_bps=slippage_bps,
                use_db_data=use_db_data,
                db_path=db_path,
                start_date=start_date,
                end_date=end_date,
                plot=False,
            )
        except Exception as exc:
            skipped.append({"ticker": ticker, "error": str(exc)})
            continue

        result_summary = {
            "ticker": ticker,
            "trade_count": result["trade_count"],
            "total_pnl": result["total_pnl"],
            "final_value": result["final_value"],
            "start_value": result["start_value"],
        }
        results.append(result_summary)
        aggregate_pnl += result["total_pnl"]
        aggregate_trades += result["trade_count"]
        total_final_value += result["final_value"]

        print(
            f"{ticker:>8} | trades={result['trade_count']:>3} | pnl={result['total_pnl']:>11.2f} | final={result['final_value']:>11.2f}"
        )

    average_final = total_final_value / len(results) if results else 0.0
    average_pnl = aggregate_pnl / len(results) if results else 0.0
    average_trades = aggregate_trades / len(results) if results else 0.0
    aggregate_total_return = (
        aggregate_pnl / (initial_capital * len(results)) if results and initial_capital else 0.0
    )

    summary = {
        "tickers": [r["ticker"] for r in results],
        "results": results,
        "skipped": skipped,
        "aggregate_pnl": aggregate_pnl,
        "aggregate_trades": aggregate_trades,
        "average_final_value": average_final,
        "average_pnl": average_pnl,
        "average_trade_count": average_trades,
        "aggregate_total_return": aggregate_total_return,
    }
    return summary


def _format_percentage(value: float) -> str:
    return f"{value * 100:,.2f}%"


def _print_category_summary(category: str, batch_result: Dict[str, Any], initial_capital: float) -> None:
    total_tickers = len(batch_result["results"])
    total_return = (
        batch_result["aggregate_pnl"] / (initial_capital * total_tickers)
        if total_tickers and initial_capital
        else 0.0
    )

    print(f"\nSummary for category: {category}")
    print("-" * 78)
    print(f"  Tickers evaluated : {total_tickers}")
    print(f"  Total trades      : {batch_result['aggregate_trades']}")
    print(f"  Aggregate PnL     : {batch_result['aggregate_pnl']:.2f}")
    print(f"  Total return      : {_format_percentage(total_return)}")

    if batch_result["skipped"]:
        print("  Skipped tickers   :")
        for skipped in batch_result["skipped"]:
            print(f"    - {skipped['ticker']}: {skipped['error']}")


def _compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    macd_line = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "ema_200": df["close"].ewm(span=200, adjust=False).mean(),
        },
        index=df.index,
    )


def _plot_backtest(df: pd.DataFrame, strategy: bt.Strategy) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    try:
        from mplfinance.original_flavor import candlestick_ohlc
    except ImportError as exc:
        raise ImportError(
            "mplfinance is required to draw candlestick charts. Install it with `pip install mplfinance`."
        ) from exc

    indicators = _compute_macd(df)
    equity_curve = pd.DataFrame(strategy.equity_curve)
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x_indices = np.arange(len(df))
    candlestick_data = np.column_stack([
        x_indices,
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
    ])
    candlestick_ohlc(
        axes[0],
        candlestick_data,
        width=0.6,
        colorup="green",
        colordown="red",
        alpha=0.9,
    )

    axes[0].plot(
        x_indices,
        indicators["ema_200"].values,
        label="EMA 200",
        color="royalblue",
        linewidth=1.8,
        antialiased=True,
        solid_capstyle="round",
    )

    # Plot buy/sell execution moments recorded by the strategy.
    entry_points = getattr(strategy, "entry_points", [])
    exit_points = getattr(strategy, "exit_points", [])

    buy_points = [p for p in entry_points + exit_points if p["side"] == "buy"]
    sell_points = [p for p in entry_points + exit_points if p["side"] == "sell"]

    # Add points matching dates directly to their corresponding integer index
    buy_indices = []
    buy_prices = []
    for p in buy_points:
        dt = pd.to_datetime(p["date"])
        # We find the nearest index in the dataframe (if dt is a date only, we might need a daily match)
        # Using tz-naive date matching
        match = df.index[df.index.normalize() == dt] if not len(df.index[df.index == dt]) else df.index[df.index == dt]
        if len(match) > 0:
            idx = df.index.get_loc(match[0])
            buy_indices.append(idx)
            buy_prices.append(p["price"])

    sell_indices = []
    sell_prices = []
    for p in sell_points:
        dt = pd.to_datetime(p["date"])
        match = df.index[df.index.normalize() == dt] if not len(df.index[df.index == dt]) else df.index[df.index == dt]
        if len(match) > 0:
            idx = df.index.get_loc(match[0])
            sell_indices.append(idx)
            sell_prices.append(p["price"])

    if buy_indices:
        axes[0].scatter(
            buy_indices,
            buy_prices,
            marker="^",
            color="lime",
            edgecolors="black",
            linewidths=1.5,
            s=250,
            label="Buy Execution",
            zorder=10,
        )
    if sell_indices:
        axes[0].scatter(
            sell_indices,
            sell_prices,
            marker="v",
            color="red",
            edgecolors="black",
            linewidths=1.5,
            s=250,
            label="Sell Execution",
            zorder=10,
        )

    axes[0].set_title("Buy/Sell Price Chart with Candlesticks and EMA Curve")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    
    def format_date(x, pos=None):
        if x < 0 or x >= len(df):
            return ""
        return df.index[int(x)].strftime("%Y-%m-%d %H:%M")
        
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_date))

    axes[1].plot(
        x_indices,
        indicators["macd"].values,
        label="MACD",
        color="tab:blue",
        linewidth=2.0,
        antialiased=True,
        solid_capstyle="round",
    )
    axes[1].plot(
        x_indices,
        indicators["signal"].values,
        label="Signal",
        color="tab:orange",
        linewidth=2.0,
        antialiased=True,
        solid_capstyle="round",
    )
    axes[1].axhline(0.0, color="black", linewidth=1.5, linestyle="--", alpha=0.8, label="Zero Line")
    axes[1].set_title("MACD Trend Line Chart")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def run_standalone(
    parameters: Dict[str, Any] | None = None,
    ticker: str = "AAPL",
    periods: int = 1200,
    interval: str = "1d",
    lookback_days: int = 60,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.005,
    slippage_bps: float = 0.0,
    use_db_data: bool = False,
    db_path: str = "data/backtest.db",
    start_date: str | None = None,
    end_date: str | None = None,
    plot: bool = True,
) -> Dict[str, Any]:
    parameters = parameters or {}
    if interval != "1d" and use_db_data:
        print("Warning: intraday price loading from DB is not supported; falling back to yfinance intraday data.")
        use_db_data = False

    if use_db_data:
        try:
            data = _load_db_daily_data(
                ticker=ticker,
                periods=periods,
                db_path=db_path,
                start_date=start_date,
                end_date=end_date,
            )
        except ValueError as exc:
            print(f"Warning: {exc}. Falling back to yfinance daily data.")
            data = _download_yfinance_daily_data(
                ticker=ticker,
                periods=periods,
                start_date=start_date,
                end_date=end_date,
            )
    elif interval == "1d":
        data = _download_yfinance_daily_data(
            ticker=ticker,
            periods=periods,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        data = _download_yfinance_intraday_data(
            ticker=ticker,
            interval=interval,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
        )

    if not data.empty:
        actual_start = data.index.min().date()
        actual_end = data.index.max().date()
        print(f"Using data range: {actual_start} to {actual_end} ({len(data)} bars)")

    # Align with the web app backtest engine's expected feed format.
    feed_data = data.reset_index()
    feed_data = feed_data.rename(columns={feed_data.columns[0]: "date"})

    cerebro = bt.Cerebro(preload=True, runonce=True, optdatas=True, optreturn=False, stdstats=False)
    cerebro.broker.set_cash(initial_capital)
    cerebro.addstrategy(MACDStrategy().create_backtrader_strategy(parameters))
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=feed_data,
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
        )
    )
    cerebro.broker.setcommission(commission=commission_rate)
    if slippage_bps > 0.0:
        cerebro.broker.set_slippage_perc(slippage_bps / 10000.0)

    results = cerebro.run()
    strategy = results[0]

    final_value = cerebro.broker.get_value()
    trades = getattr(strategy, "trades", [])
    equity_curve = getattr(strategy, "equity_curve", [])

    if plot:
        _plot_backtest(data, strategy)

    return {
        "final_value": final_value,
        "start_value": initial_capital,
        "trade_count": len(trades),
        "total_pnl": sum([t.get("pnl", 0.0) for t in trades]),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MACD strategy standalone with daily data.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol to use for the backtest.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated list of tickers for batch execution.")
    parser.add_argument("--batch", action="store_true", help="Run the strategy across the default ticker universe and aggregate results.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories to run: stocks,forex,commodities,funds.")
    parser.add_argument("--periods", type=int, default=300, help="Number of bars to use.")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval: 1d, 1h, 30m, 15m.")
    parser.add_argument("--lookback-days", type=int, default=60, help="Lookback days for intraday yfinance data.")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting capital for the backtest.")
    parser.add_argument("--commission-rate", type=float, default=0.005, help="Broker commission rate used in the backtest.")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage basis points used in the backtest.")
    parser.add_argument("--use-db-data", action="store_true", help="Load price data from the local app database instead of yfinance.")
    parser.add_argument("--db-path", type=str, default="data/backtest.db", help="Path to the local SQLite DB containing price_daily history.")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD) for price history.")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD) for price history.")
    parser.add_argument("--macd-fast", type=int, default=None, help="Override the MACD fast period.")
    parser.add_argument("--macd-slow", type=int, default=None, help="Override the MACD slow period.")
    parser.add_argument("--macd-signal", type=int, default=None, help="Override the MACD signal period.")
    parser.add_argument("--ema-period", type=int, default=None, help="Override the EMA trend period.")
    parser.add_argument("--risk-pct", type=float, default=None, help="Override the per-trade risk percentage.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting and only print summary output.")
    args = parser.parse_args()

    parameters: Dict[str, Any] = {}
    if args.macd_fast is not None:
        parameters["macd_fast"] = args.macd_fast
    if args.macd_slow is not None:
        parameters["macd_slow"] = args.macd_slow
    if args.macd_signal is not None:
        parameters["macd_signal"] = args.macd_signal
    if args.ema_period is not None:
        parameters["ema_period"] = args.ema_period
    if args.risk_pct is not None:
        parameters["risk_pct"] = args.risk_pct

    if args.batch or args.tickers:
        if args.tickers:
            category_batches = {"custom": [t.strip().upper() for t in args.tickers.split(",") if t.strip()]}
        else:
            selected_categories = [
                c.strip().lower()
                for c in (args.categories or ",".join(DEFAULT_MARKET_TICKER_CATEGORIES.keys())).split(",")
                if c.strip()
            ]
            category_batches = {
                c: DEFAULT_MARKET_TICKER_CATEGORIES[c]
                for c in selected_categories
                if c in DEFAULT_MARKET_TICKER_CATEGORIES
            }

        if not category_batches:
            print("No valid categories selected. Available categories: ", ", ".join(DEFAULT_MARKET_TICKER_CATEGORIES.keys()))
            return

        print("\n" + "=" * 78)
        print("Batch backtest starting")
        print(f"Categories         : {', '.join(category_batches.keys())}")
        print(f"Interval           : {args.interval}")
        print(f"Bars requested     : {args.periods}")
        print(f"Initial capital    : {args.initial_capital:.2f}")
        print("=" * 78)

        overall_aggregate_pnl = 0.0
        overall_aggregate_trades = 0
        overall_total_final = 0.0
        overall_tickers_evaluated = 0

        for category, tickers in category_batches.items():
            print("\n" + "-" * 78)
            print(f"Category: {category} ({len(tickers)} tickers)")
            print("-" * 78)
            batch_result = run_ticker_batch(
                parameters=parameters,
                tickers=tickers,
                interval=args.interval,
                lookback_days=args.lookback_days,
                periods=args.periods,
                initial_capital=args.initial_capital,
                commission_rate=args.commission_rate,
                slippage_bps=args.slippage_bps,
                use_db_data=args.use_db_data,
                db_path=args.db_path,
                start_date=args.start_date,
                end_date=args.end_date,
            )

            _print_category_summary(category, batch_result, args.initial_capital)

            overall_aggregate_pnl += batch_result["aggregate_pnl"]
            overall_aggregate_trades += batch_result["aggregate_trades"]
            overall_total_final += sum(r["final_value"] for r in batch_result["results"])
            overall_tickers_evaluated += len(batch_result["results"])

        overall_average_final = overall_total_final / overall_tickers_evaluated if overall_tickers_evaluated else 0.0
        overall_total_return = (
            overall_aggregate_pnl / (args.initial_capital * overall_tickers_evaluated)
            if overall_tickers_evaluated
            else 0.0
        )

        print("\n" + "=" * 78)
        print("Overall batch summary")
        print("=" * 78)
        print(f"  Categories evaluated  : {len(category_batches)}")
        print(f"  Total tickers         : {overall_tickers_evaluated}")
        print(f"  Total trades          : {overall_aggregate_trades}")
        print(f"  Aggregate PnL         : {overall_aggregate_pnl:.2f}")
        print(f"  Total return          : {_format_percentage(overall_total_return)}")
        print(f"  Average final value   : {overall_average_final:.2f}")
    else:
        result = run_standalone(
            parameters=parameters,
            ticker=args.ticker,
            periods=args.periods,
            interval=args.interval,
            lookback_days=args.lookback_days,
            initial_capital=args.initial_capital,
            commission_rate=args.commission_rate,
            slippage_bps=args.slippage_bps,
            use_db_data=args.use_db_data,
            db_path=args.db_path,
            start_date=args.start_date,
            end_date=args.end_date,
            plot=not args.no_plot,
        )

        print("\n" + "=" * 78)
        print("Standalone MACD backtest finished")
        print("=" * 78)
        print(f"Start Value      : {result['start_value']:.2f}")
        print(f"Final Value      : {result['final_value']:.2f}")
        print(f"Trade Count      : {result['trade_count']}")
        print(f"Total PnL        : {result['total_pnl']:.2f}")


if __name__ == "__main__":
    main()
