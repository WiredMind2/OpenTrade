"""
Backtest execution engine for the Trading Backtester API.
"""
import json
import sqlite3
import numpy as np
import pandas as pd
import backtrader as bt
from datetime import datetime
from typing import Dict, Any

from backend.logging_config import get_app_logger
from .websocket import broadcast_websocket_message


logger = get_app_logger()


async def run_backtest_background(
    backtest_id: str,
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    parameters: Dict[str, Any],
    app_state: Dict[str, Any]
):
    """Run backtest in background using Backtrader."""
    try:
        logger.info(f"Running background backtest: {strategy_name} (ID: {backtest_id})")

        # Create Backtrader strategy class
        class SentimentStrategy(bt.Strategy):
            params = (
                ('initial_capital', initial_capital),
                ('max_position_pct', parameters.get('max_position_pct', 0.1)),
                ('stop_loss_pct', parameters.get('stop_loss_pct', 0.05)),
                ('take_profit_pct', parameters.get('take_profit_pct', 0.15)),
            )

            def __init__(self):
                self.equity_curve = []
                self.trades = []
                self.predictions = {}

            def next(self):
                # Get predictions for current date
                current_date = self.datas[0].datetime.date(0).isoformat()

                # Load predictions from database for this date
                conn = sqlite3.connect(app_state["database_path"])
                cur = conn.cursor()
                cur.execute("""
                    SELECT ticker, suggested_position_pct
                    FROM trading_model_predictions
                    WHERE dt = ?
                """, (current_date,))
                preds = cur.fetchall()
                conn.close()

                if not preds:
                    return

                # Calculate target allocations
                allocations = {}
                for ticker, pct in preds:
                    allocations[ticker] = pct

                # Enforce position limits
                total_exposure = sum(abs(pct) for pct in allocations.values())
                if total_exposure > self.params.max_position_pct:
                    scale = self.params.max_position_pct / total_exposure
                    allocations = {t: v * scale for t, v in allocations.items()}

                # Execute trades
                for ticker, target_pct in allocations.items():
                    # Find data feed for this ticker
                    data = None
                    for d in self.datas:
                        if hasattr(d, '_name') and d._name == ticker:
                            data = d
                            break

                    if data is None:
                        continue

                    current_position = self.getposition(data).size
                    current_value = current_position * data.close[0]
                    portfolio_value = self.broker.getvalue()
                    target_value = target_pct * portfolio_value

                    if abs(current_value - target_value) < 100:  # Minimum trade size
                        continue

                    if target_value > current_value:
                        # Buy
                        shares_to_buy = int((target_value - current_value) / data.close[0])
                        if shares_to_buy > 0:
                            self.buy(data=data, size=shares_to_buy)
                    elif target_value < current_value:
                        # Sell
                        shares_to_sell = int((current_value - target_value) / data.close[0])
                        if shares_to_sell > 0:
                            self.sell(data=data, size=shares_to_sell)

                # Record equity curve
                self.equity_curve.append({
                    'date': current_date,
                    'value': self.broker.getvalue()
                })

            def notify_trade(self, trade):
                if trade.isclosed:
                    self.trades.append({
                        'size': trade.size,
                        'price': trade.price,
                        'value': trade.value,
                        'pnl': trade.pnl,
                        'pnlcomm': trade.pnlcomm
                    })

        # Set up Backtrader
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SentimentStrategy)

        # Add data feeds for tickers that have predictions
        conn = sqlite3.connect(app_state["database_path"])
        cur = conn.cursor()

        # Get all tickers that have predictions in the date range
        cur.execute("""
            SELECT DISTINCT ticker
            FROM trading_model_predictions
            WHERE dt >= ? AND dt <= ?
        """, (start_date.date().isoformat(), end_date.date().isoformat()))

        tickers = [row[0] for row in cur.fetchall()]

        for ticker in tickers[:10]:  # Limit to 10 tickers for performance
            # Get price data for this ticker
            cur.execute("""
                SELECT date, open, high, low, close, volume
                FROM price_daily
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            """, (ticker, start_date.date().isoformat(), end_date.date().isoformat()))

            price_data = cur.fetchall()
            if not price_data:
                continue

            # Create pandas DataFrame
            df = pd.DataFrame(price_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Create Backtrader data feed
            data = bt.feeds.PandasData(dataname=df, name=ticker)
            cerebro.adddata(data)

        conn.close()

        # Set broker parameters
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=parameters.get('commission_per_share', 0.005))

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        logger.info(f"Starting Backtrader execution for {strategy_name}")
        results = cerebro.run()
        strat = results[0]

        # Extract results
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate metrics
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        annualized_return = strat.analyzers.returns.get_analysis().get('rnorm100', 0)

        # Trade analysis
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get('total', {}).get('total', 0)
        win_trades = trade_analysis.get('won', {}).get('total', 0)
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        # Calculate average trade return
        pnl_comm = [t['pnlcomm'] for t in strat.trades]
        avg_trade_return = np.mean(pnl_comm) if pnl_comm else 0

        # Calculate volatility (simplified)
        equity_values = [point['value'] for point in strat.equity_curve]
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0

        # Store results in database
        conn = sqlite3.connect(app_state["database_path"])
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO backtest_runs (
                id, name, params, started_at, completed_at, initial_capital,
                final_value, total_return, annualized_return, sharpe_ratio,
                max_drawdown, win_rate, total_trades, avg_trade_return,
                volatility, equity_curve, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            backtest_id,
            strategy_name,
            json.dumps(parameters),
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
            json.dumps(strat.equity_curve),
            json.dumps({
                "backtest_id": backtest_id,
                "status": "completed",
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "avg_trade_return": avg_trade_return,
                "volatility": volatility
            })
        ))

        conn.commit()
        conn.close()

        logger.info(f"Background backtest completed: {strategy_name} (ID: {backtest_id})")
        logger.info(f"Results - Final Value: ${final_value:.2f}, Total Return: {total_return:.2%}")

        # Broadcast backtest status update
        await broadcast_websocket_message({
            "type": "backtest_status",
            "data": {
                "strategy_name": strategy_name,
                "start_date": start_date,
                "end_date": datetime.utcnow(),
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
                    "volatility": volatility
                },
                "equity_curve": strat.equity_curve
            }
        })

    except Exception as e:
        logger.error(f"Background backtest failed: {str(e)}")

        # Store failure in database
        try:
            conn = sqlite3.connect(app_state["database_path"])
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO backtest_runs (id, name, params, started_at, completed_at, metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                backtest_id,
                strategy_name,
                json.dumps(parameters),
                start_date.isoformat(),
                datetime.utcnow().isoformat(),
                json.dumps({"status": "failed", "error": str(e)})
            ))
            conn.commit()
            conn.close()

            # Broadcast backtest failure status update
            await broadcast_websocket_message({
                "type": "backtest_status",
                "data": {
                    "strategy_name": strategy_name,
                    "start_date": start_date,
                    "end_date": datetime.utcnow(),
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
                    "metrics": {
                        "backtest_id": backtest_id,
                        "status": "failed",
                        "error": str(e)
                    },
                    "equity_curve": []
                }
            })
        except Exception as db_e:
            logger.error(f"Failed to store backtest failure: {str(db_e)}")