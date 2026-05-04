"""
MACD Strategy with Trend Filter and Position Sizing.

This module implements a dynamic MACD strategy using an EMA trend filter,
Zero-line pullbacks, and 1% risk position sizing.
"""

from typing import Dict, Any, Type
import backtrader as bt

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
                                    price=price,
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
                                    price=price,
                                    stopprice=stop_loss_price,
                                    limitprice=take_profit_price,
                                    exectype=bt.Order.Market
                                )

            def notify_order(self, order):
                pass
                
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
