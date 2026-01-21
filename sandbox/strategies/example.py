from .base import Strategy, Signal
import pandas as pd
from typing import Dict, Any

class RSIScalper(Strategy):
    """
    A simple Mean Reversion strategy.
    Buy if RSI < 30.
    Sell if RSI > 70.
    """
    def __init__(self, rsi_low=30, rsi_high=70):
        super().__init__(name=f"RSIScalper({rsi_low}-{rsi_high})")
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

    def on_candle(self, slice_df, indicators, portfolio_state) -> Signal:
        rsi = indicators['rsi']
        price = indicators['price']
        inventory = portfolio_state['inventory']
        
        # Entry Logic
        if inventory == 0:
            if rsi < self.rsi_low:
                return Signal(direction='BUY', size=1.0, reason=f"RSI Oversold ({rsi:.1f})")
        
        # Exit Logic
        elif inventory > 0:
            if rsi > self.rsi_high:
                return Signal(direction='SELL', reason=f"RSI Overbought ({rsi:.1f})")
            
            # Stop Loss (Manual Logic)
            avg_entry = portfolio_state['avg_entry']
            if price < avg_entry * 0.98: # 2% Hard Stop
                 return Signal(direction='SELL', reason="Stop Loss")
                 
        return Signal(direction='HOLD')
