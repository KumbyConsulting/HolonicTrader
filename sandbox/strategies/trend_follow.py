from .base import Strategy, Signal
import pandas as pd
import numpy as np

class TrendFollowStrategy(Strategy):
    """
    Classic Trend Following.
    Long if EMA 12 > EMA 26 (MACD Proxy) and RSI > 50.
    """
    def __init__(self, fast=12, slow=26):
        super().__init__(name=f"TrendFollow({fast}/{slow})")
        self.fast = fast
        self.slow = slow

    def on_candle(self, slice_df, indicators, portfolio_state) -> Signal:
        # Calculate EMAs on the fly (or pre-calc if optimized)
        # Sandbox passes slice, so we calc off tail
        closes = slice_df['close']
        
        if len(closes) < self.slow + 2: return Signal('HOLD')
        
        ema_fast = closes.ewm(span=self.fast, adjust=False).mean().iloc[-1]
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean().iloc[-1]
        
        prev_fast = closes.ewm(span=self.fast, adjust=False).mean().iloc[-2]
        prev_slow = closes.ewm(span=self.slow, adjust=False).mean().iloc[-2]
        
        # Crossover Event
        crossover_up = (prev_fast <= prev_slow) and (ema_fast > ema_slow)
        
        inventory = portfolio_state['inventory']
        
        if inventory == 0:
            if crossover_up and indicators['rsi'] > 50:
                 return Signal('BUY', reason="Golden Cross + Momentum")
                 
        elif inventory > 0:
            # Exit on Death Cross
            if ema_fast < ema_slow:
                 return Signal('SELL', reason="Death Cross")
                 
        return Signal('HOLD')
