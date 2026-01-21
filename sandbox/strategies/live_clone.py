from .base import Strategy, Signal
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import Original Holons to use their Logic
# Note: We are importing the CLASSES, not the agents themselves
from HolonicTrader.agent_entropy import EntropyHolon
import config

class LiveCloneStrategy(Strategy):
    """
    A faithful recreation of the Live ENTRY logic (Phase 4).
    Uses EntropyHolon for Regime Detection.
    Uses basic Indicator Logic (RSI/BB) from Oracle.
    """
    def __init__(self):
        super().__init__(name="LiveClone(Producción)")
        # We instantiate a helper EntropyHolon. 
        # It's stateless for calculation purposes usually.
        self.entropy_agent = EntropyHolon()
        self.regime = 'ORDERED'
        self.entropy_val = 0.0

    def on_candle(self, slice_df, indicators, portfolio_state) -> Signal:
        # 1. Update Regime (Entropy)
        # Needs last 50 returns
        if len(slice_df) < 50:
             return Signal(direction='HOLD')
             
        # Extract returns window
        # Ensure 'returns' is calculated in playground
        returns_window = slice_df['returns'].tail(50).dropna()
        if len(returns_window) < 20: return Signal(direction='HOLD')
        
        # Calculate Entropy
        self.entropy_val = self.entropy_agent.calculate_shannon_entropy(returns_window)
        self.regime = self.entropy_agent.determine_regime(self.entropy_val)
        
        # 2. Extract Indicators
        rsi = indicators['rsi']
        price = indicators['price']
        bb_lower = indicators['bb_lower']
        bb_upper = indicators['bb_upper']
        
        # 3. Apply Production Entry Rules (from EntryOracleHolon)
        # Rules:
        # A. REGIME CHECK: Only trade in ORDERED or TRANSITION (Block CHAOTIC)
        #    (Unless UNLEASHED - but let's assume Standard)
        
        if self.regime == 'CHAOTIC':
            return Signal(direction='HOLD', reason="Regime: CHAOTIC")
            
        inventory = portfolio_state['inventory']
        
        # ENTRY LOGIC
        if inventory == 0:
            # Rule 1: Oversold (RSI < 35) AND Price < BB Lower * 1.01 (1% Buffer)
            # Relaxed for more signals
            if rsi < 35 and price < (bb_lower * 1.01):
                return Signal(
                    direction='BUY', 
                    size=1.0, 
                    reason=f"Mean Reversion (RSI {rsi:.1f}, Regime {self.regime})"
                )
                
        # EXIT LOGIC
        # In production, Guardian handles this. We simulate Guardian here.
        elif inventory > 0:
            # Rule 1: Overbought (RSI > 70) OR Price > BB Upper
            if rsi > 70 or price > bb_upper:
                 return Signal(
                     direction='SELL',
                     reason=f"Target Reached (RSI {rsi:.1f})"
                 )
            
            # Rule 2: Stop Loss (2% Hard)
            avg_entry = portfolio_state['avg_entry']
            if price < avg_entry * 0.98:
                 return Signal(direction='SELL', reason="Stop Loss (Hard)")
                 
        return Signal(direction='HOLD')
