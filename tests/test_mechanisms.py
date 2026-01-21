
import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import time

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Config
sys.modules['config'] = MagicMock()
import config
# Satellite Params
config.SATELLITE_BBW_EXPANSION_THRESHOLD = 0.0001
config.SATELLITE_RVOL_THRESHOLD = 0.1 # Relax to focus on logic flow
config.SATELLITE_ENTRY_RSI_CAP = 80 # High cap to ensure RSI doesn't block, so we test the Breakout Logic removal
config.SATELLITE_ASSETS = ['BTC/USDT']
config.GMB_THRESHOLD = 0.40 # Fix for TypeError in apply_asset_personality
config.IRON_BANK_BUFFER_PCT = 0.10
config.IRON_BANK_RATCHET_PCT = 0.50
config.VOL_SCALAR_MIN = 0.5
config.VOL_SCALAR_MAX = 2.0
config.KELLY_LOOKBACK = 20
config.CORRELATION_CHECK = True
# Governor Params
config.GOVERNOR_COOLDOWN_SECONDS = 60
config.ACC_SANITY_THRESHOLD = 0.5 # Fix for update_accumulator
config.ACC_DRAWDOWN_LIMIT = 0.99 # Fix for update_accumulator
config.GOVERNOR_MIN_STACK_DIST = 0.01
config.MIN_ORDER_VALUE = 10.0
config.IRON_BANK_ENABLED = False
# Other imports
config.USE_INTEL_GPU = False
config.POLYMARKET_PATIENCE_MINUTES = 3 
config.PERSONALITY_BTC_ATR_FILTER = 0.8
config.PERSONALITY_DOGE_RVOL = 2.0
config.PERSONALITY_SOL_RSI_LONG = 40
config.PERSONALITY_SOL_RSI_SHORT = 60
config.PHYSICS_MIN_RVOL = 1.0


# Mock Deps
sys.modules['HolonicTrader.agent_ppo'] = MagicMock()
sys.modules['performance_tracker'] = MagicMock()

from HolonicTrader.agent_oracle import EntryOracleHolon as OracleHolon
from HolonicTrader.agent_governor import GovernorHolon

class TestMechanisms(unittest.TestCase):

    def setUp(self):
        self.oracle = OracleHolon("TestOracle")
        self.governor = GovernorHolon("TestGov", initial_balance=100.0)
        self.governor.available_balance = 100.0
        self.governor.DEBUG = True # Enable debug prints to verify output

    def test_satellite_logic_fix(self):
        """
        Verify Satellite Strategy fires on a Dip (Price NOT breaking out).
        Previously, logic required Price > UpperBand (Breakout) which conflicted with RSI Cap.
        We verify that a Dip (Price < Upper) now generates a signal.
        """
        symbol = 'BTC/USDT'
        
        # 1. 1H Data (Bullish Trend for Alignment)
        dates = pd.date_range(start="2025-01-01", periods=250, freq="1H")
        df_1h = pd.DataFrame({
            'timestamp': dates,
            'close': np.linspace(100, 200, 250), # Strong uptrend
            'volume': [100]*250
        })
        
        # 2. 15m Data (Bullish Trend but Currently Dipping)
        dates_15 = pd.date_range(start="2025-01-09", periods=100, freq="15min")
        # EMA50 approx 190. We want price > 190 but < UpperBand.
        # Regular linear growth
        closes = np.linspace(180, 200, 100) 
        # Add a dip at the end:
        # Last values drop from 200 to 195. 
        # Upper Band (approx 200+). Price (195) is definitely < Upper.
        closes[-10:] = np.linspace(200, 195, 10)
        
        df_15m = pd.DataFrame({
            'timestamp': dates_15,
            'close': closes,
            'volume': np.array([100]*90 + [500]*10) # Spike volume at end for RVOL
        })
        
        # Mock Observer
        observer = MagicMock()
        observer.fetch_market_data.return_value = df_15m
        
        # Run
        signal = self.oracle.analyze_satellite_entry(symbol, df_1h, observer)
        
        self.assertIsNotNone(signal, "Satellite Signal should generate on Dip (Breakout check removed).")
        if signal:
            self.assertEqual(signal.direction, 'BUY')
            print("\n✅ Satellite Logic Fix Verified: Signal generated on Dip.")

    def test_governor_cooldown(self):
        """Verify Governor blocks trades during cooldown."""
        symbol = 'BTC/USDT'
        self.governor.last_trade_time[symbol] = time.time() # Just traded
        
        allowed = self.governor.is_trade_allowed(symbol, 50000)
        self.assertFalse(allowed, "Governor should block check inside cooldown")
        
        # Fast forward time (mock logic or simple manual check logic)
        self.governor.last_trade_time[symbol] = time.time() - 400
        allowed_p = self.governor.is_trade_allowed(symbol, 50000)
        self.assertTrue(allowed_p, "Governor should allow after cooldown")
        print("\n✅ Governor Cooldown Verified.")

    def test_governor_solvency(self):
        """Verify Governor blocks execution if bankrupt."""
        symbol = 'BTC/USDT'
        self.governor.available_balance = 5.0 # Below Min $10
        self.governor.last_trade_time[symbol] = 0
        
        allowed = self.governor.is_trade_allowed(symbol, 50000)
        self.assertFalse(allowed, "Governor should block if insolvent")
        print("\n✅ Governor Solvency Gate Verified.")

    def test_governor_risk_budget_update(self):
        """Verify set_live_balance updates Risk Budget (The Fix)."""
        config.IRON_BANK_ENABLED = True # Enable for this test
        self.governor.fortress_balance = 50.0
        # Initial budget is 0
        self.assertEqual(self.governor.risk_budget, 0.0)
        
        # Simulate Live Update (Balance 100 > Fortress 50) -> Budget should be 50.
        # BUT: Ratchet triggers (Threshold 55). Surplus 50. Lock 50% -> 25.
        # New Fortress 75. New Budget 25.
        self.governor.set_live_balance(100.0, 100.0)
        
        self.assertEqual(self.governor.risk_budget, 25.0, "Risk Budget should update (and Ratchet) after set_live_balance")
        print("\n✅ Governor Risk Budget Update Verified.")

if __name__ == '__main__':
    unittest.main()
