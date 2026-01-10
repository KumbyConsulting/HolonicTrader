import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Set Path: Parent of 'tests' should be in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"DEBUG: sys.path[0] = {sys.path[0]}")

try:
    import config
    print("DEBUG: Successfully imported config")
    from HolonicTrader.agent_oracle import EntryOracleHolon
    print("DEBUG: Successfully imported EntryOracleHolon")
    from HolonicTrader.agent_governor import GovernorHolon
    print("DEBUG: Successfully imported GovernorHolon")
except ImportError as e:
    print(f"FATAL: Import Error: {e}")
    sys.exit(1)

class TestHolisticPhysics(unittest.TestCase):

    def setUp(self):
        self.oracle = EntryOracleHolon()
        self.governor = GovernorHolon()
        # Mock dependencies
        self.governor.positions = {}
        self.governor.balance = 1000.0
        
        # Disable logging for cleaner test output
        self.oracle._safe_print = MagicMock()
        self.governor._safe_print = MagicMock()

    def test_memecoin_override(self):
        """Test that Memecoins with High RVOL bypass Macro Veto."""
        print("\nTesting Memecoin Physics...")
        
        symbol = 'DOGE/USDT' # Should be in MEMECOIN_ASSETS
        
        # 1. Setup Bearish Macro
        structure_ctx = {'macro_trend': 'BEARISH'}
        
        # 2. Mock Market Data with High RVOL (> 3.0)
        # Create 20 candles
        data = pd.DataFrame({
            'close': [100.0] * 20,
            'volume': [1000.0] * 20
        })
        # Pump the last volume to 5000 (RVOL = 5.0)
        data.iloc[-1, data.columns.get_loc('volume')] = 5000.0 
        
        # Mock other inputs for analyze_for_entry
        bb_vals = {'upper': 105, 'lower': 95, 'middle': 100}
        
        # Override config just in case
        config.MEMECOIN_ASSETS = ['DOGE/USDT']
        config.MEMECOIN_PUMP_RVOL = 3.0
        
        # 3. Run Analysis
        # We need to spy on the internal logic about 'can_long'.
        # Since analyze_for_entry is complex, we'll check if the log message was printed
        
        # Patch the method to inspect locals? No, safer to rely on _safe_print logs.
        
        self.oracle.analyze_for_entry(
            symbol, data, bb_vals, obv_slope=0.1, metabolism_state='PREDATOR', structure_ctx=structure_ctx
        )
        
        # Check logs for the Override message
        self.oracle._safe_print.assert_any_call(
            f"[{self.oracle.name}] 🚀 SECTOR PHYSICS: {symbol} Decoupling from Macro (RVOL 5.0)"
        )
        
    def test_sentiment_regulation(self):
        """Test that Extreme Fear reduces position size."""
        print("\nTesting Emotional Regulation...")
        
        symbol = 'BTC/USDT'
        price = 50000.0
        
        # 1. Normal Sentiment (0.0)
        approved, qty_normal, _ = self.governor.calc_position_size(
            symbol, price, current_atr=100, atr_ref=100, conviction=0.5, sentiment_score=0.0
        )
        
        # 2. Fear Sentiment (-0.8)
        approved, qty_fear, _ = self.governor.calc_position_size(
            symbol, price, current_atr=100, atr_ref=100, conviction=0.5, sentiment_score=-0.8
        )
        
        # Expectation: qty_fear should be ~80% of qty_normal
        # (Assuming risk_multiplier and balance didn't change logic too much between calls)
        
        print(f"Normal Qty: {qty_normal}, Fear Qty: {qty_fear}")
        
        self.assertLess(qty_fear, qty_normal, "Fear should reduce Quantity")
        self.assertAlmostEqual(qty_fear, qty_normal * 0.8, places=4, msg="Should be exactly 20% reduction")

    def test_actuator_zero_price(self):
        """Test that Actuator handles zero price gracefully."""
        print("\nTesting Actuator Safety...")
        from HolonicTrader.agent_actuator import ActuatorHolon
        actuator = ActuatorHolon()
        actuator.exchange = MagicMock() # Mock CCXT
        
        # Check Liquidity with Price = 0
        try:
            result = actuator.check_liquidity("BTC/USDT", "BUY", 1.0, 0.0)
            self.assertTrue(result, "Should return True (Fail Open) on invalid price")
        except ZeroDivisionError:
            self.fail("Actuator raised ZeroDivisionError on price=0")

if __name__ == '__main__':
    unittest.main()
