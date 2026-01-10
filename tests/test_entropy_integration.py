import unittest
from unittest.mock import MagicMock
import pandas as pd
import sys
import os

# Set Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.agent_oracle import EntryOracleHolon

class TestEntropyIntegration(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon()
        # Mock ML models to return fixed high probability
        self.oracle.predict_trend_xgboost = MagicMock(return_value=0.9)
        self.oracle.predict_trend_lstm = MagicMock(return_value=0.9)
        self.oracle._safe_print = MagicMock()
        self.oracle.get_kalman_estimate = MagicMock(return_value=100.0)
        self.oracle.get_market_bias = MagicMock(return_value=0.5)

    def test_chaotic_dampening(self):
        print("\nTesting Chaotic Dampening...")
        symbol = "BTC/USDT"
        
        # Mock Data (Window needed for feature eng)
        data = pd.DataFrame({
            'close': [100.0] * 30,
            'high': [101.0] * 30,
            'low': [99.0] * 30,
            'volume': [1000.0] * 30
        })
        
        # Context with CHAOTIC Regime
        ctx = {'entropy_regime': 'CHAOTIC'}
        bb = {'upper': 110, 'lower': 90, 'middle': 100}
        
        # Call Analyze
        # This calls predict_trend_xgboost (returns 0.9)
        # Chaos logic should dampen it: 0.5 + (0.9 - 0.5)*0.5 = 0.5 + 0.2 = 0.7
        
        self.oracle.analyze_for_entry (
            symbol, data, bb, 0.1, 'PREDATOR', structure_ctx=ctx
        )
        
        # Check if the dampening log was printed
        # "🌪️ CHAOS DETECTED (BTC/USDT): Dampening Confidence (XGB 0.90->0.70)"
        self.oracle._safe_print.assert_any_call(
            f"[{self.oracle.name}] 🌪️ CHAOS DETECTED ({symbol}): Dampening Confidence (XGB 0.90->0.70)"
        )
        print("✅ Log check passed.")
        
        # Verify stored probe value
        stored_prob = self.oracle.last_probes[symbol]['xgb']
        self.assertAlmostEqual(stored_prob, 0.7, places=2)
        print(f"✅ Probe Value Verified: {stored_prob:.2f} (Expected 0.7)")

    def test_ordered_no_change(self):
        print("\nTesting Ordered Regime (No Change)...")
        symbol = "ETH/USDT"
        data = pd.DataFrame({'close': [100.0]*30, 'high':[101]*30, 'low':[99]*30, 'volume':[1000]*30})
        
        # ORDERED Regime
        ctx = {'entropy_regime': 'ORDERED'}
        bb = {'upper': 110, 'lower': 90, 'middle': 100}
        
        self.oracle.analyze_for_entry(
            symbol, data, bb, 0.1, 'PREDATOR', structure_ctx=ctx
        )
        
        # Verify NO change (0.9 stays 0.9)
        stored_prob = self.oracle.last_probes[symbol]['xgb']
        self.assertAlmostEqual(stored_prob, 0.9, places=2)
        print(f"✅ Probe Value Verified: {stored_prob:.2f} (Expected 0.9)")

if __name__ == '__main__':
    unittest.main()
