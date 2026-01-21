import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent
from HolonicTrader.agent_oracle import EntryOracleHolon

class TestWhaleShadow(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon("TestOracle")
        
    def test_divergence_trigger(self):
        # Construct Bullish Divergence
        # Point A (Index 5): Price 100.0, Vol 1000 (Sell but absorbed?)
        # Point B (Index 15): Price 95.0 (LL), Vol 5000 (Huge Buy volume on way up)
        
        # We manually construct OBV via volume/price moves
        # To get OBV higher at point B, we need positive candles between A and B
        
        prev_price = 110.0
        data = []
        obv_val = 0
        
        # Generate 35 candles (Min 30 for logic)
        for i in range(35):
            price = 110.0
            vol = 100
            
            # First Valley (Index 5)
            if i == 5: 
                price = 100.0 # Low 1
                vol = 1000 # Sell vol
                
            # Second Valley (Index 15)
            if i == 15:
                price = 95.0 # Low 2 (Lower Acc Low)
                vol = 1000
                
            # Between valleys (Index 6-14), create BUY pressure to raise OBV
            if 6 <= i <= 14:
                price = 105.0 # Higher than valley
                vol = 5000 # Massive buying
            
            # Append
            # OBV logic: if close > prev_close -> +vol
            row = {
                'open': price, 'high': price+1, 'low': price-1, 'close': price, 'volume': vol
            }
            data.append(row)
            prev_price = price
            
        df = pd.DataFrame(data)
        
        # Run detection
        is_shadow = self.oracle.detect_whale_shadow("TEST", df)
        
        # Debug info if failed
        # Calculate OBV and Fractals manually to check
        # obv = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()
        # print("OBV:", obv[5], obv[15])
        
        self.assertTrue(is_shadow, "Should detect Bullish Divergence (Whale Shadow)")

    def test_no_divergence(self):
        # Scenario: Price Lower Low, OBV Lower Low (Confirmation) -> Bearish
        data = []
        for i in range(25):
            price = 110.0
            vol = 100
            
            if i == 5: price = 100.0 # Low 1
            if i == 15: price = 95.0 # Low 2
            
            if 6 <= i <= 14:
                 price = 105.0 
                 vol = 10 # Tiny buying, so OBV won't recover
                 
            # Also ensure selling into Low 2 is heavy to drop OBV
            if i == 15: vol = 50000
            
            row = {'open': price, 'high': price+1, 'low': price-1, 'close': price, 'volume': vol}
            data.append(row)
            
        df = pd.DataFrame(data)
        
        is_shadow = self.oracle.detect_whale_shadow("TEST", df)
        self.assertFalse(is_shadow, "Should NOT detect divergence in standard downtrend")

if __name__ == '__main__':
    unittest.main()
