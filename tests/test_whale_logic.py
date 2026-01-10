
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Go up to the root directory where config.py is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.agent_oracle import EntryOracleHolon
import config as config

class TestWhaleLogic(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon(name="TestOracle")
        
    def create_mock_df(self, length=30, trend='FLAT', volume_scenario='NORMAL'):
        prices = []
        volumes = []
        base_price = 100.0
        
        for i in range(length):
            # Price Logic
            if trend == 'FLAT':
                p = base_price + np.random.uniform(-0.1, 0.1)
            elif trend == 'UP':
                p = base_price + (i * 0.5)
            elif trend == 'DOWN':
                p = base_price - (i * 0.5)
            elif trend == 'DIP_RECOVER':
                if i < length - 2: p = base_price - i # Drop
                else: p = base_price # Recovery
            
            # Volume Logic
            if volume_scenario == 'NORMAL':
                v = 1000.0
            elif volume_scenario == 'SPIKE_END':
                v = 1000.0 if i < length - 1 else 5000.0 # 5x spike
            elif volume_scenario == 'ACCUMULATION':
                v = 3000.0 # Consistently high
                
            prices.append(p)
            volumes.append(v)
            
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=length, freq='15min'),
            'open': prices,
            'high': [p + 0.2 for p in prices],
            'low': [p - 0.2 for p in prices],
            'close': prices,
            'volume': volumes
        })
        return df

    def test_detect_accumulation(self):
        # Scenario: High Volume, Low Volatility (Tight Range)
        # Start with NORMAL volume (1000) to establish a low baseline
        df = self.create_mock_df(length=30, trend='FLAT', volume_scenario='NORMAL')
        
        # Manually set High/Low to be very tight for the last few candles
        df.iloc[-3:, df.columns.get_loc('high')] = 100.1
        df.iloc[-3:, df.columns.get_loc('low')] = 99.9
        df.iloc[-3:, df.columns.get_loc('close')] = 100.0
        
        # Manually set Volume to be MEANINGFULLY high only for the accumulation phase (last 5 bars)
        # Baseline (prev 25) = 1000. Recent (5) = 4000.
        # Rolling Avg (20) approx 1750. Current 4000. RVOL ~ 2.2 > 2.0
        df.iloc[-5:, df.columns.get_loc('volume')] = 4000.0
        
        atr = 0.1
        avg_atr = 0.5 # High compression
        
        # Override config for test stability using the config module we imported
        config.WHALE_ACCUMULATION_RVOL = 2.0
        
        is_acc = self.oracle.detect_accumulation(df, atr, avg_atr)
        self.assertTrue(is_acc, "Should detect Accumulation (High Vol + Tight Range)")
        
        # Negative Test
        df_normal = self.create_mock_df(length=30, trend='UP', volume_scenario='NORMAL')
        is_acc_fail = self.oracle.detect_accumulation(df_normal, 1.0, 1.0)
        self.assertFalse(is_acc_fail, "Should NOT detect Accumulation on normal data")

    def test_book_pressure(self):
        # Bid Wall
        book = {
            'bids': [[99, 1000], [98, 1000]], # Vol 2000
            'asks': [[101, 100], [102, 100]]  # Vol 200
        }
        ratio = self.oracle.calculate_book_pressure(book)
        self.assertEqual(ratio, 10.0)
        self.assertTrue(ratio > config.WHALE_ORDER_IMBALANCE_RATIO)
        
        # Ask Wall
        book_bear = {
            'bids': [[99, 100]],
            'asks': [[101, 1000]]
        }
        ratio_bear = self.oracle.calculate_book_pressure(book_bear)
        self.assertEqual(ratio_bear, 0.1)

    def test_short_squeeze(self):
        # Squeeze Condition: Negative Funding + Bullish/Neutral
        is_squeeze = self.oracle.detect_short_squeeze(-0.001, 'NEUTRAL')
        self.assertTrue(is_squeeze)
        
        is_squeeze2 = self.oracle.detect_short_squeeze(-0.001, 'BULLISH')
        self.assertTrue(is_squeeze2)
        
        # Fail Condition: Bearish Trend (Just a normal dip)
        is_fail = self.oracle.detect_short_squeeze(-0.001, 'BEARISH')
        self.assertFalse(is_fail)
        
        # Fail Condition: Positive Funding
        is_fail2 = self.oracle.detect_short_squeeze(0.001, 'BULLISH')
        self.assertFalse(is_fail2)

if __name__ == '__main__':
    unittest.main()
