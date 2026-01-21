import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent
from HolonicTrader.agent_oracle import EntryOracleHolon, GlobalTradeSignal

class TestAIWiring(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon("TestOracle")
        
    def test_lstm_trigger(self):
        # Mock LSTM
        self.oracle.model = MagicMock() # Exist
        self.oracle.predict_trend_lstm = MagicMock(return_value=0.88) # Bullish
        
        # Create Data
        data = pd.DataFrame({
            'open': [100.0]*100,
            'high': [101.0]*100,
            'low': [99.0]*100,
            'close': [100.0]*100,
            'volume': [1000.0]*100,
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='15min')
        })
        
        # Run Analysis
        # Note: analyze_for_entry requires pack_stats now
        pack_stats = {'mean': 0.0, 'std': 0.01}
        # And ticker_data dict
        ticker_data = {'change_24h': 0.0}
        
        # Mock structural context to avoid errors
        # analyze_for_entry calls get_structural_context which might fail if no observer
        # So we mock get_structural_context
        self.oracle.get_structural_context = MagicMock(return_value={'regime': 'BULLISH'})
        self.oracle.get_market_bias = MagicMock(return_value=0.6) # Neutral bias
        
        # Prevent personality from blocking
        self.oracle.apply_asset_personality = MagicMock(side_effect=lambda s, sig: sig)
        
        signal = self.oracle.analyze_for_entry("BTC/USDT", data, {}, None, ticker_data, pack_stats)
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.metadata.get('strategy'), 'NEURAL_HYBRID')
        self.assertTrue('LSTM' in signal.metadata.get('reason'))
        print(f"Signal Reason: {signal.metadata.get('reason')}")

if __name__ == '__main__':
    unittest.main()
