import unittest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_executor import TradeSignal

class TestPivotVeto(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon("TestOracle")
        # Mock dependencies to avoid side effects
        self.oracle.market_state = {'entropy': 0.5, 'correlation': None}
        self.oracle.get_market_bias = MagicMock(return_value=0.5)
        
    def test_pivot_veto_logic(self):
        # Scenario: Weak Buy Signal (Conviction 0.5), Price (100) < Pivot P (105)
        # Should be VETOED.
        
        symbol = "TEST/USDT"
        structure_ctx = {
            'pivots': {'P': 105.0} # Pivot is ABOVE price -> Bearish Zone
        }
        
        sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=100.0)
        sig.conviction = 0.5 # Weak
        sig.metadata = {'structure': structure_ctx, 'rvol': 2.0} # High RVOL to pass Energy check
        
        # Run Physics
        result = self.oracle.apply_market_physics(symbol, sig)
        
        self.assertIsNone(result, "Weak Long below Pivot should be Vetoed")
        
    def test_pivot_allow_strong(self):
        # Scenario: Strong Buy Signal (Conviction 0.9), Price (100) < Pivot P (105)
        # Should NOT be vetoed (Bucking the trend is allowed if conviction is high)
        
        symbol = "TEST/USDT"
        structure_ctx = {
            'pivots': {'P': 105.0}
        }
        
        sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=100.0)
        sig.conviction = 0.9 # Strong
        sig.metadata = {'structure': structure_ctx, 'rvol': 2.0}
        
        result = self.oracle.apply_market_physics(symbol, sig)
        
        self.assertIsNotNone(result, "Strong Long below Pivot should allowed")
        self.assertEqual(result.symbol, symbol)

    def test_pivot_allow_above(self):
        # Scenario: Weak Buy Signal (Conviction 0.5), Price (110) > Pivot P (105)
        # Should NOT be vetoed (We are in Bullish Zone)
        
        symbol = "TEST/USDT"
        structure_ctx = {
            'pivots': {'P': 105.0}
        }
        
        sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=110.0)
        sig.conviction = 0.5 # Weak but OK
        sig.metadata = {'structure': structure_ctx, 'rvol': 2.0}
        
        result = self.oracle.apply_market_physics(symbol, sig)
        
        self.assertIsNotNone(result, "Weak Long ABOVE Pivot should be allowed")

if __name__ == '__main__':
    unittest.main()
