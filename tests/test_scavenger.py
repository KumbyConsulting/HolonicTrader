import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent
from HolonicTrader.agent_oracle import EntryOracleHolon

class TestScavengerTrap(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon("TestOracle")
        
    def test_detect_trap_success(self):
        # Setup: Pivot S1 at 100.0
        structure_ctx = {'pivots': {'S1': 100.0}}
        
        # Data: Candle Low 99.0 (Below), Close 100.2 (Above)
        data = pd.DataFrame({
            'high': [100.0, 101.0],
            'low': [98.0, 99.0],
            'close': [99.0, 100.2],
            'open': [99.0, 100.5]
        })
        
        is_trap, level = self.oracle.detect_scavenger_trap("TEST", data, structure_ctx)
        
        self.assertTrue(is_trap, "Should detect trap pattern")
        self.assertEqual(level, "S1")
        
    def test_no_trap_breakdown(self):
        # Setup: S1 at 100.0
        structure_ctx = {'pivots': {'S1': 100.0}}
        
        # Data: Low 99.0, Close 99.5 (BOTH Below) -> Breakdown, not Trap
        data = pd.DataFrame({
            'high': [101.0],
            'low': [99.0],
            'close': [99.5],
            'open': [100.5]
        })
        
        is_trap, level = self.oracle.detect_scavenger_trap("TEST", data, structure_ctx)
        
        self.assertFalse(is_trap, "Should NOT detect trap on breakdown")

    def test_no_trap_hover(self):
        # Setup: S1 at 100.0
        structure_ctx = {'pivots': {'S1': 100.0}}
        
        # Data: Low 100.1, Close 100.5 (Never touched S1)
        data = pd.DataFrame({
            'high': [101.0],
            'low': [100.1],
            'close': [100.5],
            'open': [100.5]
        })
        
        is_trap, level = self.oracle.detect_scavenger_trap("TEST", data, structure_ctx)
        
        self.assertFalse(is_trap, "Should NOT detect trap if level not tested")

if __name__ == '__main__':
    unittest.main()
