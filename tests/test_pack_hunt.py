import unittest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent
from HolonicTrader.agent_oracle import EntryOracleHolon

class TestPackHunt(unittest.TestCase):
    def setUp(self):
        self.oracle = EntryOracleHolon("TestOracle")
        # Mock get_market_bias
        self.oracle.get_market_bias = MagicMock()
        
    def test_pack_hunt_trigger(self):
        # Scenario: Strong Bull Market (0.8), Asset is Laggard (Z = -2.0)
        self.oracle.get_market_bias.return_value = 0.8
        
        pack_stats = {'mean': 5.0, 'std': 2.0} # Pack up 5% +/- 2%
        ticker_data = {'percentage': 0.0}       # Asset flat (Lagging bad)
        
        # Z = (0 - 5) / 2 = -2.5. Should trigger.
        is_hunt = self.oracle.detect_pack_laggard("TEST", ticker_data, pack_stats)
        
        self.assertTrue(is_hunt, "Should detect Pack Laggard")

    def test_pack_hunt_weak_market(self):
        # Scenario: Markets are choppy (0.5), Asset is Laggard
        self.oracle.get_market_bias.return_value = 0.5
        
        pack_stats = {'mean': 5.0, 'std': 2.0}
        ticker_data = {'percentage': 0.0}
        
        is_hunt = self.oracle.detect_pack_laggard("TEST", ticker_data, pack_stats)
        
        self.assertFalse(is_hunt, "Should NOT hunt in weak market")

    def test_not_laggard(self):
        # Scenario: Strong Bull, Asset is performing OK (Z = -0.5)
        self.oracle.get_market_bias.return_value = 0.8
        
        pack_stats = {'mean': 5.0, 'std': 2.0}
        ticker_data = {'percentage': 4.0} # Up 4% vs Mean 5%
        
        # Z = (4 - 5) / 2 = -0.5. Not laggard enough.
        is_hunt = self.oracle.detect_pack_laggard("TEST", ticker_data, pack_stats)
        
        self.assertFalse(is_hunt, "Should NOT trigger for mild underperformance")

if __name__ == '__main__':
    unittest.main()
