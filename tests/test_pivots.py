import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent
from HolonicTrader.agent_structure import CTKSStrategicHolon

class TestStructurePivots(unittest.TestCase):
    def test_pivot_calc(self):
        holon = CTKSStrategicHolon("TestStructure")
        
        # Mock Dataframe: 2 days (so we have a 'previous' day)
        # Day 1 (Previous): High=110, Low=100, Close=105
        # Day 2 (Current): ...
        data = {
            'high': [110.0, 112.0],
            'low': [100.0, 108.0],
            'close': [105.0, 110.0],
            'open': [102.0, 105.0],
            'volume': [1000, 1200]
        }
        df = pd.DataFrame(data)
        
        pivots = holon._calculate_pivots(df)
        
        # Expected Values for Day 1
        # P = (110 + 100 + 105) / 3 = 105.0
        self.assertAlmostEqual(pivots['P'], 105.0)
        
        # R1 = 2*P - L = 210 - 100 = 110.0
        self.assertAlmostEqual(pivots['R1'], 110.0)
        
        # S1 = 2*P - H = 210 - 110 = 100.0
        self.assertAlmostEqual(pivots['S1'], 100.0)
        
        # Fib R1 = P + 0.382 * (H-L) = 105 + 0.382 * 10 = 105 + 3.82 = 108.82
        self.assertAlmostEqual(pivots['Fib_R1'], 108.82)
        
        print(f"Computed Pivots: {pivots}")

if __name__ == '__main__':
    unittest.main()
