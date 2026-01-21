
import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

import config
from HolonicTrader.agent_regime import RegimeController
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_executor import TradeSignal

class TestVolWindow(unittest.TestCase):
    def setUp(self):
        # Reset Config for testing
        config.VOL_WINDOW_MIN_VOLATILITY = 0.45
        config.VOL_WINDOW_FUNDING_THRESHOLD = 0.03
        config.VOL_WINDOW_SPREAD_THRESHOLD = 0.004
        config.VOL_WINDOW_RISK_PCT = 0.02
        config.VOL_WINDOW_MAX_POSITIONS = 3
        
        self.regime = RegimeController()
        self.governor = GovernorHolon()
        self.governor.regime_controller = self.regime # Link
        self.guardian = ExitGuardianHolon()

    def test_regime_trigger(self):
        """Test that VOL_WINDOW triggers on high volatility and funding."""
        # 1. Normal Conditions (Low Vol)
        data = {'btc_vol': 0.20, 'avg_funding': 0.01, 'avg_spread': 0.001}
        cond = self.regime.check_vol_window_conditions(data)
        self.assertFalse(cond, "Should NOT trigger on low volatility")
        
        # 2. High Vol only
        data = {'btc_vol': 0.50, 'avg_funding': 0.01, 'avg_spread': 0.001}
        cond = self.regime.check_vol_window_conditions(data)
        self.assertFalse(cond, "Should NOT trigger on High Vol alone (needs Funding)")
        
        # 3. All Conditions Met
        data = {'btc_vol': 0.50, 'avg_funding': 0.04, 'avg_spread': 0.001}
        cond = self.regime.check_vol_window_conditions(data)
        self.assertTrue(cond, "SHOULD trigger when all conditions met")
        
        # 4. Spread Veto
        data = {'btc_vol': 0.50, 'avg_funding': 0.04, 'avg_spread': 0.005}
        cond = self.regime.check_vol_window_conditions(data)
        self.assertFalse(cond, "Should VETO on high spread")

    def test_regime_switching(self):
        """Test entry and exit of VOL_WINDOW regime."""
        # Force Entry
        data_entry = {'btc_vol': 0.60, 'avg_funding': 0.05, 'avg_spread': 0.001}
        self.regime.attempt_vol_window_entry(data_entry)
        self.assertEqual(self.regime.current_regime, 'VOL_WINDOW')
        
        # Check Governor Permission
        # VOL_WINDOW allows overrides
        
        # Force Exit (Conditions lost)
        data_exit = {'btc_vol': 0.30, 'avg_funding': 0.01, 'avg_spread': 0.001}
        self.regime.attempt_vol_window_entry(data_exit)
        # Should revert to MICRO (default)
        self.assertEqual(self.regime.current_regime, 'MICRO')

    def test_governor_sizing_vol_window(self):
        """Test Position Sizing in VOL_WINDOW."""
        self.regime.current_regime = 'VOL_WINDOW'
        self.governor.balance = 1000.0
        
        # Risk 2% = $20. 
        # Stop Distance default 1%.
        # Position Size = $20 / 0.01 = $2000.
        # Leverage = 5x. Max Gross = 1000 * 5 = 5000.
        # $2000 is within limits.
        
        approved, qty, leverage = self.governor.calc_position_size(
            'BTC/USDT', asset_price=100.0, current_atr=1.0, atr_ref=1.0
        )
        
        self.assertTrue(approved)
        self.assertEqual(leverage, config.VOL_WINDOW_LEVERAGE)
        # Expected Qty = $2000 / 100 = 20.0
        # Wait, inside logic: stop_dist = (ATR*1.5)/Price = 1.5/100 = 1.5%
        # Size = $20 / 0.015 = $1333.33
        # Qty = 1333.33 / 100 = 13.33
        
        expected_size = (1000 * 0.02) / ((1.0 * 1.5)/100.0)
        expected_qty = expected_size / 100.0
        self.assertAlmostEqual(qty, expected_qty, places=1)
        
    def test_governor_max_pos_vol_window(self):
        """Test Max Positions Cap in VOL_WINDOW."""
        self.regime.current_regime = 'VOL_WINDOW'
        self.governor.balance = 1000.0
        
        # Fill positions
        self.governor.positions = {
            'A': {'quantity': 1}, 
            'B': {'quantity': 1}, 
            'C': {'quantity': 1}
        } # 3 positions
        
        approved, _, _ = self.governor.calc_position_size('D', 100.0)
        self.assertFalse(approved, "Should reject 4th position (Max 3)")
        
    def test_guardian_exit_vol_window(self):
        """Test Guardian Exit Logic for VOL_WINDOW."""
        regime = 'VOL_WINDOW'
        symbol = 'BTC/USDT'
        entry = 100.0
        
        # 1. Trailing Stop Hit
        # High Watermark = 110. Current = 108.
        # ATR = 1.0. Stop Dist = 1.5 * 1.0 = 1.5.
        # Stop Price = 110 - 1.5 = 108.5.
        # Current 108 <= 108.5 -> EXIT.
        
        self.guardian.update_watermark(symbol, 110.0, entry) # Set High
        signal = self.guardian.analyze_for_exit(
            symbol, current_price=108.0, entry_price=entry, 
            bb={}, atr=1.0, metabolism_state='PREDATOR', 
            direction='BUY', regime=regime
        )
        
        self.assertIsNotNone(signal)
        self.assertEqual(signal.direction, 'SELL')
        self.assertEqual(signal.metadata['reason'], 'VOL_TRAIL')

if __name__ == '__main__':
    unittest.main()
