import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Heavy ML Dependencies
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['xgboost'] = MagicMock()
sys.modules['openvino'] = MagicMock()
sys.modules['openvino.runtime'] = MagicMock()

# Mock Config
class MockConfig:
    PRINCIPAL = 10.0
    MAX_RISK_PCT = 0.05
    VOL_SCALAR_MIN = 0.5
    VOL_SCALAR_MAX = 2.0
    KELLY_LOOKBACK = 10
    MICRO_CAPITAL_MODE = True
    NANO_CAPITAL_THRESHOLD = 50.0
    NANO_MAX_LEVERAGE = 20.0
    IMMUNE_MAX_LEVERAGE_RATIO = 10.0
    MIN_ORDER_VALUE = 2.0
    MIN_TRADE_QTY = {}
    MICRO_GUARD_GROSS_LEVERAGE = 5.0
    MICRO_GUARD_CASH_PRESERVATION_THRESHOLD = 50.0
    MICRO_GUARD_CASH_PRESERVATION_LEVERAGE = 1.0
    MICRO_GUARD_PORTFOLIO_NOTIONAL_MULT = 1.5
    MICRO_GUARD_SINGLE_NOTIONAL_MULT = 0.5
    PHYSICS_MIN_RVOL = 1.0
    PERSONALITY_BTC_ATR_FILTER = 0.5
    PERSONALITY_DOGE_RVOL = 1.5
    PERSONALITY_SOL_RSI_LONG = 40
    PERSONALITY_SOL_RSI_SHORT = 60
    SATELLITE_ASSETS = []
    CRISIS_SAFE_HAVENS = []
    CRISIS_RISK_ASSETS = []
    FAMILY_L1 = []
    FAMILY_PAYMENT = []
    FAMILY_MEME = []
    REGIME_PERMISSIONS = {'MICRO': {'max_stacks': 5}} # Test Config

# Apply Mock Config
sys.modules['config'] = MockConfig()
import config

# Mock Holon Core to avoid huge dependencies
class MockHolon:
    def __init__(self, name, disposition=None):
        self.name = name
        self.balance = 100.0
        self.available_balance = 100.0
        self.equity = 100.0
        self.positions = {}
        self.last_trade_time = {}
        self.last_specific_entry = {}
        self.DEBUG = True
        self.regime_controller = None
    
    def receive_message(self, s, c): pass

# Patch imports in Governor/Oracle
with patch('HolonicTrader.holon_core.Holon', MockHolon):
    from HolonicTrader.agent_governor import GovernorHolon
    from HolonicTrader.agent_oracle import EntryOracleHolon

@dataclass
class MockSignal:
    symbol: str
    direction: str
    conviction: float
    metadata: dict

class TestDynamicLogic(unittest.TestCase):
    
    def setUp(self):
        self.gov = GovernorHolon(name="TestGov")
        self.oracle = EntryOracleHolon(name="TestOracle")
        
    def test_kelly_sizing(self):
        print("\n--- Testing Kelly Sizing (Governor) ---")
        balance = 100.0
        
        # Scenario A: Low Win Rate (40%), R:R 2.0
        # Kelly = (0.4 * 3 - 1) / 2 = (1.2 - 1) / 2 = 0.1 (10%)
        # Half-Kelly = 5%
        # Safe Sizing = 5% of $100 = $5.00
        
        # We enforce mock win rate via calculate_recent_win_rate patch
        with patch.object(self.gov, 'calculate_recent_win_rate', return_value=0.40):
            risk_usd = self.gov.calculate_kelly_size(balance)
            print(f"WR 40%: Risk ${risk_usd:.4f} (Expected ~$5.00)")
            self.assertAlmostEqual(risk_usd, 5.0, delta=0.5)

        # Scenario B: High Win Rate (60%), R:R 2.0
        # Kelly = (0.6 * 3 - 1) / 2 = (1.8 - 1) / 2 = 0.4 (40%)
        # Half-Kelly = 20%
        # Gov Caps to 20% max safe -> $20.00
        with patch.object(self.gov, 'calculate_recent_win_rate', return_value=0.60):
            risk_usd = self.gov.calculate_kelly_size(balance)
            print(f"WR 60%: Risk ${risk_usd:.4f} (Expected ~$20.00)")
            self.assertAlmostEqual(risk_usd, 20.0, delta=0.5)
            
    def test_oracle_probabilistic_weighting(self):
        print("\n--- Testing Oracle Probabilistic Weighting ---")
        sig = MockSignal(symbol="BTC/USDT", direction="SELL", conviction=0.8, metadata={'rvol': 2.0})
        
        # Scenario A: Strong Bull Bias (0.9) vs Short Signal
        # Penalty = (0.9 - 0.6) * 1.5 = 0.3 * 1.5 = 0.45
        # New Conviction = 0.8 - 0.45 = 0.35
        
        with patch.object(self.oracle, 'get_market_bias', return_value=0.9):
            result = self.oracle.apply_market_physics("BTC/USDT", sig)
            print(f"Bull Bias 0.9 vs Short: Conviction 0.8 -> {sig.conviction:.4f} (Expected ~0.35)")
            self.assertLess(sig.conviction, 0.5)
            
        # Reset
        sig.conviction = 0.8
        
        # Scenario B: Neutral Bias (0.5) vs Short
        # Penalty = 0 (Bias < 0.6)
        with patch.object(self.oracle, 'get_market_bias', return_value=0.5):
            result = self.oracle.apply_market_physics("BTC/USDT", sig)
            print(f"Neutral Bias 0.5 vs Short: Conviction 0.8 -> {sig.conviction:.4f} (Expected 0.8)")
            self.assertEqual(sig.conviction, 0.8)

if __name__ == '__main__':
    unittest.main()
