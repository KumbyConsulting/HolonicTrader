import unittest
from unittest.mock import MagicMock, patch
import time
from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.agent_monitor import MonitorHolon
import config

class TestSystemStability(unittest.TestCase):
    
    def test_circuit_breaker_activates(self):
        print("\n--- Testing Circuit Breaker ---")
        actuator = ActuatorHolon(name="TestActuator")
        # Mock exchange
        actuator.exchange = MagicMock()
        
        # Simulate 5 Failures
        print("1. Simulating 5 consecutive API errors...")
        for i in range(5):
             actuator.report_failure("503 Service Unavailable")
             
        self.assertTrue(actuator.circuit_open, "Circuit should be OPEN after 5 failures")
        self.assertTrue(actuator.hibernate_until > time.time(), "Hibernate time should be set")
        
        # Verify blockade
        print("2. Verifying blockade...")
        bal = actuator.get_account_balance()
        self.assertEqual(bal, 0.0, "Balance call should trigger return 0.0 immediately")
        
        # Verify no calls made to exchange
        actuator.exchange.fetch_balance.assert_not_called()
        print("✅ Circuit Breaker Test Passed")
        
    def test_monitor_fever_check(self):
        print("\n--- Testing Monitor Fever Check ---")
        monitor = MonitorHolon(name="TestMonitor", principal=1000.0)
        
        # Setup clean state
        monitor.daily_start_balance = 1000.0
        monitor.last_day_reset = time.time()
        monitor.is_system_healthy = True
        
        # 1. Healthy Check
        print("1. Testing Healthy Equity ($990)...")
        health, msg = monitor.perform_live_check(990.0)
        self.assertTrue(health, "Should be healthy (1% drawdown)")
        
        # 2. Fever Check
        print("2. Testing Fever Equity ($850)...")
        # config.IMMUNE_MAX_DAILY_DRAWDOWN is 0.05 (5%)
        # 1000 -> 850 is 15% drawdown
        health, msg = monitor.perform_live_check(850.0)
        self.assertFalse(health, "Should be UNHEALTHY (15% drawdown)")
        self.assertIn("FEVER", msg)
        self.assertFalse(monitor.is_system_healthy, "Internal state should be unhealthy")
        
        print(f"   Msg: {msg}")
        print("✅ Monitor Fever Check Test Passed")

if __name__ == '__main__':
    # Suppress normal output for clean test run
    unittest.main()
