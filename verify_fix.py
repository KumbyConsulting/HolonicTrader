import sys
import os

# Ensure we can import from local dir
sys.path.append(os.getcwd())

# Mock Config
import config
config.INITIAL_CAPITAL = 10.0
config.TRADING_MODE = 'PAPER'

# Mock DB
class MockDB:
    def get_portfolio(self):
        return {'balance_usd': 145397.39} # The "Bad" State
    def get_last_block(self): return None
    def save_portfolio(self, bal, assets, meta): pass

from HolonicTrader.agent_executor import ExecutorHolon

def test_sync_fix():
    print("--- Starting Verification ---")
    
    # 1. Initialize Executor (simulating startup with stale DB)
    executor = ExecutorHolon(initial_capital=10.0, db_manager=MockDB())
    
    print(f"Initial Balance (from DB): ${executor.balance_usd}")
    
    if executor.balance_usd != 145397.39:
        print("FAIL: Did not load stale state correctly.")
        return
        
    # 2. Apply Fix
    REAL_CAPITAL = 18.73
    print(f"Applying sync_balance({REAL_CAPITAL})...")
    executor.sync_balance(REAL_CAPITAL)
    
    # 3. Verify
    print(f"New Balance: ${executor.balance_usd}")
    
    if abs(executor.balance_usd - REAL_CAPITAL) < 0.01:
        print("SUCCESS: Balance Synchronized correctly.")
    else:
        print(f"FAIL: Balance mismatch. Expected {REAL_CAPITAL}, got {executor.balance_usd}")

if __name__ == "__main__":
    test_sync_fix()
