
import sys
import os
import time
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Logic to handle running from 'HolonicTrader/HolonicTrader' or 'HolonicTrader'
# We want to add the REPO ROOT to the path so we can import 'HolonicTrader.agent_...'
# If we are in '.../HolonicTrader/HolonicTrader/tests', we need to go up 2 levels.
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, repo_root)

# Also try adding the package root directly if running inside package
package_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, package_root)

try:
    from HolonicTrader.HolonicTrader.agent_oracle import EntryOracleHolon
    from HolonicTrader.HolonicTrader.agent_executor import ExecutorHolon, Disposition
    from HolonicTrader.HolonicTrader.agent_governor import GovernorHolon
    from HolonicTrader.HolonicTrader.agent_structure import StructureHolon, TradeSignal
    import HolonicTrader.HolonicTrader.config as config
except ImportError:
    # Fallback for different directory structure
    try:
        from HolonicTrader.agent_oracle import EntryOracleHolon
        from HolonicTrader.agent_executor import ExecutorHolon, Disposition
        from HolonicTrader.agent_governor import GovernorHolon
        from HolonicTrader.agent_structure import StructureHolon, TradeSignal
        import HolonicTrader.config as config
    except ImportError:
         # Direct import if inside package
        from agent_oracle import EntryOracleHolon
        from agent_executor import ExecutorHolon, Disposition
        from agent_governor import GovernorHolon
        from agent_structure import StructureHolon, TradeSignal
        import config

def run_stress_test():
    print("\n🔥 STARTING STRESS TEST: 'UNLEASHED' PROTOCOL 🔥")
    print("===================================================")
    
    # 1. Mock Agents
    print("\n[1] Initializing Agents...")
    oracle = EntryOracleHolon("Oracle_Test")
    oracle.predict_trend_lstm = MagicMock(return_value=0.85) # High conviction Bullish
    oracle.predict_trend_xgboost = MagicMock(return_value=0.85)
    oracle.get_kalman_estimate = MagicMock(return_value=100.0) # Fair value 100
    oracle.get_market_bias = MagicMock(return_value=0.6) # Bullish market
    
    executor = ExecutorHolon("Executor_Test")
    # Mock Ledger
    executor.ledger = MagicMock()
    executor.ledger.add_block = MagicMock(return_value=MagicMock(hash="block_123"))
    
    governor = GovernorHolon("Governor_Test")
    governor.balance = 50.0 # Scavenger Level Balance
    governor.positions = {}
    governor.last_trade_time = {}
    
    # 2. Scenario A: The "Flash Crash" Panic Buy
    # Price crashes to 90 (Kalman is 100) -> 10% Discount
    # RSI is 15 (Extreme Oversold)
    # Entropy is 2.1 (Chaos)
    print("\n[2] Scenario A: 'Flash Crash' (Entropy 2.1, Price -10%, RSI 15)")
    
    # Mock Data
    df_panic = pd.DataFrame({
        'close': [100.0]*18 + [95.0, 90.0],
        'high': [101.0]*20,
        'low': [99.0]*18 + [94.0, 89.0],
        'volume': [1000.0]*20
    })
    
    # Oracle Decision
    oracle.last_macro_state = {'BTC/USDT': 'BEARISH'} # Trend is against us
    
    # Manually Inject RSI logic simulation since we can't easily mock the internal RSI calculation 
    # inside analyze_for_entry without data structure overhead.
    # Instead, we will rely on the fact that we updated the code to prioritize "trigger_panic_buy"
    # We need to construct a window_data that results in RSI < 20.
    # A straight drop does that.
    
    print("   -> Oracle Analyzing...")
    # Mocking the internal calculator to ensure RSI is seen as 15
    # We can't mock local variables in a method easily. 
    # We'll rely on the logic: The previous code we wrote uses the dataframe.
    # Our dataframe has a sharp drop.
    
    signal = oracle.analyze_for_entry('BTC/USDT', df_panic, metabolism_state='SCAVENGER', structure_ctx={})
    
    if signal and signal.direction == 'BUY':
        print(f"   ✅ PASS: Oracle generated BUY signal during Crash. (Reason: {signal.metadata.get('reason', 'Unknown')})")
    else:
        print(f"   ❌ FAIL: Oracle stayed silent. (Signal: {signal})")

    # Executor Decision (Entropy Check)
    entropy_score = 2.1
    print(f"   -> Executor Checking Entropy {entropy_score} (Threshold is {config.TRADER_MAX_WORKERS if hasattr(config, 'TRADER_MAX_WORKERS') else 'Unknown'})... Wait, Threshold is in code.")
    
    decision = executor.decide_trade(signal, 'CHAOTIC', entropy_score)
    
    if decision.action != 'HALT':
        print(f"   ✅ PASS: Executor allowed trade in Chaos (Action: {decision.action}, Autonomy: {decision.disposition.autonomy:.2f})")
    else:
        print(f"   ❌ FAIL: Executor HALTED the trade. (Action: {decision.action})")
        
    # Governor Decision (Risk Check)
    # Balance 50 -> Scavenger. Should size aggressively but safely.
    print("   -> Governor Sizing...")
    approved, size, lev = governor.validate_trade(signal, decision)
    
    if approved:
         print(f"   ✅ PASS: Governor Approved. Size: {size}, Lev: {lev}")
         expected_margin = 10.0 # approximate 10 bullet
         if size > 500: # 50 * 10 leverage? No Scavenger is fixed margin.
              # Scavenger logic: margin = min(10, balance*0.6) * lev * scalar
              print("      (Note: Check if size is reasonable)")
    else:
         print("   ❌ FAIL: Governor Rejected.")

    # 3. Scenario B: The "Blow-Off Top" (Counter Trend Short)
    print("\n[3] Scenario B: 'Blow-Off Top' (RSI 95, Price +20%)")
    # Massive pump. Trend is Bullish. We want to SHORT.
    df_pump = pd.DataFrame({
        'close': [100.0]*15 + [110.0, 120.0, 130.0, 135.0, 140.0], # massive gain
        'high': [100.0]*15 + [110.0, 120.0, 130.0, 135.0, 140.0],
        'low': [100.0]*20,
        'volume': [5000.0]*20 # Whale volume
    })
    
    # Oracle Context
    # We need to simulate the Unified Logic trigger: trigger_panic_sell
    # RSI should be high.
    
    signal_pump = oracle.analyze_for_entry('SOL/USDT', df_pump, metabolism_state='SCAVENGER', structure_ctx={})
    
    if signal_pump and signal_pump.direction == 'SELL':
         print(f"   ✅ PASS: Oracle generated SELL signal into Pump. (Reason: {signal_pump.metadata.get('reason', 'Unknown')})")
    else:
         print(f"   ❌ FAIL: Oracle missed the top. (Signal: {signal_pump})")

if __name__ == "__main__":
    run_stress_test()
