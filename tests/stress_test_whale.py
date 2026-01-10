
import unittest
import pandas as pd
import numpy as np
import sys
import os
import time
import random

import sys
import os

# Assume running from root or inner dir
# Add the PARENT of the 'tests' directory to sys.path
# This allows 'from HolonicTrader import ...' to work if we are in the outer root
# OR 'import agent_oracle' if we are in the inner root.

# Force add the OUTER root (AEHML/HolonicTrader)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_root)

# Force add the INNER root (AEHML/HolonicTrader/HolonicTrader)
inner_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, inner_root)

try:
    # Try importing as if we are inside the package
    from agent_oracle import EntryOracleHolon
    import config
    print("✅ Imported from Inner Root context")
except ImportError:
    # Try importing as package
    try:
        from HolonicTrader.agent_oracle import EntryOracleHolon
        import config # Config is in root, not inside HolonicTrader package
        print("✅ Imported from Outer Root context (Package)")
    except ImportError as e:
        print(f"❌ Import Failed: {e}")
        print(f"Current Sys Path: {sys.path}")
        raise

def run_stress_test():
    print("🐳 STARTING WHALE STRATEGY STRESS TEST...")
    
    oracle = EntryOracleHolon(name="StressTestOracle")
    
    # Mock Data Generator
    def generate_candle_batch(length=100):
        base_price = 100.0
        prices = [base_price + np.random.uniform(-1, 1) for _ in range(length)]
        volumes = [random.uniform(1000, 5000) for _ in range(length)]
        
        # Inject Whale Signals
        # 1. Accumulation (Low Price Volatility, High Volume)
        for i in range(80, 90):
            prices[i] = 100.0 # Flat
            volumes[i] = 10000.0 # Huge Volume
            
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=length, freq='15min'),
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': volumes
        })
        return df

    # Run Loop
    iterations = 50
    start_time = time.time()
    
    print(f"🔄 Executing {iterations} iterations of FULL ANALYSIS pipeline...")
    
    errors = 0
    whales_found = 0
    
    for i in range(iterations):
        try:
            df = generate_candle_batch()
            
            # Mock Context
            bb_vals = {'upper': 105, 'middle': 100, 'lower': 95}
            structure_ctx = {'macro_trend': 'BULLISH', 'minutes_into_candle': 10}
            book_data = {'bids': [[99, 5000]], 'asks': [[101, 1000]]} # Bullish Book
            
            # Call Analyze
            signal = oracle.analyze_for_entry(
                symbol="BTC/USDT",
                window_data=df,
                bb_vals=bb_vals,
                obv_slope=0.5,
                metabolism_state='PREDATOR',
                structure_ctx=structure_ctx,
                book_data=book_data,
                funding_rate=-0.006 # Squeeze
            )
            
            if signal and signal.metadata.get('is_whale'):
                whales_found += 1
                
            if i % 10 == 0:
                print(f"   Step {i}: OK")
                
        except Exception as e:
            print(f"❌ CRASH at step {i}: {e}")
            errors += 1
            import traceback
            traceback.print_exc()
            
    total_time = time.time() - start_time
    print(f"\n✅ STRESS TEST COMPLETE in {total_time:.2f}s")
    print(f"   - Errors: {errors}")
    print(f"   - Whale Signals Generated: {whales_found}")
    
    if errors == 0:
        print("🎉 SYSTEM STABLE.")
    else:
        print("💥 SYSTEM UNSTABLE.")

if __name__ == '__main__':
    run_stress_test()
