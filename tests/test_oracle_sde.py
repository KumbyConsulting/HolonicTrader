import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.agent_oracle import EntryOracleHolon
import config

def test_oracle_sde_logic():
    print("\n--- Testing Oracle SDE Integration ---")
    oracle = EntryOracleHolon()
    
    # 1. Create synthetic "Stretched" data (Mean = 1.0, Price = 1.25)
    # sigma of returns is approx 0.004 per 15m candle in previous test.
    # We want a log-distance of > 2 sigma.
    # Log(1.25) - Log(1.0) = 0.223.
    # 0.223 / 0.004 = 55 sigma? That's definitely stretched.
    
    steps = 100
    prices = np.linspace(1.0, 1.25, steps) # Strong upward push
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=steps, freq='15min'),
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.normal(1000, 100, steps)
    })
    
    bb_vals = {
        'upper': 1.3,
        'middle': 1.1,
        'lower': 0.9
    }
    
    # 2. Run analysis
    sig = oracle.analyze_for_entry(
        symbol='BTC/USDT',
        window_data=df,
        bb_vals=bb_vals,
        obv_slope=0.1,
        metabolism_state='PREDATOR'
    )
    
    if sig:
        print(f"ORACLE SIGNAL: {sig.direction} REASON: {sig.metadata.get('reason')}")
        
        # 3. Test Governor Validation
        from HolonicTrader.agent_governor import GovernorHolon
        governor = GovernorHolon(initial_balance=100.0)
        governor.available_balance = 100.0
        
        # Inject metadata into a mock validation request
        print("Consulting Governor for position sizing and Ruin Guard...")
        approved, qty, lev = governor.calc_position_size(
            symbol='BTC/USDT',
            asset_price=1.25,
            current_atr=0.01,
            atr_ref=0.01,
            conviction=1.0,
            direction=sig.direction,
            metadata=sig.metadata
        )
        
        print(f"GOVERNOR RESULT: Approved={approved}, Qty={qty}, Lev={lev}")
        if not approved:
            print("DONE: Governor correctly validated/vetoed based on Physics/Ruin.")
    else:
        print("RESULT: No Oracle Signal")

if __name__ == "__main__":
    test_oracle_sde_logic()
