import sys
import os
import traceback

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("--- IMPORTING ---")
    import config
    from HolonicTrader.agent_governor import GovernorHolon
    print("Import OK")
    
    print("--- INSTANTIATING ---")
    gov = GovernorHolon()
    gov.balance = 1000.0
    gov._safe_print = print # Hook print
    print("Init OK")
    
    print("--- CALLING METHOD (NORMAL) ---")
    _, qty_normal, _ = gov.calc_position_size("BTC/USDT", 50000.0, 100, 100, 0.5, "BUY", 0.0, 0.0)
    print(f"Normal Qty: {qty_normal}")
    
    print("--- CALLING METHOD (FEAR) ---")
    _, qty_fear, _ = gov.calc_position_size("BTC/USDT", 50000.0, 100, 100, 0.5, "BUY", 0.0, -0.8)
    print(f"Fear Qty: {qty_fear}")
    
    if qty_fear < qty_normal:
        print(f"✅ SUCCESS: Fear Reduced Size by {100 - (qty_fear/qty_normal)*100:.1f}%")
    else:
        print(f"❌ FAILURE: No Reduction (Ratio {qty_fear/qty_normal:.2f})")
        
    print("\n--- TESTING ACTUATOR SAFETY ---")
    from HolonicTrader.agent_actuator import ActuatorHolon
    from unittest.mock import MagicMock
    actuator = ActuatorHolon(exchange_id='kraken') # Must init to load class
    actuator.exchange = MagicMock() # Mock real exchange
    
    print("Checking Liquidity with Price=0...")
    try:
        res = actuator.check_liquidity("BTC/USDT", "BUY", 1.0, 0.0)
        if res is True:
            print("✅ SUCCESS: Actuator handled Zero Price.")
        else:
             print("❌ FAILURE: Actuator returned False.")
    except Exception as e:
        print(f"❌ FAILURE: Actuator Crashed: {e}")
        traceback.print_exc()

except Exception:
    traceback.print_exc()
