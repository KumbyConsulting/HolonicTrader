import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
import time

def test_radar():
    print("📡 Testing Order Flow Radar (Sniper Mode)...")
    
    # 1. Setup
    observer = ObserverHolon(exchange_id='kucoin') # Default
    oracle = EntryOracleHolon()
    symbol = 'BTC/USDT'
    
    # 2. Test Fetch (First Call)
    print(f"\n[1] Pinging {symbol} Trades (API Call)...")
    start = time.time()
    res = oracle.analyze_order_flow(symbol, observer)
    elapsed = time.time() - start
    print(f"Result: {res}")
    print(f"Latency: {elapsed:.2f}s")
    
    if res['vol_processed'] == 0:
        print("❌ No trades fetched. Check API or Symbol.")
        return

    # 3. Test Cache (Second Call)
    print(f"\n[2] Pinging {symbol} Trades (Cache Check)...")
    start = time.time()
    res2 = oracle.analyze_order_flow(symbol, observer)
    elapsed2 = time.time() - start
    print(f"Result: {res2}")
    print(f"Latency: {elapsed2:.2f}s")
    
    if elapsed2 > 1.0:
         print("⚠️ Cache did not seem to work (Latency > 1s)")
    else:
         print("✅ Cache HIT. Latency minimal.")

if __name__ == "__main__":
    try:
        test_radar()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Test Failed: {e}")
