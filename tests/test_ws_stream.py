
import time
import asyncio
from HolonicTrader.agent_observer import ObserverHolon
import config

async def test_ws_stream():
    print("--- 📡 WebSocket Stream Verification (Kraken Futures) 📡 ---")
    observer = ObserverHolon(exchange_id='krakenfutures')
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    print(f"Starting WS for: {symbols}")
    
    observer.start_ws(symbols)
    
    print("Waiting 10 seconds for events...")
    for i in range(10):
        # Check cache
        prices = {}
        for s in symbols:
            p = observer.get_latest_price(s)
            if p > 0:
                prices[s] = p
        
        if prices:
            print(f"[{i}s] Real-time Prices: {prices}")
        else:
            print(f"[{i}s] Waiting for stream...")
            
        time.sleep(1)

    # Final Check
    valid = True
    for s in symbols:
        p = observer.get_latest_price(s)
        if p <= 0:
            print(f"❌ {s}: NO STREAM DATA")
            valid = False
        else:
            print(f"✅ {s}: Latest Price {p}")
            
    if valid:
        print("\n[SUCCESS] WebSocket streaming verified.")
    else:
        print("\n[FAILED] WebSocket stream did not populate cache.")

if __name__ == "__main__":
    try:
        # Since Observer starts its own loop in a thread, we just run the test sync/async
        asyncio.run(test_ws_stream())
    except Exception as e:
        print(f"Test Error: {e}")
