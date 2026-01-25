
import time
import pandas as pd
from HolonicTrader.agent_observer import ObserverHolon
import config

def test_matrix_ingestion():
    print("--- ⚡ Matrix Ingestion Test ⚡ ---")
    observer = ObserverHolon(exchange_id='krakenfutures')
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    print(f"Fetching Matrix for: {symbols}")
    start = time.perf_counter()
    matrix = observer.fetch_matrix_data(symbols)
    end = time.perf_counter()
    
    duration = end - start
    print(f"\n[Matrix Fetch Complete in {duration:.2f}s]")
    
    for sym in symbols:
        data = matrix.get(sym)
        if not data:
            print(f"❌ {sym}: FAILED (No Data)")
            continue
            
        print(f"\n✅ {sym}:")
        print(f"  - 15m OHLCV: {len(data['df_15m'])} bars")
        print(f"  - 1h OHLCV:  {len(data['df_1h'])} bars")
        print(f"  - OrderBook: {len(data['book']['bids'])} bids, {len(data['book']['asks'])} asks")
        print(f"  - Funding:   {data['funding']:.6f}")
        
    # Validation
    assert len(matrix) == len(symbols), "Matrix size mismatch"
    print("\n[SUCCESS] Matrix integrity verified.")

if __name__ == "__main__":
    try:
        test_matrix_ingestion()
    except Exception as e:
        print(f"\n[FAILED] Test Error: {e}")
