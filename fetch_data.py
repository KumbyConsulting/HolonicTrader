import sys
import os
import time

# Ensure Python path includes current dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from HolonicTrader.agent_observer import ObserverHolon

SYMBOLS = ['ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'ADA/USDT']
PERIOD_1H = '1h'
PERIOD_15M = '15m'

# Need enough 1m/15m data to support 1500 hours simulation
# 1500 hours = 1500 * 4 = 6000 15m candles
LIMIT_1H = 2000
LIMIT_15M = 8000

def fetch_data():
    observer = ObserverHolon()
    
    print(f"📡 [DataFetcher] Optimizing Cache for Gauntlet Assets: {SYMBOLS}")
    
    for symbol in SYMBOLS:
        print(f"\n🔄 Synching {symbol}...")
        
        # 1. Fetch Primary (1H)
        try:
            print(f"   Getting {PERIOD_1H} (Limit {LIMIT_1H})...")
            df = observer.fetch_market_data(symbol=symbol, timeframe=PERIOD_1H, limit=LIMIT_1H)
            if not df.empty:
                print(f"   ✅ Saved {len(df)} candles for {symbol} {PERIOD_1H}")
            else:
                print(f"   ❌ Failed to fetch/save {PERIOD_1H}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        # 2. Fetch Secondary (15m)
        try:
            print(f"   Getting {PERIOD_15M} (Limit {LIMIT_15M})...")
            df_sec = observer.fetch_market_data(symbol=symbol, timeframe=PERIOD_15M, limit=LIMIT_15M)
            if not df_sec.empty:
                print(f"   ✅ Saved {len(df_sec)} candles for {symbol} {PERIOD_15M}")
            else:
                print(f"   ❌ Failed to fetch/save {PERIOD_15M}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
    print("\n✅ Data Sync Complete.")

if __name__ == "__main__":
    fetch_data()
