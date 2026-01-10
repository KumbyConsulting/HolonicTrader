
import ccxt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add parent to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def analyze():
    print(">> Initializing Kraken Client...")
    
    # Try both Spot and Futures
    exchange_spot = ccxt.kraken()
    exchange_futures = ccxt.krakenfutures()
    
    target_symbols = [
        'WBTC/USD', 'BTC/USD', 'TBTC/USD', 'XAUT/USD', 'LSETH/USD', 'TAO/USD'
    ]
    
    # 1. Verify Symbols
    print("\n>> Verifying Symbols on Kraken...")
    spot_markets = exchange_spot.load_markets()
    futures_markets = exchange_futures.load_markets()
    
    valid_pairs = {}
    
    for sym in target_symbols:
        found = False
        # Check Spot
        if sym in spot_markets:
            print(f"✅ Found {sym} on SPOT")
            valid_pairs[sym] = ('spot', sym)
            found = True
        
        # Check Futures
        if not found:
            # Try to find futures equivalent
            # e.g. BTC/USD:USD
            match = None
            for f_sym in futures_markets:
                if f_sym.startswith(sym.replace('/USD', '/USD:USD')):
                     match = f_sym
                     break
                if f_sym == sym:
                     match = f_sym
                     break
            
            if match:
                 print(f"✅ Found {sym} on FUTURES ({match})")
                 valid_pairs[sym] = ('futures', match)
                 found = True
                 
        if not found:
            print(f"❌ Could not find {sym} on Kraken.")

    if not valid_pairs:
        print("No valid pairs found to analyze.")
        return

    # 2. Fetch Data
    print("\n>> Fetching 1h Data (Last 500 hours)...")
    dfs = {}
    
    for user_sym, (venue, market_sym) in valid_pairs.items():
        try:
            ex = exchange_spot if venue == 'spot' else exchange_futures
            ohlcv = ex.fetch_ohlcv(market_sym, '1h', limit=500)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            dfs[user_sym] = df['close']
            print(f"   -> Fetched {len(df)} rows for {user_sym}")
        except Exception as e:
            print(f"   -> Failed to fetch {user_sym}: {e}")

    # 3. Correlation Analysis
    if len(dfs) < 2:
        print("Not enough data for correlation.")
        return

    combined = pd.DataFrame(dfs)
    combined = combined.dropna()
    
    print("\n>> Correlation Matrix (Pearson):")
    corr = combined.corr()
    print(corr)
    
    # 4. Spread Analysis (Peg Check)
    # Check BTC Pegs
    btc_variants = [s for s in ['BTC/USD', 'WBTC/USD', 'TBTC/USD'] if s in combined.columns]
    if len(btc_variants) > 1:
        print("\n>> BTC Peg Analysis (Spread from BTC/USD):")
        base = combined['BTC/USD']
        for var in btc_variants:
            if var == 'BTC/USD': continue
            spread = combined[var] - base
            z_score = (spread - spread.mean()) / spread.std()
            print(f"   {var}: Mean Spread ${spread.mean():.2f}, Current Z-Score: {z_score.iloc[-1]:.2f}")
            latest_spread = spread.iloc[-1]
            status = "DEPEGGED" if abs(z_score.iloc[-1]) > 2.0 else "PEGGED"
            print(f"   -> Status: {status} (Spread: ${latest_spread:.2f})")

    # 5. Output to File
    with open('correlation_report.txt', 'w', encoding='utf-8') as f:
        f.write("KRAKEN ASSET CORRELATION REPORT\n")
        f.write("===============================\n\n")
        f.write("1. Correlation Matrix:\n")
        f.write(corr.to_string())
        f.write("\n\n2. Prices (Latest):\n")
        f.write(combined.iloc[-1].to_string())
    
    print("\n>> Analysis complete. Saved to correlation_report.txt")

if __name__ == "__main__":
    analyze()
