import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import linregress

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'HolonicTrader', 'market_data')
TIMEFRAME = '15m'
MIN_DATA_POINTS = 1000

def load_data():
    """Load and align Closes for all assets."""
    print(f"Loading data from {DATA_DIR}...")
    close_prices = {}
    
    pattern = os.path.join(DATA_DIR, f"*_{TIMEFRAME}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No data files found!")
        return None

    for f in files:
        try:
            # Extract symbol check
            base = os.path.basename(f).split('_')[0]
            if base.endswith('USDT'):
                clean_base = base[:-4]
                symbol = f"{clean_base}/USDT"
            else:
                symbol = f"{base}/USDT"
            
            df = pd.read_csv(f)
            if 'timestamp' not in df.columns or 'close' not in df.columns:
                continue
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            close_prices[symbol] = df['close']
        except Exception as e:
            print(f"  ⚠️ Error loading {f}: {e}")

    # Combine into single DataFrame
    print("Aligning timestamps...")
    aligned_df = pd.DataFrame(close_prices)
    aligned_df = aligned_df.fillna(method='ffill', limit=5)
    aligned_df = aligned_df.dropna(thresh=int(len(aligned_df.columns)*0.8)) 
    print(f"Aligned Data Shape: {aligned_df.shape}")
    return aligned_df

def calculate_hurst(series):
    """
    Calculate the Hurst Exponent.
    """
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def write_report(content):
    with open("mean_reversion_proof.md", "a", encoding='utf-8') as f:
        f.write(content + "\n")

if __name__ == "__main__":
    with open("mean_reversion_proof.md", "w", encoding='utf-8') as f:
        f.write("# Mean Reversion Mathematical Proofs\n")
        f.write(f"Generated from {TIMEFRAME} candles.\n\n")

    df = load_data()
    if df is not None and not df.empty:
        results = []
        for col in df.columns:
            # Check Hurst Exponent instead of subtle Half-Life
            # Log prices
            h = calculate_hurst(np.log(df[col].values))
            results.append((col, h))
            
        results.sort(key=lambda x: x[1])
        
        write_report("## Hurst Exponent Validation")
        write_report("H < 0.5 implies Mean Reversion. Lower is stronger.\n")
        
        write_report("### 🦅 Validated Scavengers (H < 0.5)")
        for r in results:
             if r[1] < 0.5:
                 write_report(f"- **{r[0]}**: H={r[1]:.3f} (Mean Reverting)")
             else:
                 pass # Don't list random walks here
                 
        write_report("\n### 🎲 Random Walk / Trending (H >= 0.5)")
        for r in results:
             if r[1] >= 0.5:
                 write_report(f"- **{r[0]}**: H={r[1]:.3f} (Trending)")

        print("Proof generated: mean_reversion_proof.md")
    else:
        print("Data load failed.")
