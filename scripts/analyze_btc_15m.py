
import ccxt
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table

# Ensure we can import local modules
sys.path.append(os.getcwd())

try:
    from HolonicTrader.kalman import KalmanFilter1D
    # Simple Entropy Implementation if Holon import fails or is too complex
    def calculate_entropy(returns, window=20):
        entropies = []
        for i in range(len(returns)):
            if i < window:
                entropies.append(0.0)
                continue
            
            # Discretize returns into bins
            hist, _ = np.histogram(returns[i-window:i], bins=10, density=False)
            # Shannon Entropy of the histogram (Discrete)
            p = hist / np.sum(hist)
            p = p[p > 0]
            ent = -np.sum(p * np.log2(p))
            entropies.append(ent)
        return entropies
        
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def fetch_and_analyze():
    console = Console()
    console.print("[bold yellow]🚀 Connecting to Kraken Futures...[/bold yellow]")
    
    # Initialize Exchange (No keys needed for public data)
    ex = ccxt.krakenfutures()
    symbol = 'BTC/USD:USD'
    timeframe = '15m'
    limit = 200
    
    try:
        # Fetch OHLCV
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        console.print(f"[green]✅ Fetched {len(df)} candles for {symbol} ({timeframe})[/green]")
        
        # 1. Kalman Filter (Certainty)
        kf = KalmanFilter1D(process_noise=0.001, measurement_noise=0.1) # Adjusted for BTC price scale? 
        # Actually standard params might lag on BTC price (100k). 
        # Better to run on log price or diff, but let's try raw price first with more adaptation.
        # Let's use the default params but maybe run it twice or warm it up.
        
        kalman_estimates = []
        for price in df['close']:
            k_val = kf.update(price)
            kalman_estimates.append(k_val)
            
        df['kalman'] = kalman_estimates
        
        # 2. Panic Spread (Price - Certainty)
        df['panic_spread'] = df['close'] - df['kalman']
        df['panic_pct'] = (df['panic_spread'] / df['kalman']) * 100
        
        # 3. Entropy (Regime)
        df['returns'] = df['close'].pct_change().fillna(0.0) # Fill NaNs from first row
        
        # Guard against Infinite returns if division by zero occurred
        df['returns'] = df['returns'].replace([np.inf, -np.inf], 0.0)
        
        df['entropy'] = calculate_entropy(df['returns'].values, window=20)
        
        # Display Results using Rich Table
        table = Table(title=f"BTC/USD Futures Analysis ({timeframe} Windows)")
        table.add_column("Time (UTC)", style="cyan")
        table.add_column("Price", style="white")
        table.add_column("Kalman (Certainty)", style="blue")
        table.add_column("Panic Spread", justify="right")
        table.add_column("Entropy", justify="right")
        table.add_column("Regime", justify="center")
        
        # Show last 20 candles
        subset = df.tail(20)
        
        for _, row in subset.iterrows():
            # Color Spread
            spread_val = row['panic_spread']
            spread_style = "green" # Normal
            if spread_val > 500: spread_style = "bold red" # FOMO
            elif spread_val < -500: spread_style = "bold red" # FEAR
            
            # Color Regime
            ent = row['entropy']
            regime = "ORDERED"
            reg_style = "green"
            if ent > 2.5: 
                regime = "CHAOTIC"
                reg_style = "bold red"
            elif ent > 2.0:
                regime = "TRANSITION"
                reg_style = "yellow"
                
            table.add_row(
                row['timestamp'].strftime('%H:%M'),
                f"${row['close']:.2f}",
                f"${row['kalman']:.2f}",
                f"[{spread_style}]${spread_val:+.2f}[/{spread_style}]",
                f"{ent:.3f}",
                f"[{reg_style}]{regime}[/{reg_style}]"
            )
            
        console.print(table)
        
        # Summary Stats
        avg_entropy = df['entropy'].tail(50).mean()
        max_spread = df['panic_spread'].abs().max()
        console.print(f"\n[bold]Summary Analysis:[/bold]")
        console.print(f"Current Entropy: {df['entropy'].iloc[-1]:.3f} (Avg: {avg_entropy:.3f})")
        console.print(f"Max Panic Deviance: ${max_spread:.2f}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fetch_and_analyze()
