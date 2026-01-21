
import pandas as pd
import pandas_ta as ta
import numpy as np
from rich.console import Console
from rich.table import Table

def run_backtest():
    console = Console()
    console.print("[bold yellow]🧪 Starting Fractal Resonance Backtest (BTC 2 Years)...[/bold yellow]")
    
    # 1. Load Data
    # 15m Data (Target)
    try:
        df_15m = pd.read_csv('market_data/BTCUSDT_15m.csv')
        df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'])
        df_15m.set_index('timestamp', inplace=True)
    except FileNotFoundError:
        console.print("[red]❌ BTCUSDT_15m.csv not found. Run fetching script first.[/red]")
        return

    # Resample 15m to 1h for Macro Trend (Guarantees alignment)
    df_1h = df_15m.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    console.print(f"Loaded {len(df_15m)} 15m candles and {len(df_1h)} 1h candles.")

    # 2. Indicators
    # Macro (1h)
    df_1h['EMA50'] = ta.ema(df_1h['close'], length=50)
    df_1h['Trend'] = np.where(df_1h['close'] > df_1h['EMA50'], 1, -1) # 1 Bull, -1 Bear
    
    # Micro (15m)
    df_15m['RSI'] = ta.rsi(df_15m['close'], length=14)
    
    # Merge Trend back to 15m
    # Forward fill the 1h trend to the 15m candles
    # Reset index to merge
    df_15m_reset = df_15m.reset_index()
    df_1h_reset = df_1h[['Trend']].reset_index()
    
    # Merge asof (matches nearest previous 1h timestamp)
    df_merged = pd.merge_asof(df_15m_reset, df_1h_reset, on='timestamp', direction='backward')
    
    # 3. Strategy Logic
    # Baseline: Simple RSI Mean Reversion
    # Buy RSI < 30, Sell RSI > 70
    
    # Fractal: Buy RSI < 30 AND Trend == 1, Sell RSI > 70 AND Trend == -1
    
    trades_base = []
    trades_fractal = []
    
    position_base = 0 # 0, 1, -1
    entry_price_base = 0.0
    
    position_fractal = 0
    entry_price_fractal = 0.0
    
    # Iterate
    # Simplified loop for speed
    equity_base = 1000.0
    equity_fractal = 1000.0
    
    for i, row in df_merged.iterrows():
        price = row['close']
        rsi = row['RSI']
        trend = row['Trend']
        
        if pd.isna(rsi) or pd.isna(trend): continue
        
        # --- BASELINE STRATEGY ---
        if position_base == 0:
            if rsi < 30: # Buy
                position_base = 1
                entry_price_base = price
            elif rsi > 70: # Sell
                position_base = -1
                entry_price_base = price
        elif position_base == 1:
            # Exit Conditions (RSI neutral or Stop)
            if rsi > 50:
                pnl = (price - entry_price_base) / entry_price_base
                equity_base *= (1 + pnl)
                trades_base.append(pnl)
                position_base = 0
        elif position_base == -1:
            if rsi < 50:
                pnl = (entry_price_base - price) / entry_price_base
                equity_base *= (1 + pnl)
                trades_base.append(pnl)
                position_base = 0
                
        # --- FRACTAL STRATEGY ---
        if position_fractal == 0:
            if rsi < 30 and trend == 1: # Buy ONLY if Bull Trend
                position_fractal = 1
                entry_price_fractal = price
            elif rsi > 70 and trend == -1: # Sell ONLY if Bear Trend
                position_fractal = -1
                entry_price_fractal = price
        elif position_fractal == 1:
            if rsi > 50:
                pnl = (price - entry_price_fractal) / entry_price_fractal
                equity_fractal *= (1 + pnl)
                trades_fractal.append(pnl)
                position_fractal = 0
        elif position_fractal == -1:
            if rsi < 50:
                pnl = (entry_price_fractal - price) / entry_price_fractal
                equity_fractal *= (1 + pnl)
                trades_fractal.append(pnl)
                position_fractal = 0

    # 4. Results
    table = Table(title="Strategy Comparison (2 Years BTC 15m)")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline (RSI Only)", style="magenta")
    table.add_column("Fractal (RSI + 1H Trend)", style="green")
    
    win_base = len([t for t in trades_base if t > 0])
    win_fractal = len([t for t in trades_fractal if t > 0])
    
    wr_base = (win_base / len(trades_base)) * 100 if trades_base else 0
    wr_fractal = (win_fractal / len(trades_fractal)) * 100 if trades_fractal else 0
    
    table.add_row("Total Trades", str(len(trades_base)), str(len(trades_fractal)))
    table.add_row("Win Rate", f"{wr_base:.2f}%", f"{wr_fractal:.2f}%")
    table.add_row("Final Equity ($1000)", f"${equity_base:.2f}", f"${equity_fractal:.2f}")
    table.add_row("Profit Factor", f"{equity_base/1000:.2f}x", f"{equity_fractal/1000:.2f}x")
    
    # Capture Console Output to File
    with open("backtest_results.txt", "w", encoding="utf-8") as f:
         f.write(f"Strategy Comparison (2 Years BTC 15m)\n")
         f.write(f"Metric | Baseline (RSI) | Fractal (RSI+Trend)\n")
         f.write(f"--- | --- | ---\n")
         f.write(f"Trades | {len(trades_base)} | {len(trades_fractal)}\n")
         f.write(f"Win Rate | {wr_base:.2f}% | {wr_fractal:.2f}%\n")
         f.write(f"Equity | ${equity_base:.2f} | ${equity_fractal:.2f}\n")
         f.write(f"Profit Factor | {equity_base/1000:.2f}x | {equity_fractal/1000:.2f}x\n")
         
         if equity_fractal > equity_base:
             f.write("\n✅ Hypothesis Confirmed: Fractal Filtering improved performance.\n")
         else:
             f.write("\n❌ Hypothesis Failed.\n")

    console.print(table)

if __name__ == "__main__":
    run_backtest()
