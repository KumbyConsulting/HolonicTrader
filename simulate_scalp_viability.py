
import ccxt
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

def simulate_scalp():
    console = Console()
    console.print("[bold yellow]🧪 Simulating 'Volatility Trap' Viability (15m)...[/bold yellow]")
    
    ex = ccxt.krakenfutures()
    symbol = 'BTC/USD:USD'
    timeframe = '15m'
    
    try:
        # Fetch data
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Indicators
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Simulation Parameters
        SCALP_HALFWIDTH_ATR = 0.5 # Buy @ -0.5 ATR, Sell @ +0.5 ATR
        STOP_LOSS_ATR = 1.0       # Stop if price moves > 1.0 ATR against
        
        results = []
        balance = 100.0
        wins = 0
        losses = 0
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            atr = prev['atr']
            open_price = row['open']
            
            # Setup Trap based on Open
            buy_limit = open_price - (atr * SCALP_HALFWIDTH_ATR)
            sell_limit = open_price + (atr * SCALP_HALFWIDTH_ATR)
            
            # Check execution
            # 1. Did we hit Buy Limit?
            filled_buy = row['low'] <= buy_limit
            # 2. Did we hit Sell Limit?
            filled_sell = row['high'] >= sell_limit
            
            pnl = 0.0
            note = "NO FILL"
            
            # Logic: If we fill, do we close profitable or stop out?
            # Approximation: Use Close price as exit (Simulating "End of Patience" close)
            # or check if we hit Stop Loss within the candle.
            
            if filled_buy:
                # We bought. Did we crash?
                stop_price = buy_limit - (atr * STOP_LOSS_ATR)
                if row['low'] <= stop_price:
                    pnl -= 1.0 # 1R Loss
                    note = "BUY STOPPED"
                    losses += 1
                else:
                    # We hold to close (Reversion check)
                    gain = (row['close'] - buy_limit) / buy_limit
                    if gain > 0: 
                        pnl += gain * 10 
                        wins += 1
                        note = "BUY WIN"
                    else: 
                        pnl += gain * 10
                        losses += 1
                        note = "BUY LOSS"
                        
            if filled_sell:
                # We sold. Did we pump?
                stop_price = sell_limit + (atr * STOP_LOSS_ATR)
                if row['high'] >= stop_price:
                    pnl -= 1.0 # 1R Loss
                    note += " & SELL STOPPED"
                    losses += 1
                else:
                    gain = (sell_limit - row['close']) / sell_limit
                    if gain > 0: 
                        pnl += gain * 10
                        wins += 1
                        note += " & SELL WIN"
                    else: 
                        pnl += gain * 10
                        losses += 1
                        note += " & SELL LOSS"
            
            balance += pnl
            results.append({'time': row['timestamp'], 'pnl': pnl, 'note': note, 'balance': balance})
            
        # Stats
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        console.print(f"[bold]Simulation Results (Last {len(df)} candles):[/bold]")
        console.print(f"Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
        console.print(f"Final Score: {balance:.2f} (Start 100.0)")
        
        # Show recent trades
        table = Table(title="Recent Scalp setups")
        table.add_column("Time")
        table.add_column("Result")
        table.add_column("Balance")
        
        for r in results[-10:]:
             style = "green" if r['pnl'] > 0 else "red"
             table.add_row(r['time'].strftime('%H:%M'), f"[{style}]{r['note']} ({r['pnl']:.2f})[/{style}]", f"{r['balance']:.2f}")
             
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    simulate_scalp()
