
import ccxt
import time
import pandas as pd
from datetime import datetime
from rich.console import Console

def check_data_availability():
    console = Console()
    console.print("[bold yellow]🔎 Checking Kraken Futures Data Depth...[/bold yellow]")
    
    ex = ccxt.krakenfutures()
    symbol = 'BTC/USD:USD'
    timeframe = '15m'
    
    # Target: 2 years ago
    # 2 years = ~730 days
    now_ms = int(time.time() * 1000)
    target_start_ms = now_ms - (730 * 24 * 60 * 60 * 1000)
    
    console.print(f"Goal: Fetch data back to {datetime.fromtimestamp(target_start_ms/1000)}")
    
    try:
        # Fetch earliest available candle via pagination simulation
        # Kraken Futures typically supports 'since' or 'until' pagination.
        
        # Test 1: Fetch far back
        since = target_start_ms
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
        
        if ohlcv:
            first_candle_time = ohlcv[0][0]
            console.print(f"[green]✅ Success! Earliest fetched: {datetime.fromtimestamp(first_candle_time/1000)}[/green]")
            
            if first_candle_time <= target_start_ms + (24*3600*1000): # Allow 1 day tolerance
                console.print("[bold green]DATA TRUTH: Kraken Futures HOLDS 2 years of 15m data.[/bold green]")
            else:
                console.print(f"[yellow]Warning: Req {datetime.fromtimestamp(target_start_ms/1000)}, Got {datetime.fromtimestamp(first_candle_time/1000)}[/yellow]")
        else:
             console.print("[red]❌ No data returned for that far back.[/red]")
             
        # Test 2: Verify Pagination Limit (How big are chunks?)
        # Fetch recent block
        ohlcv_recent = ex.fetch_ohlcv(symbol, timeframe, limit=5000) # Request massive chunk
        console.print(f"Max candles per call test: Requested 5000, Got {len(ohlcv_recent)}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    check_data_availability()
