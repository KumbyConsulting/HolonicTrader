
import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta, timezone
from rich.console import Console
from rich.progress import Progress

def fetch_history():
    console = Console()
    console.print("[bold yellow]📥 Starting Deep History Fetch (2 Years / 15m)...[/bold yellow]")
    
    # Initialize Exchange
    ex = ccxt.krakenfutures({
        'enableRateLimit': True, # CCXT handles basic rate limits
    })
    
    # Target Assets (Full Portfolio)
    # Mapping internal symbol -> Kraken Futures ticker
    symbols_map = {
        'BTC/USDT': 'BTC/USD:USD',
        'ETH/USDT': 'ETH/USD:USD',
        'SOL/USDT': 'SOL/USD:USD',
        'XRP/USDT': 'XRP/USD:USD',
        'ADA/USDT': 'ADA/USD:USD',
        'DOGE/USDT': 'DOGE/USD:USD',
        'LINK/USDT': 'LINK/USD:USD',
        'LTC/USDT': 'LTC/USD:USD',
        'XTZ/USDT': 'XTZ/USD:USD',
        'AVAX/USDT': 'AVAX/USD:USD',
        'DOT/USDT': 'DOT/USD:USD',
        'PAXG/USDT': 'PAXG/USD:USD'
    }
    
    timeframe = '15m'
    limit_per_call = 1500 # Safe buffer below 2000
    
    # 2 Years ago
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=730)
    start_ts = int(start_time.timestamp() * 1000)
    
    data_dir = os.path.join(os.getcwd(), 'market_data')
    os.makedirs(data_dir, exist_ok=True)
    
    for display_sym, api_sym in symbols_map.items():
        console.print(f"\n[cyan]Fetching {display_sym} ({api_sym})...[/cyan]")
        all_candles = []
        current_since = start_ts
        
        # Calculate approx calls needed
        total_time_ms = int(end_time.timestamp() * 1000) - start_ts
        ms_per_candle = 15 * 60 * 1000
        total_candles = total_time_ms / ms_per_candle
        total_calls = int(total_candles / limit_per_call) + 2
        
        with Progress() as progress:
            task = progress.add_task(f"[green]Downloading {display_sym}...", total=total_calls)
            
            while True:
                try:
                    ohlcv = ex.fetch_ohlcv(api_sym, timeframe, since=current_since, limit=limit_per_call)
                    
                    if not ohlcv:
                        break
                        
                    all_candles.extend(ohlcv)
                    
                    # Update 'since' to the last timestamp + 1ms
                    last_ts = ohlcv[-1][0]
                    current_since = last_ts + 1
                    
                    progress.update(task, advance=1)
                    
                    # Manual Rate Limit (Safe Mode)
                    # Kraken limit is generous, but let's be polite: 0.5s sleep
                    time.sleep(0.5)
                    
                    # Check if we reached now
                    if last_ts >= int(end_time.timestamp() * 1000):
                        break
                        
                except Exception as e:
                    console.print(f"[red]Error fetching chunk:[/red] {e}")
                    time.sleep(5) # Backoff on error
                    
        # Save to CSV
        if all_candles:
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Clean header name for filename
            clean_sym = display_sym.replace('/', '').replace(':', '')
            filename = f"{clean_sym}_15m.csv"
            filepath = os.path.join(data_dir, filename)
            
            df.to_csv(filepath, index=False)
            console.print(f"[bold green]✅ Saved {len(df)} candles to {filepath}[/bold green]")
        else:
            console.print(f"[red]❌ No data collected for {display_sym}[/red]")

if __name__ == "__main__":
    fetch_history()
