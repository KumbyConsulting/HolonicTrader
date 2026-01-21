import ccxt
import config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import time

console = Console()

def fetch_and_display_account():
    """
    Connects to Kraken Futures and displays a full account preview.
    """
    console.print("[bold blue]🔍 Connecting to Kraken Futures...[/bold blue]")
    
    # 1. Initialize Exchange
    try:
        exchange = ccxt.krakenfutures({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
        })
        exchange.load_markets()
    except Exception as e:
        console.print(f"[bold red]❌ Connection Failed:[/bold red] {e}")
        return

    # 2. Fetch Data
    try:
        balance = exchange.fetch_balance()
        positions = exchange.fetch_positions()
        orders = exchange.fetch_open_orders()
    except Exception as e:
        console.print(f"[bold red]❌ Data Fetch Error:[/bold red] {e}")
        return

    # 3. Process Balance
    # Kraken Futures structure: 'info', 'USDT', 'total', 'free', 'used'
    # We care about the 'total' USD value (Equity) and 'free' (Available Margin)
    
    # Try different keys usually found in futures balances
    total_equity = 0.0
    free_margin = 0.0
    
    # Check for specific futures wallet structure (often 'USDT' or 'USD')
    # CCXT unifies this, but let's debug if needed.
    if 'total' in balance:
        total_equity = balance['total'].get('USD', balance['total'].get('USDT', 0.0))
        free_margin = balance['free'].get('USD', balance['free'].get('USDT', 0.0))
    else:
        # Fallback inspection
        total_equity = float(balance['info'].get('totalWalletBalance', 0.0)) # Example
    
    # If using 'krakenfutures', 'total' usually contains the unified stats
    # Specific to Kraken Futures (cf):
    # 'total': {'USD': ...}
    
    # Calculate Drawdown
    start_cap = config.INITIAL_CAPITAL
    pnl = total_equity - start_cap
    pnl_pct = (pnl / start_cap) * 100 if start_cap > 0 else 0.0
    
    # 4. Display Balance Panel
    bal_table = Table(title="🏦 Account Overview", show_header=True, header_style="bold magenta")
    bal_table.add_column("Metric", style="cyan")
    bal_table.add_column("Value", style="bold white", justify="right")
    
    bal_table.add_row("Total Equity", f"${total_equity:.2f}")
    bal_table.add_row("Available Margin", f"${free_margin:.2f}")
    bal_table.add_row("Initial Capital", f"${start_cap:.2f}")
    
    pnl_color = "green" if pnl >= 0 else "red"
    bal_table.add_row("Total PnL", f"[{pnl_color}]${pnl:.2f} ({pnl_pct:+.2f}%)[/{pnl_color}]")
    
    console.print(bal_table)
    
    # 5. Display Positions
    pos_table = Table(title="📦 Active Positions", show_header=True, header_style="bold yellow")
    pos_table.add_column("Symbol")
    pos_table.add_column("Side")
    pos_table.add_column("Size")
    pos_table.add_column("Entry Price")
    pos_table.add_column("Mark Price")
    pos_table.add_column("Leverage")
    pos_table.add_column("Unrealized PnL", justify="right")
    
    active_positions = [p for p in positions if float(p['contracts']) > 0]
    
    if active_positions:
        for p in active_positions:
            sym = p['symbol']
            side = p['side'].upper() # 'long' or 'short'
            size = float(p['contracts'])
            entry = float(p['entryPrice'])
            mark = float(p['markPrice']) if p.get('markPrice') else entry # Fallback
            leverage = p['leverage']
            unrealized = float(p['unrealizedPnl']) if p.get('unrealizedPnl') else 0.0
            
            pnl_style = "green" if unrealized >= 0 else "red"
            side_style = "green" if side == 'LONG' else "red"
            
            pos_table.add_row(
                sym,
                f"[{side_style}]{side}[/{side_style}]",
                f"{size:.4f}",
                f"${entry:.2f}",
                f"${mark:.2f}",
                f"{leverage}x",
                f"[{pnl_style}]${unrealized:.2f}[/{pnl_style}]"
            )
        console.print(pos_table)
    else:
        console.print(Panel("✅ No Active Positions", style="green"))

    # 6. Display Open Orders
    order_table = Table(title="📝 Open Orders", show_header=True, header_style="bold blue")
    order_table.add_column("Symbol")
    order_table.add_column("Type")
    order_table.add_column("Side")
    order_table.add_column("Price")
    order_table.add_column("Amount")
    
    if orders:
        for o in orders:
            order_table.add_row(
                o['symbol'],
                o['type'],
                o['side'].upper(),
                f"${o['price']}",
                f"{o['amount']}"
            )
        console.print(order_table)
    else:
        console.print(Panel("💤 No Open Orders", style="dim"))

if __name__ == "__main__":
    fetch_and_display_account()
