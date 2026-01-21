import ccxt
import config
import time
import sys

def run_diagnostics():
    print("=== KRAKEN FUTURES DIAGNOSTICS ===")
    
    # Setup Exchange
    exchange_id = 'krakenfutures'
    apiKey = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
    secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
    
    print(f"Connecting to {exchange_id}...")
    exchange = getattr(ccxt, exchange_id)({
        'apiKey': apiKey,
        'secret': secret,
        'enableRateLimit': True,
    })
    
    try:
        # 1. Fetch Balance
        print("\n--- 1. ACCOUNT BALANCE ---")
        balance = exchange.fetch_balance()
        # print("Raw Balance Keys:", balance.keys()) # Debug
        
        info = balance.get('info', {})
        accounts = info.get('accounts', {})
        flex = accounts.get('flex', {})
        
        # Helper to safely get float
        def get_f(d, k):
            val = d.get(k)
            if val is None: return 0.0
            return float(val)
        
        margin_equity = get_f(flex, 'marginEquity')
        avail_margin = get_f(flex, 'availableMargin')
        used_margin = get_f(flex, 'initialMargin') # 'usedMargin' might not exist, usually initialMargin
        
        if margin_equity == 0.0:
            # Fallback to checking 'total'
            print("Flex account empty/zero. Checking 'total'...")
            margin_equity = float(balance.get('total', {}).get('USD', 0.0))
            avail_margin = float(balance.get('free', {}).get('USD', 0.0))
        
        print(f"EQUITY (Net Worth):      ${margin_equity:.2f}")
        print(f"AVAILABLE (Buying Power): ${avail_margin:.2f}")
        print(f"USED MARGIN:             ${used_margin:.2f}")
        
        # 2. Fetch Open Positions
        print("\n--- 2. OPEN POSITIONS ---")
        positions = exchange.fetch_positions()
        active_pos = [p for p in positions if float(p['contracts']) > 0]
        
        if not active_pos:
            print("No active positions.")
        else:
            for p in active_pos:
                symbol = p['symbol']
                contracts = get_f(p, 'contracts')
                side = p['side']
                entry = get_f(p, 'entryPrice')
                price = get_f(p, 'markPrice')
                if price == 0: price = entry # Fallback
                
                notional = contracts * price
                print(f"• {symbol}: {side.upper()} {contracts} @ ${entry:.2f} (Value: ${notional:.2f})")

        # 3. Fetch Open Orders
        print("\n--- 3. OPEN ORDERS ---")
        orders = exchange.fetch_open_orders()
        if not orders:
            print("No open orders.")
        else:
            for o in orders:
                print(f"• {o['symbol']} {o['side'].upper()} {o['amount']} @ {o['price']}")

        # 4. Market Limits (BTC/USDT)
        print("\n--- 4. MARKET LIMITS (BTC/USDT:PF_XBTUSD) ---")
        exchange.load_markets()
        symbol = 'PF_XBTUSD' # The Kraken Futures symbol for BTC perp usually
        
        # Try to find the right symbol if standard mapping fails
        target = 'BTC/USDT'
        if target in config.KRAKEN_SYMBOL_MAP:
             target = config.KRAKEN_SYMBOL_MAP[target]
             
        # Check if target exists in markets
        market = exchange.markets.get(target)
        if not market:
            # Fallback search
            for m in exchange.markets:
                if 'BTC' in m and 'USD' in m:
                    market = exchange.markets[m]
                    print(f"Reading limits for: {m}")
                    break
        else:
            print(f"Reading limits for: {target}")
            
        if market:
            limits = market.get('limits', {})
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')
            precision = market.get('precision', {})
            
            print(f"Min Quantity: {min_amount}")
            print(f"Min Cost ($): {min_cost}")
            print(f"Precision:    {precision}")
            
            # Calculate actual min cost for entry
            price = exchange.fetch_ticker(market['symbol'])['last']
            actual_min_notional = min_amount * price if min_amount else 0
            print(f"Current Price: ${price:.2f}")
            print(f"Calculated Min Notional: ${actual_min_notional:.2f}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostics()
