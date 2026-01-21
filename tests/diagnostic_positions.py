import ccxt
import config

def check_positions():
    print("=== POSITIONS ONLY CHECK ===")
    exchange_id = 'krakenfutures'
    apiKey = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
    secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
    
    exchange = getattr(ccxt, exchange_id)({
        'apiKey': apiKey,
        'secret': secret,
        'enableRateLimit': True,
    })
    
    try:
        positions = exchange.fetch_positions()
        active = [p for p in positions if float(p.get('contracts', 0)) > 0]
        
        if not active:
            print("No Active Positions Found.")
        else:
            for p in active:
                sym = p['symbol']
                qty = float(p.get('contracts', 0))
                price = float(p.get('entryPrice', 0))
                side = p['side']
                print(f"FOUND: {sym} {side.upper()} {qty} @ ${price}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_positions()
