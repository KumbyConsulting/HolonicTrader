"""
HOLONIC TRADER: PRE-FLIGHT DIAGNOSTIC CHECK
Verifies all systems are ready for live trading.
"""

import sys
import os

print("=" * 60)
print("HOLONIC TRADER - PRE-FLIGHT DIAGNOSTIC")
print("=" * 60)

results = []

# 1. Config Check
print("\n[1/7] CONFIG VALIDATION")
try:
    import config
    checks = [
        ("TRADING_MODE", config.TRADING_MODE in ['FUTURES', 'SPOT']),
        ("PAPER_TRADING defined", hasattr(config, 'PAPER_TRADING')),
        ("INITIAL_CAPITAL > 0", config.INITIAL_CAPITAL > 0),
        ("ALLOWED_ASSETS", len(config.ALLOWED_ASSETS) > 0),
    ]
    for name, passed in checks:
        status = "OK" if passed else "WARN"
        print(f"  {name}: {status}")
        if not passed:
            results.append(("Config", "WARN"))
    results.append(("Config", "OK"))
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("Config", "FAIL"))

# 2. API Keys
print("\n[2/7] API KEYS")
try:
    has_spot = bool(getattr(config, 'API_KEY', ''))
    has_futures = bool(getattr(config, 'KRAKEN_FUTURES_API_KEY', ''))
    print(f"  Spot API Key: {'SET' if has_spot else 'MISSING'}")
    print(f"  Futures API Key: {'SET' if has_futures else 'MISSING'}")
    results.append(("API Keys", "OK" if (has_spot or has_futures) else "WARN"))
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("API Keys", "FAIL"))

# 3. Exchange Connectivity
print("\n[3/7] EXCHANGE CONNECTIVITY")
try:
    import ccxt
    if config.TRADING_MODE == 'FUTURES':
        exchange = ccxt.krakenfutures({
            'apiKey': config.KRAKEN_FUTURES_API_KEY or config.API_KEY,
            'secret': config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
        })
    else:
        exchange = ccxt.kraken({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET
        })
    
    # Quick market check (no auth needed)
    markets = exchange.load_markets()
    print(f"  Connected: {len(markets)} markets loaded")
    results.append(("Exchange", "OK"))
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("Exchange", "FAIL"))

# 4. Agent Imports
print("\n[4/7] AGENT IMPORTS")
agents = [
    "HolonicTrader.agent_trader",
    "HolonicTrader.agent_observer",
    "HolonicTrader.agent_entropy",
    "HolonicTrader.agent_oracle",
    "HolonicTrader.agent_guardian",
    "HolonicTrader.agent_governor",
    "HolonicTrader.agent_executor",
    "HolonicTrader.agent_actuator",
]
all_ok = True
for agent in agents:
    try:
        __import__(agent)
        print(f"  {agent.split('.')[-1]}: OK")
    except Exception as e:
        print(f"  {agent.split('.')[-1]}: FAIL ({e})")
        all_ok = False
results.append(("Agents", "OK" if all_ok else "FAIL"))

# 5. Rust Engine
print("\n[5/7] RUST ENGINE")
try:
    import holonic_speed
    funcs = [
        'calculate_shannon_entropy',
        'calculate_rsi',
        'calculate_bollinger_bands',
        'governor_check_cluster_risk',
        'executor_decide_trade',
        'oracle_analyze_for_entry'
    ]
    available = [f for f in funcs if hasattr(holonic_speed, f)]
    print(f"  Available Functions: {len(available)}/{len(funcs)}")
    results.append(("Rust Engine", "OK" if len(available) == len(funcs) else "PARTIAL"))
except ImportError:
    print("  NOT INSTALLED")
    results.append(("Rust Engine", "MISSING"))

# 6. Database
print("\n[6/7] DATABASE")
try:
    from database_manager import DatabaseManager
    db = DatabaseManager()
    print(f"  Connection: OK")
    db.close()
    results.append(("Database", "OK"))
except Exception as e:
    print(f"  ERROR: {e}")
    results.append(("Database", "FAIL"))

# 7. Paper Trading Mode
print("\n[7/7] TRADING MODE")
mode = "PAPER" if config.PAPER_TRADING else "LIVE"
print(f"  Current Mode: {mode}")
print(f"  Trading Mode: {config.TRADING_MODE}")
results.append(("Mode", mode))

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
all_pass = True
for name, status in results:
    icon = "OK" if status in ["OK", "PAPER", "LIVE"] else "!!"
    print(f"  [{icon}] {name}: {status}")
    if status in ["FAIL", "MISSING"]:
        all_pass = False

print("-" * 60)
if all_pass:
    print("SYSTEM READY FOR TRADING")
else:
    print("SOME ISSUES DETECTED - REVIEW ABOVE")
print("=" * 60)
