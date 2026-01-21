"""
NEXUS Configuration (Phase 15) - NANO MODE EDITION

CENTRAL STORAGE for all thresholds, leverage caps, and system parameters.
UPDATED 2026-01-13 for NANO accounts (< $50)
FIXED: Position sizing, leverage, and margin calculations for $25 account
"""

from dotenv import load_dotenv
import os

load_dotenv()

# === API KEYS ===
KRAKEN_FUTURES_API_KEY = os.getenv('KRAKEN_FUTURES_API_KEY')
KRAKEN_FUTURES_PRIVATE_KEY = os.getenv('KRAKEN_FUTURES_PRIVATE_KEY')
KRAKEN_SPOT_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_SPOT_SECRET = os.getenv('KRAKEN_PRIVATE_KEY')

# Active Keys
API_KEY = KRAKEN_FUTURES_API_KEY or KRAKEN_SPOT_KEY
API_SECRET = KRAKEN_FUTURES_PRIVATE_KEY or KRAKEN_SPOT_SECRET

# === TELEGRAM INTEGRATION ===
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# === ACCOUNT PARAMETERS ===
SCAVENGER_THRESHOLD = 90.0                # Balance <= this = Scavenger Mode
INITIAL_CAPITAL = 300.0                   # Reset for Paper Trading (2026-01-14)
MISSION_TARGET = 3000.0                   # GOAL: Grow to $3000
MISSION_NAME = "Operation Paper Centurion"
PRINCIPAL = 100.0                         # Floor for capital protection (Lowered for buying power)
PAPER_TRADING = True                      # PAPER MODE for test run

# === TIME SETTINGS ===
TIMEFRAME = '15m'                         # Default trading timeframe
DEFAULT_CYCLE_INTERVAL = 30               # Faster Response (User Request)


# === TRADING MODE ===
TRADING_MODE = 'FUTURES'

# === KRAKEN SYMBOL MAPPING (FUTURES) ===
KRAKEN_SYMBOL_MAP = {
    'BTC/USDT': 'BTC/USD',
    'ETH/USDT': 'ETH/USD',
    'SOL/USDT': 'SOL/USD',
    'XRP/USDT': 'XRP/USD',
    'ADA/USDT': 'ADA/USD',
    'DOGE/USDT': 'DOGE/USD',
    'SUI/USDT': 'SUI/USD',
    'UNI/USDT': 'UNI/USD',
    'AAVE/USDT': 'AAVE/USD',
    'SHIB/USDT': 'SHIB/USD',
    'PAXG/USDT': 'PAXG/USD',
    'LINK/USDT': 'LINK/USD',
    'BNB/USDT': 'BNB/USD',
    'LTC/USDT': 'LTC/USD',
    'XMR/USDT': 'XMR/USD',
    'XTZ/USDT': 'XTZ/USD',
    'AVAX/USDT': 'AVAX/USD',
    'DOT/USDT': 'DOT/USD',
    # MATIC REMOVED (Rebranded to POL/Delisted on KuCoin)
    'NEAR/USDT': 'NEAR/USD',
    'PEPE/USDT': 'PEPE/USD',
    'TAO/USDT': 'TAO/USD',
    'XAUT/USDT': 'XAUT/USD',
}

# === ASSET CONSTRAINTS (MOVED UP FOR DEPENDENCIES) ===
# UNLEASHED: Allow ALL mapped assets
ALLOWED_ASSETS = list(KRAKEN_SYMBOL_MAP.keys())
# ALLOWED_ASSETS = [
#     'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'DOGE/USDT',
#     'ADA/USDT', 'LINK/USDT', 'LTC/USDT', 'XTZ/USDT', 'AVAX/USDT',
#     'DOT/USDT', 'PAXG/USDT', 'TAO/USDT', 'XAUT/USDT',
# ]

# === SCOUT ARCHITECTURE ===
# The "Hot" List: Assets actively being traded/analyzed deeply
ACTIVE_WATCHLIST = ALLOWED_ASSETS.copy()

# === TRINITY STRATEGY PREFERENCES ===
# User Control: Define which assets play which role in the strategy
ASSET_PREF_PREDATOR = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SUI/USDT', 'TAO/USDT', 'XMR/USDT', 'AVAX/USDT']   # High Conviction Trenders
ASSET_PREF_SCAVENGER = ['XRP/USDT', 'DOGE/USDT', 'PEPE/USDT', 'SHIB/USDT', 'ADA/USDT', 'LINK/USDT', 'DOT/USDT'] # Volatility Harvesters

# Kraken Futures Minimums (verified)
MIN_TRADE_QTY = {
    'BTC': 0.0001,   # ~$9 at $90k BTC
    'ETH': 0.001,    # ~$3 at $3k ETH
    'SOL': 0.01,     # ~$0.15 at $15 SOL
    'XRP': 1.0,      # ~$0.50
    'ADA': 1.0,      # ~$0.50
    'DOGE': 10.0,    # ~$0.50
    'LTC': 0.1,      # ~$7
}

TICK_SIZES = {
    'BTC': 0.50,
    'ETH': 0.10,
    'SOL': 0.01,
    'ADA': 0.00001,
    'XRP': 0.00001,
}

# === REGIME SYSTEM (CAPITAL-BASED) ===
# Updated thresholds for realistic progression
REGIME_NANO_CEILING = 49.0                # NANO: $0 - $49 (YOU ARE HERE)
REGIME_MICRO_CEILING = 249.0              # MICRO: $50 - $249
REGIME_SMALL_CEILING = 999.0              # SMALL: $250 - $999
REGIME_MEDIUM_CEILING = 9999.0            # MEDIUM: $1000+

# Promotion/Demotion Rules
REGIME_PROMOTION_STABILITY_HOURS = 72
REGIME_PROMOTION_HEALTH_THRESHOLD = 0.95
REGIME_PROMOTION_MIN_TRADES = 20
REGIME_DEMOTION_BUFFER = 5.0
REGIME_DEMOTION_DRAWDOWN_PCT = 0.25

# === REGIME-SPECIFIC PERMISSIONS (UPDATED FOR NANO SAFETY) ===
# === REGIME-SPECIFIC PERMISSIONS (UPDATED FOR NANO SAFETY) ===
REGIME_PERMISSIONS = {
    'NANO': {  # FOR $0-$49 ACCOUNTS (AGGRESSIVE MODE)
        'max_positions': 5,               # Allow Multi-Asset (5 slots)
        'max_stacks': 0,                  # No stacking allowed
        'max_exposure_ratio': 10.0,       # Max 10x total exposure
        'max_leverage': 50.0,             # UNLOCKED: 50x (Kraken Futures)
        'allowed_pairs': ALLOWED_ASSETS,  # Trade EVERYTHING
        'correlation_check': True,        # Enable correlation check
        'min_order_value': 2.0,           # $2 minimum position
        'max_order_value_pct': 0.40,      # Max 40% of capital per position
        'cooldown_after_failure': 60,     # Fast retry
    },
    'MICRO': {  # UNLOCKED FOR ACTIVE TRADING
        'max_positions': 5,               # 5 positions allowed (was 2)
        'max_stacks': 0,                  # Still no pyramiding
        'max_exposure_ratio': 20.0,       # Increased for High Leverage support
        'max_leverage': 10.0,             # UNLOCKED: 10x (Progressive Tier: Nano 1.5 -> Micro 10)
        'allowed_pairs': ALLOWED_ASSETS,  # Trade all assets (was limited)
        'correlation_check': True,        # Keep cluster risk
        'min_order_value': 3.0,           # $3 minimum
        'max_order_value_pct': 0.30,      # 30% of capital
        'cooldown_after_failure': 120,
    },
    'SMALL': {
        'max_positions': 12,
        'max_stacks': 6,
        'max_exposure_ratio': 20.0,       # Increased for High Leverage support
        'max_leverage': 25.0,             # UNLOCKED: 25x (Reduced from 50x for Safety)
        'allowed_pairs': ALLOWED_ASSETS,
        'correlation_check': True,
        'min_order_value': 10.0,
        'max_order_value_pct': 0.25,
        'cooldown_after_failure': 120,
    },
    'MEDIUM': {
        'max_positions': 24,
        'max_stacks': 2,
        'max_exposure_ratio': 25.0,       # Increased for High Leverage support
        'max_leverage': 50.0,             # UNLOCKED: 50x (Kraken Futures)
        'allowed_pairs': ALLOWED_ASSETS,
        'correlation_check': True,
        'min_order_value': 20.0,
        'max_order_value_pct': 0.30,
        'cooldown_after_failure': 60,
    },
}

# === SECTOR DEFINITIONS (Holistic Phase 5b) ===
MEMECOIN_ASSETS = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
MEMECOIN_PUMP_RVOL = 3.0
POLYMARKET_PATIENCE_MINUTES = 3 

# === MACRO STRATEGY LISTS ===
CRISIS_SAFE_HAVENS = ['BTC/USDT', 'PAXG/USDT']
CRISIS_RISK_ASSETS = ['DOGE/USDT', 'ADA/USDT', 'SOL/USDT']

# === VOLATILITY WINDOW (High-Entropy Regime) ===
VOL_WINDOW_RISK_PCT = 0.02           # 2% Risk per trade
VOL_WINDOW_LEVERAGE = 5              # Max 5x Leverage
VOL_WINDOW_CYCLE_INTERVAL = 15       # 15s Cycle
VOL_WINDOW_BTC_VOL_THRESHOLD = 0.45  # > 45% Annualized Volatility
VOL_WINDOW_FUNDING_THRESHOLD = 0.0003 # > 0.03% per 8h
VOL_WINDOW_SPREAD_THRESHOLD = 0.004  # < 0.4% Spread required
VOL_WINDOW_MIN_BALANCE_SHUTOFF = 5.0 # Stop if balance < $5
VOL_WINDOW_TARGET_PROFIT = 0.39      # +39% Daily Target
VOL_WINDOW_MAX_POSITIONS = 3         # Max 3 concurrent positions
VOL_WINDOW_GROSS_RISK_CAP = 0.06     # Max 6% NAV Gross Risk
VOL_WINDOW_MIN_VOLATILITY = 0.40       # Minimum Volatility to activate window


# === SATELLITE COMMANDER (TIER 2) ===
# 🧬 APEX PREDATOR (NANO CHALLENGE 2026-01-14)
# Goal: $17.90 -> $100.00 | Achieved in Sim: $40.26 (+125% ROI)
SATELLITE_ASSETS = ['SOL/USDT', 'DOGE/USDT', 'ADA/USDT'] 
SATELLITE_MARGIN = 17.0      # Approx All-In on $17 Account (Nano Mode)
SATELLITE_LEVERAGE = 5.0     # Reduced to 5x for Survival (Winner's setting)

# Evolved Genome (RELAXED FOR RANGING MARKETS)
SATELLITE_RVOL_THRESHOLD = 1.5   # Relaxed from 2.29 for ranging markets
SATELLITE_DOGE_RVOL_THRESHOLD = 2.0  # Relaxed from 3.0
SATELLITE_BBW_EXPANSION_THRESHOLD = 0.15 # Relaxed from 0.216
SATELLITE_ENTRY_RSI_CAP = 55.0  # Relaxed from 45.15 (Buy below 55)

# Risk Management
SATELLITE_BREAKEVEN_TRIGGER = 0.02 # Move SL at +2%
SATELLITE_TAKE_PROFIT_1 = 0.51 # +51% Target (Moon Bag)
SATELLITE_STOP_LOSS = 0.057 # -5.7% Stop (Tight enough to survive 5x)

# === LEVERAGE SETTINGS (SAFETY FIRST) ===
# Global maximums - actual limits set by regime
SCAVENGER_LEVERAGE = 50                   # Cap at 50x (Unleashed)
PREDATOR_LEVERAGE = 50                    # Unlocked 50x
MICRO_HARD_LEVERAGE_LIMIT = 50.0          # Unlocked for NANO Aggression
PREDATOR_TRAILING_STOP_ATR_MULT = 2.0     # 2.0x ATR Trailing Stop


# === POSITION SIZING & RISK (UNLEASHED) ===
MIN_ORDER_VALUE = 2.0                     # Lowered from 5.0 for NANO accounts
MAX_RISK_PCT = 0.05                       # UNLEASHED: 5% Risk per trade (Was 2%)
SCAVENGER_MAX_MARGIN = 0.85               # UNLEASHED: Max 85% Margin (Was 80%)
SCAVENGER_STOP_LOSS = 0.10                # UNLEASHED: 10% stop loss (Was 5%)
SCAVENGER_SCALP_TP = 0.06                 # UNLEASHED: 6% take profit (Was 4%) to Capture bigger swings
PREDATOR_STOP_LOSS = 0.08                 # UNLEASHED: 8% stop loss (Was 5%)
PREDATOR_TAKE_PROFIT = 0.12               # UNLEASHED: 12% take profit (Was 6%) - Let winners RUN

# === MICRO-GUARD HARD LIMITS (UNLEASHED) ===
MICRO_GUARD_PORTFOLIO_NOTIONAL_MULT = 3.0  # Restored to 3.0 (Was 1.5)
MICRO_GUARD_SINGLE_NOTIONAL_MULT = 1.5     # Increased to 1.5 (Was 0.75)
MICRO_GUARD_GROSS_LEVERAGE_MULT = 3.0      # Restored to 3.0 (Was 1.5)
MICRO_GUARD_GROSS_LEVERAGE = 10.0          # UNLEASHED: 10x Cap (Was 3.0)


# Emergency protection
MICRO_GUARD_CASH_PRESERVATION_THRESHOLD = 10.0
MICRO_GUARD_CASH_PRESERVATION_LEVERAGE = 2.0

# === ORDER FAILURE PROTECTION ===
ORDER_FAILURE_COOLDOWN = {
    'insufficientAvailableFunds': 300,     # 5 minutes
    'wouldNotReducePosition': 60,          # 1 minute
    'network': 10,                         # 10 seconds
    'default': 120,                        # 2 minutes for other errors
}

MAX_CONSECUTIVE_FAILURES = 3               # Stop after 3 consecutive failures

# === SCOUT SETTINGS ===
SCOUT_CANDIDATES = ALLOWED_ASSETS  # Keep it simple for NANO
SCOUT_RVOL_THRESHOLD = 2.5
SCOUT_HYPE_THRESHOLD = 0.5
SCOUT_CACHE_TTL = 60
SCOUT_CYCLE_INTERVAL = 60

# === STRATEGY SETTINGS (RELAXED FOR RANGING MARKETS) ===
STRATEGY_RSI_OVERSOLD = 35            # Relaxed from 45
STRATEGY_RSI_OVERBOUGHT = 65          # Relaxed from 70
STRATEGY_RSI_PANIC_BUY = 30.0         # Relaxed from 40.0
STRATEGY_RSI_ENTRY_MAX = 60.0         # Relaxed from 70.0
STRATEGY_LSTM_THRESHOLD = 0.52        # Slightly raised for more entries
STRATEGY_XGB_THRESHOLD = 0.50         # Slightly raised from 0.48
STRATEGY_POST_EXIT_COOLDOWN_CANDLES = 2  # Faster re-entry

# === GOVERNANCE / RISK ===
GOVERNOR_COOLDOWN_SECONDS = 60
GOVERNOR_MIN_STACK_DIST = 0.005 # Increased to 0.5% (User Request)
GOVERNOR_MAX_MARGIN_PCT = 0.85  # UNLEASHED: 85% Utiliization (Was 0.60)
GOVERNOR_STACK_DECAY = 0.8
GOVERNOR_MAX_TREND_AGE_HOURS = 24.0
GOVERNOR_TREND_DECAY_START = 12.0
# === VOLATILITY ===
ATR_PERIOD = 14
ATR_STORM_MULTIPLIER = 3.0
ATR_STOP_LOSS_MULTIPLIER = 3.0 # UNLEASHED: 3.0x ATR Wide Stop (Was 2.0)
KELLY_LOOKBACK = 20 # Number of trades to calculate Win Rate for Kelly Criterion
VOL_SCALAR_MIN = 0.5 # Min Volatility Scalar (High Vol = 0.5x size)
VOL_SCALAR_MAX = 2.0 # Max Volatility Scalar (Low Vol = 2.0x size)

# === BOLLINGER BANDS ===
BB_PERIOD = 20
BB_STD = 2

# === SENTIMENT & NEWS ===
SENTIMENT_SOURCES = [
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://cryptopanic.com/news/rss/',
    'https://beincrypto.com/feed/',
    # === MACRO & RETAIL SOURCES === 
    'https://www.cnbc.com/id/100727362/device/rss/rss.html', 
    'https://www.investing.com/rss/commodities.rss', 
    'https://www.reddit.com/r/CryptoCurrency/top/.rss?t=hour'
]

# === PHASE 39: MARKET PHYSICS (VALIDATION LAYER) ===
PHYSICS_CORRELATION_THRESHOLD = 0.75  # Veto Longs if Corr > 0.75 & BTC Bearish
PHYSICS_MIN_RVOL = 1.1                # Veto Breakouts if Volume < 1.1x Avg
PHYSICS_MAX_ENTROPY = 1.35            # Veto All Signals if Entropy > 1.35 (Chaos)

# === CORRELATION GUARD ===
CORRELATION_CHECK = True              # Enable/Disable Correlation Matrix Checks

# === WHALE TRACKING (Project Ahab) ===
WHALE_RVOL_THRESHOLD = 3.0            # 300% Volume Spike = Whale
WHALE_SENTIMENT_WEIGHT = 0.2          # Boost sentiment if whales mentioned
# Ahab Phase 1: Stealth Accumulation
WHALE_ACCUMULATION_RVOL = 2.0         # 200% Volume (Quiet Buying)
WHALE_ACCUMULATION_ATR_FACTOR = 0.8   # Low Volatility (Price Contained)
# Ahab Phase 2: Defense & Squeeze
WHALE_DEFENSE_RVOL = 2.5              # 250% Volume (Defending Level)
WHALE_ORDER_IMBALANCE_RATIO = 2.0     # 2x Buying Pressure (Bid Wall)
WHALE_FUNDING_SQUEEZE_THRESHOLD = -0.00005 # -0.005% Funding (Shorts trapped)
SENTIMENT_WEIGHT = 0.3
SENTIMENT_THRESHOLD_BULL = 0.2
SENTIMENT_THRESHOLD_BEAR = -0.2

# === CROSS-ASSET CORRELATION ===
GMB_THRESHOLD = 0.40  # Lowered from 0.50 for neutral markets

# === SAFETY THRESHOLDS (Governor) ===
SOLVENCY_PANIC_THRESHOLD = 0.98       # UNLEASHED: 98% (Was 0.95)
SOLVENCY_PANIC_REDUCTION = 0.20       # Reduce to 20% size if panic hit

# === CONSOLIDATION ENGINE ===
CONSOLIDATION_WEIGHT_PNL = 0.30
CONSOLIDATION_WEIGHT_CONVICTION = 0.25
CONSOLIDATION_WEIGHT_LIQUIDITY = 0.15
CONSOLIDATION_WEIGHT_AGE = 0.10
CONSOLIDATION_WEIGHT_CORRELATION = 0.20

CONSOLIDATION_DUST_THRESHOLD = 1.0
CONSOLIDATION_STALE_HOURS = 24.0
CONSOLIDATION_MARGIN_BUFFER = 1.5
CONSOLIDATION_HARD_BUFFER = 1.2

# === ACCUMULATOR ===
ACC_RISK_FLOOR = 0.5
ACC_RISK_CEILING = 2.0
ACC_DRAWDOWN_LIMIT = 0.40  # Refined for SMALL/NANO Volatility (Was 0.25)
ACC_SANITY_THRESHOLD = 0.60 # Only reset HWM if drop is > 60% (Glitch protection)

# === PPO SOVEREIGN BRAIN ===
PPO_LEARNING_RATE = 0.0003
PPO_CLIP_RATIO = 0.2
PPO_REWARD_DRAWDOWN_PENALTY = 2.0

# === CONCURRENCY & RATE LIMITING ===
CCXT_RATE_LIMIT = True
CCXT_POOL_SIZE = 10         # Reduced for NANO accounts
TRADER_MAX_WORKERS = 8      # Reduced for NANO accounts

# === INTEL GPU ACCELERATION ===
USE_INTEL_GPU = True
USE_OPENVINO = True

# === IMMUNE SYSTEM & PERSONALITY ===
FAMILY_L1 = ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT']
FAMILY_PAYMENT = ['XRP/USDT', 'LTC/USDT']
FAMILY_MEME = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']

IMMUNE_MAX_DAILY_DRAWDOWN = 0.20        # UNLEASHED: 20% (Was 10%)
IMMUNE_MAX_LEVERAGE_RATIO = 5.0         # Absolute Notional Cap (5x Equity)NANO

# === LIQUIDATION ENGINE ===
MAINTENANCE_MARGIN_RATE = 0.50

# === UNIFIED CONTROL PROTOCOL ===
MICRO_CAPITAL_MODE = True
MICRO_MAX_LEVERAGE = 3.0
MICRO_MAX_EXPOSURE_RATIO = 3.5
MICRO_MAX_POSITIONS = 2

# Stacking restrictions
STACKING_MIN_EQUITY = 100.0
STACKING_BUFFER_MULTIPLIER = 5.0

# MicroMode refinements
MICRO_STARTUP_MAX_POSITIONS = 2

# === NANO-MODE CONFIG (REALITY ANCHORED) ===
NANO_CAPITAL_THRESHOLD = 50.0
NANO_ALLOCATION_PCT = 0.05           # 5% Max Position Size ($0.77 at $15)
NANO_MAX_LEVERAGE = 1.5              # 1.5x NANO Safety Leverage
NANO_MAX_POSITIONS = 1               # Max 1 Position
NANO_COOLDOWN_AFTER_FAILURE = 86400  # 24 Hours (Extreme Punishment)

# === PERSONALITY PARAMETERS ===
PERSONALITY_BTC_ATR_FILTER = 0.5
PERSONALITY_SOL_RSI_LONG = 50.0 # Relaxed from 55.0 for better sensitivity
PERSONALITY_SOL_RSI_SHORT = 45.0
PERSONALITY_DOGE_RVOL = 2.0

# === CONVICTION DECAY ===
CONVICTION_DECAY_BASE_HOURS = 48.0
CONVICTION_DECAY_CAPITAL_MULTIPLIER = 5.0

# === IRON BANK (Capital Preservation) ===
IRON_BANK_ENABLED = True
IRON_BANK_MIN_RESERVE = 100.0  # Floor (Lowered for buying power)
IRON_BANK_RATCHET_PCT = 0.00   # LOCKED: Profit Ratchet Disabled.
IRON_BANK_BUFFER_PCT = 0.05    # Keep 5% buffer above Floor before Ratcheting

# === MARKET TOPOLOGY (TDA) ===
TOPOLOGY_WARNING_THRESHOLD = 0.0001  # Validated Low (Normal: 0.003-0.02)
TOPOLOGY_WINDOW_SIZE = 50



# === GARBAGE COLLECTOR ===
GC_INTERVAL_CYCLES = 10
GC_STALE_ORDER_TIMEOUT = 180
GC_LOG_VERBOSE = True

# === API RESILIENCE ===
API_MAX_RETRIES = 15
API_HIBERNATION_TIME = 60
API_RETRY_JITTER_MIN = 1.0
API_RETRY_JITTER_MAX = 3.0
API_RATE_LIMIT_COOL = 10.0
