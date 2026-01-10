"""
NEXUS Configuration (Phase 15)

Central storage for all thresholds, leverage caps, and system parameters.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Prioritize Futures Keys if present, otherwise fallback to Spot (if unified)
KRAKEN_FUTURES_API_KEY = os.getenv('KRAKEN_FUTURES_API_KEY')
KRAKEN_FUTURES_PRIVATE_KEY = os.getenv('KRAKEN_FUTURES_PRIVATE_KEY')

# Base Keys (Spot)
KRAKEN_SPOT_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_SPOT_SECRET = os.getenv('KRAKEN_PRIVATE_KEY')

# Active Keys (Selected dynamically later or defaulted here)
API_KEY = KRAKEN_SPOT_KEY
API_SECRET = KRAKEN_SPOT_SECRET

# === TELEGRAM INTEGRATION ===
# SECURITY: Tokens must come from .env file. No hardcoded fallbacks.
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)  # Auto-disable if missing

SCAVENGER_THRESHOLD = 90.0  # Balance <= this = Scavenger Mode (PREDATOR @ $100)
INITIAL_CAPITAL = 25.0    # Updated to reflect approximate real starting balance
MISSION_TARGET = 100.0    # GOAL: Grow account to this amount
MISSION_NAME = "Operation Centurion"
PRINCIPAL = 20.0          # Protect $20 (Hard Stop)

PAPER_TRADING = False        # Set to False to enable real exchange execution

# === TIME SETTINGS ===
TIMEFRAME = '15m'            # Default trading timeframe

# === LEVERAGE SETTINGS ===
SCAVENGER_LEVERAGE = 5      # WARP SPEED: Max Alt Leverage (Reduced to 5x for safety)
PREDATOR_LEVERAGE = 10       # WARP SPEED: Max BTC Leverage (Reduced to 10x for safety)

# === POSITION SIZING & RISK ===
SCAVENGER_MAX_MARGIN = 8.0   # WARP SPEED: Risk 80% of Equity
SCAVENGER_STOP_LOSS = 0.03   # 3% wiggle room
SCAVENGER_SCALP_TP = 0.01    # 1% Quick Scalp target

PREDATOR_STOP_LOSS = 0.05    # 5% wider stop for trends
PREDATOR_TAKE_PROFIT = 0.03  # 3% target (Reduced from 5% for tighter exits)

# === TIME GATING (Kill Zones - UTC) ===
KILL_ZONES = [] # WARP SPEED: Trade 24/7

# === VOLATILITY ===
ATR_PERIOD = 14
ATR_STORM_MULTIPLIER = 3.0   # If current ATR > 3x avg, it's a storm

# === BOLLINGER BANDS ===
BB_PERIOD = 20
BB_STD = 2

# === PREDATOR TRAILING STOP ===
PREDATOR_TRAILING_STOP_ATR_MULT = 2.0

# === ORDER CONSTRAINTS ===
MIN_ORDER_VALUE = 2.0       # Minimum order size in USD (Lowered for Micro Scavenger)

# === TRADING MODE ===
# 'SPOT' or 'FUTURES'
TRADING_MODE = 'FUTURES'

# === KRAKEN SYMBOL MAPPING ===
# Maps internal USDT symbols to Kraken specific pairs.
# For FUTURES, we use the Linear Swap symbols (USD Margined) found via CCXT.
if TRADING_MODE == 'FUTURES':
    KRAKEN_SYMBOL_MAP = {
        'BTC/USDT': 'BTC/USD:USD',
        'ETH/USDT': 'ETH/USD:USD',
        'SOL/USDT': 'SOL/USD:USD',
        'XRP/USDT': 'XRP/USD:USD',
        'ADA/USDT': 'ADA/USD:USD',
        'DOGE/USDT': 'DOGE/USD:USD',
        'SUI/USDT': 'SUI/USD:USD',
        'UNI/USDT': 'UNI/USD:USD',
        'AAVE/USDT': 'AAVE/USD:USD',
        'SHIB/USDT': 'SHIB/USD:USD',
        'PAXG/USDT': 'PAXG/USD:USD',
        'LINK/USDT': 'LINK/USD:USD',
        'BNB/USDT': 'BNB/USD:USD',
        'LTC/USDT': 'LTC/USD:USD',
        'XMR/USDT': 'XMR/USD:USD',
        'XTZ/USDT': 'XTZ/USD:USD',
        'AVAX/USDT': 'AVAX/USD:USD',
        'DOT/USDT': 'DOT/USD:USD',
        'MATIC/USDT': 'MATIC/USD:USD',
        'NEAR/USDT': 'NEAR/USD:USD',
        'PEPE/USDT': 'PEPE/USD:USD',
    }
else:
    # SPOT MAPPING
    KRAKEN_SYMBOL_MAP = {
        'SUI/USDT': 'SUI/USD',
        'UNI/USDT': 'UNI/USD',
        'AAVE/USDT': 'AAVE/USD',
        'PAXG/USDT': 'PAXG/USD',
        'LINK/USDT': 'LINK/USD',
    }

# === SECTOR DEFINITIONS (Holistic Phase 5b) ===
MEMECOIN_ASSETS = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']
MEMECOIN_PUMP_RVOL = 3.0
POLYMARKET_PATIENCE_MINUTES = 3 # Fast reaction for Holistic Mode

# === ASSET CONSTRAINTS ===
# Kraken Futures Perps with good liquidity and low minimum order sizes
# Suitable for micro-accounts ($10-$100)
ALLOWED_ASSETS = [
    # Tier 1: High Liquidity (Tightest Spreads)
    'BTC/USDT',   # Bitcoin - King of liquidity
    'ETH/USDT',   # Ethereum - Second most liquid
    
    # Tier 2: Major Alts (Good Liquidity, Low Mins)
    'XRP/USDT',   # Ripple - Very liquid, ~$0.01 min
    'SOL/USDT',   # Solana - Hot momentum, low min
    'DOGE/USDT',  # Dogecoin - Meme but liquid
    'ADA/USDT',   # Cardano - Steady volume
    'LINK/USDT',  # Chainlink - DeFi blue chip
    'LTC/USDT',   # Litecoin - OG alt, liquid
    
    'XTZ/USDT',   # Tezos - ~$1 min
    'AVAX/USDT',  # Avalanche - Good volatility
    'DOT/USDT',   # Polkadot - Solid volume
    'PAXG/USDT',  # Paxos Gold - Crisis Hedge
]

# === SCOUT ARCHITECTURE (Dynamic Watchlists) ===
# The "Hot" List: Assets actively being traded/analyzed deeply
ACTIVE_WATCHLIST = ALLOWED_ASSETS.copy() 

# The "Cold" List: Assets scanned lightly for opportunities
# Top 30 Liquid Pairs + High Volatility Candidates
SCOUT_CANDIDATES = [
    'PEPE/USDT', 'RUNE/USDT', 'INJ/USDT', 'TIA/USDT', 'OP/USDT', 
    'ARB/USDT', 'NEAR/USDT', 'APT/USDT',
    'SAND/USDT', 'MANA/USDT', 'AXS/USDT', 'APE/USDT', 'CRV/USDT',
    '1INCH/USDT', 'DYDX/USDT', 'IMX/USDT', 'STX/USDT', 'LDO/USDT',
    'FIL/USDT', 'ATOM/USDT', 'VET/USDT', 'GRT/USDT', 'SNX/USDT',
    'AAVE/USDT', 'UNI/USDT', 'QNT/USDT', 'THETA/USDT'
]

# Scout Thresholds
SCOUT_RVOL_THRESHOLD = 2.5      # 250% Volume -> Trigger Alert
SCOUT_HYPE_THRESHOLD = 0.5      # Sentiment > 0.5 -> Trigger Alert
SCOUT_CACHE_TTL = 60            # Cache ticker data for 60s
SCOUT_CYCLE_INTERVAL = 60       # Run Scout Loop every 60s

# Asset Personality Thresholds
ANCHOR_ADX_THRESHOLD = 25.0     # Trend Strength for Anchors
ROCKET_RVOL_THRESHOLD = 3.0     # Volume Explosion for Rockets

FORBIDDEN_ASSETS = ['ALGO/USDT', 'SHIB/USDT']  # SHIB has poor precision

# === MACRO STRATEGY LISTS ===
CRISIS_SAFE_HAVENS = ['BTC/USDT', 'PAXG/USDT']
CRISIS_RISK_ASSETS = ['DOGE/USDT', 'ADA/USDT', 'SOL/USDT']

# === STRATEGY SETTINGS (Centralized) ===
STRATEGY_RSI_OVERSOLD = 45
STRATEGY_RSI_OVERBOUGHT = 70
STRATEGY_RSI_PANIC_BUY = 40.0
STRATEGY_RSI_ENTRY_MAX = 70.0       # RECALIBRATION: Increased from 65 (Aggressive)
STRATEGY_LSTM_THRESHOLD = 0.51      # RECALIBRATION: Lowered from 0.52
STRATEGY_XGB_THRESHOLD = 0.48    # Confidence required for XGBoost (Ensemble)
STRATEGY_POST_EXIT_COOLDOWN_CANDLES = 3 # Wait 3 candles before re-entry

# === POLYMARKET EXECUTION LOGIC ===
POLYMARKET_PATIENCE_MINUTES = 3 # Wait first 3 mins of new 15m candle (Noise Filter)
SCALP_SPREAD_PCT = 0.005 # 0.5% Spread for Volatility Trap (Disabled for now)

# === GOVERNANCE / RISK ===
GOVERNOR_COOLDOWN_SECONDS = 60 # 1 Minute Cooldown (Aggressive)
GOVERNOR_MIN_STACK_DIST = 0.002     # RECALIBRATION: Lowered from 0.005
GOVERNOR_MAX_MARGIN_PCT = 0.60      # RECALIBRATION: Increased from 0.40 (Aggressive)
GOVERNOR_STACK_DECAY = 0.8 # Reduce size by 20% each stack
GOVERNOR_MAX_TREND_AGE_HOURS = 24.0 # Trends > 24h are exhausted
GOVERNOR_TREND_DECAY_START = 12.0 # Start reducing risk after 12h

# === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
# Minimax Constraint (Game Theory)
# PRINCIPAL is now defined above to avoid duplication.
MAX_RISK_PCT = 0.02  # RECALIBRATION: Increased from 0.01

# === STRUCTURAL SUPPORT/RESISTANCE ===
SR_LOOKBACK = 50
SR_TOLERANCE = 0.01 # 1% buffer

# Modified Kelly Criterion (Half-Kelly)
KELLY_RISK_REWARD = 2.0  # Adjusted risk/reward for win-rate calculation
KELLY_LOOKBACK = 50  # Number of trades to calculate win rate
KELLY_MIN_FRACTION = 0.05  # Minimum Kelly fraction (floor)
KELLY_MAX_FRACTION = 0.25  # Maximum Kelly fraction (ceiling)

# Volatility Scalar (Inverse Variance Weighting)
VOL_SCALAR_PERIOD = 14  # ATR reference period
VOL_SCALAR_MIN = 0.5  # Minimum scalar (max position reduction)
VOL_SCALAR_MAX = 2.0  # Maximum scalar (max position increase)

# === CROSS-ASSET CORRELATION ===
GMB_THRESHOLD = 0.35  # RECALIBRATION: Lowered from 0.5

# === PHASE 22: PPO SOVEREIGN BRAIN ===
PPO_LEARNING_RATE = 0.0003
PPO_CLIP_RATIO = 0.2
PPO_REWARD_DRAWDOWN_PENALTY = 2.0 # Penalty multiplier for drawdown

# === PHASE 31: CONCURRENCY & RATE LIMITING ===
CCXT_RATE_LIMIT = True
CCXT_POOL_SIZE = 20         # Matches/exceeds TRADER_MAX_WORKERS
TRADER_MAX_WORKERS = 16    # Parallel analysis threads

# === PHASE 33: INTEL GPU ACCELERATION ===
USE_INTEL_GPU = True
USE_OPENVINO = True

# === PHASE 25: SATELLITE COMMANDER (TIER 2) ===
SATELLITE_ASSETS = ['XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'LINK/USDT']
SATELLITE_MARGIN = 10.0      # Fixed $10 Margin per trade
SATELLITE_LEVERAGE = 10.0    # 10x Fixed Leverage (Position = $100)
SATELLITE_RVOL_THRESHOLD = 1.5   # 1.5x Volume required
SATELLITE_DOGE_RVOL_THRESHOLD = 2.0 # 2.0x for DOGE specifically
SATELLITE_BBW_EXPANSION_THRESHOLD = 0.20 # 20% Expansion required
SATELLITE_BREAKEVEN_TRIGGER = 0.015  # +1.5% Move -> Move SL to BE
SATELLITE_TAKE_PROFIT_1 = 0.03       # +3.0% Move -> Close 50%

# === PHASE 35: IMMUNE SYSTEM & PERSONALITY ===
# Asset Families (Cluster Risk)
FAMILY_L1 = ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT']
FAMILY_PAYMENT = ['XRP/USDT', 'LTC/USDT', 'BCH/USDT']
FAMILY_MEME = ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT']

# Health Thresholds
IMMUNE_MAX_DAILY_DRAWDOWN = 0.05     # 5% Daily Loss Limit
IMMUNE_MAX_LEVERAGE_RATIO = 10.0     # Max 10x Total Account Leverage

# Personality Parameters
PERSONALITY_BTC_ATR_FILTER = 0.5     # Ignore if ATR < 50% of 30d Avg
PERSONALITY_SOL_RSI_LONG = 55.0      # Min RSI to Long SOL
PERSONALITY_SOL_RSI_SHORT = 45.0     # Max RSI to Short SOL
PERSONALITY_DOGE_RVOL = 2.0          # Higher RVOL for DOGE

# === PHASE 36: LIQUIDATION ENGINE ===
MAINTENANCE_MARGIN_RATE = 0.50       # Liquidation if Equity < 50% of Used Margin

# === UNIFIED CONTROL PROTOCOL (Phase 6) ===
MICRO_CAPITAL_MODE = True            # Enable for accounts < $500
MICRO_MAX_LEVERAGE = 10.0            # Hard Cap for Micro
MICRO_MAX_EXPOSURE_RATIO = 3.5       # Max 3.5x Equity Total Exposure
MICRO_MAX_EXPOSURE_RATIO = 3.5       # Max 3.5x Equity Total Exposure
MICRO_MAX_POSITIONS = 4              # User Request: Increased to 4

# Stacking Hard Gates (Request A & User Input)
STACKING_MIN_EQUITY = 100.0          # No Stacking below this equity
STACKING_BUFFER_MULTIPLIER = 5.0     # Need 5x MinOrder free margin to stack

# MicroMode Refinements (Phase 6c)
MICRO_HARD_LEVERAGE_LIMIT = 3.0      # Hard Cap for Micro Mode (Variance Control)
MICRO_STARTUP_MAX_POSITIONS = 2      # Max positions to hold on startup consolidation

# === CAPITAL REGIME MODEL (Phase 7) ===
# Regime Thresholds
REGIME_MICRO_CEILING = 49.0          # MICRO: $0 - $49
REGIME_SMALL_CEILING = 249.0         # SMALL: $50 - $249
REGIME_MEDIUM_CEILING = 999.0        # MEDIUM: $250 - $999 (PREDATOR unlocked)

# Promotion Rules (strict)
REGIME_PROMOTION_STABILITY_HOURS = 72  # Must hold threshold for 72h
REGIME_PROMOTION_HEALTH_THRESHOLD = 0.95  # Behavior integrity score
REGIME_PROMOTION_MIN_TRADES = 20       # Minimum trades for health calculation

# Demotion Rules (fast and ruthless)
REGIME_DEMOTION_BUFFER = 5.0          # Promote at $50, demote at $45
REGIME_DEMOTION_DRAWDOWN_PCT = 0.25   # 25% drawdown from peak = instant demotion

# Regime-Specific Permissions
REGIME_PERMISSIONS = {
    'MICRO': {
        'max_positions': 4,
        'max_stacks': 0,
        'max_exposure_ratio': 3.0,
        'max_leverage': 3.0,
        'allowed_pairs': ['BTC/USDT', 'ETH/USDT', 'XRP/USDT'],
        'correlation_check': False,  # Too risky with 2 positions
    },
    'SMALL': {
        'max_positions': 4,
        'max_stacks': 1,
        'max_exposure_ratio': 5.0,
        'max_leverage': 5.0,
        'allowed_pairs': ALLOWED_ASSETS,  # All Tier-1
        'correlation_check': True,
    },
    'MEDIUM': {
        'max_positions': 8,
        'max_stacks': 2,
        'max_exposure_ratio': 8.0,
        'max_leverage': 10.0,
        'allowed_pairs': ALLOWED_ASSETS,
        'correlation_check': True,
    },
}

# Consolidation Scoring Weights
CONSOLIDATION_WEIGHT_PNL = 0.30
CONSOLIDATION_WEIGHT_CONVICTION = 0.25
CONSOLIDATION_WEIGHT_LIQUIDITY = 0.15
CONSOLIDATION_WEIGHT_AGE = 0.10
CONSOLIDATION_WEIGHT_CORRELATION = 0.20  # Subtracts from score

# Consolidation Thresholds
CONSOLIDATION_DUST_THRESHOLD = 1.0     # Force Close if < $1.0 Notional
CONSOLIDATION_STALE_HOURS = 24.0      # Force Close if > 24h & Negative PnL

# Strict Directives (User Rules)
CONSOLIDATION_MARGIN_BUFFER = 1.5     # 1.5x Required Margin
CONSOLIDATION_HARD_BUFFER = 1.2       # Hard mode trigger below 1.2x
CONVICTION_DECAY_BASE_HOURS = 48.0    # Base lifespan of conviction
CONVICTION_DECAY_CAPITAL_MULTIPLIER = 5.0 # Acceleration for heavy positions

# Consolidation Hard Rules
CONSOLIDATION_DUST_THRESHOLD = 1.0    # Close if notional < $1
CONSOLIDATION_STALE_HOURS = 24        # Close if no favorable move in 24h

# === PHASE 38: SENTIMENT & NEWS ===
SENTIMENT_SOURCES = [
    'https://cointelegraph.com/rss',
    'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'https://cryptopanic.com/news/rss/',
    'https://beincrypto.com/feed/',
    # === MACRO & RETAIL SOURCES ===
    'http://feeds.reuters.com/reuters/businessNews', # Macro/War
    'https://www.cnbc.com/id/100727362/device/rss/rss.html', # Global Markets
    'https://www.investing.com/rss/commodities.rss', # Oil/Gold
    'https://www.reddit.com/r/CryptoCurrency/top/.rss?t=hour' # Retail Pulse
]
SENTIMENT_WEIGHT = 0.3 # 30% of Market Bias comes from News
SENTIMENT_THRESHOLD_BULL = 0.2
SENTIMENT_THRESHOLD_BEAR = -0.2

# === PHASE 39: MARKET PHYSICS (VALIDATION LAYER) ===
PHYSICS_CORRELATION_THRESHOLD = 0.75  # Veto Longs if Corr > 0.75 & BTC Bearish
PHYSICS_MIN_RVOL = 1.1                # Veto Breakouts if Volume < 1.1x Avg
PHYSICS_MAX_ENTROPY = 1.35            # Veto All Signals if Entropy > 1.35 (Chaos)

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

# === PHASE 42: ACCUMULATOR (CAPITAL CONTROL) ===
ACC_RISK_FLOOR = 0.5    # Minimum Risk Multiplier (Defense)
ACC_RISK_CEILING = 2.0  # Maximum Risk Multiplier (Offense)
ACC_DRAWDOWN_LIMIT = 0.25 # 25% Drawdown from High Water Mark halts trading

# === PHASE 43: GARBAGE COLLECTOR MONITOR ===
GC_INTERVAL_CYCLES = 2          # Run GC every 2 cycles (More aggressive)
GC_STALE_ORDER_TIMEOUT = 180    # Orders older than 3 minutes are stale
GC_LOG_VERBOSE = True           # Log all GC actions
