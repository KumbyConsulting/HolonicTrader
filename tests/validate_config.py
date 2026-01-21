
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
    print("✅ Config Imported Successfully.")
except ImportError as e:
    print(f"❌ Failed to import config: {e}")
    sys.exit(1)

# List of all config variables used in agent_governor.py (extracted from analysis)
required_vars = [
    'IRON_BANK_ENABLED',
    'IRON_BANK_MIN_RESERVE',
    'IRON_BANK_BUFFER_PCT',
    'IRON_BANK_RATCHET_PCT',
    'ACC_DRAWDOWN_LIMIT',
    'ACC_RISK_CEILING',
    'ACC_RISK_FLOOR',
    'ACC_SANITY_THRESHOLD',
    'GOVERNOR_MAX_MARGIN_PCT',
    'PREDATOR_LEVERAGE',
    'GOVERNOR_COOLDOWN_SECONDS',
    'GOVERNOR_MIN_STACK_DIST',
    'MIN_ORDER_VALUE',
    'SATELLITE_ASSETS',
    'SATELLITE_MARGIN',
    'SATELLITE_LEVERAGE',
    'VOL_WINDOW_MAX_POSITIONS',
    'VOL_WINDOW_RISK_PCT',
    'VOL_WINDOW_LEVERAGE',
    'MICRO_CAPITAL_MODE',
    'REGIME_PERMISSIONS',
    'MICRO_MAX_POSITIONS',
    'MICRO_MAX_EXPOSURE_RATIO',
    'STACKING_MIN_EQUITY',
    'STACKING_BUFFER_MULTIPLIER',
    'CORRELATION_CHECK',
    'SENTIMENT_THRESHOLD_BULL',
    'SCAVENGER_MAX_MARGIN',
    'SCAVENGER_LEVERAGE',
    'SCAVENGER_STOP_LOSS',
    'PREDATOR_STOP_LOSS',
    'GOVERNOR_TREND_DECAY_START',
    'GOVERNOR_MAX_TREND_AGE_HOURS',
    'GOVERNOR_STACK_DECAY',
    'MICRO_HARD_LEVERAGE_LIMIT',
    'VOL_SCALAR_MIN',
    'VOL_SCALAR_MAX',
    'MIN_TRADE_QTY',
    'PRINCIPAL',
    'KELLY_LOOKBACK', # The one that crashed earlier
    'NANO_CAPITAL_THRESHOLD' # Used in satellite override
]

missing = []
for var in required_vars:
    if not hasattr(config, var):
        # Check if it has a default via getattr in code? 
        # But we want to ensure it exists to be safe.
        missing.append(var)
    else:
        # Optional: Print value
        pass

if missing:
    print("❌ MISSING CONFIG VARIABLES:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)
else:
    print("✅ All Governor Config Dependencies Verified.")
