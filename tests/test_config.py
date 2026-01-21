import config
import sys

required_attrs = [
    'ACTIVE_WATCHLIST',
    'SCAVENGER_THRESHOLD',
    'MEMECOIN_ASSETS',
    'SATELLITE_ASSETS',
    'VOL_WINDOW_MAX_POSITIONS',
    'ACC_DRAWDOWN_LIMIT',
    'MICRO_MAX_LEVERAGE',
    'IMMUNE_MAX_LEVERAGE_RATIO',
    'VOL_WINDOW_SPREAD_THRESHOLD',
    'MIN_ORDER_VALUE',
    'REGIME_PERMISSIONS'
]

missing = []

print("Running Config Integrity Check...")
for attr in required_attrs:
    if not hasattr(config, attr):
        missing.append(attr)
        print(f"❌ MISSING: {attr}")
    else:
        print(f"✅ FOUND: {attr}")

if missing:
    print(f"\nCRITICAL: {len(missing)} attributes missing from config.py!")
    sys.exit(1)
else:
    print("\n✅ Config Integrity Check PASSED. All critical attributes present.")
    sys.exit(0)
