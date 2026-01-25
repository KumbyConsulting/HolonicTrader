
import sys
import os

print("--- DEPENDENCY CHECK ---")

# 1. Sentiment Deps
try:
    import feedparser
    print("feedparser: INSTALLED")
except ImportError:
    print("feedparser: MISSING")

try:
    from textblob import TextBlob
    print("textblob: INSTALLED")
except ImportError:
    print("textblob: MISSING")

# 2. Evolution Bridge
# Sandbox should be at ../sandbox relative to this script if script is in HolonicTrader/HolonicTrader
# Actually, based on file list:
# HolonicTrader/HolonicTrader/
#   check_deps.py
#   sandbox/
#     strategies/
#       ensemble.py
# So sandbox is in CURRENT dir?
# Let's check listing again:
# {"name":"sandbox","isDir":true,"numChildren":10}
# Yes, sandbox is in c:\Users\USER\Documents\AEHML\HolonicTrader\HolonicTrader\sandbox
# So import should be `from sandbox.strategies.ensemble import EnsembleStrategy` directly?

try:
    # Add current dir to path just in case
    sys.path.append(os.getcwd())
    from sandbox.strategies.ensemble import EnsembleStrategy
    print("EnsembleStrategy: IMPORT_OK")
except Exception as e:
    print(f"EnsembleStrategy: FAILED ({e})")
    # specific path debug
    print(f"CWD: {os.getcwd()}")
    if os.path.exists('sandbox'):
        print("sandbox dir: FOUND")
        if os.path.exists('sandbox/strategies'):
             print("sandbox/strategies dir: FOUND")
             if os.path.exists('sandbox/strategies/ensemble.py'):
                 print("sandbox/strategies/ensemble.py: FOUND")
    else:
        print("sandbox dir: NOT FOUND")

print("--- END CHECK ---")
