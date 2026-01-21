
import sys
import os
import time

# Simulate Dashboard environment
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

try:
    from run_backtest import run_backtest
    print("SUCCESS: run_backtest imported correctly.")
except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"ERROR: {e}")
