
import holonic_speed
import numpy as np
import pandas as pd

print("VERIFYING RUST INDICATORS")
print("=" * 60)

# Generate Synthetic Data
# Simple Uptrend
data_len = 50
closes = np.linspace(100, 150, data_len).tolist()
highs = [c + 1 for c in closes]
lows = [c - 1 for c in closes]
data_s = pd.Series(closes)

print("[1/3] RSI Check (Period=14)")
# Pandas/Manual Calculation for RSI
def py_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean() # Simple MA for first? 
    # Wilder's Smoothing usually starts with SMA then EMA
    # Let's compare against Rust output logic which uses Wilder's
    return 0.0 # Placeholder, hard to match exactly without library

# Rust
rsi_rust = holonic_speed.calculate_rsi(closes, 14)
print(f"  Rust RSI[-1]: {rsi_rust[-1]:.2f}")
# Simple check: Uptrend should have high RSI
if rsi_rust[-1] > 70:
    print("  sanity Check: PASS (High RSI in Uptrend)")
else:
    print("  sanity Check: FAIL")

print("\n[2/3] Bollinger Bands (20, 2.0)")
upper, mid, lower = holonic_speed.calculate_bollinger_bands(closes, 20, 2.0)
print(f"  Rust BB[-1]: U={upper[-1]:.2f}, M={mid[-1]:.2f}, L={lower[-1]:.2f}")

# Manual verification of last point
# SMA of last 20
slice_20 = closes[-20:]
mu = np.mean(slice_20)
sigma = np.std(slice_20) # Population or Sample? Rust impl uses Population (N) usually or N-1?
# Rust impl used mean, and loop.
# Let's check consistency.
# Rust impl: variance_sum / period. This is Population variance (N).
# Numpy std defaults to Population (ddof=0).
expected_u = mu + 2*sigma
expected_l = mu - 2*sigma

print(f"  Expected:    U={expected_u:.2f}, M={mu:.2f}, L={expected_l:.2f}")

if abs(upper[-1] - expected_u) < 0.1:
    print("  Accuracy: MATCH")
else:
    print(f"  Accuracy: MISMATCH (Diff: {upper[-1] - expected_u})")

print("\n[3/3] ATR (14)")
atr_rust = holonic_speed.calculate_atr(highs, lows, closes, 14)
print(f"  Rust ATR[-1]: {atr_rust[-1]:.4f}")
# Since H-L is constant 2.0, ATR should be 2.0
if abs(atr_rust[-1] - 2.0) < 0.01:
    print("  Accuracy: MATCH (Constant Range)")
else:
    print(f"  Accuracy: MISMATCH (Expected 2.0, Got {atr_rust[-1]})")

print("\n" + "="*60)
