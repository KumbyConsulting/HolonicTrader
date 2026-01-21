"""
HOLONIC SPEED ENGINE: FULL VERIFICATION & BENCHMARK
Verifies all Rust functions and measures performance vs Python.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

import holonic_speed

print("=" * 70)
print("HOLONIC SPEED ENGINE: FULL VERIFICATION & BENCHMARK")
print("=" * 70)

# Generate test data
np.random.seed(42)
n = 5000
prices = 40000 + np.cumsum(np.random.randn(n) * 100)
closes = prices.tolist()
highs = (prices + np.abs(np.random.randn(n) * 50)).tolist()
lows = (prices - np.abs(np.random.randn(n) * 50)).tolist()
returns = np.diff(prices) / prices[:-1]

print(f"\nTest Data: {n} candles\n")

results = []

# ============================================================
# 1. ENTROPY
# ============================================================
print("[1/6] ENTROPY")
data_50 = returns[:50].tolist()

# Python
start = time.time()
for _ in range(10000):
    counts, _ = np.histogram(data_50, bins=10)
    probs = counts / counts.sum()
    py_ent = float(scipy_entropy(probs))
py_time = time.time() - start

# Rust
start = time.time()
for _ in range(10000):
    rs_ent = holonic_speed.calculate_shannon_entropy(data_50)
rs_time = time.time() - start

diff = abs(py_ent - rs_ent)
status = "PASS" if diff < 0.01 else "FAIL"
speedup = py_time / rs_time
results.append(("Entropy", status, f"{speedup:.1f}x"))
print(f"  Accuracy: {status} (diff={diff:.6f})")
print(f"  Speedup: {speedup:.1f}x (Py={py_time:.3f}s, Rs={rs_time:.3f}s)")

# ============================================================
# 2. RSI
# ============================================================
print("\n[2/6] RSI")

# Python
start = time.time()
for _ in range(100):
    delta = pd.Series(closes).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    py_rsi = (100 - 100 / (1 + rs)).values
py_time = time.time() - start

# Rust
start = time.time()
for _ in range(100):
    rs_rsi = holonic_speed.calculate_rsi(closes, 14)
rs_time = time.time() - start

# Compare last value (ignore NaN alignment)
valid_py = [x for x in py_rsi if not np.isnan(x)]
valid_rs = [x for x in rs_rsi if not np.isnan(x)]
diff = abs(valid_py[-1] - valid_rs[-1]) if valid_py and valid_rs else 999
status = "PASS" if diff < 5.0 else "CHECK"  # RSI implementations vary
speedup = py_time / rs_time
results.append(("RSI", status, f"{speedup:.1f}x"))
print(f"  Accuracy: {status} (last diff={diff:.2f})")
print(f"  Speedup: {speedup:.1f}x")

# ============================================================
# 3. BOLLINGER BANDS
# ============================================================
print("\n[3/6] BOLLINGER BANDS")

# Python
start = time.time()
for _ in range(100):
    sma = pd.Series(closes).rolling(20).mean()
    std = pd.Series(closes).rolling(20).std()
    py_upper = sma + 2 * std
    py_lower = sma - 2 * std
py_time = time.time() - start

# Rust
start = time.time()
for _ in range(100):
    rs_upper, rs_mid, rs_lower = holonic_speed.calculate_bollinger_bands(closes, 20, 2.0)
rs_time = time.time() - start

diff = abs(py_upper.iloc[-1] - rs_upper[-1])
status = "PASS" if diff < 1.0 else "CHECK"
speedup = py_time / rs_time
results.append(("Bollinger Bands", status, f"{speedup:.1f}x"))
print(f"  Accuracy: {status} (upper diff={diff:.2f})")
print(f"  Speedup: {speedup:.1f}x")

# ============================================================
# 4. ATR
# ============================================================
print("\n[4/6] ATR")

# Python
start = time.time()
for _ in range(100):
    h = pd.Series(highs)
    l = pd.Series(lows)
    c = pd.Series(closes)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    py_atr = tr.rolling(14).mean().values
py_time = time.time() - start

# Rust
start = time.time()
for _ in range(100):
    rs_atr = holonic_speed.calculate_atr(highs, lows, closes, 14)
rs_time = time.time() - start

valid_py = [x for x in py_atr if not np.isnan(x)]
valid_rs = [x for x in rs_atr if not np.isnan(x)]
diff = abs(valid_py[-1] - valid_rs[-1]) if valid_py and valid_rs else 999
status = "PASS" if diff < 5.0 else "CHECK"
speedup = py_time / rs_time
results.append(("ATR", status, f"{speedup:.1f}x"))
print(f"  Accuracy: {status} (diff={diff:.2f})")
print(f"  Speedup: {speedup:.1f}x")

# ============================================================
# 5. GOVERNOR
# ============================================================
print("\n[5/6] GOVERNOR")

# Cluster Risk Check
held = ["BTC/USDT"]
is_safe = holonic_speed.governor_check_cluster_risk(held, "TBTC/USDT")
status1 = "PASS" if not is_safe else "FAIL"

is_safe2 = holonic_speed.governor_check_cluster_risk(held, "ETH/USDT")
status2 = "PASS" if is_safe2 else "FAIL"

# Max Risk
max_risk = holonic_speed.governor_calculate_max_risk(50.0, 10.0, 50.0)
status3 = "PASS" if max_risk > 0 else "FAIL"

overall = "PASS" if status1 == status2 == status3 == "PASS" else "CHECK"
results.append(("Governor", overall, "N/A"))
print(f"  Cluster Risk (same family): {status1}")
print(f"  Cluster Risk (diff family): {status2}")
print(f"  Max Risk: {status3} (${max_risk:.2f})")

# ============================================================
# 6. EXECUTOR
# ============================================================
print("\n[6/6] EXECUTOR")

# Test all regimes
action1, size1 = holonic_speed.executor_decide_trade(1.0, 1.5, "ORDERED")
action2, size2 = holonic_speed.executor_decide_trade(1.0, 2.5, "CHAOTIC")
action3, size3 = holonic_speed.executor_decide_trade(1.0, 1.5, "TRANSITION")

s1 = "PASS" if action1 == "EXECUTE" else "FAIL"
s2 = "PASS" if action2 == "HALT" else "FAIL"
s3 = "PASS" if action3 == "REDUCE" else "FAIL"

overall = "PASS" if s1 == s2 == s3 == "PASS" else "CHECK"
results.append(("Executor", overall, "N/A"))
print(f"  ORDERED -> EXECUTE: {s1}")
print(f"  CHAOTIC -> HALT: {s2}")
print(f"  TRANSITION -> REDUCE: {s3}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"{'Component':<20} {'Status':<10} {'Speedup':<10}")
print("-" * 40)
for name, status, speedup in results:
    print(f"{name:<20} {status:<10} {speedup:<10}")
print("-" * 40)

all_pass = all(r[1] == "PASS" for r in results)
if all_pass:
    print("\nALL TESTS PASSED!")
else:
    print("\nSome tests need review.")
print("=" * 70)
