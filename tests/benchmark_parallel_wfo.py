"""
Parallel WFO Benchmark: Measures speedup from Rayon parallelization.
"""

import time
import numpy as np
import holonic_speed

print("=" * 70)
print("PARALLEL WALK-FORWARD OPTIMIZATION BENCHMARK")
print("=" * 70)

# Generate synthetic data (2 years of hourly data)
print("\n[1/3] Generating synthetic data...")
np.random.seed(42)
n = 17520  # 2 years of hourly data
prices = 40000 + np.cumsum(np.random.randn(n) * 100)

timestamps = list(range(n))
opens = prices.tolist()
highs = (prices + np.abs(np.random.randn(n) * 50)).tolist()
lows = (prices - np.abs(np.random.randn(n) * 50)).tolist()
closes = (prices + np.random.randn(n) * 30).tolist()

# Compute indicators using Rust
rsi = holonic_speed.calculate_rsi(closes, 14)
bb_upper, bb_mid, bb_lower = holonic_speed.calculate_bollinger_bands(closes, 20, 2.0)
obv_slopes = [0.0] * n  # Simplified
entropy_scores = [1.5] * n  # Neutral entropy

print(f"  Data points: {n}")

# Define WFO windows (12 windows, each 1 month train + 1 week test)
print("\n[2/3] Setting up Walk-Forward Windows...")
window_configs = []
train_size = 720  # 1 month
test_size = 168   # 1 week
step_size = 168   # Roll forward 1 week

num_windows = 12
for i in range(num_windows):
    start = i * step_size
    train_end = start + train_size
    test_end = train_end + test_size
    
    if test_end <= n:
        window_configs.append((start, train_end, test_end))

print(f"  Windows: {len(window_configs)}")

# Run Sequential (Python loop)
print("\n[3/3] Running Benchmarks...")

# Sequential: Run each window one by one
start_seq = time.time()
seq_results = []
for idx, (s, te, e) in enumerate(window_configs):
    # Simulate running backtest on slice
    result = holonic_speed.run_backtest_fast(
        timestamps[te:e], opens[te:e], highs[te:e], lows[te:e], closes[te:e],
        [0] * (e - te),  # No signals for timing test
        1000.0, 0.001
    )
    seq_results.append(result)
time_seq = time.time() - start_seq

# Parallel: Use Rust Rayon
start_par = time.time()
par_results = holonic_speed.run_parallel_wfo(
    window_configs,
    timestamps, opens, highs, lows, closes,
    rsi, bb_lower, bb_upper, obv_slopes, entropy_scores,
    1000.0
)
time_par = time.time() - start_par

print("-" * 50)
print(f"  Sequential: {time_seq:.4f}s")
print(f"  Parallel:   {time_par:.4f}s")
print(f"  Speedup:    {time_seq / time_par:.2f}x")
print("-" * 50)

# Report Results
print("\n  Window Results (Parallel):")
for r in par_results[:5]:  # Show first 5
    window_id, train_pnl, test_pnl, final_bal = r
    print(f"    Window {window_id}: Test PnL {test_pnl:.2f}%, Final ${final_bal:.2f}")

if len(par_results) > 5:
    print(f"    ... and {len(par_results) - 5} more windows")

print("=" * 70)
