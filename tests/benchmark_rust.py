
import time
import numpy as np
import pandas as pd
import holonic_speed

print("🚀 BENCHMARK: Python vs Rust (Holonic Traders)")
print("=" * 60)

# --- 1. CPU STRESS TEST (Fibonacci) ---
n = 30
print(f"\n[1/3] Fibonacci({n})")

start = time.time()
def py_fib(n):
    if n <= 1: return n
    return py_fib(n-1) + py_fib(n-2)
res_py = py_fib(n)
dt_py = time.time() - start
print(f"  🐍 Python: {dt_py:.6f}s (Result: {res_py})")

start = time.time()
res_rs = holonic_speed.fibonacci(n)
dt_rs = time.time() - start
print(f"  🦀 Rust:   {dt_rs:.6f}s (Result: {res_rs})")

print(f"  ⚡ Speedup: {dt_py / dt_rs:.2f}x")


# --- 2. DATA PROCESSING (Moving Average) ---
size = 1_000_000
window = 50
print(f"\n[2/3] SMA (Window={window}, Data={size:,})")

data = np.random.rand(size).tolist()

start = time.time()
# Naive Python Loop
sma_py = []
for i in range(len(data)):
    if i < window - 1:
        sma_py.append(np.nan)
    else:
        window_slice = data[i-window+1 : i+1]
        sma_py.append(sum(window_slice) / window)
dt_py = time.time() - start
print(f"  🐍 Python (Loc): {dt_py:.6f}s")

start = time.time()
# Pandas (Optimized C)
s = pd.Series(data)
sma_pd = s.rolling(window).mean()
dt_pd = time.time() - start
print(f"  🐼 Pandas (C):   {dt_pd:.6f}s")

start = time.time()
sma_rs = holonic_speed.calculate_sma(data, window)
dt_rs = time.time() - start
print(f"  🦀 Rust:         {dt_rs:.6f}s")

print(f"  ⚡ Rust vs Py:   {dt_py / dt_rs:.2f}x")
print(f"  ⚡ Rust vs Pd:   {dt_pd / dt_rs:.2f}x")


# --- 3. BACKTEST SIMULATION (Loop) ---
print(f"\n[3/3] Backtest Execution (Loops={size:,})")
prices = np.random.uniform(100, 200, size).tolist()
signals = np.random.choice([-1, 0, 1], size).astype(int).tolist() # -1 Sell, 0 Hold, 1 Buy

start = time.time()
# Python Backtest Loop
balance = 1000.0
position = 0.0
trades = 0
for i in range(len(prices)):
    price = prices[i]
    signal = signals[i]
    
    if signal == 1 and position == 0.0:
        position = balance / price
        balance = 0.0
        trades += 1
    elif signal == -1 and position > 0.0:
        balance = position * price
        position = 0.0
        
if position > 0.0:
    balance = position * prices[-1]
    
dt_py = time.time() - start
print(f"  🐍 Python: {dt_py:.6f}s (Bal: ${balance:.2f}, Trades: {trades})")

start = time.time()
balance_rs, trades_rs = holonic_speed.fast_backtest_poc(prices, signals)
dt_rs = time.time() - start
print(f"  🦀 Rust:   {dt_rs:.6f}s (Bal: ${balance_rs:.2f}, Trades: {trades_rs})")

print(f"  ⚡ Speedup: {dt_py / dt_rs:.2f}x")

print("\n" + "="*60)
