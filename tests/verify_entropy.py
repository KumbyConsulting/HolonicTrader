
import holonic_speed
import numpy as np
import time
from scipy.stats import entropy as scipy_entropy

print("VERIFYING RUST ENTROPY ENGINE")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic returns data
data = np.random.normal(0, 0.01, 50).tolist()

print("[1/2] Accuracy Check (Shannon)")

def python_shannon(returns_series):
    counts, _ = np.histogram(returns_series, bins=10)
    total_count = counts.sum()
    if total_count == 0: return 0.0
    probabilities = counts / total_count
    return float(scipy_entropy(probabilities)) # Base e

# 1. Python Result
py_val = python_shannon(data)
print(f"  Python (Scipy): {py_val:.6f}")

# 2. Rust Result
rs_val = holonic_speed.calculate_shannon_entropy(data)
print(f"  Rust:           {rs_val:.6f}")

diff = abs(py_val - rs_val)
if diff < 1e-6:
    print("  Accuracy: MATCH")
else:
    print(f"  Accuracy: MISMATCH (Diff: {diff})")

    
print("\n[2/2] Speed Benchmark (100,000 ops)")
iterations = 100_000

start = time.time()
for _ in range(iterations):
    python_shannon(data)
dt_py = time.time() - start

start = time.time()
for _ in range(iterations):
    holonic_speed.calculate_shannon_entropy(data)
dt_rs = time.time() - start

print(f"  Python: {dt_py:.4f}s")
print(f"  Rust:   {dt_rs:.4f}s")
if dt_rs > 0:
    print(f"  Speedup: {dt_py / dt_rs:.2f}x")
else:
    print("  Speedup: Infinite (Rust took 0s)")

print("\n" + "="*60)
