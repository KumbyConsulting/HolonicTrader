import time
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.sde_engine import SDEEngine, RUST_AVAILABLE

def benchmark_sde():
    print(f"--- SDE Engine Benchmark (Rust Available: {RUST_AVAILABLE}) ---")
    
    # 1. SETUP DATA
    n_points = 1000
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, n_points))) + 100
    
    # 2. BENCHMARK ESTIMATION (OU)
    print(f"\n[1] PARAMETER ESTIMATION (n={n_points})")
    
    # Python Path (Force fallback by mocking RUST_AVAILABLE=False temporarily if needed, 
    # but easier to just measure current state since it uses Rust)
    # We'll just time it.
    
    start = time.perf_counter()
    for _ in range(100):
        params = SDEEngine.estimate_ou_parameters(prices)
    end = time.perf_counter()
    print(f"OU estimation (100 runs): {end - start:.4f}s")
    
    # 3. BENCHMARK SIMULATION (THE REAL TEST)
    print("\n[2] MONTE CARLO SIMULATION (Paths=1000, Horizon=100)")
    
    # We want to compare Python vs Rust directly.
    # We can do this by manually calling the logic or by temporarily toggling the flag.
    
    import HolonicTrader.sde_engine as sde_mod
    
    # RUST PATH
    sde_mod.RUST_AVAILABLE = True
    start = time.perf_counter()
    rust_paths = SDEEngine.simulate_paths('GBM', {'drift': 0.1, 'diffusion': 0.2}, 100.0, 100, 1000)
    end = time.perf_counter()
    rust_time = end - start
    print(f"Rust simulation:   {rust_time:.6f}s")
    
    # PYTHON PATH
    sde_mod.RUST_AVAILABLE = False
    start = time.perf_counter()
    py_paths = SDEEngine.simulate_paths('GBM', {'drift': 0.1, 'diffusion': 0.2}, 100.0, 100, 1000)
    end = time.perf_counter()
    py_time = end - start
    print(f"Python simulation: {py_time:.6f}s")
    
    print(f"\nSPEEDUP: {py_time / rust_time:.1f}x")
    
    # 5. BENCHMARK RUIN GUARD (THE OPTIMIZED CASE)
    print("\n[3] RUIN GUARD CALCULATION (Paths=1000, Horizon=100)")
    
    params = {'mu': 0.05, 'sigma': 0.2, 'lambda': 0.1}
    sl = 95.0
    tp = 110.0
    
    # RUST
    sde_mod.RUST_AVAILABLE = True
    start = time.perf_counter()
    rust_ruin = SDEEngine.calculate_ruin_probability('GBM', params, 100.0, sl, tp, 100, 1000)
    end = time.perf_counter()
    rust_ruin_time = end - start
    print(f"Rust Ruin Calc:   {rust_ruin_time:.6f}s")
    
    # PYTHON
    sde_mod.RUST_AVAILABLE = False
    start = time.perf_counter()
    py_ruin = SDEEngine.calculate_ruin_probability('GBM', params, 100.0, sl, tp, 100, 1000)
    end = time.perf_counter()
    py_ruin_time = end - start
    print(f"Python Ruin Calc: {py_ruin_time:.6f}s")
    
    print(f"RUIN SPEEDUP: {py_ruin_time / rust_ruin_time:.1f}x")

    # 6. PARITY CHECK
    print("\n[4] PARITY CHECK")
    # Means should be roughly similar (it's stochastic but should converge)
    rust_mean = np.mean(rust_paths[:, -1])
    py_mean = np.mean(py_paths[:, -1])
    print(f"Rust Final Mean:   {rust_mean:.4f}")
    print(f"Python Final Mean: {py_mean:.4f}")
    
    diff = abs(rust_mean - py_mean) / py_mean
    if diff < 0.05:
        print("VERIFIED: Statistical Parity (within 5% tolerance for stochastic simulation)")
    else:
        print("WARNING: Significant discrepancy in means.")

if __name__ == "__main__":
    benchmark_sde()
