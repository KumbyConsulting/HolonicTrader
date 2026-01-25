
import time
import pandas as pd
import numpy as np
import holonic_speed
from HolonicTrader.agent_entropy import EntropyHolon

def benchmark_tier3():
    print("--- 🦀 Rust Signal Normalization Benchmark (Tier 3) ---")
    
    # Generate mock data for 15 assets
    num_assets = 15
    history_len = 100
    symbols = [f"ASSET_{i}" for i in range(num_assets)]
    data_map = {
        s: {
            'close': np.random.normal(100, 5, history_len).tolist(),
            'high': np.random.normal(105, 5, history_len).tolist(),
            'low': np.random.normal(95, 5, history_len).tolist()
        } for s in symbols
    }
    
    # --- RUST BATCH ANALYSIS ---
    print(f"Executing Rust Batch Analysis for {num_assets} assets...")
    start_rust = time.perf_counter()
    rust_results = holonic_speed.calculate_signals_matrix(
        symbols,
        {s: d['close'] for s, d in data_map.items()},
        {s: d['high'] for s, d in data_map.items()},
        {s: d['low'] for s, d in data_map.items()}
    )
    end_rust = time.perf_counter()
    duration_rust = end_rust - start_rust
    print(f"✅ Rust Batch Total Time: {duration_rust*1000:.2f}ms")
    
    # Sample check
    sample_sym = symbols[0]
    res = rust_results.get(sample_sym, {})
    print(f"\nSample Signals for {sample_sym}:")
    for k, v in res.items():
        print(f"  - {k}: {v:.4f}")
    
    # --- PYTHON SEQUENTIAL (Simulated Legacy) ---
    print(f"\nExecuting Python Sequential Analysis (Simulation)...")
    entropy_agent = EntropyHolon()
    start_py = time.perf_counter()
    
    for s in symbols:
        p = np.array(data_map[s]['close'])
        returns = np.diff(p) / p[:-1]
        entropy_val = entropy_agent.calculate_shannon_entropy(pd.Series(returns))
        
        # Simple moving average for RSI/ATR simulation
        # In reality, pandas-ta is used, but we'll just sum for cost simulation
        _sum = sum(p[-14:]) 
        _tr = max(data_map[s]['high'][-1] - data_map[s]['low'][-1], 0.1)
        
    end_py = time.perf_counter()
    duration_py = end_py - start_py
    print(f"✅ Python Sequential Time: {duration_py*1000:.2f}ms")
    
    print(f"\n🚀 TIER 3 SPEEDUP: {duration_py / duration_rust:.1f}x")

if __name__ == "__main__":
    benchmark_tier3()
