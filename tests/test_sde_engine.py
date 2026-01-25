import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.sde_engine import SDEEngine

def test_ou_estimation():
    print("\n--- Testing OU Parameter Estimation ---")
    
    # 1. Generate Synthetic OU Path
    # dX = lambda * (mu - X) * dt + sigma * dW
    true_lambda = 50.0
    true_mu = 1.2
    true_sigma = 0.2
    dt = 1/35040 # 15 min
    steps = 50000
    
    x = np.zeros(steps)
    x[0] = true_mu # Start at mean
    
    for t in range(1, steps):
        dw = np.random.normal(0, np.sqrt(dt))
        x[t] = x[t-1] + true_lambda * (true_mu - x[t-1]) * dt + true_sigma * dw
        
    # 2. Estimate
    params = SDEEngine.estimate_ou_parameters(x, dt=dt)
    
    # 3. Verify
    print(f"Results: {params}")
    print(f"True Lambda: {true_lambda:.2f} | Est: {params['lambda']:.2f}")
    print(f"True Mu:     {true_mu:.2f} | Est: {params['mu']:.2f}")
    print(f"True Sigma:  {true_sigma:.2f} | Est: {params['sigma']:.2f}")
    print(f"Half-Life:  {params['half_life']:.4f} years")

    # Loose tolerance due to stochastic nature and finite sample
    assert abs(params['mu'] - true_mu) < 0.2
    assert abs(params['lambda'] - true_lambda) < 20.0
    assert abs(params['sigma'] - true_sigma) < 0.1
    print("DONE: OU Estimation Test Passed.")

def test_gbm_estimation():
    print("\n--- Testing GBM Parameter Estimation ---")
    
    # 1. Generate Synthetic GBM Path
    # dS = mu * S * dt + sigma * S * dW
    true_drift = 0.15 # 15% annual drift
    true_vol = 0.8 # 80% annual vol
    dt = 1/35040
    steps = 2000
    
    s = np.zeros(steps)
    s[0] = 100.0
    
    for t in range(1, steps):
        dw = np.random.normal(0, np.sqrt(dt))
        # Exact solution for simulation
        s[t] = s[t-1] * np.exp((true_drift - 0.5 * true_vol**2) * dt + true_vol * dw)
        
    # 2. Estimate
    params = SDEEngine.estimate_gbm_parameters(s, dt=dt)
    
    # 3. Verify
    print(f"Results: {params}")
    print(f"True Drift: {true_drift:.2f} | Est: {params['drift']:.2f}")
    print(f"True Vol:   {true_vol:.2f} | Est: {params['diffusion']:.2f}")

    # Drift is notoriously hard to estimate in short windows - relax tolerance
    assert abs(params['diffusion'] - true_vol) < 0.2
    assert abs(params['drift'] - true_drift) < 2.0
    print("DONE: GBM Estimation Test Passed.")

if __name__ == "__main__":
    test_ou_estimation()
    test_gbm_estimation()
