import holonic_speed
import numpy as np
import time

def test_aehml_2_0():
    print("Testing AEHML 2.0 Rust Integration...")
    
    # 1. Generate Synthetic Data
    # White Noise (High Entropy)
    noise = np.random.normal(0, 1, 1000).tolist()
    # Sine Wave (Low Entropy)
    t = np.linspace(0, 10 * np.pi, 1000)
    sine = np.sin(t).tolist()
    
    # 2. Test Permutation Entropy
    pe_noise = holonic_speed.calculate_permutation_entropy(noise, 3, 1)
    pe_sine = holonic_speed.calculate_permutation_entropy(sine, 3, 1)
    
    print(f"Permutation Entropy (Noise): {pe_noise:.4f} (Expected ~1.0)")
    print(f"Permutation Entropy (Sine):  {pe_sine:.4f} (Expected < 0.5)")
    
    assert pe_noise > pe_sine, "Noise should have higher PE than Sine"
    
    # 3. Test RCMWPE (Multiscale)
    scales = 5
    mse_noise = holonic_speed.calculate_multiscale_entropy(noise, scales, 2)
    mse_sine = holonic_speed.calculate_multiscale_entropy(sine, scales, 2)
    
    print(f"MSE (Noise): {mse_noise}")
    print(f"MSE (Sine):  {mse_sine}")
    
    # Check Scale 1 vs Scale 5
    # For white noise, entropy usually decreases slightly or stays high across scales
    # For complex noise (1/f), it stays constant.
    
    # 4. Test Topological Entropy (Persistent Entropy)
    # Crash Simulation: Structure collapses -> Lower Betti score (Low Entropy/Complexity)
    # ... Wait, our Rust implementation returns "Complexity Score" (MST Weight / N)
    # Higher Score = More spread out / higher dimension.
    # Collapse = Lower Score.
    
    te_noise = holonic_speed.calculate_persistent_entropy(noise, 50, 3, 1)
    te_sine = holonic_speed.calculate_persistent_entropy(sine, 50, 3, 1) # Sine is a 1D loop embedded in 3D
    
    print(f"Topology Score (Noise): {te_noise:.4f}")
    print(f"Topology Score (Sine):  {te_sine:.4f}")
    
    print("✅ AEHML 2.0 Verification Passed!")

if __name__ == "__main__":
    test_aehml_2_0()
