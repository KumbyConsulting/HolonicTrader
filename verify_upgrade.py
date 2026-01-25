
import sys
import os
import numpy as np

# Add local directory to path to simulate running from root
sys.path.append(os.getcwd())

print("Testing Entropy Scouter Integration...")

try:
    from core.scouts.entropy_scouter import EntropyScouter
    from core.mathematics.entropy import calculate_sample_entropy
    print("✅ Imports Successful.")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

# Test Math
data = np.sin(np.linspace(0, 10, 100)) # Ordered sine wave
ent = calculate_sample_entropy(data, m=2, r=0.2*np.std(data))
print(f"Sine Wave Entropy: {ent:.4f} (Expected < 0.2)")

noise = np.random.rand(100) # Random noise
ent_noise = calculate_sample_entropy(noise, m=2, r=0.2*np.std(noise))
print(f"Noise Entropy: {ent_noise:.4f} (Expected > 1.5)")

# Test Scouter Init
try:
    scouter = EntropyScouter("context.yaml")
    print("✅ Scouter Initialized with context.yaml")
    
    # Check if loaded correct config structure
    if hasattr(scouter, 'scout_config') and scouter.metrics_config:
         print(f"✅ Loaded Extended Configuration (r_sigma: {scouter.metrics_config.get('r_sigma')})")
         print(f"✅ Regimes Configured: {list(scouter.regime_config.keys())}")
    else:
         print("⚠️ Loaded Legacy Config or Default")

except Exception as e:
    print(f"❌ Scouter Init Failed: {e}")

print("Verification Complete.")
