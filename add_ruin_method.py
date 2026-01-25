import os

path = r'c:\Users\USER\Documents\AEHML\HolonicTrader\HolonicTrader\HolonicTrader\agent_governor.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

method_to_add = """
    def calculate_ruin_probability(self, symbol: str, entry_price: float, direction: str, stop_loss: float, take_profit: float, metadata: Dict[str, Any]) -> float:
        \"\"\"
        Monte Carlo Ruin Guard:
        Simulate 200 future paths using SDE parameters to estimate 
        the probability of hitting Stop Loss before Take Profit/Horizon.
        \"\"\"
        if not metadata or 'sde_physics' not in metadata:
            return 0.5 # Unknown risk
            
        try:
            from HolonicTrader.sde_engine import SDEEngine
            
            sde = metadata['sde_physics']
            # We use GBM for trend-following and OU for mean-reversion
            # Let's default to GBM as it's more conservative for ruin
            params = {
                'drift': sde.get('drift', 0.15),
                'diffusion': sde.get('sigma', 0.8) # sigma from OU is generally a good inst-vol proxy
            }
            
            # Simulate 100 steps (approx 25 hours if dt=15m)
            paths = SDEEngine.simulate_paths('GBM', params, entry_price, horizon=100, paths=200)
            
            # Calculate failures (hitting SL)
            failures = 0
            for p in range(len(paths)):
                path = paths[p]
                if direction == 'BUY':
                    if np.any(path < stop_loss):
                        failures += 1
                else:
                    if np.any(path > stop_loss):
                        failures += 1
                        
            ruin_prob = failures / len(paths)
            return float(ruin_prob)
            
        except Exception as e:
            if self.DEBUG: print(f"[{self.name}] Ruin Guard Error: {e}")
            return 0.5
"""

# Insert before check_cluster_risk
target = "    def check_cluster_risk"
if target in content:
    new_content = content.replace(target, method_to_add + "\n" + target)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Method added successfully")
else:
    print("Target not found")
