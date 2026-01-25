from evolution_lab import EvolutionLab
import config
import logging
import sys

# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)

def profile_doge():
    print("🐕 STARTING DOGE PROFILING...")
    print("Goal: Evolve optimal parameters for DOGE/USDT")
    
    # 1. Initialize Lab specifically for DOGE
    # We restrict the symbol list to JUST DOGE to force the algo to specialize.
    lab = EvolutionLab(
        symbols=['DOGE/USDT'],
        initial_capital=1000.0,
        leverage=5.0, # Test with reasonable leverage
        population_size=50 # Smaller population for speed
    )
    
    # 2. Load Data (History is crucial for profiling)
    print("📡 Loading Historical Data for DOGE...")
    lab.load_data(history_limit=2000, secondary_limit=5000)
    
    # 3. Seed Population (Random start to find natural fit)
    lab.seed_population()
    
    # 4. Evolve
    # Run for 10 generations to let it converge
    print("🧬 Evolving Strategy (10 Generations)...")
    winner = lab.evolve(generations=10, mutation_rate=0.2, basket=['DOGE/USDT'])
    
    print("\n\n" + "="*40)
    print("🏆 DOGE PROFILE COMPLETE 🏆")
    print("="*40)
    print(f"Final Fitness: {winner['fitness']:.2f}")
    print(f"ROI: {winner['roi']*100:.2f}%")
    print(f"Sharpe: {winner.get('sharpe', 0):.2f}")
    print(f"Trades: {winner['trades']}")
    print("-" * 20)
    print("OPTIMAL GENOME (Copy these to config/strategy):")
    for k, v in winner['genome'].items():
        print(f"  {k}: {v}")
    print("="*40)

if __name__ == "__main__":
    profile_doge()
