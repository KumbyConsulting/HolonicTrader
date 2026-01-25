
import evolution_lab
import config
import logging

# Setup minimal logging
logging.basicConfig(level=logging.INFO)

def test_evo():
    print("🧪 Testing Evolution Engine with Warp Speed...")
    try:
        # Use a single asset for speed
        lab = evolution_lab.EvolutionLab(
            population_size=10, 
            symbols=['BTC/USDT'],
            verbose=True
        )
        
        # Load minimal data
        lab.load_data(history_limit=200, secondary_limit=0)
        
        lab.seed_population()
        
        print("🚀 Starting Evolution...")
        winner = lab.evolve(generations=2)
        
        print(f"\n✅ Evolution Test Success!")
        print(f"Leader Fitness: {winner['fitness']:.2f}")
        print(f"ROI: {winner['roi']*100:.2f}%")
        print(f"Trades: {winner['trades']}")
        
    except Exception as e:
        print(f"❌ Evolution Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evo()
