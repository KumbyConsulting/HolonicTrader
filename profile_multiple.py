from evolution_lab import EvolutionLab
import config
import logging
import sys

# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)

def profile_asset(symbol):
    print(f"\n\n>> STARTING PROFILING FOR {symbol}...")
    
    # 1. Initialize Lab specifically for this asset
    lab = EvolutionLab(
        symbols=[symbol],
        initial_capital=1000.0,
        leverage=5.0, 
        population_size=40 # Moderate size
    )
    
    # 2. Load Data 
    print(f"> Loading Historical Data for {symbol}...")
    lab.load_data(history_limit=2000, secondary_limit=5000)
    
    # 3. Seed Population
    lab.seed_population()
    
    # 4. Evolve
    print(f"* Evolving Strategy for {symbol} (8 Generations)...")
    winner = lab.evolve(generations=8, mutation_rate=0.2, basket=[symbol])
    
    return winner

def main():
    # Dynamic Target Selection: Profile ALL allowed assets
    targets = config.ALLOWED_ASSETS
    print(f"🚀 STARTING MASS PROFILING SESSION FOR {len(targets)} ASSETS...")
    
    results = {}
    
    # Open file for writing results physically
    with open('profile_results_auto.txt', 'w', encoding='utf-8') as f:
        f.write(f"PROFILING REPORT - {len(targets)} ASSETS\n")
        f.write("="*50 + "\n\n")

        for symbol in targets:
            try:
                winner = profile_asset(symbol)
                results[symbol] = winner
                
                # Console Output
                print(f"\n[OK] {symbol} PROFILING COMPLETE")
                print("-" * 30)
                print(f"Fitness: {winner['fitness']:.2f}")
                print(f"Sharpe: {winner.get('sharpe', 0):.2f}")
                
                # File Output (Incremental)
                f.write(f"--- {symbol} ---\n")
                f.write(f"Fitness: {winner['fitness']:.2f} | Sharpe: {winner.get('sharpe', 0):.2f}\n")
                f.write("Params:\n")
                for k, v in winner['genome'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("-" * 30 + "\n")
                f.flush() # Ensure write
                
            except Exception as e:
                print(f"[ERR] ERROR Profiling {symbol}: {e}")
                import traceback
                traceback.print_exc()

    print("\n\n" + "="*50)
    print("FINISHED GLOBAL PROFILE REPORT (Saved to profile_results_auto.txt)")
    print("="*50)
    
    for symbol, winner in results.items():
        print(f"\n--- {symbol} ---")
        print(f"Fitness: {winner['fitness']:.2f} | Sharpe: {winner.get('sharpe', 0):.2f}")
        print("Params:")
        g = winner['genome']
        print(f"  RSI Buy: {g.get('rsi_buy'):.1f} | Sell: {g.get('rsi_sell'):.1f}")
        print(f"  StopLoss: {g.get('stop_loss')*100:.2f}% | TP: {g.get('take_profit')*100:.2f}%")
        print(f"  Sat RVOL: {g.get('sat_rvol'):.2f} | Sat BB: {g.get('sat_bb_expand'):.2f}")

if __name__ == "__main__":
    main()
