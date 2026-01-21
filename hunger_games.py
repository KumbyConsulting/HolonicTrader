import logging
import json
from evolution_lab import EvolutionLab

# Configuration
POPULATION_SIZE = 30
GENERATIONS = 5
STARTING_CAPITAL = 17.90
TARGET_CAPITAL = 100.0
LEVERAGE = 5.0 # Survival Mode
GAUNTLET_SYMBOLS = ['ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'ADA/USDT'] 

def run_hunger_games():
    print(f"🔥 THE HUNGER GAMES: $100 CHALLENGE 🔥")
    print(f"Goal: Turn ${STARTING_CAPITAL} -> ${TARGET_CAPITAL} using {LEVERAGE}x Leverage")
    print(f"Gauntlet: {GAUNTLET_SYMBOLS}\n")
    
    # 1. Initialize Lab
    lab = EvolutionLab(
        symbols=GAUNTLET_SYMBOLS,
        initial_capital=STARTING_CAPITAL,
        leverage=LEVERAGE,
        population_size=POPULATION_SIZE,
        fitness_mode='AGGRESSIVE',
        verbose=True
    )
    
    # 2. Load Data
    lab.load_data(history_limit=1500, secondary_limit=6000)
    
    # 3. Spawn Tributes (Seed Population)
    # We can inject a specific apex genome if we want, or start fresh.
    # For the challenge, let's inject the known "Apex" if we have it, or otherwise let the lab generate randoms.
    apex_genome = {
        'rsi_buy': 49,
        'rsi_sell': 70,
        'stop_loss': 0.04, 
        'take_profit': 0.09,
        'use_satellite': True,
        'sat_rvol': 2.7,
        'sat_bb_expand': 0.135
    }
    lab.seed_population(elite_genome=apex_genome)
    
    # 4. Evolve
    print(f"\n[System] Commencing Evolution for {GENERATIONS} generations...\n")
    winner = lab.evolve(generations=GENERATIONS)
    
    print("\n" + "="*60)
    print("👑 THE APEX PREDATOR ($100 CHALLENGE) 👑")
    print("="*60)
    print(f"Final Equity: ${winner['final_equity']:.2f}")
    print(f"ROI:          {winner['roi']*100:.2f}%")
    print(f"Win Rate:     {winner['win_rate']*100:.1f}%")
    print(f"Trades:       {winner['trades']}")
    print(f"Fitness:      {winner['fitness']:.2f}")
    print(f"Genome:       {winner['genome']}")
    print("="*60)
    
    # Save Winner
    with open('winner_challenge.json', 'w') as f:
        json.dump(winner, f, indent=4)
        print("💾 Challenge Winner saved to winner_challenge.json")

if __name__ == "__main__":
    # Configure logger to output to console for the game
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_hunger_games()
