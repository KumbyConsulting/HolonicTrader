import time
import json
import os
import logging
import numpy as np
from evolution_lab import EvolutionLab

# Config
LOOP_INTERVAL = 60
GENOME_FILE = 'live_genome.json'
WINNER_ARCHIVE = 'winner_challenge.json'
STAGNATION_THRESHOLD = 3
MAX_STAGNATION = 10

# Import Config
import config
config.ALLOWED_ASSETS = list(config.KRAKEN_SYMBOL_MAP.keys())

import sys
# FORCE UTF-8 for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

import sys
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# FORCE UTF-8 for Windows Console (Safety)
sys.stdout.reconfigure(encoding='utf-8')

# --- BLOOMBERG TERMINAL THEME ---
bloomberg_theme = Theme({
    "logging.level.info": "cyan",           # Info is Cyan
    "logging.level.warning": "bold yellow", # Warnings scream
    "logging.level.error": "bold white on red", # Critical
    "repr.number": "bold gold3",            # Numbers are Gold
    "repr.str": "white",                    # Strings are White
})

console = Console(theme=bloomberg_theme)

# 1. File Handler (Text / CSV-like)
file_handler = logging.FileHandler('evo_engine.log', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s [%(name)s] %(message)s')
file_handler.setFormatter(file_formatter)

# 2. Rich Console Handler (Bloomberg Style)
# Rich handles timestamps and levels natively, so we just pass the message.
rich_handler = RichHandler(
    console=console, 
    rich_tracebacks=True,
    markup=True,
    show_time=True,
    show_path=False
)
rich_handler.setFormatter(logging.Formatter("%(message)s"))

# 3. Setup Root Logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, rich_handler]
)

class Island:
    """A distinct population with unique evolutionary rules."""
    def __init__(self, name: str, strategy_bias: str, mutation_rate: float, min_leverage: float):
        self.name = name
        self.strategy_bias = strategy_bias # 'AGGRESSIVE', 'BALANCED', 'CONSERVATIVE'
        self.mutation_rate = mutation_rate
        self.average_leverage = min_leverage
        self.champion = None
        self.stagnation = 0

class ArchipelagoEngine:
    def __init__(self):
        self.total_cycles = 0
        self.migration_interval = 3
        
        # 1. Initialize Islands (Parallel Evolution)
        self.islands = [
            Island("🌋 Volcano Peak", "AGGRESSIVE", 0.35, 1.0),
            Island("🏰 Iron Fort", "CONSERVATIVE", 0.05, 1.0),
            Island("🏝️ Darwin's Rock", "BALANCED", 0.15, 1.0)
        ]
        
        # Shared Knowledge
        self.hall_of_fame = []
        self.best_global_fitness = 0.0
        
        # Labs (One per Island concept - logic handled in loop)
        self.lab = EvolutionLab(
            symbols=config.ALLOWED_ASSETS,
            initial_capital=config.INITIAL_CAPITAL,
            leverage=20.0 # Base leverage, overridden by Island
        )
        self.lab.load_data(history_limit=1500, secondary_limit=6000)
        
    def load_seed(self):
        # ... logic to load seed ...
        try:
             if os.path.exists(GENOME_FILE):
                with open(GENOME_FILE, 'r') as f: return json.load(f).get('genome')
        except: pass
        return None

    def save_apex(self, result, source_island):
        result['timestamp'] = time.time()
        result['source_island'] = source_island
        
        # Save Active Genome
        with open(GENOME_FILE, 'w') as f:
            json.dump(result, f, indent=4)
            
        # Save Hall of Fame (Top 10)
        # Deduplicate and Sort
        unique_hof = {g['fitness']: g for g in self.hall_of_fame}.values()
        sorted_hof = sorted(unique_hof, key=lambda x: x['fitness'], reverse=True)[:10]
        
        with open('hall_of_fame.json', 'w') as f:
            json.dump(list(sorted_hof), f, indent=4)
            
        logging.info(f"💾 GLOBAL APEX SAVED from {source_island} (Fit: {result.get('fitness', 0):.2f})")
        logging.info(f"💾 HOF UPDATED ({len(sorted_hof)} Kings)")

    def run(self):
        logging.info("🏝️ ARCHIPELAGO ENGINE STARTING...")
        
        # Seed Initial Populations
        seed = self.load_seed()
        self.lab.seed_population(seed)
        
        while True:
            try:
                self.total_cycles += 1
                logging.info(f"=== Archipelago Cycle {self.total_cycles} ===")
                
                # 1. Rotate through Islands
                for island in self.islands:
                    logging.info(f"⛵ Sailing to {island.name}...")
                    
                    # A. Adjust Lab Environment for this Island
                    # (In a real parallel system, these would serve simultaneous processes)
                    self.lab.leverage = island.average_leverage
                    
                    # B. Portfolio Baskets (Robustness Test)
                    # Randomly select a basket of 10 assets to test generalization
                    import random
                    basket_size = 10
                    # Ensure we pick valid symbols that have loaded data
                    valid_syms = [s for s in self.lab.symbols if s in self.lab.datasets]
                    if len(valid_syms) > basket_size:
                        basket = random.sample(valid_syms, basket_size)
                    else:
                        basket = valid_syms
                    
                    # C. Evolve
                    winner = self.lab.evolve(
                        generations=3, 
                        mutation_rate=island.mutation_rate,
                        basket=basket
                    )
                    
                    # D. Assess
                    current_fitness = winner.get('fitness', 0)
                    
                    if current_fitness > self.best_global_fitness:
                        logging.info(f"🚀 {island.name} produced a NEW GLOBAL CHAMPION! (Fit {current_fitness:.2f})")
                        self.best_global_fitness = current_fitness
                        island.champion = winner
                        self.save_apex(winner, island.name)
                        self.hall_of_fame.append(winner)
                    else:
                        # Local Improvement check?
                        pass
                        
                # 2. Migration Phase (Mixing)
                if self.total_cycles % self.migration_interval == 0:
                    logging.info("🕊️ MIGRATION EVENT: Sharing best genomes between islands.")
                    # In this sequential implementation, self.lab already has the mixed population.
                    # We just need to make sure we inject HOF members back into the gene pool
                    # if they were lost.
                    if self.hall_of_fame:
                         # Reinject Top 3 to prevent Drift forgetting
                         hof_top = sorted(self.hall_of_fame, key=lambda x: x['fitness'], reverse=True)[:3]
                         for elite in hof_top:
                             if elite['genome'] not in self.lab.population:
                                 self.lab.population.append(elite['genome'])
                         logging.info(f"💉 Re-injected {len(hof_top)} Ancient Kings from Hall of Fame.")

                logging.info(f"Sleeping {LOOP_INTERVAL}s...")
                time.sleep(LOOP_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"Archipelago Error: {e}", exc_info=True)
                time.sleep(60)

def main():
    engine = ArchipelagoEngine()
    engine.run()

if __name__ == "__main__":
    main()