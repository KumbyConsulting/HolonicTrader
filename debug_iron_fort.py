
import sys
import os
import logging
from sandbox.playground import Playground
from sandbox.strategies.evo_strategy import EvoStrategy
from evolution_lab import EvolutionLab
import config

# Setup Logging
logging.basicConfig(level=logging.INFO)

def debug_iron_fort():
    print("🏰 Debugging Iron Fort Logic...")
    
    # Init Lab (mimic Iron Fort settings)
    lab = EvolutionLab(
        initial_capital=300.0,
        leverage=1.0 # Iron Fort uses 1x
    )
    
    # Load Data (limit to same as evo engine)
    lab.load_data(history_limit=1500)
    
    # Monitor the Monster Genome (from HOF)
    genome = {
        "rsi_buy": 49.063522239318004,
        "rsi_sell": 93.09282009211583,
        "stop_loss": 0.015414218530806089,
        "take_profit": 0.10310406649624065,
        "use_satellite": True,
        "sat_rvol": 6.510221828039415,
        "sat_bb_expand": 0.22503835975332287,
        "leverage_cap": 2.27, # 2.27x
        "trailing_activation": 2.96,
        "trailing_distance": 0.23
    }
    
    print("\n🧐 Running Gauntlet with MONSTER GENOME...")
    stats = lab.run_gauntlet(genome)
    
    print(f"\n📊 Results:")
    print(f"Fitness: {stats['fitness']:.2f}")
    print(f"ROI: {stats['roi']*100:.2f}%")
    print(f"Equity: ${stats['final_equity']:.2f}")
    print(f"Trades: {stats['trades']}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    
    # Drill Down into individual Arenas to find the 1000x Asset
    print("\n🔍 Asset Inspection:")
    for sym, arena in lab.datasets.items():
        if not arena.trades:
            continue
            
        # Re-calc ROI per arena to find the anomaly
        final_eq = arena.equity_curve[-1]['equity'] if arena.equity_curve else 1000.0
        roi = (final_eq - 1000.0) / 1000.0
        
        print(f"  {sym}: ROI {roi*100:.2f}% | Trades: {len(arena.trades)} | Final Eq: ${final_eq:.2f}")
        
        if roi > 1.0: # > 100% ROI (lowered threshold to catch any spike)
            print(f"  🚨 ANOMALY DETECTED IN {sym}!")
            print("  printing first 5 and last 5 trades...")
            for t in arena.trades[:5]: print(f"    {t}")
            print("    ...")
            for t in arena.trades[-5:]: print(f"    {t}")

if __name__ == "__main__":
    debug_iron_fort()
