import random
import copy
import json
import logging
import os
from typing import List, Dict, Optional
from sandbox.playground import Playground
from sandbox.playground import Playground
from sandbox.strategies.evo_strategy import EvoStrategy
import config

# Logger configured by parent (run_evolution_loop.py)
logger = logging.getLogger("EvolutionLab")

class EvolutionLab:
    def __init__(self, 
                 symbols: List[str] = None, # Default to None, load from config if missing
                 initial_capital: float = None,
                 leverage: float = 20.0, # REDUCED: 50x -> 20x for Stability
                 population_size: int = 60,
                 fitness_mode: str = 'BALANCED', # 'BALANCED' (Safety) or 'AGGRESSIVE' (Growth)
                 verbose: bool = False):
        
        self.symbols = symbols if symbols else list(config.KRAKEN_SYMBOL_MAP.keys())
        self.initial_capital = initial_capital if initial_capital else config.INITIAL_CAPITAL
        self.leverage = leverage
        self.population_size = population_size
        self.fitness_mode = fitness_mode
        self.verbose = verbose
        
        self.datasets: Dict[str, Playground] = {}
        self.population: List[Dict] = []
        self.hall_of_fame: List[Dict] = [] # Memory: Top Genomes
        self.best_overall: Optional[Dict] = None

    def load_data(self, history_limit: int = 1500, secondary_limit: int = 6000):
        """Pre-load data for all arenas."""
        logger.info(f"📡 Pre-loading Gauntlet Data for {self.symbols}...")
        for sym in self.symbols:
            p = Playground(symbol=sym, initial_capital=self.initial_capital, verbose=False, leverage=self.leverage)
            p.load_data(limit=history_limit)
            p.load_secondary_data(limit=secondary_limit)
            self.datasets[sym] = p
        logger.info("✅ Data Loaded.")

    def generate_random_genome(self) -> Dict:
        """Spawn random genome suitable for benchmark survival."""
        return {
            'rsi_buy': random.uniform(20, 50),     # Buy Dip
            'rsi_sell': random.uniform(75, 95),    # Sell Rip
            'stop_loss': random.uniform(0.01, 0.05), # 1% - 5% (Safe for 1x)
            'take_profit': random.uniform(0.05, 0.20), # Swing targets
            'use_satellite': True,
            'sat_rvol': random.uniform(2.0, 5.0),
            'sat_bb_expand': random.uniform(0.05, 0.25),
            
            # --- INSTITUTIONAL GENES ---
            'leverage_cap': 1.0, # FORCE 1x LEVERAGE (Benchmark Mode)
            'trailing_activation': random.uniform(1.5, 3.0),
            'trailing_distance': random.uniform(0.2, 0.5)
        }



    def seed_population(self, elite_genome: Optional[Dict] = None):
        """Initialize population, optionally injecting an Elite (e.g., current live brain)."""
        self.population = [self.generate_random_genome() for _ in range(self.population_size)]
        
        if elite_genome:
            # Validate elite keys
            valid = True
            for k in self.population[0].keys():
                if k not in elite_genome:
                    valid = False
            
            # Merge Elite with defaults to ensure all keys exist
            if valid:
                self.population[0] = copy.deepcopy(elite_genome)
                logger.info(f"🧬 Injected Elite Genome: {elite_genome}")
            else:
                # Patch missing keys instead of ignoring
                defaults = self.generate_random_genome()
                defaults.update(elite_genome) # Overlay elite values
                self.population[0] = defaults
                logger.info("🧬 Injected Elite Genome (Patched with new genes).")
        
        # SANITIZATION: Enforce Physics
        # If High Leverage, FORCE Tight Stops.
        # This prevents "Toxic Legacy Genes" (e.g. 6% SL) from bankrupting the simulation instantly.
        elite = self.population[0]
        if elite.get('leverage_cap', 1.0) > 20.0:
            if elite.get('stop_loss', 0.05) > 0.02: # 2% max for 50x
                logger.warning(f"⚠️ Sanitizing Toxic Stop Loss: {elite['stop_loss']} -> 0.015")
                elite['stop_loss'] = 0.015
                self.population[0] = elite

    def run_gauntlet(self, genome: Dict, basket: List[str] = None) -> Dict:
        """Run the Gauntlet: Compounds capital across asset basket."""
        # Use provided basket or full symbol list
        active_symbols = basket if basket else self.symbols
        
        current_capital = self.initial_capital
        total_trades = 0
        wins = 0
        
        # Track global drawdown across the sequence
        max_dd = 0.0
        peak_equity = self.initial_capital

        peak_equity = self.initial_capital
        
        # Shuffle symbols to test portfolio robustness (Sequence Risk)
        # Note: If basket is passed, we assume order matters? No, usually random is better.
        test_sequence = active_symbols.copy()
        # random.shuffle(test_sequence) # Optional: Enable for sequence testing

        for symbol in test_sequence:
            # Safety check: if symbol not in datasets
            if symbol not in self.datasets: continue
            
            arena = self.datasets[symbol]
            
            # Skip empty arenas (e.g. data load failed)
            if arena.df is None or arena.df.empty:
                # logger.warning(f"Skipping {symbol} (No Data)")
                continue
                
            arena.capital = current_capital
            arena.inventory = 0.0
            arena.entry_price = 0.0
            arena.trades = []
            arena.equity_curve = []
            
            # FORCE LEVERAGE from GENOME (1.0x)
            arena.leverage = genome.get('leverage_cap', 1.0)
            
            arena.inject_strategy(EvoStrategy(genome))
            
            arena.run()
            
            current_capital = arena.capital
            
            # Harvest Stats
            # Harvest Stats
            trades = [t for t in arena.trades if t['type']=='SELL']
            total_trades += len(trades)
            wins += len([t for t in trades if t['pnl']>0])
            
            # === WALK-FORWARD VALIDATION (Time Travel) ===
            # Split Data: 80% Train (Past) | 20% Test (Future)
            # We enforce that the strategy must perform in the Test set.
            total_candles = len(arena.df)
            split_idx = int(total_candles * 0.8)
            if split_idx < len(arena.df):
                split_ts = arena.df.iloc[split_idx]['timestamp']
                
                # Split Equity Curve
                train_curve = [e for e in arena.equity_curve if e['timestamp'] < split_ts]
                test_curve = [e for e in arena.equity_curve if e['timestamp'] >= split_ts]
                
                # Analyze Test Performance
                if test_curve:
                    start_test = train_curve[-1]['equity'] if train_curve else self.initial_capital
                    end_test = test_curve[-1]['equity']
                    test_roi = (end_test - start_test) / start_test if start_test > 0 else 0
                    
                    # Count Test Phase Trades (Consistency Check)
                    test_trades_count = len([t for t in arena.trades if t['time'] >= split_ts and t['type'] == 'SELL'])

                    # Test Drawdown
                    test_peak = start_test
                    test_max_dd = 0.0
                    for item in test_curve:
                        if item['equity'] > test_peak: test_peak = item['equity']
                        dd = (test_peak - item['equity']) / test_peak
                        if dd > test_max_dd: test_max_dd = dd
                        
                    # TEST SCORE: Survival * ROI
                    test_survival = 1.0 - (test_max_dd ** 2)
                    
                    # CONSISTENCY GATE:
                    # If 0 trades in Test Phase, Fitness = 0 (Dormant Strategy)
                    if test_trades_count == 0:
                         test_multiplier = 0.0 # FAIL: Did not trade in future
                    elif test_roi < 0:
                        test_multiplier = 0.1 # FAIL: Lost money in future
                    else:
                        test_multiplier = 1.0 + test_roi # PASS: Profited in future
                else:
                    test_multiplier = 1.0 # No test data (short run)
            else:
                 test_multiplier = 1.0
            
            # Update Global Drawdown Tracking (Standard)
            if arena.equity_curve:
                for item in arena.equity_curve:
                    eq = item['equity']
                    if eq > peak_equity: 
                        peak_equity = eq
                    
                    # Calculate DD from global peak
                    dd = (peak_equity - eq) / peak_equity
                    if dd > max_dd: 
                        max_dd = dd
            
            # Bankruptcy check
            if current_capital < 2.0:
                current_capital = 0
                max_dd = 1.0 # bankrupt
                test_multiplier = 0.0 # Failed
                break
                
        roi = (current_capital - self.initial_capital) / self.initial_capital
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # --- ADVANCED FITNESS FUNCTION (Risk-Adjusted) ---
        # Objective: Reward consistency, Penalize volatility and deep drawdowns.
        # Fitness = Sortino * Calmar * (1 - non_linear_dd_penalty)
        
        import numpy as np
        
        fitness = 0.0
        
        if arena.equity_curve and len(arena.equity_curve) > 2:
            # 1. Calculate Returns Series
            eq_series = [e['equity'] for e in arena.equity_curve]
            returns = np.diff(eq_series) / eq_series[:-1]
            returns = returns[returns != 0] # Filter idle
            
            if len(returns) > 5:
                # 2. Risk Metrics
                avg_ret = np.mean(returns)
                std_dev = np.std(returns)
                downside_returns = returns[returns < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_dev
                
                # Annualized (Assuming 15m/1H candles? - Heuristic approx)
                # Volatility Adjustment
                
                # Sharpe / Sortino Calculation (Safe)
                # Annualized (Assuming 15m/1H candles? - Heuristic approx)
                
                std_dev_safe = std_dev if std_dev > 1e-9 else 1e-9
                downside_std_safe = downside_std if downside_std > 1e-9 else 1e-9
                
                sharpe = avg_ret / std_dev_safe
                sortino = avg_ret / downside_std_safe
                
                # CAP EXTREME VALUES (Heal the math)
                sortino = min(sortino, 50.0) 
                sharpe = min(sharpe, 20.0)

                # 3. Drawdown Penalty (Non-Linear)
                # DD < 20% -> Linear Penalty
                # DD > 20% -> Exponential Penalty (The "Death Zone")
                # DD > 50% -> Fitness Obliteration
                
                dd_penalty = 0.0
                if max_dd <= 0.20:
                    dd_penalty = max_dd * 1.5
                elif max_dd <= 0.50:
                    dd_penalty = 0.30 + ((max_dd - 0.20) * 3.0) # 20-50% scales fast
                else:
                    dd_penalty = 1.0 # >50% DD is instant failure for institutional grade
                    
                survival_score = 1.0 - dd_penalty
                if survival_score < 0: survival_score = 0
                
                # 4. Profit Factor (Gross Win / Gross Loss)
                # Add check for zero loss
                gross_win = sum([t['pnl'] for t in trades if t['pnl'] > 0])
                gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
                profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 1.0
                profit_factor = min(profit_factor, 10.0) # Cap extreme profit factors
                
                # 5. Composite Score
                # Weighted mix of Stability (Sortino), Profitability (Profit Factor), and Survival
                
                # Base: ROI damped by Time
                # We want ROI, but not at the cost of logic.
                
                # New Fitness Formula:
                # Fit = (Sortino^2) * Log(Total_Return) * Survival_Score
                
                log_ret = np.log1p(max(0, roi)) # Log return to dampen millions%
                
                fitness = (sortino * 2.0) + (profit_factor * 1.0) + (log_ret * 5.0)
                fitness *= survival_score
                
                # Final hard clamp for bankruptcy
                if max_dd > 0.60: fitness = 0.0
                
            else:
                fitness = 0.0 # Not enough trading activity
        else:
            fitness = 0.0

        # Test Multiplier (Future Validation)
        fitness *= test_multiplier
        
        # Dead strategy penalty
        if total_trades == 0: fitness = 0.0
        
        return {
            'genome': genome,
            'final_equity': current_capital,
            'roi': roi,
            'win_rate': win_rate,
            'trades': total_trades,
            'max_dd': max_dd,
            'sharpe': sharpe if 'sharpe' in locals() else 0.0,
            'sortino': sortino if 'sortino' in locals() else 0.0,
            'validation_roi': test_roi if 'test_roi' in locals() else 0.0,
            'validation_trades': test_trades_count if 'test_trades_count' in locals() else 0,
            'entropy': 0.0,
            'test_score': test_multiplier, 
            'fitness': fitness,
            'sharpe': float(sharpe) if 'sharpe' in locals() else 0.0,
            'sortino': float(sortino) if 'sortino' in locals() else 0.0
        }

    def evolve(self, generations: int = 5, mutation_rate: float = 0.1, population_size: int = None, basket: List[str] = None) -> Dict:
        """
        Run the evolution loop.
        :param basket: Optional subset of assets to train on (Portfolio Level Testing)
        """
        target_pop_size = population_size if population_size else self.population_size
        basket_tag = "Full Market" if not basket else f"Basket({len(basket)})"
        
        logger.info(f"🏹 Starting Evolution: {generations} Gens, {target_pop_size} Pop, {basket_tag}")
        
        for gen in range(generations):
            results = []
            for genome in self.population:
                res = self.run_gauntlet(genome, basket=basket)
                results.append(res)
                
            results.sort(key=lambda x: x['fitness'], reverse=True)
            best_gen = results[0]
            
            # --- MEMORY: Update Hall of Fame ---
            # Keep top 10 unique genomes
            # Logic: If fits > worst_hof, add. Then dedup.
            self.hall_of_fame.append(best_gen)
            # Sort HOF
            self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
            # Keep Top 10
            self.hall_of_fame = self.hall_of_fame[:10]
            # -----------------------------------
            
            logger.info(f"Gen {gen+1} Leader: Fit {best_gen['fitness']:.2f} | Eq ${best_gen['final_equity']:.2f} (DD {best_gen['max_dd']*100:.1f}%) | Sharpe {best_gen.get('sharpe', 0):.2f} | Sortino {best_gen.get('sortino', 0):.2f}")
            
            if self.best_overall is None or best_gen['fitness'] > self.best_overall['fitness']:
                self.best_overall = best_gen
                
            # === PARETO REPRODUCTION (80/20 Rule) ===
            # Only the Vital Few (Top 20%) reproduce.
            # The Trivial Many (Bottom 80%) are discarded.
            # === PARETO REPRODUCTION (80/20 Rule) ===
            # Only the Vital Few (Top 20%) reproduce.
            # The Trivial Many (Bottom 80%) are discarded.
            elite_count = max(2, int(target_pop_size * 0.20)) # Min 2
            survivors = results[:elite_count] 
            new_pop = [s['genome'] for s in survivors]
            
            # 1. 🛸 ALIENS (20% Fresh Random) - Escape Local Optima
            alien_count = int(target_pop_size * 0.2)
            for _ in range(alien_count):
                new_pop.append(self.generate_random_genome())
                
            # 2. 🧬 CROSSOVER & MUTATION of the ELITES
            while len(new_pop) < target_pop_size:
                # Strictly breed from the Top 20%
                parent_a = random.choice(survivors)['genome']
                parent_b = random.choice(survivors)['genome']
                
                # Crossover (50/50 mix)
                child = {}
                for k in parent_a.keys():
                    child[k] = parent_a[k] if random.random() > 0.5 else parent_b[k]
                
                # Hyper-Mutation (Adapted to Mutation Rate)
                # mutation_rate ~ 0.15 means ~15% of genes mutate? 
                # Or just use it as a scaler for "Intensity". 
                # Let's say Intensity = 1 + (Rate * 10) -> 0.15 * 10 = 1.5 -> 1-2 genes.
                # If Rate is 0.35 -> 3.5 -> 3-4 genes.
                intensity_base = int(mutation_rate * 10)
                mutation_intensity = max(1, random.randint(1, intensity_base + 1))
                
                for _ in range(mutation_intensity):
                    key = random.choice(list(child.keys()))
                    if not isinstance(child[key], bool):
                        # Wider variance: 0.8x to 1.2x (Scaled by rate?)
                        # Let's keep variance standard but frequency dynamic
                        child[key] *= random.uniform(0.8, 1.2)
                        
                        # --- SAFETY CLAMPING ---
                        # Enforce Physics: Stop Loss * Leverage < 1.0 (ideally < 0.8 for buffer)
                        if key == 'stop_loss':
                            lev = child.get('leverage_cap', self.leverage)
                            max_safe_sl = 0.8 / lev if lev > 0 else 1.0
                            if child['stop_loss'] > max_safe_sl:
                                child['stop_loss'] = max_safe_sl
                        # Safety Clamp: Leverage Cap
                        if key == 'leverage_cap':
                            child['leverage_cap'] = max(1.0, min(child['leverage_cap'], 5.0)) # Clamp 1x-5x

                        # Safety Clamp: Stop Loss vs Leverage
                        if key == 'rsi_buy':
                            child['rsi_buy'] = max(10, min(60, child['rsi_buy'])) # 10 < Buy < 60
                        if key == 'rsi_sell':
                            child['rsi_sell'] = max(40, min(95, child['rsi_sell'])) # 40 < Sell < 95
                            
                new_pop.append(child)
                
            self.population = new_pop
            
        logger.info(f"👑 Apex Predator Found: ${self.best_overall['final_equity']:.2f}")
        return self.best_overall

if __name__ == "__main__":
    # Test Run
    lab = EvolutionLab(leverage=5.0)
    lab.load_data()
    lab.seed_population()
    winner = lab.evolve(generations=3)
    print(winner)
