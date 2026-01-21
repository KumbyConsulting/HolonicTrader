from .base import Strategy, Signal
from .evo_strategy import EvoStrategy
import json
import os
import logging
import statistics

class EnsembleStrategy(Strategy):
    """
    🎭 TRIUMVIRATE ENSEMBLE 🎭
    Aggregates the wisdom of the Top 3 'Ancient Kings' from the Hall of Fame.
    """
    def __init__(self, hof_path='hall_of_fame.json', top_n=3):
        super().__init__(name="Ensemble-Triumvirate")
        self.strategies = []
        self.logger = logging.getLogger("Ensemble")
        
        # Path resolution: If relative, assume sibling of runs? 
        # Or look in current CWD
        load_path = hof_path if os.path.exists(hof_path) else os.path.join(os.getcwd(), hof_path)
        
        if os.path.exists(load_path):
            try:
                with open(load_path, 'r') as f:
                    hof = json.load(f)
                    
                    # Ensure list
                    if isinstance(hof, list):
                        # Sort by fitness desc
                        hof.sort(key=lambda x: x.get('fitness', 0), reverse=True)
                        
                        top_genes = hof[:top_n]
                        for i, entry in enumerate(top_genes):
                            genome = entry.get('genome')
                            if genome:
                                strat = EvoStrategy(genome)
                                strat.name = f"Member-{i+1}"
                                self.strategies.append(strat)
            except Exception as e:
                self.logger.error(f"Failed to load HOF: {e}")
        else:
             self.logger.warning(f"HOF File not found at {load_path}")
                    
        self.logger.info(f"🎭 Ensemble Loaded {len(self.strategies)} Strategies")

    def on_candle(self, slice_df, indicators, portfolio_state, secondary_slice_df=None) -> Signal:
        if not self.strategies:
            return Signal('HOLD', reason="Empty Ensemble")
            
        votes = []
        sizes = []
        stops = []
        take_profits = []
        
        score = 0 # Net Sentiment
        buy_votes = 0
        sell_votes = 0
        
        for s in self.strategies:
            sig = s.on_candle(slice_df, indicators, portfolio_state, secondary_slice_df)
            
            val = 0
            if sig.direction == 'BUY': 
                val = 1
                buy_votes += 1
                if sig.size: sizes.append(sig.size)
                if sig.stop_loss: stops.append(sig.stop_loss)
                if sig.take_profit: take_profits.append(sig.take_profit)
                
            elif sig.direction == 'SELL': 
                val = -1
                sell_votes += 1
            
            votes.append(sig)
            score += val

        # === CONSENSUS LOGIC ===
        
        # ENTRY LOGIC (Need Strong Majority)
        # If 3 Members: Need 2 Votes.
        if portfolio_state['inventory'] == 0:
            if buy_votes >= 2:
                # Average Sizing
                avg_size = statistics.mean(sizes) if sizes else 0.2
                # Conservative Stop (Tightest? Or Average?)
                # Average is distinct representation of group risk tolerance
                avg_sl = statistics.mean(stops) if stops else 0.02
                avg_tp = statistics.mean(take_profits) if take_profits else 0.05
                
                return Signal('BUY', size=avg_size, stop_loss=avg_sl, take_profit=avg_tp, 
                              reason=f"Ensemble Buy ({buy_votes}/{len(self.strategies)})")
        
        # EXIT LOGIC
        elif portfolio_state['inventory'] > 0:
            # 1. RISK VETO (Any Member hitting Stop Loss triggers exit)
            # This prevents the "Holders" from dragging the group into deep water
            for v in votes:
                if v.direction == 'SELL' and ("Stop Loss" in v.reason or "Risk" in v.reason):
                    return Signal('SELL', reason=f"Ensemble Risk Veto: {v.reason}")
            
            # 2. PROFIT CONSENSUS (Need Majority to Take Profit)
            # If 1 says TP, but 2 say HOLD, we might override and HOLD?
            # Or should we scale out?
            # Current Engine is All-or-Nothing.
            # Safety First: If Majority Sell (TP or otherwise), Exit.
            if sell_votes >= 2:
                 return Signal('SELL', reason=f"Ensemble Consensus Exit ({sell_votes}/{len(self.strategies)})")
                 
            # 3. UNANIMOUS HOLD (Implicit)
            
        return Signal('HOLD')
