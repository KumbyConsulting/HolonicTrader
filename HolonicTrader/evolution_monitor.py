import logging
import time
from typing import Dict, List, Any

logger = logging.getLogger("EvoMonitor")

class EvolutionMonitor:
    def __init__(self):
        self.alert_thresholds = {
            'fitness_inflation': 2.0,    # 2x jump in one cycle
            'equity_divergence': 0.5,     # fitness up, equity down
            'sharpe_ceiling': 4.0        # suspected overfitting
        }
        self.history = {} # island_name -> history_dicts

    def check_health(self, island_name: str, current_metrics: Dict[str, Any]) -> List[str]:
        alerts = []
        if island_name not in self.history:
            self.history[island_name] = []
            self.history[island_name].append(current_metrics)
            return alerts
            
        prev = self.history[island_name][-1]
        
        # 1. Fitness Inflation (without ROI growth)
        if current_metrics['fitness'] > prev['fitness'] * self.alert_thresholds['fitness_inflation']:
            if current_metrics['roi'] <= prev['roi']:
                alerts.append(f"☢️ FITNESS INFLATION: {island_name} fitness doubled while ROI stalled!")
                
        # 2. Equity/Fitness Divergence
        if current_metrics['fitness'] > prev['fitness'] and current_metrics.get('final_equity', 0) < prev.get('final_equity', 0):
             alerts.append(f"📉 DIVERGENCE: {island_name} fitness up, but equity down.")
             
        # 3. Sharpe Ceiling
        if current_metrics.get('sharpe', 0) > self.alert_thresholds['sharpe_ceiling']:
             alerts.append(f"🚩 OVERFIT SUSPECT: {island_name} Sharpe {current_metrics['sharpe']:.2f} exceeds ceiling.")

        # Update History
        self.history[island_name].append(current_metrics)
        return alerts

    def get_summary(self):
        summary = "--- EVO MONITOR SUMMARY ---\n"
        for island, logs in self.history.items():
            if logs:
                last = logs[-1]
                summary += f"{island}: Fit {last['fitness']:.2f} | ROI {last['roi']*100:.1f}%\n"
        return summary
