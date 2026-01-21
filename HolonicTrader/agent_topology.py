"""
TopologyHolon - The "Structure" Brain (AEHML 2.0)

Specialized in Topological Data Analysis (TDA) to detect market crashes
before they happen by monitoring the collapse of high-dimensional structure.

Logic:
    Healthy Market = Complex Topology (High Persistent Entropy)
    Crash Precursor = Topology Collapse (Low Persistent Entropy)
"""

import pandas as pd
from typing import Any, Dict
import holonic_speed
from HolonicTrader.holon_core import Holon, Disposition
import config

class TopologyHolon(Holon):
    def __init__(self, name: str = "TopologyAgent"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.4))
        self.last_entropy = 0.0
        self.crash_warning = False
        self.embedding_dim = 3
        self.delay = 1
        self.window_size = 50

    def analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze topological structure of the market.
        Returns: {'status': 'STABLE'|'CRITICAL', 'score': float}
        """
        if len(df) < self.window_size:
            return {'status': 'WAITING', 'score': 0.0}

        # Extract close prices
        # We need a list of floats
        prices = df['close'].values.tolist()
        
        # Calculate Persistent Entropy (TDA Score)
        # Low Score (< 0.1) = Structure Collapse = CRASH RISK
        tda_score = holonic_speed.calculate_persistent_entropy(
            prices, 
            self.window_size, 
            self.embedding_dim, 
            self.delay
        )
        
        self.last_entropy = tda_score
        
        # Threshold Tuning:
        # Based on verification script: Sine Wave (Perfect Order) ~ 0.03
        # Noise ~ 0.7
        # Healthy Market ~ 0.4 - 0.6?
        # Crash Precursor ~ < 0.15 (Sudden Simplification)
        
        warning_threshold = getattr(config, 'TOPOLOGY_WARNING_THRESHOLD', 0.20)
        
        status = 'STABLE'
        if tda_score < warning_threshold:
            status = 'CRITICAL'
            self.crash_warning = True
            print(f"[{self.name}] 🚨 TOPOLOGY ALERT: Structure Collapse (Score {tda_score:.4f} < {warning_threshold})")
        else:
            self.crash_warning = False

        return {
            'status': status,
            'score': tda_score,
            'crash_warning': self.crash_warning
        }

    def get_health(self) -> dict:
        return {
            'status': 'ACTIVE',
            'last_score': f"{self.last_entropy:.4f}",
            'warning': self.crash_warning
        }
        
    def receive_message(self, sender: Any, content: Any) -> None:
        pass
