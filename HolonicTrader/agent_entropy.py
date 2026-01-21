"""
EntropyHolon - The Entropy Engine (Phase 3)

This agent acts as the 'risk manager' of the AEHML core.
It calculates Shannon Entropy on market returns to judge
market order vs. chaos.

High entropy means the market is too random to trade safely.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Any, Literal
import pandas as pd

from HolonicTrader.holon_core import Holon, Disposition


class EntropyHolon(Holon):
    """
    EntropyHolon is the 'Brain' that judges market order vs. chaos.
    It calculates Shannon Entropy on a returns series and classifies
    the market regime as ORDERED, CHAOTIC, or TRANSITION.
    """

    def __init__(self, name: str = "EntropyAgent"):
        # Initialize with default disposition
        default_disposition = Disposition(autonomy=0.7, integration=0.6)
        super().__init__(name=name, disposition=default_disposition)


    def calculate_shannon_entropy(self, returns_series: pd.Series) -> float:
        """
        Calculate Shannon Entropy using Rust Engine (Holonic Speed) if available.
        Fallback to Python/Scipy if not.
        """
        # Try Rust Path (100x Faster)
        try:
            import holonic_speed
            # Rust expects a flat list of floats
            # We must ensure data is clean (no NaNs/Infs) or Rust might panic/return NaN
            data = returns_series.dropna().values.tolist()
            if not data: return 0.0
            return float(holonic_speed.calculate_shannon_entropy(data))
            
        except ImportError:
            # Fallback to Legacy Python
            counts, bin_edges = np.histogram(returns_series, bins=10)
            total_count = counts.sum()
            if total_count == 0: return 0.0
            probabilities = counts / total_count
            return float(scipy_entropy(probabilities))
        except Exception as e:
            # Safety Net
            # print(f"Rust Entropy Error: {e}") 
            counts, bin_edges = np.histogram(returns_series, bins=10)
            total_count = counts.sum()
            if total_count == 0: return 0.0
            probabilities = counts / total_count
            return float(scipy_entropy(probabilities))

    def calculate_renyi_entropy(self, returns_series: pd.Series, alpha: float = 2.0) -> float:
        """
        Calculate Rényi Entropy using Rust Engine.
        """
        try:
            import holonic_speed
            data = returns_series.dropna().values.tolist()
            if not data: return 0.0
            return float(holonic_speed.calculate_renyi_entropy(data, alpha))
            
        except ImportError:
            # Fallback
            counts, _ = np.histogram(returns_series, bins=10)
            total_count = counts.sum()
            if total_count == 0: return 0.0
            probabilities = counts / total_count
            if np.isclose(alpha, 1.0):
                return self.calculate_shannon_entropy(returns_series)
            sum_p_alpha = np.sum(probabilities ** alpha)
            if sum_p_alpha == 0: return 0.0
            return float((1.0 / (1.0 - alpha)) * np.log(sum_p_alpha))

    def calculate_multiscale_entropy(self, returns_series: pd.Series, max_scale: int = 10, m: int = 2) -> list:
        """
        AEHML 2.0: Calculate Multiscale Entropy (MSE/RCMWPE).
        Returns a list of entropy values for scales 1 to max_scale.
        """
        try:
            import holonic_speed
            data = returns_series.dropna().values.tolist()
            if not data: return [0.0] * max_scale
            return holonic_speed.calculate_multiscale_entropy(data, max_scale, m)
        except Exception as e:
            print(f"[{self.name}] RCMWPE Error: {e}")
            # Fallback: Just return naive Shannon repeated or zeros
            return [self.calculate_shannon_entropy(returns_series)] * max_scale

    def calculate_permutation_entropy(self, returns_series: pd.Series, m: int = 3, delay: int = 1) -> float:
        """
        AEHML 2.0: Calculate Permutation Entropy (Structural Complexity).
        """
        try:
            import holonic_speed
            data = returns_series.dropna().values.tolist()
            if not data: return 0.0
            return float(holonic_speed.calculate_permutation_entropy(data, m, delay))
        except Exception as e:
            # Fallback? PE is hard to replicate simply in numpy without loop.
            return 0.0

    def determine_regime(self, entropy_value: float) -> Literal['ORDERED', 'CHAOTIC', 'TRANSITION']:
        """
        Determine market regime based on entropy value.
        
        THRESHOLDS (Calibrated Phase 34 - TUNED RELAXATION):
            ORDERED:    < 1.00 (Allowed for complex structure)
            CHAOTIC:    > 1.35 (Just below Gaussian Noise ~1.4)
            TRANSITION: 1.00 - 1.35
            
        Args:
            entropy_value: The calculated Shannon Entropy.

        Returns:
            'ORDERED' if entropy < 1.00
            'CHAOTIC' if entropy > 1.35
            'TRANSITION' otherwise
        """
        if entropy_value < 1.00:
            return 'ORDERED'
        elif entropy_value > 1.35:
            return 'CHAOTIC'
        else:
            return 'TRANSITION'

    def get_health(self) -> dict:
        """Report agent health status."""
        return {
            'status': 'OK',
            'last_entropy': 'N/A' # Could track last value
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            pass
        else:
            pass
