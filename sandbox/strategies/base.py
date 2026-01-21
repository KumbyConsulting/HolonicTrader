from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import pandas as pd

@dataclass
class Signal:
    """Standardized output for a strategy decision."""
    direction: Literal['BUY', 'SELL', 'HOLD']
    size: float = 1.0  # Percentage of capital/stack to use (0.0 - 1.0)
    strength: float = 0.0 # Signal conviction/magnitude (e.g. -2.0 to 2.0)
    reason: str = "Strategy"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

class Strategy(ABC):
    """
    Abstract Base Class for Holonic Strategies.
    Implement `on_candle` to define your logic.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.position = None # Track current position state if needed locally

    @abstractmethod
    def on_candle(self, 
                  slice_df: pd.DataFrame, 
                  indicators: Dict[str, Any], 
                  portfolio_state: Dict[str, Any],
                  secondary_slice_df: Optional[pd.DataFrame] = None) -> Signal:
        """
        Called on every candle close.
        
        Args:
            slice_df: DataFrame containing price history up to current candle.
            indicators: Pre-calculated indicators for the current candle (RSI, BB, etc).
            portfolio_state: Dict with 'balance', 'inventory', 'avg_entry'.
            secondary_slice_df: Optional slice of lower-timeframe data (e.g. 15m) for satellite checks.
            
        Returns:
            Signal object dictating action.
        """
        pass
    
    def set_position(self, position):
        """Update internal position tracking (optional hook)."""
        self.position = position
