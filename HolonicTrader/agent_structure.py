"""
CTKSStrategicHolon - The "Institution of One"
(Structure Overrides Momentum)

Objective:
- Map "Stanfield Levels" (SLS) using Higher Timeframe (HTF) Data.
- Enforce BORSOG Protocol (Buy On Red, Sell On Green).
- Override Impulse/Momentum signals if they conflict with Structure.

Methodology:
- Tier 1: Weekly/Daily Trend & Levels.
- Tier 2: 4H Market Structure.
- Tier 3: Execution (handled by Oracle/Executor).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from HolonicTrader.holon_core import Holon, Disposition
import config

class CTKSStrategicHolon(Holon):
    def __init__(self, name: str = "StructureBoss"):
        # High Autonomy (It sets the rules), High Integration (Must be obeyed)
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.8))
        self.sls_levels = {} # {symbol: {'support': [], 'resistance': []}}
        self.htf_bias = {}   # {symbol: 'BULLISH' | 'BEARISH' | 'RANGING'}

    def receive_message(self, sender: Any, content: Any) -> None:
        """Process incoming messages (CTKS Protocol)."""
        pass # Currently pull-based architecture

    def get_structural_context(self, symbol: str, observer: Any) -> Dict[str, Any]:
        """
        Primary Output: Analyze HTF structure and return constraints.
        """
        # 1. Tier 1 Analysis (Daily/Weekly)
        # We need Observer to fetch HTF data.
        if not observer: 
            return {'structure_mode': 'UNKNOWN', 'sls_zone': 'NONE', 'bias': 'NEUTRAL'}

        try:
            # Fetch Daily Data (Tier 2/1 Hybrid for Crypto)
            df_daily = observer.fetch_market_data(timeframe='1d', limit=60, symbol=symbol)
            if df_daily.empty:
                return {'structure_mode': 'NO_DATA'}
            
            # 2. Map Stanfield Levels (SLS)
            # Simplified: Use Swing Highs/Lows and Volume Nodes
            supports, resistances = self._map_sls_levels(df_daily)
            self.sls_levels[symbol] = {'support': supports, 'resistance': resistances}
            
            current_price = df_daily['close'].iloc[-1]
            
            # 3. Determine Bias (Higher Highs / Lower Lows)
            bias = self._determine_bias(df_daily)
            self.htf_bias[symbol] = bias
            
            # 4. Calculate Pivot Points (New)
            pivots = self._calculate_pivots(df_daily)
            
            # 5. Check BORSOG Alignment
            # Are we at Support (Buy Zone) or Resistance (Sell Zone)?
            
            # Find nearest levels
            nearest_sup = max([s for s in supports if s < current_price], default=0)
            nearest_res = min([r for r in resistances if r > current_price], default=float('inf'))
            
            dist_to_sup = (current_price - nearest_sup) / current_price if nearest_sup > 0 else 1.0
            dist_to_res = (nearest_res - current_price) / current_price if nearest_res < float('inf') else 1.0
            
            # SLS Zone Definition (e.g. within 1.5% of level)
            zone_threshold = 0.015 
            
            sls_zone = 'NEUTRAL'
            if dist_to_sup < zone_threshold:
                sls_zone = 'SUPPORT'
            elif dist_to_res < zone_threshold:
                sls_zone = 'RESISTANCE'
                
            dist_to_sup_pct = (current_price - nearest_sup) / nearest_sup if nearest_sup > 0 else 0
            
            # Context Object
            ctx = {
                'structure_mode': 'Valid',
                'macro_trend': bias,
                'sls_zone': sls_zone,
                'nearest_support': nearest_sup,
                'nearest_resistance': nearest_res,
                'dist_to_sup_pct': -abs(dist_to_sup_pct), # Always negative/zero
                'dist_to_res_pct': dist_to_res,
                'pivots': pivots
            }
            
            pivot_str = f"| Piv: {pivots.get('P',0):.2f}" if pivots else ""
            print(f"[{self.name}] 🏛️ {symbol} Structure: {bias} | Zone: {sls_zone} | Sup: {nearest_sup:.2f}, Res: {nearest_res:.2f} {pivot_str}")
            return ctx

        except Exception as e:
            print(f"[{self.name}] Error analyzing structure: {e}")
            return {}

    def _map_sls_levels(self, df: pd.DataFrame) -> (List[float], List[float]):
        """
        Identify Significant Swing Points (Fractals).
        """
        highs = df['high']
        lows = df['low']
        
        # Simple Fractal Logic (5-candle)
        # High is higher than 2 previous and 2 next
        swing_highs = []
        swing_lows = []
        
        # We scan up to -3 to allow for '2 next' confirmation
        # Ideally we use a rolling window
        for i in range(2, len(df)-2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                swing_highs.append(highs.iloc[i])
                
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                swing_lows.append(lows.iloc[i])
                
        # Filter: Only keep levels that have multiple rejections?
        # For now, return recent significant ones (last 5)
        return sorted(swing_lows)[-5:], sorted(swing_highs)[-5:]

    def _determine_bias(self, df: pd.DataFrame) -> str:
        # Simple SMA Check + Structure
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        price = df['close'].iloc[-1]
        
        # Check Lower Highs / Higher Lows?
        # For robustness, use SMA alignment
        if price > sma50: return 'BULLISH'
        else: return 'BEARISH'

    def _calculate_pivots(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Standard, Fibonacci, and Camarilla Pivot Points.
        Uses the PREVIOUS completed day (iloc[-2]).
        """
        if len(df) < 2: return {}
        
        # Previous Day Data
        prev = df.iloc[-2]
        H = prev['high']
        L = prev['low']
        C = prev['close']
        
        # 1. Standard (Floor) Pivots
        P = (H + L + C) / 3
        R1 = (2 * P) - L
        S1 = (2 * P) - H
        R2 = P + (H - L)
        S2 = P - (H - L)
        R3 = H + 2 * (P - L)
        S3 = L - 2 * (H - P)
        
        # 2. Fibonacci Pivots
        range_hl = H - L
        Fib_R1 = P + (0.382 * range_hl)
        Fib_S1 = P - (0.382 * range_hl)
        Fib_R2 = P + (0.618 * range_hl)
        Fib_S2 = P - (0.618 * range_hl)
        Fib_R3 = P + (1.000 * range_hl)
        Fib_S3 = P - (1.000 * range_hl)
        
        # 3. Camarilla Pivots (Mean Reversion)
        Cam_R3 = C + (range_hl * 1.1 / 4)
        Cam_S3 = C - (range_hl * 1.1 / 4)
        Cam_R4 = C + (range_hl * 1.1 / 2) # Breakout
        Cam_S4 = C - (range_hl * 1.1 / 2) # Breakdown

        return {
            'P': P,
            'R1': R1, 'S1': S1,
            'R2': R2, 'S2': S2,
            'R3': R3, 'S3': S3,
            'Fib_R1': Fib_R1, 'Fib_S1': Fib_S1,
            'Fib_R2': Fib_R2, 'Fib_S2': Fib_S2,
            'Fib_R3': Fib_R3, 'Fib_S3': Fib_S3,
            'Cam_R3': Cam_R3, 'Cam_S3': Cam_S3,
            'Cam_R4': Cam_R4, 'Cam_S4': Cam_S4
        }
