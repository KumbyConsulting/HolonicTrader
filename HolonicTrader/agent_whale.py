"""
WhaleHolon - The "Hunter" Brain (Phase X)

Specialized in:
1. Detecting massive Bid/Ask Walls (Whale intent).
2. "Whale-Scalper" Strategy execution.
3. High-frequency 1-minute time checks.
"""

from typing import Any, Dict, List, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config
import time

class WhaleHolon(Holon):
    def __init__(self, name: str = "WhaleHunter"):
        # Medium Autonomy, High Integration (Works closely with Oracle/Actuator)
        super().__init__(name=name, disposition=Disposition(autonomy=0.6, integration=0.8))
        self.last_scan_time = 0.0
        self.active_whales = {} # {symbol: {price: float, side: 'bid'|'ask', size: float}}

    def check_bid_wall(self, symbol: str, depth: Dict, daily_vol: float = 0.0) -> Optional[Dict]:
        """
        Check for a "Bid Wall" condition:
        - Wall Size >= 0.5% of 24h Volume (Dynamic)
        - Floor: $50,000 (To avoid noise in illiquid pairs)
        - Wall Price <= Mid Price + 0.2% (Close to market)
        
        Returns wall details if found, else None.
        """
        if not depth or 'bids' not in depth:
            return None
            
        bids = depth['bids']
        if not bids: return None
        
        best_bid = bids[0][0]
        
        # 1. Dynamic Threshold Logic
        # If daily_vol is provided (USD notional), use 0.5% of it.
        # Fallback to $500k if 0.
        
        if daily_vol > 0:
            wall_threshold_usd = max(50000.0, daily_vol * 0.005) # 0.5% or $50k min
        else:
            wall_threshold_usd = 500000.0 # Legacy fallback
            
        # print(f"[{self.name}] 🔍 Debug {symbol}: 24hVol=${daily_vol:,.0f} -> WallThresh=${wall_threshold_usd:,.0f}")
        
        for price, vol in bids[:15]:
            notional = price * vol
            
            if notional >= wall_threshold_usd:
                # 2. Distance Check
                # Is it close? (Within 0.5%)
                dist = (best_bid - price) / best_bid
                if dist <= 0.005: 
                    return {
                        'type': 'BID_WALL',
                        'price': price,
                        'vol': vol,
                        'notional': notional,
                        'distance': dist,
                        'threshold_used': wall_threshold_usd
                    }
                    
        return None

    def check_scout_entry(self, symbol: str, observer_data: Dict) -> Optional[Dict]:
        """
        Main Logic Check.
        Called by Trader during 1m loop.
        """
        # 1. Depth Check
        depth = observer_data.get(f"depth_{symbol}")
        curr_price = observer_data.get('price', 0.0)
        
        if not depth or curr_price == 0:
            return None
            
        wall = self.check_bid_wall(symbol, depth)
        
        if wall:
            print(f"[{self.name}] 🐋 WHALE DETECTED on {symbol}: ${wall['notional']:,.0f} Bid Wall @ {wall['price']}")
            
            # CONFIRMATION: Price Action must be bouncing off it or holding
            # For now, we signal entry if we are just above it (front-run the wall)
            
            # Setup: BUY just above wall
            entry_price = wall['price'] * 1.001 # +0.1% front-run
            
            if curr_price <= entry_price * 1.002: # Ensure we haven't missed it
                return {
                    'signal': 'BUY',
                    'symbol': symbol,
                    'price': curr_price,
                    'reason': f"Whale Wall Support (${wall['notional']/1000:.0f}k)",
                    'stop_loss': wall['price'] * 0.995, # tight stop below wall
                    'target': curr_price * 1.015 # 1.5% scalp target
                }
                
        return None

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
