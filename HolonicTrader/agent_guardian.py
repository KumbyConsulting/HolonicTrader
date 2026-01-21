"""
ExitGuardianHolon - The "Defense" Brain (Phase 16)

Specialized in:
1. Unrealized PnL Protection
2. Time-at-Risk Management
3. Volatility-based Trailing Stops
"""

from typing import Any, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config
import os
import json

class ExitGuardianHolon(Holon):
    def __init__(self, name: str = "ExitGuardian"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.4))
        self.last_exit_times = {} # {symbol: timestamp}
        self.trade_watermarks = {} # {symbol: {high: float, low: float}}
        self.partial_exits = {} # {symbol: True} if scalp taken

    def update_watermark(self, symbol: str, current_price: float, entry_price: float = None):
        """Update the High/Low watermark for true trailing."""
        if symbol not in self.trade_watermarks:
            # Initialize with Entry Price comparison to ensure we capture the full range
            high = max(current_price, entry_price) if entry_price else current_price
            low = min(current_price, entry_price) if entry_price else current_price
            self.trade_watermarks[symbol] = {'high': high, 'low': low}
        else:
            if current_price > self.trade_watermarks[symbol]['high']:
                self.trade_watermarks[symbol]['high'] = current_price
            if current_price < self.trade_watermarks[symbol]['low']:
                self.trade_watermarks[symbol]['low'] = current_price

    def reset_watermark(self, symbol: str):
         if symbol in self.trade_watermarks:
             del self.trade_watermarks[symbol]
         if symbol in self.partial_exits:
             del self.partial_exits[symbol]

    def _load_genome(self):
        """Load the latest evolved parameters from disk."""
        try:
            path = os.path.join(os.getcwd(), 'live_genome.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                     return json.load(f)
        except Exception:
            pass # Silent fail is fine, use defaults
        return None

    def manage_satellite_positions(self, symbol: str, current_price: float, entry_price: float, direction: Literal['BUY', 'SELL']):
        """
        Hit & Run Management for Satellite Assets.
        Breakeven at +1.5%, Take Profit 50% at +3%.
        """
        from .agent_executor import TradeSignal
        
        if entry_price <= 0: return None
        
        # Update Watermark
        self.update_watermark(symbol, current_price, entry_price)
        
        # PnL Calculation
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        # 1. Breakeven Trigger (Move SL to Entry + 0.1%)
        # Logic: If we are > 1.5% profit, we assume the actuator moves the SL.
        # But here, we simulate the "Close" if price drops back.
        # Ideally, actuator handles hard stops. Guardian handles 'decisions'.

        pnl_pct = (current_price - entry_price) / entry_price if direction == 'BUY' else (entry_price - current_price) / entry_price
        
        # Load Config
        genome = self._load_genome()
        if genome:
             stop_limit = genome.get('satellite_stop_loss', config.SATELLITE_STOP_LOSS)
             tp_target = genome.get('satellite_take_profit', config.SATELLITE_TAKE_PROFIT_1)
        else:
             stop_limit = config.SATELLITE_STOP_LOSS
             tp_target = config.SATELLITE_TAKE_PROFIT_1
             
        # 1. Take Profit (Sniper Moonshot)
        if pnl_pct >= tp_target:
             print(f"[{self.name}] 🎯 SATELLITE SNIPER HIT: {symbol} (+{pnl_pct*100:.2f}%)")
             self.reset_watermark(symbol)
             return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'SATELLITE_TP'})

        # 2. Breakeven (Safety) - Static for now, or could evolve
        # be_trigger = config.SATELLITE_BREAKEVEN_TRIGGER
        # ... logic for BE is complex state, keeping simple for now ...
             
        # 3. Hard Stop (Genome Evolved)
        # Note: stop_limit is positive number (e.g. 0.05), so we check <= -0.05
        if pnl_pct <= -stop_limit:
             print(f"[{self.name}] 💥 SATELLITE STOP LOSS (EVO): {symbol} ({pnl_pct*100:.2f}%)")
             self.reset_watermark(symbol)
             return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'SATELLITE_SL'})
             
        # 4. RSI Exit? (Optional, if in genome)
        # genome['rsi_sell'] check would require passing Indicators to Guardian.
        # For now, we rely on PnL exits mainly for robustness.
        
        return None

    def analyze_for_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        bb: dict,
        atr: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR'],
        position_age_hours: float = 0.0,
        direction: Literal['BUY', 'SELL'] = 'BUY',
        regime: str = 'MICRO'
    ):
        
        if symbol in config.SATELLITE_ASSETS:
            return self.manage_satellite_positions(symbol, current_price, entry_price, direction)
        # -----------------------------------

        from .agent_executor import TradeSignal
        
        if entry_price <= 0:
            return None
            
        # Update Watermark (Initialize with Entry if needed)
        self.update_watermark(symbol, current_price, entry_price)
            
        # PnL Calculation
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else: # SELL (Short)
            pnl_pct = (entry_price - current_price) / entry_price
            
        # --- VOL-WINDOW EXIT LOGIC ---
        if regime == 'VOL_WINDOW':
            # Tighter trailing stop: 1.5 ATR (High Turnover)
            if atr > 0:
                watermark = self.trade_watermarks.get(symbol, {'high': current_price, 'low': current_price})
                trail_dist = atr * 1.5 
                
                if direction == 'BUY':
                    stop_price = watermark['high'] - trail_dist
                    if current_price <= stop_price:
                         print(f"[{self.name}] ⚡ VOL_WINDOW TRAIL HIT: {symbol} @ {current_price:.4f} (High {watermark['high']:.4f})")
                         self.reset_watermark(symbol)
                         return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata={'reason': 'VOL_TRAIL'})
                else: # SELL
                    stop_price = watermark['low'] + trail_dist
                    if current_price >= stop_price:
                         print(f"[{self.name}] ⚡ VOL_WINDOW TRAIL HIT: {symbol} @ {current_price:.4f} (Low {watermark['low']:.4f})")
                         self.reset_watermark(symbol)
                         return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata={'reason': 'VOL_TRAIL'})
            
            # Hard Stop (1%)
            if pnl_pct <= -0.01:
                 print(f"[{self.name}] ⚡ VOL_WINDOW HARD STOP: {symbol} ({pnl_pct*100:.2f}%)")
                 self.reset_watermark(symbol)
                 return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'VOL_STOP'})
                 
            return None # Skip standard logic
        # -----------------------------

        
        # 1. HARD STOP LOSS (Circuit Breaker)
        sl_target = config.SCAVENGER_STOP_LOSS if metabolism_state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
        if pnl_pct <= -sl_target:
            print(f"[{self.name}] 🚨 EMERGENCY EXIT: {symbol} ({direction}) PnL {pnl_pct*100:.2f}%")
            self.reset_watermark(symbol)
            return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)

        if metabolism_state == 'SCAVENGER':
            # Scavenger Exits: Quick Mean Reversion
            if position_age_hours >= 12.0:
                print(f"[{self.name}] ⏳ TIME EXIT: {symbol} (12h reached)")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
            
            if pnl_pct >= config.SCAVENGER_SCALP_TP:
                if not self.partial_exits.get(symbol, False):
                    print(f"[{self.name}] 💰 SCALP-TO-PYRAMID: {symbol} (+{pnl_pct*100:.2f}%) -> Banking 50% House Money")
                    self.partial_exits[symbol] = True
                    # Do NOT reset watermark - let the runner trail
                    return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=0.5, price=current_price, metadata={'reason': 'PARTIAL_TP'})
                else:
                    # Already scalped - Let it ride! (Handled by Trailing Stop or Time Exit)
                    pass
                
            # Mean Reversion: Hit BB Middle
            if direction == 'BUY' and current_price >= bb['middle']:
                print(f"[{self.name}] 🔄 MEAN REVERSION (Long): {symbol} hit BB Middle")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
            elif direction == 'SELL' and current_price <= bb['middle']:
                print(f"[{self.name}] 🔄 MEAN REVERSION (Short): {symbol} hit BB Middle")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)

        else: # PREDATOR
            # Predator Exits: Momentum Following
            if position_age_hours >= 24.0:
                print(f"[{self.name}] ⏳ TREND EXPIRY: {symbol} (24h reached)")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
                
            if pnl_pct >= config.PREDATOR_TAKE_PROFIT:
                print(f"[{self.name}] 💰 PREDATOR TP: {symbol} (+{pnl_pct*100:.2f}%)")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
            
            # --- ADAPTIVE TRAILING STOP (High-Watermark) ---
            if atr > 0:
                watermark = self.trade_watermarks[symbol]
                
                if direction == 'BUY':
                    # Trail from Highest High
                    highest = watermark['high']
                    # Use multiplier (default 2.0 ATR)
                    trail_dist = atr * config.PREDATOR_TRAILING_STOP_ATR_MULT
                    trailing_stop = highest - trail_dist
                    
                    # Ensure Stop never moves DOWN (only ratchets up) - implicitly handled by high watermark
                    # But we also must cap it at entry if watermark is low (breakeven logic handled elsewhere)
                    
                    if current_price <= trailing_stop:
                        print(f"[{self.name}] 🛡️ ADAPTIVE STOP HIT (Long): {symbol} @ {current_price:.4f} (High: {highest:.4f}, Trail: {trail_dist:.4f})")
                        self.reset_watermark(symbol)
                        return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
                        
                else: # SELL
                    # Trail from Lowest Low
                    lowest = watermark['low']
                    trail_dist = atr * config.PREDATOR_TRAILING_STOP_ATR_MULT
                    trailing_stop = lowest + trail_dist
                    
                    if current_price >= trailing_stop:
                        print(f"[{self.name}] 🛡️ ADAPTIVE STOP HIT (Short): {symbol} @ {current_price:.4f} (Low: {lowest:.4f}, Trail: {trail_dist:.4f})")
                        self.reset_watermark(symbol)
                        return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)

        return None


    def check_liquidity_health(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, order_book: dict) -> str:
        """
        Analyze order book depth to ensure safe exit.
        direction: 'SELL' to exit Long, 'BUY' to cover Short.
        """
        if quantity <= 0 or not order_book: return "UNKNOWN"
        
        # If exiting Long (SELL), we look at BIDS (buyers)
        # If covering Short (BUY), we look at ASKS (sellers)
        side = 'bids' if direction == 'SELL' else 'asks'
        book_side = order_book.get(side, [])
        
        if not book_side: return "NO_LIQUIDITY"
        
        # Calculate cumulative liquidity within ~0.5% slippage
        best_price = book_side[0][0]
        slippage_limit = best_price * 0.005 # 0.5% tolerance
        
        available_vol = 0.0
        weighted_price_sum = 0.0
        
        for price, vol in book_side:
            if direction == 'SELL':
                if price < (best_price - slippage_limit): break
            else:
                if price > (best_price + slippage_limit): break
                
            available_vol += vol
            weighted_price_sum += (price * vol)
            
            if available_vol >= quantity * 1.5: # Found enough + 50% buffer
                break
                
        if available_vol < quantity:
            return "CRITICAL_ILLIQUIDITY" # Can't fill without massive slippage
        elif available_vol < quantity * 1.5:
            return "WARNING_THIN_BOOK"
        
        return "HEALTHY"

    def record_exit(self, symbol: str, timestamp: Any):
        self.last_exit_times[symbol] = timestamp
        # Clean up watermarks
        if symbol in self.trade_watermarks:
            del self.trade_watermarks[symbol]

    def get_health(self) -> dict:
        return {
            'status': 'OK',
            'exits_tracked': len(self.last_exit_times)
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
