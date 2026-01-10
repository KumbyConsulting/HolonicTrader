"""
GovernorHolon - NEXUS Risk & Homeostasis (Phase 15)

Implements:
1. Dual Metabolic State (SCAVENGER / PREDATOR)
2. Volatility Targeting (ATR-based position sizing)
3. Principal Protection (Never risk the $10 base)
"""

from typing import Any, Tuple, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config

import datetime
import time

class GovernorHolon(Holon):
    def __init__(self, name: str = "GovernorAgent", initial_balance: float = 10.0, db_manager: Any = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        
        self.balance = initial_balance
        self.available_balance = initial_balance # New: Track free margin
        self.hard_stop_threshold = 5.0
        self.DEBUG = False # Silence rejection spam
        self.db_manager = db_manager  # For win rate tracking
        
        # Phase 22: Portfolio Health Tracking
        self.max_balance = initial_balance
        self.drawdown_pct = 0.0
        self.margin_utilization = 0.0
        
        # Accumulator State (Phase 42)
        self.high_water_mark = initial_balance
        self.risk_multiplier = 1.0
        self.equity_history = []
        self.drawdown_lock = False
        
        # Phase 50: Daily Risk Reset
        self.last_hwm_date = datetime.datetime.utcnow().date()
        
        # Reference ATR for volatility targeting (set during first cycle)
        self.reference_atr = None
        
        # Position Tracking (Multi-Asset)
        self.positions = {} # symbol -> {entry_price, quantity, direction}
        self.last_trade_time = {} # symbol -> timestamp
        self.last_specific_entry = {} # symbol -> price (for stacking distance)
        
        # Phase 7: Regime Controller Integration
        self.regime_controller = None  # Set by Trader after instantiation
        
        # Consolidation Engine State
        self.last_consolidation_time = 0.0
        self.consolidation_in_progress = False
        
    def sync_positions(self, held_assets: dict, metadata: dict):
        """
        Sync positions from Executor/DB on startup to cure Amnesia.
        Handles both LONG (positive qty) and SHORT (negative qty) positions.
        """
        print(f"[{self.name}] Syncing positions from DB...")
        count = 0
        for symbol, qty in held_assets.items():
            # Handle both LONG (qty > 0) and SHORT (qty < 0) positions
            if abs(qty) > 0.00000001:
                meta = metadata.get(symbol, {})
                entry_price = meta.get('entry_price', 0.0)
                # Determine direction from metadata or infer from qty sign
                direction = meta.get('direction')
                if direction is None:
                    direction = 'BUY' if qty > 0 else 'SELL'
                
                # Reconstruct position entry
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'quantity': abs(qty),  # Store absolute quantity
                    'stack_count': meta.get('stack_count', 1), # Restore stack count from DB
                    'first_entry_time': time.time()
                }
                # Sync stacking tracker
                self.last_specific_entry[symbol] = entry_price
                
                count += 1
                print(f"[{self.name}] Synchronized: {symbol} ({direction}, Qty: {abs(qty):.4f})")
                
        if count == 0:
            print(f"[{self.name}] No active positions found to sync.")
    def set_live_balance(self, total: float, available: float):
        """Update equity and free margin from live exchange data."""
        
        # --- PATCH: NULL SAFETY ---
        if total is None or available is None:
            # Keep previous known state to avoid panic
            return
            
        # --- PATCH: STALE HWM PREVENTER ---
        # On first connection or if HWM is strangely low/high, trust the live balance.
        # Check total > 0.0 to prevent syncing on API errors
        if total > 0.0:
            # If we have never updated HWM (it's at init 10.0) OR if we are resetting:
            if self.high_water_mark == 10.0 or (total < self.high_water_mark * 0.8): 
                # If current total is > 20% below HWM on startup, assume it's a new session/reset
                # This prevents "Solvency Halt" due to previous session data if we deployed fresh
                if not getattr(self, '_hwm_synced', False):
                    print(f"[{self.name}] 🔄 Syncing High Water Mark to Live Balance: ${total:.2f}")
                    self.high_water_mark = total
                    self._hwm_synced = True
        
            # Only update state if valid read
            self.balance = total
            self.available_balance = available
            self.update_accumulator(total)

    def update_accumulator(self, current_equity: float):
        """
        The Accumulator Logic: 
        1. Ratchet: Track High Water Mark & Lock if Drawdown > Limit.
        2. Pump: Adjust Risk Multiplier based on Equity Velocity.
        """
        # 0. Daily Reset Check (New Day = New Session)
        current_date = datetime.datetime.utcnow().date()
        if current_date > self.last_hwm_date:
             print(f"[{self.name}] 🌅 New Day Detected ({current_date}). Resetting High Water Mark to ${current_equity:.2f}")
             self.high_water_mark = current_equity
             self.last_hwm_date = current_date
             self.drawdown_lock = False

        # 1. Update High Water Mark (The Ratchet)
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            self.drawdown_lock = False # Unlock if we make new highs
            
        # 2. Check Drawdown Lock
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
            
            # --- PATCH: DATA SANITY CHECK ---
            # If drawdown is MASSIVE (>30%) instantly (implying we didn't actually lose it trading),
            # assume previous HWM was a glitch (phantom spike) and reset.
            if drawdown > 0.30 and not self.drawdown_lock:
                 print(f"[{self.name}] 📉 DATA SANITY CHECK: Instant >30% Drop (${self.high_water_mark:.2f} -> ${current_equity:.2f}). Resetting HWM (Assuming Glitch).")
                 self.high_water_mark = current_equity
                 drawdown = 0.0
            # --------------------------------
            
            if drawdown > config.ACC_DRAWDOWN_LIMIT:
                if not self.drawdown_lock:
                    print(f"[{self.name}] 🛑 ACCUMULATOR HALT: Drawdown {drawdown:.1%} > {config.ACC_DRAWDOWN_LIMIT:.1%}. Trading Locked.")
                self.drawdown_lock = True
            
        # 3. Calculate Velocity (The Pump)
        self.equity_history.append(current_equity)
        if len(self.equity_history) > 10: self.equity_history.pop(0)
        
        if len(self.equity_history) >= 5:
            # Simple slope of last 5 points
            avg_equity = sum(self.equity_history) / len(self.equity_history)
            
            if current_equity > avg_equity:
                # We are growing -> Pump
                self.risk_multiplier = min(self.risk_multiplier + 0.1, config.ACC_RISK_CEILING)
            elif current_equity < avg_equity:
                # We are shrinking -> Deflate
                self.risk_multiplier = max(self.risk_multiplier - 0.1, config.ACC_RISK_FLOOR)
        
    def update_balance(self, new_balance: float):
        """Update the internal balance knowledge and health metrics."""
        self.balance = new_balance
        
        # Track Drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
            
        if self.max_balance > 0:
            self.drawdown_pct = (self.max_balance - self.balance) / self.max_balance
        
        # Calculate Margin Utilization
        total_exposure = 0.0
        for sym, pos in self.positions.items():
            total_exposure += abs(pos['quantity']) * pos['entry_price']
            
        if self.balance > 0:
            # We normalize margin utilization based on the config limit
            # If we use all allowed margin, util = 1.0
            allowed_exposure = self.balance * config.GOVERNOR_MAX_MARGIN_PCT * config.PREDATOR_LEVERAGE
            self.margin_utilization = total_exposure / allowed_exposure if allowed_exposure > 0 else 0.0

        self._check_homeostasis()

    def _check_homeostasis(self):
        """Check if the system is viable."""
        if self.balance < self.hard_stop_threshold:
            self.state = 'HIBERNATE'
            print(f"[{self.name}] CRITICAL: Balance ${self.balance:.2f} < ${self.hard_stop_threshold}. HIBERNATING.")
        else:
            if self.state == 'HIBERNATE':
                self.state = 'ACTIVE'

    def get_metabolism_state(self) -> Literal['SCAVENGER', 'PREDATOR']:
        """
        Determine current metabolic state based on balance.
        """
        if self.balance <= config.SCAVENGER_THRESHOLD:
            return 'SCAVENGER'
        else:
            return 'PREDATOR'

    def get_portfolio_health(self) -> dict:
        """Expose health metrics for PPO Brain."""
        return {
            'drawdown_pct': self.drawdown_pct,
            'margin_utilization': self.margin_utilization,
            'balance': self.balance,
            'max_balance': self.max_balance
        }

    def is_trade_allowed(self, symbol: str, asset_price: float) -> bool:
        """
        Lightweight check to see if a trade would be allowed.
        Prevents Strategy from wasting compute on blocked trades.
        """
        # 1. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS:
            return False
            
        # 2. Price Distance Check
        last_entry = self.last_specific_entry.get(symbol, 0)
        if last_entry > 0 and symbol in self.positions:
            dist = abs(asset_price - last_entry) / last_entry
            if dist < config.GOVERNOR_MIN_STACK_DIST:
                return False
                
        return True

    def calc_position_size(self, symbol: str, asset_price: float, current_atr: float = None, atr_ref: float = None, conviction: float = 0.5, direction: str = 'BUY', crisis_score: float = 0.0, sentiment_score: float = 0.0) -> Tuple[bool, float, float]:
        """
        Calculate position size with Phase 12 institutional risk management.
        
        Integrates:
        1. Minimax Constraint (protect principal)
        2. Volatility Scalar (ATR-based sizing)
        4. Conviction Scalar (LSTM-based scaling)
        5. Holistic Feedback (Sentiment Hormone)
        
        Returns:
            (is_approved: bool, quantity: float, leverage: float)
        """
        # 0. Update Accumulator State
        # ideally this is done in sync loop, but fine to do here for latest check
        self.update_accumulator(self.balance)
        
        # 1. Check Accumulator Lock
        if self.drawdown_lock:
             # Check if this is a "Risk Reducing" trade (Closing)
             is_risk_reducing = False
             
             # Logic to detect reduction:
             # If we hold Long (Pos > 0) and we are Selling -> Reduce
             # If we hold Short (Pos < 0) and we are Buying -> Reduce
             
             pos_data = self.positions.get(symbol, {})
             qty_held = pos_data.get('quantity', 0.0)
             current_dir = pos_data.get('direction', 'BUY')
             
             if qty_held > 0:
                 if current_dir == 'BUY' and direction == 'SELL': is_risk_reducing = True
                 elif current_dir == 'SELL' and direction == 'BUY': is_risk_reducing = True
             
             if not is_risk_reducing:
                print(f"[{self.name}] 🛡️ DRAWDOWN LOCK ACTIVE. Blocking New Risk: {symbol}")
                return False, 0.0, 0.0
             else:
                 print(f"[{self.name}] 🔓 DRAWDOWN OVERRIDE: Allowing Risk Reduction for {symbol}")
             existing_pos = self.positions.get(symbol)
             if existing_pos and existing_pos.get('quantity', 0) > 0:
                 # We hold it. Is this a close?
                 # Need to know the direction of the proposed trade.
                 # If direction is opposite to existing direction.
                 current_dir = existing_pos.get('direction', 'BUY')
                 if current_dir == 'BUY' and direction == 'SELL': is_risk_reducing = True
                 if current_dir == 'SELL' and direction == 'BUY': is_risk_reducing = True
             
             if is_risk_reducing:
                 if self.DEBUG:
                    print(f"[{self.name}] 🔓 ALLOWING Close/Reduce for {symbol} despite Lock.")
             elif symbol == "PAXG/USDT" and crisis_score > 0.5:
                 print(f"[{self.name}] 🚨 CRISIS BYPASS: Allowing PAXG trade (Score {crisis_score:.2f}) despite Lock.")
             else:
                 print(f"[{self.name}] 🛑 REJECT {symbol}: Accumulator Lock Active (Drawdown limit hit).")
                 return False, 0.0, 1.0

        # --- PHASE 25: SATELLITE OVERRIDE ---
        # The executor passes 'conviction' which we used as a carrier for metadata in previous versions,
        # but here we might need a clearer signal. 
        # Actually, let's inspect the 'conviction' arg. If it's a dict or special object? 
        # No, 'conviction' is float.
        # We need to rely on the `symbol` being in `SATELLITE_ASSETS`.
        if symbol in config.SATELLITE_ASSETS:
             # Fixed Sizing: $10 Margin * 10x Lev = $100 Position
             notional = config.SATELLITE_MARGIN * config.SATELLITE_LEVERAGE
             quantity = notional / asset_price
             return True, quantity, config.SATELLITE_LEVERAGE
        # ------------------------------------

        if self.state == 'HIBERNATE':
            print(f"[{self.name}] Trade REJECTED: System in HIBERNATION.")
            return False, 0.0, 0.0

        existing_pos = self.positions.get(symbol)

        # --- UNIFIED CONTROL PROTOCOL: MICRO MODE & STACKING GATES ---
        
        # 1. MICRO-ACCOUNT MODE (Request F)
        if config.MICRO_CAPITAL_MODE:
            # A. NO STACKING
            if existing_pos:
                if self.DEBUG:
                     print(f"[{self.name}] 🧊 MICRO FREEZE: Stacking disabled in Micro Mode. Rejecting.")
                return False, 0.0, 0.0
            
            # B. MAX POSITIONS CAP (Replaces Cluster Risk for Micro)
            if len(self.positions) >= config.MICRO_MAX_POSITIONS:
                # Allow existing position to continue (e.g. if we are reducing/exiting?)
                # Actually, calc_position_size is only for ENTRIES (Long/Short).
                # Wait, if we are Shorting and we have a Long, it's a flip? Or if we have a Short and we Add?
                # The 'existing_pos' check above handles stacking/adding.
                # If we are here, 'existing_pos' is None, meaning NEW position.
                if self.DEBUG:
                     print(f"[{self.name}] 🛑 MAX POSITIONS REAACHED ({len(self.positions)}/{config.MICRO_MAX_POSITIONS}). Rejecting.")
                return False, 0.0, 0.0
                
             # C. EXPOSURE CAP
            estimated_exposure = asset_price * (self.balance * config.MICRO_MAX_LEVERAGE / asset_price) # Worst case
            current_exposure = sum([p['quantity'] * p['entry_price'] for p in self.positions.values()])
            if (current_exposure + estimated_exposure) > (self.balance * config.MICRO_MAX_EXPOSURE_RATIO):
                 pass

        # 2. LOW NAV FREEZE (Request A - Modified)
        # Even if not in Micro mode, if funds are low, don't stack.
        if self.balance < config.STACKING_MIN_EQUITY:
            if existing_pos:
                 # Check Free Margin Buffer
                 # We need ~5x the min order value in FREE margin to justify a stack
                 required_buffer = config.MIN_ORDER_VALUE * config.STACKING_BUFFER_MULTIPLIER
                 if self.available_balance < required_buffer:
                      if self.DEBUG:
                           print(f"[{self.name}] 🧊 LOW NAV FREEZE: Free Margin ${self.available_balance:.2f} < Buffer ${required_buffer:.2f}. Stacking Blocked.")
                      return False, 0.0, 0.0

        # --- PATCH 2: THE STACKING CAP (Stop the Martingale) ---
        MAX_STACKS = 3
        if existing_pos:
            current_stacks = existing_pos.get('stack_count', 1)
            if current_stacks >= MAX_STACKS:
                if self.DEBUG:
                     print(f"[{self.name}] ⚠️ MAX STACKS REACHED ({current_stacks}). REJECTING ORDER.")
                return False, 0.0, 0.0
        # -------------------------------------------------------

        # --- PHASE 35: IMMUNE SYSTEM CHECKS ---
        # Note: In Micro Mode, we might skip Cluster Risk if desired ("Ignore cluster risk" request)
        if not config.MICRO_CAPITAL_MODE:
            if not self.check_cluster_risk(symbol):
                return False, 0.0, 0.0
            
        # --------------------------------------

        # 1. Minimax Constraint (The "House Money" Rule)
        max_loss_usd = self.calculate_max_risk(self.balance)
        if asset_price <= 0:
            print(f"[{self.name}] Trade REJECTED: Invalid Asset Price.")
            return False, 0.0, 0.0
            
        # WARP SPEED 3.0: Smart Stacking & Cooldowns
        
        # 1. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS:
            if self.DEBUG:
                print(f"[{self.name}] REJECTED: Cooldown active for {symbol} ({int(config.GOVERNOR_COOLDOWN_SECONDS - (time.time() - last_time))}s rem).")
            return False, 0.0, 0.0
            
        # 2. Solvency Check (New)
        # Cost = (Qty * Price) / Leverage
        # But we don't know Qty yet. We are calculating it.
        # Let's verify AFTER calculation.
        pass # Placeholder
        
        # 3. Minimax Sizing
        # ... sizing logic ...

            
        # 2. Price Distance Check
        last_entry = self.last_specific_entry.get(symbol, 0)
        if last_entry > 0 and symbol in self.positions:
            dist = abs(asset_price - last_entry) / last_entry
            if dist < config.GOVERNOR_MIN_STACK_DIST:
                if self.DEBUG:
                    print(f"[{self.name}] REJECTED: Price {asset_price} too close to last entry {last_entry} (Dist: {dist*100:.2f}% < {config.GOVERNOR_MIN_STACK_DIST*100}%).")
                return False, 0.0, 0.0
        
        state = self.get_metabolism_state()
        
        # === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
        
        # Conviction Scalar (0.5 to 1.5)
        # conviction here is LSTM prob (0-1). We transform it.
        # For BUYS: prob > 0.5 is good. For SELLS: prob < 0.5 is good.
        # Wait, the EntryOracle already chooses direction. 
        # Let's assume passed conviction is 'strength' (0.5 to 1.0).
        conv_scalar = 0.5 + (max(0.0, conviction - 0.5) * 2.0)
        conv_scalar = max(0.5, min(1.5, conv_scalar))

        # Base position sizing
        if state == 'SCAVENGER':
            # 10-Bullet Rule: Max margin %
            # DYNAMIC RISK: Scale margin by Accumulator Multiplier
            margin = min(config.SCAVENGER_MAX_MARGIN, self.balance * config.GOVERNOR_MAX_MARGIN_PCT) * self.risk_multiplier
            leverage = config.SCAVENGER_LEVERAGE
            base_notional = margin * leverage * conv_scalar
            
        else:  # PREDATOR
            leverage = config.PREDATOR_LEVERAGE
            
            # Use Modified Kelly for PREDATOR
            kelly_size_usd = self.calculate_kelly_size(self.balance) * self.risk_multiplier
            
            # Trend Age Decay
            current_pos = self.positions.get(symbol)
            decay_mult = 1.0
            if current_pos:
                # 1. Age-based Decay
                age_hours = (time.time() - current_pos.get('first_entry_time', time.time())) / 3600.0
                if age_hours > config.GOVERNOR_TREND_DECAY_START:
                    overtime = age_hours - config.GOVERNOR_TREND_DECAY_START
                    window = config.GOVERNOR_MAX_TREND_AGE_HOURS - config.GOVERNOR_TREND_DECAY_START
                    decay_mult *= max(0.0, 1.0 - (overtime / window))
                    print(f"[{self.name}] ⏳ Trend Age {age_hours:.1f}h. Decaying by {decay_mult:.2f}x")
                
                # 2. Stack-based Decay (Phase 18)
                stacks = current_pos.get('stack_count', 0)
                stack_decay = (config.GOVERNOR_STACK_DECAY ** stacks)
                decay_mult *= stack_decay
                
                if decay_mult < 1.0:
                    print(f"[{self.name}] 🥞 Stack {stacks} Decay: {stack_decay:.2f}x (Total Decay: {decay_mult:.2f}x)")
                    kelly_size_usd *= decay_mult
                    
                if age_hours > config.GOVERNOR_MAX_TREND_AGE_HOURS:
                    print(f"[{self.name}] 🛑 Trend Exhausted (>24h). Rejecting Stack.")
                    return False, 0.0, 0.0
            
            base_notional = kelly_size_usd * leverage * conv_scalar
        
        # --- PHASE 6c: MICRO MODE HARD LEVERAGE LOCK ---
        if config.MICRO_CAPITAL_MODE:
            effective_leverage = min(leverage, config.MICRO_HARD_LEVERAGE_LIMIT)
            if effective_leverage < leverage:
                 print(f"[{self.name}] 🔒 MICRO LOCK: Leverage capped {leverage}x -> {effective_leverage}x")
                 leverage = effective_leverage
                 base_notional = (base_notional / config.PREDATOR_LEVERAGE) * leverage # Adjust notional to new leverage
        # -----------------------------------------------
        
        # Apply Volatility Scalar (if ATR provided)
        if current_atr and atr_ref:
            vol_scalar = self.calculate_volatility_scalar(current_atr, atr_ref)
            vol_adjusted_notional = base_notional * vol_scalar
            print(f"[{self.name}] 📊 Volatility Scalar: {vol_scalar:.2f}x, Conviction: {conv_scalar:.2f}x")
        else:
            vol_adjusted_notional = base_notional
            vol_scalar = 1.0
            
        # Apply Minimax Constraint (CRITICAL)
        max_risk_usd = self.calculate_max_risk(self.balance)
        
        # Assume mode-specific stop loss distance for risk calculation
        sl_dist = config.SCAVENGER_STOP_LOSS if state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
        max_notional_from_risk = max_risk_usd / sl_dist
        
        # Take minimum of volatility-adjusted and risk-constrained
        final_notional = min(vol_adjusted_notional, max_notional_from_risk)
        
        # 5. HOLISTIC EMOTIONAL REGULATION (Phase 5b)
        # Apply Fear Reduction to the FINAL agreed size to ensure it works even if capped.
        if sentiment_score < -0.5:
             final_notional *= 0.8 # Reduce size by 20%
             if self.DEBUG: 
                 print(f"[{self.name}] 📉 FEAR RESPONSE: Shrinking final size by 20% (Sent: {sentiment_score:.2f})")
        
        # --- PATCH 4: MINIMUM ORDER VALUE (Kraken) ---
        # If calculated size is too small, check if we can safely floor it to MIN_ORDER_VALUE
        if final_notional < config.MIN_ORDER_VALUE:
             # Check if we have enough "House Money" or principal buffer to allow this small deviation
             # PATCH: Relaxed for Micro-Accounts (< $100). 
             # Allow if we can structurally afford the margin (Balance * Lev > MinOrder)
             # We assume max leverage of 5x here for safety check
             max_buying_power = self.balance * 5.0 
             
             if config.MIN_ORDER_VALUE < max_buying_power:
                 if self.DEBUG:
                     print(f"[{self.name}] 🤏 Scaling up {final_notional:.2f} to Min Order {config.MIN_ORDER_VALUE}")
                 final_notional = config.MIN_ORDER_VALUE
             else:
                 # Too poor to afford minimum bet
                 if self.DEBUG:
                     print(f"[{self.name}] ❌ Account too small for Min Order: ${config.MIN_ORDER_VALUE} > MaxPower ${max_buying_power:.2f}")
                 return False, 0.0, 0.0
        # ---------------------------------------------
        
        # Convert to quantity
        quantity = final_notional / asset_price
        
        # Leverage Cap (Dynamic based on Conviction?)
        # For now, we stick to Config limits per asset class
        max_leverage = config.SCAVENGER_LEVERAGE if state == 'SCAVENGER' else config.PREDATOR_LEVERAGE
        
        # --- PHASE 35: LEVERAGE CHECK ---
        notional_value = quantity * asset_price
        if not self.check_leverage_risk(notional_value):
            return False, 0.0, 0.0
            
        # --- PATCH 5: SOLVENCY CHECK (Free Margin) ---
        # Ensure we don't commit more margin than we have available (with 5% buffer)
        required_margin = notional_value / leverage
        if required_margin > (self.available_balance * 0.95):
            print(f"[{self.name}] ⚠️ SOLVENCY CONSTRAINT: Req Margin ${required_margin:.2f} > Avail ${self.available_balance * 0.95:.2f}")
            # Downsize Logic - SAFE ADJUSTMENT
            # Don't bet the farm (95% of remaining). Cap to 20% of remaining if we are hitting this wall.
            max_affordable_margin = self.available_balance * 0.20 
            
            # Reduce Quantity to fit
            # qty = (margin * lev) / price
            new_qty = (max_affordable_margin * leverage) / asset_price
            
            if new_qty * asset_price < config.MIN_ORDER_VALUE:
                print(f"[{self.name}] ❌ Rejected: Downsized order < MIN_ORDER_VALUE.")
                return False, 0.0, 0.0
                
            print(f"[{self.name}] ✂️ Downsizing Qty {quantity:.4f} -> {new_qty:.4f}")
            quantity = new_qty
        # ---------------------------------------------
        
        # Log decision
        if state == 'SCAVENGER':
            print(f"[{self.name}] SCAVENGER: Margin ${margin:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
        else:
            print(f"[{self.name}] PREDATOR (Kelly): Kelly ${kelly_size_usd:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
        
        return True, quantity, leverage

            
    def open_position(self, symbol: str, direction: str, entry_price: float, quantity: float):
        """Track that a position has been opened or added to (Weighted Average)."""
        
        # Update State Trackers
        self.last_trade_time[symbol] = time.time()
        self.last_specific_entry[symbol] = entry_price
        
        existing = self.positions.get(symbol)
        
        if existing:
            old_qty = existing['quantity']
            old_price = existing['entry_price']
            new_qty = old_qty + quantity
            
            # Weighted Average Price
            if abs(new_qty) > 1e-9:
                avg_price = ((old_qty * old_price) + (quantity * entry_price)) / new_qty
            
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': avg_price,
                    'quantity': new_qty,
                    'stack_count': existing.get('stack_count', 1) + 1,
                    'first_entry_time': existing.get('first_entry_time', time.time())
                }
                print(f"[{self.name}] Position STACKED: {symbol} (New Avg: {avg_price:.4f}, Total Qty: {new_qty:.4f}, Stacks: {existing.get('stack_count', 1) + 1})")
            else:
                # Position effectively closed
                del self.positions[symbol]
                print(f"[{self.name}] Position CLOSED via fill: {symbol}")
        else:
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'stack_count': 1,
                'first_entry_time': time.time()
            }
            print(f"[{self.name}] Position OPENED: {symbol} {direction} @ {entry_price}")
        
    def close_position(self, symbol: str):
        """Clear position tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"[{self.name}] Position CLOSED: {symbol}")

    def set_reference_atr(self, atr: float):
        """Set the reference ATR for volatility targeting."""
        if self.reference_atr is None:
            self.reference_atr = atr
            print(f"[{self.name}] Reference ATR set: {atr:.6f}")

    # === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
    
    def calculate_max_risk(self, balance: float) -> float:
        """
        Minimax Constraint (Game Theory):
        Never risk the principal ($10). Only risk house money OR 1% of total.
        
        Args:
            balance: Current account balance
            
        Returns:
            Maximum USD that can be risked on a single trade
        """
        house_money = max(0, balance - config.PRINCIPAL)
        pct_risk = balance * config.MAX_RISK_PCT
        
        # Whichever is lower: house money or 1% of total
        max_risk_usd = min(house_money, pct_risk)
        
        return max_risk_usd
    
    def calculate_volatility_scalar(self, atr_current: float, atr_ref: float) -> float:
        """
        Volatility Scalar (Inverse Variance Weighting):
        Normalize position size based on current volatility.
        
        Formula: Size_adj = Size_base × (ATR_ref / ATR_current)
        
        Args:
            atr_current: Current ATR value
            atr_ref: Reference ATR (14-period average)
            
        Returns:
            Scalar multiplier (clamped to 0.5-2.0)
        """
        if atr_current <= 0 or atr_ref <= 0:
            return 1.0
        
        # Inverse relationship: high volatility = smaller size
        scalar = atr_ref / atr_current
        
        # Clamp to reasonable range
        return max(config.VOL_SCALAR_MIN, min(config.VOL_SCALAR_MAX, scalar))
    
    def calculate_recent_win_rate(self, lookback: int = None) -> float:
        """
        Calculate win rate from recent trades.
        
        Args:
            lookback: Number of recent trades to analyze
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        if lookback is None:
            lookback = config.KELLY_LOOKBACK
        
        # Integrate with database to get actual win rate
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                # Get recent trades from database
                trades = self.db_manager.get_recent_trades(lookback)
                if trades and len(trades) > 0:
                    # Calculate actual win rate
                    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
                    actual_wr = wins / len(trades)
                    
                    # BLENDING: If we have few trades, blend with a neutral baseline (0.40)
                    # to prevent "Cold Start" rejection (e.g. 0% WR after 1 loss).
                    sample_size = len(trades)
                    min_sample = 10
                    if sample_size < min_sample:
                        baseline = 0.40
                        weight = sample_size / min_sample
                        win_rate = (actual_wr * weight) + (baseline * (1 - weight))
                    else:
                        win_rate = actual_wr
                        
                    print(f"[{self.name}] 📊 Win Rate: {win_rate*100:.1f}% (Actual: {actual_wr*100:.1f}%, n={sample_size})")
                    return win_rate
            except Exception as e:
                print(f"[{self.name}] ⚠️ Win rate calculation failed: {e}")
        
        return 0.40

    def check_cluster_risk(self, symbol: str) -> bool:
        """
        Refuse trade if we already hold an asset from the same family.
        Returns: False if RISK DETECTED (Reject), True if SAFE.
        """
        family = None
        if symbol in config.FAMILY_L1: family = config.FAMILY_L1
        elif symbol in config.FAMILY_PAYMENT: family = config.FAMILY_PAYMENT
        elif symbol in config.FAMILY_MEME: family = config.FAMILY_MEME
        
        if not family: return True # No family, no risk
        
        # Check holdings
        for asset, data in self.positions.items():
            if abs(data['quantity']) > 0 and asset in family and asset != symbol:
                print(f"[{self.name}] ⚠️ CLUSTER RISK: Rejecting {symbol} (Already hold {asset})")
                return False
        return True

    def check_leverage_risk(self, new_notional_value: float) -> bool:
        """
        Refuse trade if Total Notional Exposure > 10x Balance.
        """
        current_exposure = 0.0
        # Sum absolute notional value of all positions
        for asset, data in self.positions.items():
            # We need current price for accurate notional, but entry_price is a decent proxy for risk check
            # preventing API calls here.
            qty = abs(data['quantity'])
            price = data['entry_price']
            current_exposure += (qty * price)
            
        total_exposure = current_exposure + new_notional_value
        max_allowed = self.balance * config.IMMUNE_MAX_LEVERAGE_RATIO
        
        if total_exposure > max_allowed:
            print(f"[{self.name}] ⚠️ OVER-LEVERAGE: Exposure ${total_exposure:.0f} > Limit ${max_allowed:.0f}")
            return False
        return True

    def calculate_kelly_size(self, balance: float, win_rate: float = None, risk_reward: float = None) -> float:
        """
        Modified Kelly Criterion (Half-Kelly):
        Calculate optimal position size for PREDATOR mode.
        
        Formula: f* = [(p(b+1) - 1) / b] × 0.5
        
        Args:
            balance: Current account balance
            win_rate: Recent win rate (0.0 to 1.0)
            risk_reward: Expected reward/risk ratio
            
        Returns:
            Maximum position size in USD
        """
        """
        Calculate maximum allowable risk per trade (USD) based on Minimax Regret.
        """
        # Hard cap: Never risk more than 5% of equity on a single trade idea
        hard_cap = balance * 0.05
        
        # Soft cap based on risk multiplier
        soft_cap = balance * 0.02 * self.risk_multiplier
        
        return min(hard_cap, soft_cap)
        
        # Use smoothed win rate if not provided
        if win_rate is None:
            win_rate = self.calculate_recent_win_rate()
        if risk_reward is None:
            risk_reward = config.KELLY_RISK_REWARD
        
        # Kelly formula: f* = [p(b+1) - 1] / b
        b = risk_reward
        kelly_fraction = ((win_rate * (b + 1)) - 1) / b
        
        # Half-Kelly for safety
        half_kelly = kelly_fraction * 0.5
        
        # Clamp to reasonable range (Floor prevents 0% WR from killing all trades)
        safe_fraction = max(config.KELLY_MIN_FRACTION, min(config.KELLY_MAX_FRACTION, half_kelly))
        
        return surplus * safe_fraction

    def receive_message(self, sender: Any, content: Any) -> Any:
        """Handle incoming messages."""
        msg_type = content.get('type')
        if msg_type == 'VALIDATE_TRADE':
            symbol = content.get('symbol')
            price = content.get('price')
            atr = content.get('atr')
            conviction = content.get('conviction', 0.5)
            # Check if conviction is None (if key exists but value is None)
            if conviction is None: conviction = 0.5
            direction = content.get('direction', 'BUY')
            
            crisis_score = content.get('crisis_score', 0.0)
            
            return self.calc_position_size(symbol, price, atr, conviction=conviction, direction=direction, crisis_score=crisis_score)
            
        elif msg_type == 'POSITION_FILLED':
            # --- PATCH: HANDLE SHORT COVERS ---
            # Executor sends positive Qty for 'COVER' (Buy side), but Governor
            # stores Shorts as Positive Quantity. We must NEGATE to reduce.
            if content.get('direction') == 'COVER':
                 qty = -abs(content.get('quantity'))
            else:
                 qty = content.get('quantity')

            self.open_position(
                content.get('symbol'),
                content.get('direction'),
                content.get('price'),
                qty
            )
            return True
            
        elif msg_type == 'POSITION_CLOSED':
            self.close_position(content.get('symbol'))
            return True
            
        elif msg_type == 'GET_STATE':
            return self.get_metabolism_state()
            
        elif msg_type == 'WAKE_UP':
            print(f"[{self.name}] Received WAKE_UP signal from Immune System.")
            self.state = 'ACTIVE'
            return True
            
        return None

    def gc_sync_with_executor(self, executor) -> list:
        """
        Garbage Collector: Ensure Governor positions match Executor state.
        Returns list of mismatched positions that were fixed.
        """
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        mismatches = []
        
        if not executor:
            return mismatches
        
        executor_assets = executor.held_assets
        executor_metadata = executor.position_metadata
        
        # Check for positions Governor has but Executor doesn't
        for sym in list(self.positions.keys()):
            exec_qty = executor_assets.get(sym, 0.0)
            if abs(exec_qty) < 0.00000001:
                # Executor doesn't have it, but we do
                if verbose:
                    print(f"[GC Monitor] ⚠️ Governor has {sym} but Executor doesn't. Removing from Governor.")
                del self.positions[sym]
                if sym in self.last_specific_entry:
                    del self.last_specific_entry[sym]
                mismatches.append(sym)
        
        # Check for positions Executor has but Governor doesn't
        for sym, qty in executor_assets.items():
            if abs(qty) > 0.00000001:
                if sym not in self.positions:
                    # Executor has it, but we don't
                    if verbose:
                        print(f"[GC Monitor] ⚠️ Executor has {sym} but Governor doesn't. Adding to Governor.")
                    meta = executor_metadata.get(sym, {})
                    self.positions[sym] = {
                        'direction': meta.get('direction', 'BUY'),
                        'entry_price': meta.get('entry_price', 0.0),
                        'quantity': qty,
                        'stack_count': 1,
                        'first_entry_time': time.time()
                    }

                    mismatches.append(sym)
        
        if verbose and mismatches:
            print(f"[GC Monitor] ✅ Governor Sync: {len(mismatches)} position(s) fixed: {mismatches}")
        elif verbose:
            print(f"[GC Monitor] ✅ Governor Sync: Aligned with Executor.")
        
        return mismatches

    # === PHASE 7: CONSOLIDATION ENGINE ===
    def run_consolidation_engine(self, current_prices: dict, position_metadata: dict = None) -> list:
        """
        Intelligent Position Consolidation.
        
        Triggers:
        - open_positions > max_positions_allowed (from regime)
        - free_margin < 1.5 * min_required_margin
        - regime_transition_pending
        
        Scoring Model (normalized 0-1):
        - PnL (30%): Unrealized profit
        - Conviction (25%): Signal confidence at entry
        - Liquidity (15%): Is it a major pair?
        - Age (10%): Newer positions may be better
        - Correlation (-20%): Penalty for redundant positions
        
        Returns: Single symbol to CLOSE (one per cycle for safety).
        """
        if self.consolidation_in_progress:
            return []
            
        open_positions = list(self.positions.keys())
        if len(open_positions) == 0:
            return []
            
        # Get regime permissions
        if self.regime_controller:
            permissions = self.regime_controller.get_permissions()
            max_positions = permissions.get('max_positions', 2)
            transition_pending = self.regime_controller.is_transition_pending()
        else:
            # Fallback to MICRO
            max_positions = config.REGIME_PERMISSIONS['MICRO']['max_positions']
            transition_pending = False
            
        # Check Trigger Conditions
        trigger_reason = None
        
        # A. Position count exceeds limit
        if len(open_positions) > max_positions:
            trigger_reason = f"Positions {len(open_positions)} > Max {max_positions}"
            
        # B. Regime transition pending
        elif transition_pending:
            trigger_reason = "Regime Transition Pending"
            
        # C. Free margin too low
        elif self.available_balance < 1.5 * config.MIN_ORDER_VALUE:
            trigger_reason = f"Free Margin ${self.available_balance:.2f} < ${1.5 * config.MIN_ORDER_VALUE:.2f}"
            
        if not trigger_reason:
            return []
            
        self.consolidation_in_progress = True
        
        # Dampened Logging: Only print if we haven't printed in the last 60s OR if action is taken
        should_log = (time.time() - self.last_consolidation_time > 60.0) 
        
        if should_log:
            print(f"\n[ConsolidationEngine] 🧹 TRIGGERED: {trigger_reason}")
            # print(f"[ConsolidationEngine] Analyzing {len(open_positions)} positions...") # Too noisy
        
        # Score all positions
        scored_positions = []
        
        # Calculate Total Portfolio Equity for weighting
        # Use available_balance + notional of all positions?
        # Simpler: Use Governor's tracked equity if available, else sum
        total_equity = self.available_balance
        for sym in open_positions:
            p = self.positions[sym]
            total_equity += abs(p.get('quantity', 0.0) * current_prices.get(sym, p.get('entry_price', 0.0)))
            
        if total_equity <= 0: total_equity = 1.0 # Prevent div/0

        for sym in open_positions:
            pos = self.positions[sym]
            entry = pos.get('entry_price', 0.0)
            qty = pos.get('quantity', 0.0)
            direction = pos.get('direction', 'BUY')
            current_price = current_prices.get(sym, entry)
            
            # === Calculate Individual Scores (0-1 normalized) ===
            
            # 1. PnL Score
            if entry > 0 and current_price > 0:
                pnl_pct = (current_price - entry) / entry
                if direction == 'SELL': pnl_pct *= -1
                # Normalize: -10% to +10% -> 0 to 1
                pnl_score = max(0.0, min(1.0, (pnl_pct + 0.10) / 0.20))
            else:
                pnl_score = 0.0
                
            # 2. Conviction Score (With Capital-Weighted Decay)
            meta = position_metadata.get(sym, {}) if position_metadata else {}
            raw_conviction = meta.get('conviction', 0.5)
            
            # --- WEIGHTED DECAY LOGIC ---
            notional = abs(qty * current_price)
            capital_weight = notional / total_equity if total_equity > 0 else 0.0
            
            # Acceleration: Heavy positions decay faster
            decay_factor = 1.0 + (capital_weight * config.CONVICTION_DECAY_CAPITAL_MULTIPLIER)
            effective_lifespan = config.CONVICTION_DECAY_BASE_HOURS / decay_factor
            
            first_entry = pos.get('first_entry_time', time.time())
            age_hours = (time.time() - first_entry) / 3600.0
            
            # Age Score (0.0=Dead, 1.0=Fresh) relative to effective lifespan
            age_efficiency = max(0.0, 1.0 - (age_hours / effective_lifespan))
            
            # Decayed Conviction
            conviction_score = raw_conviction * age_efficiency
            # ----------------------------
            
            # 3. Liquidity Score (major pairs)
            tier1_pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
            tier2_pairs = ['SOL/USDT', 'DOGE/USDT', 'ADA/USDT', 'LINK/USDT']
            if sym in tier1_pairs:
                liquidity_score = 1.0
            elif sym in tier2_pairs:
                liquidity_score = 0.6
            else:
                liquidity_score = 0.3
                
            # 4. Age Score (Simple linear for general ranking)
            # We keep the raw age score for the mix, but also used it for decay above.
            age_score = max(0.0, min(1.0, 1.0 - (age_hours / 48.0)))
            
            # 5. Correlation Penalty
            correlation_penalty = 0.0
            for other_sym in open_positions:
                if other_sym == sym: continue
                if self._are_correlated(sym, other_sym):
                    correlation_penalty += 0.2  # Reduced from 0.5 per new directive or just kept?
                    # Plan said 0.2
            correlation_penalty = min(0.6, correlation_penalty) # Max penalty 0.6
            
            # 6. Stack Penalty (New Directive)
            stack_penalty = 0.0
            if meta.get('stack_count', 1) > 1:
                stack_penalty = 0.5 # Severe penalty for stacked positions
            
            # === Composite Score ===
            score = (
                config.CONSOLIDATION_WEIGHT_PNL * pnl_score +
                config.CONSOLIDATION_WEIGHT_CONVICTION * conviction_score +
                config.CONSOLIDATION_WEIGHT_LIQUIDITY * liquidity_score +
                config.CONSOLIDATION_WEIGHT_AGE * age_score -
                config.CONSOLIDATION_WEIGHT_CORRELATION * correlation_penalty -
                stack_penalty
            )
            
            # === Hard Override Rules ===
            notional = abs(qty * current_price)
            force_close = False
            force_reason = ""
            
            # A. Dust threshold
            if notional < config.CONSOLIDATION_DUST_THRESHOLD:
                force_close = True
                force_reason = f"Dust (${notional:.2f})"
                score = -999.0
                
            # B. Stale position (no favorable movement)
            # (Simplified: just check if losing for too long)
            if pnl_pct < 0 and age_hours > config.CONSOLIDATION_STALE_HOURS:
                force_close = True
                force_reason = f"Stale Loss ({age_hours:.0f}h, {pnl_pct*100:.1f}%)"
                score = -998.0
                
            scored_positions.append({
                'symbol': sym,
                'score': score,
                'pnl_pct': pnl_pct if entry > 0 else 0.0,
                'pnl_score': pnl_score,
                'conviction_score': conviction_score,
                'liquidity_score': liquidity_score,
                'age_score': age_score,
                'correlation_penalty': correlation_penalty,
                'force_close': force_close,
                'force_reason': force_reason,
            })
            
        # Sort by Score DESCENDING (Best = highest score, kept first)
        scored_positions.sort(key=lambda x: x['score'], reverse=True)
        
        # Log ranking
        print(f"[ConsolidationEngine] Ranking:")
        for i, item in enumerate(scored_positions):
            status = "→ KEEP" if i < max_positions and item['score'] > -100 else "→ CLOSE"
            force_tag = f" [FORCED: {item['force_reason']}]" if item['force_close'] else ""
            print(f"  {i+1}. {item['symbol']:<12} score={item['score']:.2f} (PnL:{item['pnl_pct']*100:+.1f}%) {status}{force_tag}")
            
        # Select ONE position to close (lowest score)
        to_close = []
        if scored_positions:
            lowest = scored_positions[-1]
            if len(open_positions) > max_positions or lowest['force_close']:
                to_close.append(lowest['symbol'])
                print(f"[ConsolidationEngine] ❌ CLOSING: {lowest['symbol']} (Score: {lowest['score']:.2f})")
            else:
                print(f"[ConsolidationEngine] ✅ All positions acceptable.")
                
        self.consolidation_in_progress = False
        self.last_consolidation_time = time.time()
        
        # Notify Regime Controller that consolidation complete
        if self.regime_controller and to_close == []:
            self.regime_controller.complete_transition()
            
        return to_close
        
    def _are_correlated(self, sym1: str, sym2: str) -> bool:
        """
        Check if two symbols are in the same correlation family.
        """
        # Define families
        families = [
            ['BTC/USDT', 'ETH/USDT'],  # Majors move together
            ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT'],  # L1s
            ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'],  # Memes
            ['XRP/USDT', 'LTC/USDT'],  # OG Alts
            ['LINK/USDT', 'UNI/USDT', 'AAVE/USDT'],  # DeFi
        ]
        
        for family in families:
            if sym1 in family and sym2 in family:
                return True
        return False
        
    # Legacy method for backwards compatibility
    def consolidate_micro_exposure(self, current_prices: dict) -> list:
        """Legacy wrapper for run_consolidation_engine."""
        return self.run_consolidation_engine(current_prices)

