"""
GovernorHolon - NEXUS Risk & Homeostasis (Phase 15)

Implements:
1. Dual Metabolic State (SCAVENGER / PREDATOR)
2. Volatility Targeting (ATR-based position sizing)
3. Principal Protection (Never risk the $10 base)
"""

from typing import Any, Tuple, Literal, Dict, List
from HolonicTrader.holon_core import Holon, Disposition
import config
from HolonicTrader.agent_ppo import PPOHolon # Phase 22: The Monolith

try:
    from performance_tracker import DatabaseManager, get_performance_data
except ImportError:
    # Fallback mock if not found during dev
    class DatabaseManager: 
        def get_win_rate(self): return 0.5
    def get_performance_data(): return {'win_rate': 50.0}

import datetime
import time

class GovernorHolon(Holon):
    def __init__(self, name: str = "GovernorAgent", initial_balance: float = None, db_manager: Any = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        
        # FIX: Startup Budget sync
        if initial_balance is None:
            initial_balance = getattr(config, 'INITIAL_CAPITAL', 10.0)
            
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
        
        # FIX 3: Stack Timeout Tracker (for 5-minute reduction trigger)
        self.stack_timeout_tracker = {} # symbol -> first_blocked_timestamp
        
        # Phase 7: Regime Controller Integration
        self.regime_controller = None  # Set by Trader after instantiation
        
        # IRON BANK STATE (Capital Preservation)
        self.fortress_balance = getattr(config, 'IRON_BANK_MIN_RESERVE', 100.0) # The Floor
        self.risk_budget = 0.0 # Tradeable Capital
        self.last_ratchet_time = 0
        
        # Consolidation Engine State
        self.last_consolidation_time = 0.0
        self.consolidation_in_progress = False

        
        # Primary Budget Sync
        self.manage_iron_bank()
        
        # Phase 22: The Monolith (PPO Brain)
        # We give it a small autonomy to start (it suggests, rule-based decides)
        self.ppo = PPOHolon(name="Monolith")
        print(f"[{self.name}] 🧠 Monolith (PPO) Online.")

    def sync_positions(self, held_assets: dict, metadata: dict):
        """
        Sync positions from Executor/DB on startup to cure Amnesia.
        Handles both LONG (positive qty) and SHORT (negative qty) positions.
        """
        print(f"[{self.name}] Syncing positions from DB...")
        
        # FIX: Clear old phantoms!
        self.positions.clear() 
        self.last_specific_entry.clear()
        
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
            print(f"[{self.name}] Aucun position active trouvée pour synchronisation.")

    def sync_fortress(self, stored_floor: float):
        """Restore the Iron Bank floor from DB."""
        if stored_floor is not None and stored_floor > 0:
            self.fortress_balance = max(self.fortress_balance, stored_floor)
            print(f"[{self.name}] 🏰 Iron Bank Floor Restored: ${self.fortress_balance:.2f}")
        else:
            # Fallback for uninitialized DB or None
            self.fortress_balance = config.PRINCIPAL
            print(f"[{self.name}] 🏰 Iron Bank Floor Initialized (Fallback): ${self.fortress_balance:.2f}")
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
            
            # FIX: Force Recalculate Risk Budget after Balance Update
            self.manage_iron_bank()

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
            if drawdown > config.ACC_SANITY_THRESHOLD and not self.drawdown_lock:
                 print(f"[{self.name}] 📉 DATA SANITY CHECK: Instant >{config.ACC_SANITY_THRESHOLD:.0%} Drop (${self.high_water_mark:.2f} -> ${current_equity:.2f}). Resetting HWM (Assuming Glitch).")
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

        # IRON BANK CHECK (Every Balance Update)
        self.manage_iron_bank()

        self._check_homeostasis()

    def manage_iron_bank(self):
        """
        The Iron Bank: Secure Profits & Enforce Risk Floor.
        Called every cycle to update Risk Budget and Ratchet Fortress.
        """
        if not config.IRON_BANK_ENABLED:
            self.risk_budget = self.balance # Unrestricted
            return

        # 0. RESET OVERRIDE: If Ratchet disabled, force floor to Min Reserve
        # This allows users to "Unlock" profits by setting Ratchet to 0.0
        if config.IRON_BANK_RATCHET_PCT <= 0.0:
            self.fortress_balance = config.IRON_BANK_MIN_RESERVE

        # 1. Update Risk Budget
        # Risk Budget = Equity - Fortress Floor
        # If negative, we are "Underwater" relative to Fortress -> HALT.
        # We use self.balance which is synced to total equity in set_live_balance
        raw_budget = self.balance - self.fortress_balance
        self.risk_budget = max(0.0, raw_budget)
        
        # 2. Ratchet Logic (Lock Profits)
        # If we have significantly exceeded the Fortress (+ Buffer), raise the Floor.
        ratchet_threshold = self.fortress_balance * (1.0 + config.IRON_BANK_BUFFER_PCT)
        
        if self.balance > ratchet_threshold:
            profit_surplus = self.balance - self.fortress_balance
            
            # We lock a % of the TOTAL surplus (simple ratchet)
            # Or lock incremental? Simple Ratchet: Raise floor by X% of surplus.
            lock_amount = profit_surplus * config.IRON_BANK_RATCHET_PCT
            
            # Only ratchet if meaningful (> $1)
            if lock_amount > 1.0:
                old_fortress = self.fortress_balance
                self.fortress_balance += lock_amount
                self.last_ratchet_time = time.time()
                
                print(f"[{self.name}] 🏰 IRON BANK RATCHET: Locked ${lock_amount:.2f} Profits. New Floor: ${self.fortress_balance:.2f} (Equity: ${self.balance:.2f})")
                
                # Re-calc budget after ratchet
                self.risk_budget = max(0.0, self.balance - self.fortress_balance)

        # 3. Status Check
        if self.risk_budget < config.MIN_ORDER_VALUE:
             pass # Silent unless critical fail? No, Executor will handle "Insufficient Funds" errors implicitly via budget check.

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
            'max_balance': self.max_balance,
            'risk_budget': self.risk_budget,
            'fortress_balance': self.fortress_balance
        }

    def is_trade_allowed(self, symbol: str, asset_price: float) -> bool:
        """
        Lightweight check to see if a trade would be allowed.
        Prevents Strategy from wasting compute on blocked trades.
        """
        # 1. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS:
            print(f"[{self.name}] ⏳ Cooldown Active for {symbol} ({int(time.time() - last_time)}s < {config.GOVERNOR_COOLDOWN_SECONDS}s)")
            return False
            
        # 2. Price Distance Check (with Timeout Reduction Trigger)
        last_entry = self.last_specific_entry.get(symbol, 0)
        if last_entry > 0 and symbol in self.positions:
            dist = abs(asset_price - last_entry) / last_entry
            if dist < config.GOVERNOR_MIN_STACK_DIST:
                # FIX 3: Stack Timeout Tracking
                now = time.time()
                if symbol not in self.stack_timeout_tracker:
                    self.stack_timeout_tracker[symbol] = now
                    print(f"[{self.name}] 📏 Stack Too Close for {symbol}: Price ${asset_price:.2f} vs Entry ${last_entry:.2f} (Dist {dist*100:.2f}% < {config.GOVERNOR_MIN_STACK_DIST*100:.2f}%)")
                else:
                    elapsed = now - self.stack_timeout_tracker[symbol]
                    timeout_seconds = getattr(config, 'STACK_TIMEOUT_SECONDS', 300)  # Default 5 min
                    if elapsed > timeout_seconds:
                        # TIMEOUT TRIGGERED - Recommend 50% reduction
                        print(f"[{self.name}] ⏰ STACK TIMEOUT ({elapsed/60:.1f}min): {symbol} stuck at entry. RECOMMENDING 50% REDUCTION.")
                        # Flag for potential auto-reduction in Trader/Executor
                        self.positions[symbol]['needs_reduction'] = True
                        # Reset entry basis to current price to unstick
                        self.last_specific_entry[symbol] = asset_price
                        self.stack_timeout_tracker.pop(symbol, None)
                return False
            else:
                # Price moved away - clear the timeout tracker
                self.stack_timeout_tracker.pop(symbol, None)

        # 3. Solvency Gate (Fail Fast)
        if self.available_balance < config.MIN_ORDER_VALUE:
             print(f"[{self.name}] 💸 INSOLVENCY GATE: Available ${self.available_balance:.2f} < Min ${config.MIN_ORDER_VALUE}")
             return False

        # 4. IRON BANK GATE (Capital Preservation)

        if config.IRON_BANK_ENABLED:
            # If Risk Budget is zero (or dust), we are at the Floor.
            # We allow reducing positions (checked in calc_position_size), but this is a pre-check.
            # Ideally we pass 'direction' to is_trade_allowed to know if it's a reduce?
            # is_trade_allowed is usually called by Strategy looking for ENTRIES.
            # So blocking here is correct for ENTRIES.
            if self.risk_budget < (config.MIN_ORDER_VALUE * 0.10): # Assuming minimal 10% stop
                 print(f"[{self.name}] 🏰 IRON BANK GATE: Risk Budget ${self.risk_budget:.2f} too low. Protection Mode Active.")
                 return False

        return True

    def calc_position_size(self, symbol: str, asset_price: float, current_atr: float = None, atr_ref: float = None, conviction: float = 0.5, direction: str = 'BUY', crisis_score: float = 0.0, sentiment_score: float = 0.0, whale_confirmed: bool = False, market_bias: float = 0.5, metadata: Dict[str, Any] = None) -> Tuple[bool, float, float]:
        """
        Calculate position size with Phase 12 institutional risk management.
        
        Integrates:
        1. Minimax Constraint (protect principal)
        2. Volatility Scalar (ATR-based sizing)
        4. Conviction Scalar (LSTM-based scaling)
        5. Holistic Feedback (Sentiment Hormone)
        6. Dynamic Flotilla Sizing (Market Bias)
        
        Returns:
            (is_approved: bool, quantity: float, leverage: float)
        """
        is_override = False
        is_risk_reducing = False # Fix for UnboundLocalError
        final_notional = 0.0
        leverage = config.PREDATOR_LEVERAGE # Default
        
        # 0. Update Accumulator State
        # ideally this is done in sync loop, but fine to do here for latest check
        self.update_accumulator(self.balance)
        state = self.get_metabolism_state()
        
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

        # --- PHASE 25: SATELLITE OVERRIDE (High Value Snipers) ---
        # NANO GUARD: Disable Override in Nano Mode to ensure strict check at Patch 5a
        nano_active = (self.balance < config.NANO_CAPITAL_THRESHOLD)
        
        if symbol in config.SATELLITE_ASSETS and not nano_active:
             # Target Margin from config, but capped by available funds
             target_margin = getattr(config, 'SATELLITE_MARGIN', 10.0)
             
             # Dynamic Cap: Use 75% of available funds for these snipers
             # This ensures we don't hit "Insufficient Funds" on Kraken.
             safe_max_margin = (self.available_balance - 1.0) * 0.75
             safe_max_margin = max(0.0, safe_max_margin)
             
             actual_margin = min(target_margin, safe_max_margin)
             leverage = getattr(config, 'SATELLITE_LEVERAGE', 5.0)
             
             final_notional = actual_margin * leverage
             quantity = final_notional / asset_price
             
             if actual_margin < target_margin:
                  print(f"[{self.name}] 🎯 SATELLITE SNIPER: Capping Margin {target_margin:.2f} -> {actual_margin:.2f} (Solvency)")
             else:
                  print(f"[{self.name}] 🎯 SATELLITE SNIPER: Targeting ${actual_margin:.2f} Margin ({leverage}x)")
                  
             is_override = True
        # ------------------------------------

        # --- VOL-WINDOW REGIME OVERRIDE ---
        regime = self.regime_controller.get_current_regime() if self.regime_controller else 'MICRO'
        if regime == 'VOL_WINDOW':
            # 1. Check Max Positions
            if len(self.positions) >= config.VOL_WINDOW_MAX_POSITIONS:
                 print(f"[{self.name}] 🛑 VOL_WINDOW CAP: Max {config.VOL_WINDOW_MAX_POSITIONS} positions reached.")
                 return False, 0.0, 0.0
                 
            # 2. Sizing: Fixed Risk %
            # Risk Amount = Balance * Risk % (e.g. 2%)
            risk_amt_usd = self.balance * config.VOL_WINDOW_RISK_PCT
            
            # Determine Stop Distance (Need to know setup or use default)
            # We assume a tight stop for Vol Window (e.g. 1% or ATR based)
            # If ATR provided, use 1.5 ATR. Else use 1%.
            stop_dist_pct = 0.01
            if current_atr and asset_price > 0:
                 stop_dist_pct = (current_atr * 1.5) / asset_price
            
            # Position Size (Gross) = Risk / Stop%
            gross_size_usd = risk_amt_usd / stop_dist_pct
            
            # Cap Leverage
            max_gross = self.balance * config.VOL_WINDOW_LEVERAGE
            gross_size_usd = min(gross_size_usd, max_gross)
            
            quantity = gross_size_usd / asset_price

            # 🐳 WHALE SIZING BOOST
            if whale_confirmed:
                quantity *= 1.5 # 50% Size Boost for Whale trades
                print(f"[{self.name}] 🐳 WHALE SIZING: Boosting Position Size by 1.5x")
            
            print(f"[{self.name}] ⚡ VOL_WINDOW SIZING: Risk ${risk_amt_usd:.2f} (Dist {stop_dist_pct:.1%}) -> Pos ${gross_size_usd:.2f}")
            return True, quantity, config.VOL_WINDOW_LEVERAGE
        # ----------------------------------

        if is_override:
            # Skip primary logic, proceed to final solvency check at end
            pass
        elif self.state == 'HIBERNATE':
            print(f"[{self.name}] Trade REJECTED: System in HIBERNATION.")
            return False, 0.0, 0.0

        existing_pos = self.positions.get(symbol)

        # --- UNIFIED CONTROL PROTOCOL: MICRO MODE & STACKING GATES ---
        
        # 1. MICRO-ACCOUNT MODE (Request F)
        if config.MICRO_CAPITAL_MODE:
            # A. NO STACKING
            # A. REGIME STACKING CHECK
            regime_key = self.regime_controller.get_current_regime() if self.regime_controller else 'MICRO'
            max_stacks = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_stacks', 0)
            
            if existing_pos and max_stacks == 0:
                print(f"[{self.name}] 🧊 MICRO FREEZE: Stacking disabled (Regime {regime_key}, Max Stacks {max_stacks}). Rejecting.")
                return False, 0.0, 0.0
            
            # B. MAX POSITIONS CAP (Replaces Cluster Risk for Micro)
            base_pos_limit = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_positions', config.MICRO_MAX_POSITIONS)
            
            # --- DYNAMIC FLOTILLA SIZING (User Request) ---
            # Bull Market = More Slots
            bonus_slots = 0
            if market_bias > 0.8: bonus_slots = 4    # Euphoria (+4)
            elif market_bias > 0.6: bonus_slots = 2  # Bullish (+2)
            
            max_pos_limit = base_pos_limit + bonus_slots
            
            if len(self.positions) >= max_pos_limit:
                # Allow existing position to continue (e.g. if we are reducing/exiting?)
                # Actually, calc_position_size is only for ENTRIES (Long/Short).
                # Wait, if we are Shorting and we have a Long, it's a flip? Or if we have a Short and we Add?
                # The 'existing_pos' check above handles stacking/adding.
                # If we are here, 'existing_pos' is None, meaning NEW position.
                print(f"[{self.name}] 🛑 MAX POSITIONS REACHED ({len(self.positions)}/{max_pos_limit}). Rejecting.")
                return False, 0.0, 0.0
                
            # C. EXPOSURE CAP
            # Check if adding a MINIMUM SIZED trade triggers the cap. 
            # Real sizing happens later, this is just a gate.
            estimated_exposure = config.MIN_ORDER_VALUE * 1.1 # 10% buffer
            current_exposure = sum([p['quantity'] * p['entry_price'] for p in self.positions.values()])
            
            # --- PATCH: CORRECT NANO LIMIT ---
            if regime_key == 'NANO':
                 # Use NANO specific limit (10.0x) from permissions
                 nano_limit_ratio = config.REGIME_PERMISSIONS['NANO']['max_exposure_ratio']
                 max_allowed = self.balance * nano_limit_ratio
            else:
                 # Use Dynamic Limit based on Regime
                 regime_limit_ratio = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_exposure_ratio', config.MICRO_MAX_EXPOSURE_RATIO)
                 max_allowed = self.balance * regime_limit_ratio
            
            if (current_exposure + estimated_exposure) > max_allowed:
                  print(f"[{self.name}] 🛑 EXPOSURE CAP REACHED (Pre-Check): Current ${current_exposure:.2f} + MinTrade > Limit ${max_allowed:.2f}")
                  return False, 0.0, 0.0

        # 2. LOW NAV FREEZE (Request A - Modified)
        # Even if not in Micro mode, if funds are low, don't stack.
        if self.balance < config.STACKING_MIN_EQUITY:
            if existing_pos:
                 # Check Free Margin Buffer
                 # We need ~5x the min order value in FREE margin to justify a stack
                 required_buffer = config.MIN_ORDER_VALUE * config.STACKING_BUFFER_MULTIPLIER
                 if self.available_balance < required_buffer:
                       print(f"[{self.name}] 🧊 LOW NAV FREEZE: Free Margin ${self.available_balance:.2f} < Buffer ${required_buffer:.2f}. Stacking Blocked.")
                       return False, 0.0, 0.0

        # --- PATCH 2: THE STACKING CAP (Stop the Martingale) ---
        # FIXED: Removed hardcoded MAX_STACKS = 3. Rely purely on Regime permissions.
        # --- PATCH 2: DYNAMIC STACKING (Profit-Financed Risk) ---
        # "Earn Your Stacks": Max Stacks = Base Limit + Floor(Open Profit / 1R)
        if existing_pos:
            regime_key = self.regime_controller.get_current_regime() if self.regime_controller else 'MICRO'
            base_limit = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_stacks', 0)
            
            # 🐳 WHALE BONUS
            if whale_confirmed: 
                base_limit += 2
                
            # 💰 PROFIT BONUS
            # Calculate 1R (Risk Unit)
            risk_unit = max(1.0, self.balance * config.MAX_RISK_PCT)
            
            # Calculate Open Profit (Unrealized)
            open_profit = 0.0
            entry_p = existing_pos.get('entry_price', 0.0)
            qty_held = existing_pos.get('quantity', 0.0)
            pos_dir = existing_pos.get('direction', 'BUY')
            
            if entry_p > 0 and asset_price > 0:
                if pos_dir == 'BUY':
                    open_profit = (asset_price - entry_p) * qty_held
                else:
                    open_profit = (entry_p - asset_price) * qty_held
            
            # Bonus = Floor(Profit / 1R)
            bonus_stacks = 0
            if open_profit > 0:
                bonus_stacks = int(open_profit / risk_unit)
                
            dynamic_limit = base_limit + bonus_stacks
            
            # HARD CAP (Sanity)
            HARD_CAP = 10
            dynamic_limit = min(dynamic_limit, HARD_CAP)
            
            current_stacks = existing_pos.get('stack_count', 1)
            
            if current_stacks >= dynamic_limit:
                 print(f"[{self.name}] 🛑 STACKING CAP: {current_stacks} >= {dynamic_limit} (Base {base_limit} + Bonus {bonus_stacks}). Profit ${open_profit:.2f} (1R=${risk_unit:.2f}). Rejecting.")
                 return False, 0.0, 0.0
            else:
                 print(f"[{self.name}] 🥞 DYNAMIC STACK APPROVED: {current_stacks+1}/{dynamic_limit} (Bonus {bonus_stacks} from ${open_profit:.2f} profit)")
                 
        # -------------------------------------------------------
        # -------------------------------------------------------

        # --- PHASE 35: IMMUNE SYSTEM CHECKS ---
        # Note: In Micro Mode, we might skip Cluster Risk if desired ("Ignore cluster risk" request)
        if not config.MICRO_CAPITAL_MODE:
            if not self.check_cluster_risk(symbol):
                return False, 0.0, 0.0
            
        # --- PHASE 40: CORRELATION GUARD (The Hedge) ---
        # Prevent "All Eggs in One Basket"
        # If we already hold an asset highly correlated (>0.85) to the candidate, VETO it.
        # Exception: If directions are opposite (Hedge).
        
        # SMART CORRELATION (Phase 45): Relax constraints during Bull Runs
        should_check_correlation = getattr(config, 'CORRELATION_CHECK', True)
        
        if sentiment_score > getattr(config, 'SENTIMENT_THRESHOLD_BULL', 0.2):
             if self.DEBUG: print(f"[{self.name}] 🐂 BULL MARKET OVERRIDE: Disabling Correlation Check (Sent {sentiment_score:.2f} > 0.2)")
             should_check_correlation = False

        # 🐳 WHALE BYPASS (Optimized)
        if whale_confirmed:
             print(f"[{self.name}] 🐳 WHALE OVERRIDE: Disabling Correlation Guard for {symbol} (Whale Sighted)")
             should_check_correlation = False
             
        if len(self.positions) > 0 and should_check_correlation:
             # We need a correlation matrix. For now, we use "Family" variants as proxies or 
             # the PPO Brain's memory if available. 
             # Simpler: Hardcoded Map for Phase 1.
             
             # Map: {Asset: Family}
             # BTC, ETH -> 'CRYPTO_MAJOR'
             # SOL, AVAX -> 'L1_ROTATOR'
             # DOGE, PEPE -> 'MEME_BASKET'
             
             families = {
                 'BTC': 'BITCOIN', 'WBTC': 'BITCOIN',
                 'ETH': 'ETHEREUM',
                 'SOL': 'SOLANA', 'SUI': 'MOVE_L1', 'AVAX': 'EVM_L1', 'ADA': 'LEGACY_L1',
                 'DOGE': 'MEME', 'SHIB': 'MEME', 'PEPE': 'MEME',
                 'XRP': 'LEGACY_PAYMENT', 'LTC': 'LEGACY_PAYMENT'
             }
             
             cand_base = symbol.split('/')[0]
             cand_fam = families.get(cand_base, 'OTHER')
             
             for pos_sym, pos in self.positions.items():
                 # Allow stacking of the SAME asset (this is handled by Stack Limits, not Correlation)
                 if pos_sym == symbol: continue
                 
                 pos_base = pos_sym.split('/')[0]
                 pos_fam = families.get(pos_base, 'OTHER')
                 
                 # If in same family AND same direction -> BLOCK
                 if cand_fam != 'OTHER' and cand_fam == pos_fam:
                     existing_dir = pos.get('direction', 'BUY')
                     
                     if existing_dir == direction:
                         print(f"[{self.name}] 🔗 CORRELATION VETO: Rejecting {symbol} ({cand_fam}). Too similar to {pos_sym}.")
                         return False, 0.0, 0.0
                     else:
                         print(f"[{self.name}] ⚖️ HEDGE DETECTED: Allowing {symbol} ({direction}) vs {pos_sym} ({existing_dir})")
        # -----------------------------------------------

        # 1. Minimax Constraint (The "House Money" Rule)
        max_loss_usd = self.calculate_max_risk(self.balance)
        if asset_price <= 0:
            print(f"[{self.name}] Trade REJECTED: Invalid Asset Price.")
            return False, 0.0, 0.0
            
        # WARP SPEED 3.0: Smart Stacking & Cooldowns
        
        # 1. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS:
            print(f"[{self.name}] REJECTED: Cooldown active for {symbol} ({int(config.GOVERNOR_COOLDOWN_SECONDS - (time.time() - last_time))}s rem).")
            return False, 0.0, 0.0
            
        # 2. Solvency Check (New)
        # Cost = (Qty * Price) / Leverage
        # But we don't know Qty yet. We are calculating it.
        # Let's verify AFTER calculation.
        pass # Placeholder
        
        # 3. Minimax Sizing
        # ... sizing logic ...

            
        if not is_override:
            # 2. Price Distance Check
            last_entry = self.last_specific_entry.get(symbol, 0)
            if last_entry > 0 and symbol in self.positions:
                dist = abs(asset_price - last_entry) / last_entry
                if dist < config.GOVERNOR_MIN_STACK_DIST:
                    print(f"[{self.name}] REJECTED: Price {asset_price} too close to last entry {last_entry} (Dist: {dist*100:.2f}% < {config.GOVERNOR_MIN_STACK_DIST*100}%).")
                    return False, 0.0, 0.0
            
            # state calculated at start of function
        
        if not is_override:
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

                # --- PATCH: SNIPER MODE (Concentrated Fire) ---
                if config.MICRO_CAPITAL_MODE and config.MICRO_MAX_POSITIONS == 1:
                    # Use 90% of Available Balance for the Single Bullet
                    # We leave 10% buffer for fees/slippage
                    sniper_margin = self.available_balance * 0.90
                    margin = max(margin, sniper_margin) # Override if bigger
                    print(f"[{self.name}] 🎯 SNIPER MODE: Allocating ${margin:.2f} (90% of Free Margin)")
                # ----------------------------------------------

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
        
            # Apply Volatility/Physics Scalar
            physics_scalar = self.calculate_sde_physics_scalar(metadata, direction=direction)
            
            if current_atr and atr_ref:
                vol_scalar = self.calculate_volatility_scalar(current_atr, atr_ref)
                vol_adjusted_notional = base_notional * vol_scalar * physics_scalar
                print(f"[{self.name}] 📊 Volatility Scalar: {vol_scalar:.2f}x, Physics: {physics_scalar:.2f}x, Conviction: {conv_scalar:.2f}x")
            else:
                vol_adjusted_notional = base_notional * physics_scalar
                vol_scalar = 1.0
            
            # Apply Minimax Constraint (CRITICAL)
            max_risk_usd = self.calculate_max_risk(self.balance)
            
            # --- IRON BANK CAP ---
            if config.IRON_BANK_ENABLED and not is_risk_reducing: # Allow closing trades to ignore this
                 # Cap risk at the specific Risk Budget for this cycle
                 if self.risk_budget < max_risk_usd:
                     if self.DEBUG: print(f"[{self.name}] 🏰 IRON BANK CAP: Limiting Risk ${max_risk_usd:.2f} -> ${self.risk_budget:.2f}")
                     max_risk_usd = self.risk_budget
            # ---------------------
            
            # --- PHASE 40: MONTE CARLO RUIN GUARD ---
            # Estimate probability of hitting SL before TP using SDE
            sl_price = self.calculate_stop_loss(symbol, direction, asset_price, current_atr)
            tp_dist = config.SCAVENGER_SCALP_TP if state == 'SCAVENGER' else config.PREDATOR_TAKE_PROFIT
            tp_price = asset_price * (1 + tp_dist) if direction == 'BUY' else asset_price * (1 - tp_dist)
            
            ruin_prob = self.calculate_ruin_probability(symbol, asset_price, direction, sl_price, tp_price, metadata)
            
            # RECALIBRATION: Dynamic Ruin Threshold based on Win Rate
            perf = get_performance_data()
            wr = perf.get('win_rate', 50.0) / 100.0
            
            # User Audit: Loosen if performing poorly (<30%), Tighten if performing well (>50%)
            # Anchored to config.PHYSICS_MAX_RUIN_PROBABILITY (0.60)
            ruin_threshold = config.PHYSICS_MAX_RUIN_PROBABILITY
            if wr < 0.30: 
                ruin_threshold += 0.05 # Even looser (0.65) to prevent spiral
            elif wr > 0.50:
                ruin_threshold -= 0.10 # Tighter (0.50) to protect win streak
                
            if ruin_prob > ruin_threshold:
                 # --- SOFT VETO (Throttle) ---
                 # If Ruin Prob is high (e.g. 100% due to GBM assumption crossing SL),
                 # but we are trading Mean Reversion, we don't want to block entirely.
                 # Instead, we THROTTLE the size to 10% of intended risk.
                 throttle_factor = 0.10
                 print(f"[{self.name}] 🎲 RUIN GUARD THROTTLE: {symbol} Prob {ruin_prob:.1%} > {ruin_threshold:.1%}. Throttling size to {throttle_factor*100:.0f}%")
                 
                 # Apply Throttle
                 max_risk_usd *= throttle_factor
                 vol_adjusted_notional *= throttle_factor
            # ----------------------------------------
        
            # Assume mode-specific stop loss distance for risk calculation
            sl_dist = config.SCAVENGER_STOP_LOSS if state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
            
            # PROFILE OVERRIDE:
            profiles = getattr(config, 'ASSET_PROFILES', {})
            if symbol in profiles:
                # Use satellite stop if in satellite mode?
                # For sizing, we assume we use the profile's preferred stop if available
                # My profiles distinguish 'stop_loss' (general) from 'satellite_stop' (often same)
                # Let's prefer 'satellite_stop' if we are considering it a "Satellite" asset (in config list)
                p_stop = profiles[symbol].get('satellite_stop')
                if p_stop: 
                    sl_dist = p_stop
                    # print(f"[{self.name}] 🧬 Using Profile Stop Loss for {symbol}: {sl_dist:.1%}")

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

            # --- QTY FLOOR (Kraken Tiers) ---
            base_asset = symbol.split('/')[0]
            min_qty = config.MIN_TRADE_QTY.get(base_asset, 0.0)
        
            if quantity < min_qty:
                 new_notional = min_qty * asset_price
                 # Ensure we can actually afford this bump
                 # Max Power = Equity * HardMaxLev(e.g. 5 or 10)
                 max_power = self.balance * 10 
             
                 if new_notional < max_power:
                     if self.DEBUG: 
                         print(f"[{self.name}] 🏗️ Upgrading Qty {quantity:.5f} -> {min_qty} (Min Tier)")
                     quantity = min_qty
                     final_notional = new_notional
                 else:
                     if self.DEBUG:
                         print(f"[{self.name}] ❌ Min Tier {min_qty} too expensive (${new_notional:.2f} > Power ${max_power:.2f})")
                     return False, 0.0, 0.0
            # -------------------------------
        
            # Leverage Cap (Dynamic based on Conviction?)
            # For now, we stick to Config limits per asset class
            max_leverage = config.SCAVENGER_LEVERAGE if state == 'SCAVENGER' else config.PREDATOR_LEVERAGE
        
            # --- PHASE 35: LEVERAGE CHECK ---
            notional_value = quantity * asset_price
            if not self.check_leverage_risk(notional_value):
                return False, 0.0, 0.0
            
            # --- PATCH 5a: NANO-MODE (REALITY CHECK) ---
            if self.balance < config.NANO_CAPITAL_THRESHOLD:
                 if self.DEBUG: print(f"[{self.name}] 🔬 NANO MODE: Capital ${self.balance:.2f} < ${config.NANO_CAPITAL_THRESHOLD}")
             
                 # 1. Use Central Calculation (The Truth Source)
                 nano_calc = config.calculate_nano_position(self.balance, symbol, asset_price)
                 
                 if nano_calc['quantity'] <= 0:
                      print(f"[{self.name}] 🛑 NANO REJECT: Calculation returned 0 qty (Risk/Min limit).")
                      return False, 0.0, 0.0
                      
                 # 2. Override Values
                 quantity = nano_calc['quantity']
                 leverage = nano_calc['leverage']
                 final_notional = nano_calc['notional']
                 
                 print(f"[{self.name}] 🔬 NANO REALITY: Qty {quantity:.6f} | Lev {leverage}x | Margin ${nano_calc['margin']:.2f}")
                 
                 # 3. Force One-at-a-Time (Strict)
                 if len(self.positions) >= config.NANO_MAX_POSITIONS:
                      # If this is a new position (existing_pos is None), BLOCK.
                      if not existing_pos:
                          print(f"[{self.name}] 🛑 NANO CAP: Max {config.NANO_MAX_POSITIONS} position(s). Rejecting new trade.")
                          return False, 0.0, 0.0

                 # 4. Solvency - 50% Margin Check (from User Matrix)
                 # We verify that roughly 50% of equity remains as free margin
                 required_margin_est = nano_calc['margin']
                 free_margin_after = self.available_balance - required_margin_est
                 min_required_free = self.balance * 0.50
                 
                 if free_margin_after < min_required_free:
                      print(f"[{self.name}] 🛑 NANO MARGIN CHECK FAIL: Free After ${free_margin_after:.2f} < 50% Equity (${min_required_free:.2f})")
                      return False, 0.0, 0.0


        # --- PATCH 5b: SOLVENCY CHECK (Available Margin Cap) ---
        # Ensure we don't commit more margin than we have available (with significant buffer)
        # Rule: Single trade initial margin cannot exceed 75% of Available Margin for small accounts
        # This provides room for maintenance margin, slippage buffers, and fees.
        usage_limit = 0.75 if self.balance < 100 else 0.85
        max_trade_margin = (self.available_balance - 1.0) * usage_limit # $1 reserve + 75/85% limit
        max_trade_margin = max(0.0, max_trade_margin)
        
        required_margin = final_notional / leverage
        
        if required_margin > max_trade_margin:
            # Downsize Logic
            max_allowed_notional = max_trade_margin * leverage
            
            print(f"[{self.name}] ⚠️ MARGIN CAP: Req Margin ${required_margin:.2f} > {usage_limit*100:.0f}% Avail (-$1 reserve: ${max_trade_margin:.2f}). Downsizing.")
            
            final_notional = max_allowed_notional
            quantity = final_notional / asset_price
            
            # Re-Verify against Min Order Value
            if final_notional < config.MIN_ORDER_VALUE:
                 print(f"[{self.name}] ❌ Rejected: Downsized order ${final_notional:.2f} < MIN_ORDER_VALUE (${config.MIN_ORDER_VALUE}).")
                 return False, 0.0, 0.0

        # Normal Solvency (95% check for safety fallback)
        if required_margin > (self.available_balance * 0.95):
            # This should technically be caught by logic above, but keeping as fail-safe
            can_release_margin = (config.MICRO_CAPITAL_MODE and not getattr(config, 'MARGIN_RELEASE_OFF', False))
            if can_release_margin:
                 print(f"[{self.name}] ⚠️ Solvency Warning: Margin Low (${self.available_balance:.2f} < ${required_margin:.2f}), but deferring to Executor for Margin Release.")
            else:
                print(f"[{self.name}] ⚠️ SOLVENCY CONSTRAINT: Req Margin ${required_margin:.2f} > Avail ${self.available_balance * 0.95:.2f}")
                
                 # --- MATH-BASED DYNAMIC DOWNSIZE ---
                # No arbitrary "20% Panic". We solve for the exact max size that fits.
                # Max Margin = Available Balance - (1% Safety Buffer for Fees)
                exact_max_margin = max(0.0, self.available_balance * 0.99)
                
                # Calculate new Quantity based on exact fit
                # new_qty = (max_margin * leverage) / price
                new_qty = (exact_max_margin * leverage) / asset_price
                
                if new_qty * asset_price < config.MIN_ORDER_VALUE:
                    print(f"[{self.name}] ❌ Rejected: Exact Fit (${new_qty*asset_price:.2f}) < MIN_ORDER_VALUE.")
                    return False, 0.0, 0.0
                    
                # Verify against Contract Floor
                base_asset = symbol.split('/')[0]
                min_contract = config.MIN_TRADE_QTY.get(base_asset, 0.0)
                if new_qty < min_contract:
                    print(f"[{self.name}] ❌ Rejected: Exact Fit {new_qty} < MinContract {min_contract}")
                    return False, 0.0, 0.0
                    
                print(f"[{self.name}] 📉 DYNAMIC FIT: Resizing {quantity:.4f} -> {new_qty:.4f} (Utilizing 100% of ${exact_max_margin:.2f})")
                quantity = new_qty
        # ---------------------------------------------
        
        # --- PATCH: CONTRACT FLOOR ENFORCEMENT ---
        base_asset = symbol.split('/')[0]
        min_contract = config.MIN_TRADE_QTY.get(base_asset, 0.0)
        if quantity < min_contract:
             # If we are in Nano Sniper mode (which we are if <$50), we should BOOST to min_contract
             # if the cost is within reasonable limits (already checked by Sniper logic).
             # But let's verify Solvency one last time for the FLOOR.
             if self.balance < config.NANO_CAPITAL_THRESHOLD:
                  print(f"[{self.name}] 🆙 Boosting Nano Dust {quantity:.6f} -> MinContract {min_contract}")
                  quantity = min_contract
             else:
                  # For normal accounts, we reject dust
                  print(f"[{self.name}] ❌ REJECT: Qty {quantity:.6f} < MinContract {min_contract}")
                  return False, 0.0, 0.0
        # -----------------------------------------

        # Log decision
        if not is_override:
            if state == 'SCAVENGER':
                print(f"[{self.name}] SCAVENGER: Margin ${margin:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
            else:
                print(f"[{self.name}] PREDATOR (Kelly): Kelly ${kelly_size_usd:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
        
        # --- PATCH: MICRO GUARD RAIL ---
        # "Evaluated AFTER Kelly/Scavenger/Sniper sizing, BEFORE order leaves the agent"
        allowed, quantity = self.apply_micro_guard_rail(symbol, quantity, asset_price, leverage)
        if not allowed:
             return False, 0.0, 0.0
        # -------------------------------
             
        return True, quantity, leverage

    def calculate_stop_loss(self, symbol: str, direction: str, entry_price: float, atr: float = None) -> float:
        """
        Calculate Dynamic Stop Loss Price based on ATR or Fallback.
        """
        # Use Config Multiplier
        mult = getattr(config, 'ATR_STOP_LOSS_MULTIPLIER', 2.0)
        
        # --- NANO OVERRIDE: Tighten Stops ---
        # 20x Leverage = Liquidation at ~4.5% move. 
        # We MUST stop out before liquidating. Target 1.5% max risk distance.
        is_nano = getattr(config, 'MICRO_CAPITAL_MODE', False) and self.balance < getattr(config, 'NANO_CAPITAL_THRESHOLD', 50.0)
        
        if is_nano:
             mult = 1.0 # Tighter ATR
             max_dist_pct = 0.015 # Max 1.5% distance
             
             # --- PATCH: GENOME OVERRIDE ---
             sat_stop = getattr(config, 'SATELLITE_STOP_LOSS', None)
             if sat_stop and sat_stop > 0.02:
                 # If we have a specific Strategy Stop, use it (we already capped leverage for safety)
                 max_dist_pct = sat_stop
             # ------------------------------
        else:
             max_dist_pct = 0.05 # Max 5% distance
             
        # Fallback if no ATR (safety net)
        if not atr or atr <= 0:
            pct = max_dist_pct 
            delta = entry_price * pct
        else:
            delta = atr * mult
            # Clamp Delta to Max Distance
            if delta > (entry_price * max_dist_pct):
                delta = entry_price * max_dist_pct
        
        # --- NANO PATCH: MINIMUM NOISE FLOOR ---
        # Prevent stops tighter than 1.0% (Noise/Spread/Slippage protection)
        min_floor_pct = 0.01
        if delta < (entry_price * min_floor_pct):
             if self.DEBUG: print(f"[{self.name}] 🧘 Relaxing Stop to 1.0% Floor (Was {delta/entry_price:.2%})")
             delta = entry_price * min_floor_pct
        # ---------------------------------------
            
        if direction == 'BUY':
            sl = entry_price - delta
            # Logic Check: SL must be below entry for Long
            if sl >= entry_price: sl = entry_price * (1.0 - max_dist_pct)
            return sl
        else:
            sl = entry_price + delta
            # Logic Check: SL must be above entry for Short
            if sl <= entry_price: sl = entry_price * (1.0 + max_dist_pct)
            return sl
            
    def open_position(self, symbol: str, direction: str, entry_price: float, quantity: float):
        """Track that a position has been opened or added to (Weighted Average)."""
        
        # Update State Trackers
        self.last_trade_time[symbol] = time.time()
        self.last_specific_entry[symbol] = entry_price
        
        existing = self.positions.get(symbol)
        
        if existing:
            old_qty = existing['quantity']
            old_dir = existing.get('direction', direction)
            old_price = existing['entry_price']
            
            # Normalize Direction for Comparison
            def normalize(d):
                d = d.upper()
                if d == 'LONG': return 'BUY'
                if d == 'SHORT': return 'SELL'
                return d

            is_same_dir = (normalize(old_dir) == normalize(direction))
            
            if is_same_dir:
                # Additive (Stacking)
                new_qty = old_qty + quantity
                # Weighted Average Price
                avg_price = ((old_qty * old_price) + (quantity * entry_price)) / new_qty if new_qty > 1e-9 else entry_price
                new_dir = direction
                stack_inc = 1
            else:
                # Subtractive (Reduction/Flip)
                # Assuming quantity passed is Positive (Absolute Size of new order)
                print(f"[{self.name}] 📉 Netting Position: {symbol} (Old: {old_qty}, New Action: {quantity})")
                net_qty = old_qty - quantity 
                
                if net_qty > 1e-9:
                    # Partial Close (Reduced but same direction)
                    new_qty = net_qty
                    new_dir = old_dir
                    avg_price = old_price # Entry price doesn't change on reduction
                    stack_inc = 0 
                elif net_qty < -1e-9:
                    # Flip (Closed and Reversed)
                    new_qty = abs(net_qty)
                    new_dir = direction # Flipped to new
                    avg_price = entry_price # New cost basis
                    stack_inc = 1 # Reset stack
                else:
                    # Exact Close
                    new_qty = 0.0
                    avg_price = 0.0
                    new_dir = direction
                    stack_inc = 0

            # Store or Delete
            if new_qty > 1e-9:
                # --- PATCH: INVALID PRICE GUARD ---
                if avg_price <= 0:
                    avg_price = entry_price # Fallback to latest entry
                
                self.positions[symbol] = {
                    'direction': new_dir,
                    'entry_price': avg_price,
                    'quantity': new_qty,
                    'stack_count': existing.get('stack_count', 1) + stack_inc,
                    'first_entry_time': existing.get('first_entry_time', time.time())
                }
                action_tag = "STACKED" if is_same_dir else "REDUCED"
                print(f"[{self.name}] Position {action_tag}: {symbol} (New Avg: {avg_price:.8f}, Total Qty: {new_qty:.4f})")
            else:
                # Position effectively closed
                del self.positions[symbol]
                print(f"[{self.name}] Position CLOSED via fill: {symbol}")
        else:
            # --- PATCH: INVALID PRICE GUARD ---
            if entry_price <= 0: 
                 # Critical: If we open fresh with 0, try to find ANY recent price
                 entry_price = self.last_specific_entry.get(symbol, 0.0)

            self.positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'stack_count': 1,
                'first_entry_time': time.time()
            }
            print(f"[{self.name}] Position OPENED: {symbol} {direction} @ {entry_price:.8f}")
        
    def close_position(self, symbol: str):
        """Clear position tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"[{self.name}] Position CLOSED: {symbol}")

    def apply_micro_guard_rail(self, symbol: str, proposed_qty: float, price: float, leverage: float) -> Tuple[bool, float]:
        """
        MICRO-MODE HARD GUARD-RAIL v1.0
        
        Purpose: stop the bot from strangling itself with notional > 150 % of NAV
        Scope: evaluated AFTER Kelly/Scavenger/Sniper sizing
        """
        # 0. Check Regime
        regime = "MICRO"
        if self.regime_controller: 
            regime = self.regime_controller.get_current_regime()
            
        # ONLY apply Micro Guard Rail if we are truly in NANO or MICRO regime
        if regime not in ['NANO', 'MICRO']:
            return True, proposed_qty

        if not config.MICRO_CAPITAL_MODE:
            return True, proposed_qty

        # --- NANO OVERRIDE: Allow Aggressive Leverage ---
        if self.balance < config.NANO_CAPITAL_THRESHOLD:
             if self.DEBUG: print(f"[{self.name}] 🛡️ MICRO GUARD: Nano Mode Override Active (Relaxing Limits)")
             # We allow up to NANO_MAX_LEVERAGE (e.g. 20x) and Max Exposure Ratio
             # Just verify against those, don't use strict 1.5x Micro cap
             
             max_allowed_lev = config.NANO_MAX_LEVERAGE
             if leverage > max_allowed_lev:
                 if self.DEBUG: print(f"[{self.name}] 🛡️ GUARD: Leverage {leverage}x > Nano Max {max_allowed_lev}x. Capping.")
                 # Reduce Qty to fit leverage
                 allowed_qty = (proposed_qty / leverage) * max_allowed_lev
                 return True, allowed_qty
                 
             return True, proposed_qty
        # ------------------------------------------------
            
        nav = self.balance
        proposed_notional = proposed_qty * price
        
        # 5. Emergency kill-switch: if NAV < 50 USD → set MAX_GROSS_LEVERAGE_MICRO = 1.0
        max_gross_lev = config.MICRO_GUARD_GROSS_LEVERAGE
        if nav < config.MICRO_GUARD_CASH_PRESERVATION_THRESHOLD:
             max_gross_lev = config.MICRO_GUARD_CASH_PRESERVATION_LEVERAGE
             if self.DEBUG: print(f"[{self.name}] 🚨 CASH PRESERVATION: NAV ${nav:.2f} < $50. Max Lev -> {max_gross_lev}x")
             
        # Calculate Current Notional Exposure (Sum of Abs)
        current_gross_notional = 0.0
        for s, p in self.positions.items():
            if s != symbol: # Exclude current symbol if we are updating it? No, calc_size is for new/adds
                current_gross_notional += (p['quantity'] * p['entry_price'])
                
        # 1. Max portfolio notional <= 1.5 * NAV
        # The new total gross notional
        new_total_gross = current_gross_notional + proposed_notional
        limit_portfolio = config.MICRO_GUARD_PORTFOLIO_NOTIONAL_MULT * nav
        
        # 2. Max single-symbol net notional
        # CHECK STACKING OVERRIDE
        is_stacking = symbol in self.positions
        single_mult = config.MICRO_GUARD_SINGLE_NOTIONAL_MULT
        if is_stacking:
             # Use the higher of Config Limit or 1.0 (Don't downgrade a loose config)
             single_mult = max(config.MICRO_GUARD_SINGLE_NOTIONAL_MULT, 1.0) 
             if self.DEBUG: print(f"[{self.name}] 🥞 STACKING OVERRIDE: Using max({config.MICRO_GUARD_SINGLE_NOTIONAL_MULT}, 1.0) = {single_mult}x NAV")
             
        limit_single = single_mult * nav
        
        # 3. Max gross leverage <= Limit
        limit_leverage_notional = nav * max_gross_lev
        
        # FIND THE BINDING CONSTRAINT
        # We need to reduce proposed_notional such that:
        # A) current + proposed <= limit_portfolio
        # B) proposed <= limit_single
        # C) current + proposed <= limit_leverage_notional
        
        max_allowed_notional_A = max(0, limit_portfolio - current_gross_notional)
        max_allowed_notional_B = limit_single
        max_allowed_notional_C = max(0, limit_leverage_notional - current_gross_notional)
        
        final_allowed_notional = min(max_allowed_notional_A, max_allowed_notional_B, max_allowed_notional_C)
        
        if proposed_notional > final_allowed_notional:
            # DOWN-SIZE
            new_qty = final_allowed_notional / price
            print(f"[{self.name}] 🛡️ MICRO_GUARD: Down-sizing {symbol} {proposed_qty:.4f}->{new_qty:.4f} (${proposed_notional:.2f}->${final_allowed_notional:.2f})")
            print(f"    Constraint: PortLimit=${limit_portfolio:.2f}, SingleLimit=${limit_single:.2f}, LevLimit=${limit_leverage_notional:.2f}")
            
            # Check Min Notional
            if new_qty * price < config.MIN_ORDER_VALUE:
                 print(f"[{self.name}] ❌ Governor: Micro-guard veto – min_notional breach (${new_qty*price:.2f} < ${config.MIN_ORDER_VALUE})")
                 return False, 0.0
                 
            return True, new_qty
            
        return True, proposed_qty


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
        Uses Rust Engine (holonic_speed) if available.
        """
        try:
            import holonic_speed
            return holonic_speed.governor_calculate_max_risk(
                balance, config.PRINCIPAL, balance
            )
        except ImportError:
            # Fallback to Python
            house_money = max(0, balance - config.PRINCIPAL)
            pct_risk = balance * config.MAX_RISK_PCT
            return min(house_money, pct_risk)
    
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

    def calculate_sde_physics_scalar(self, metadata: Dict[str, Any], direction: str = 'BUY') -> float:
        """
        Physics-Based Position Scaling (SDE Layer).
        Dynamically adjusts size based on SDE drift and diffusion.
        """
        if not metadata or 'sde_physics' not in metadata:
            return 1.0
            
        sde = metadata['sde_physics']
        precision_physics_scalar = 1.0
        
        # 1. Diffusion-Alpha (Noise Sensitivity)
        # If Instantaneous SDE Sigma is spiking relative to ATR, we reduce size.
        inst_vol = sde.get('sigma', 0.0)
        if inst_vol > 1.5: # Extreme Volatility (> 150% annual)
             precision_physics_scalar *= 0.8
             
        # 2. Quantum Reversion Scaling
        reason = metadata.get('reason', '')
        if reason in ['QUANTUM', 'QUANTUM_SELL']:
            # For Quantum Reversion, we scale by the conviction provided by the Oracle
            # (which is based on distance from the mean)
            q_conv = metadata.get('quantum_conviction', 1.0)
            precision_physics_scalar *= q_conv
            
        # 3. Drift Check (Optional Bonus)
        # We don't want to over-size, so we clamp the bonus tightly
        drift = sde.get('drift', 0.0)
        if direction == 'BUY' and drift > 0.5:
            precision_physics_scalar *= 1.1 # 10% Bonus for strong positive drift
        elif direction == 'SELL' and drift < -0.5:
            precision_physics_scalar *= 1.1
            
        return max(0.5, min(1.5, precision_physics_scalar))
    
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


    def calculate_ruin_probability(self, symbol: str, entry_price: float, direction: str, stop_loss: float, take_profit: float, metadata: Dict[str, Any]) -> float:
        """
        Monte Carlo Ruin Guard:
        Uses optimized SDEEngine (Rust accelerated) to estimate 
        the probability of hitting Stop Loss before Take Profit/Horizon.
        """
        if not metadata or 'sde_physics' not in metadata:
            return 0.5 
            
        try:
            from HolonicTrader.sde_engine import SDEEngine
            sde = metadata['sde_physics']
            # Parameters from Oracle
            params = {
                'mu': sde.get('mu', 0.0),
                'sigma': sde.get('sigma', 0.1),
                'lambda': sde.get('lambda', 0.1)
            }
            
            # Use Rust-accelerated calculation
            return SDEEngine.calculate_ruin_probability(
                'GBM', # Default model
                params, 
                entry_price, 
                stop_loss, 
                take_profit, 
                horizon=100, 
                paths=500
            )
            
        except Exception as e:
            if self.DEBUG: print(f"[{self.name}] Ruin Guard Error: {e}")
            return 0.5

    def check_cluster_risk(self, symbol: str) -> bool:
        """
        Refuse trade if we already hold an asset from the same family.
        Returns: False if RISK DETECTED (Reject), True if SAFE.
        Uses Rust Engine (holonic_speed) if available.
        """
        try:
            import holonic_speed
            # Get currently held symbols
            held_symbols = [s for s, d in self.positions.items() if abs(d.get('quantity', 0)) > 0]
            result = holonic_speed.governor_check_cluster_risk(held_symbols, symbol)
            if not result:
                print(f"[{self.name}] CLUSTER RISK: Rejecting {symbol} (Same family as held)")
            return result
        except ImportError:
            # Fallback to Python
            family = None
            if symbol in config.FAMILY_L1: family = config.FAMILY_L1
            elif symbol in config.FAMILY_PAYMENT: family = config.FAMILY_PAYMENT
            elif symbol in config.FAMILY_MEME: family = config.FAMILY_MEME
            
            if not family: return True
            
            for asset, data in self.positions.items():
                if abs(data['quantity']) > 0 and asset in family and asset != symbol:
                    print(f"[{self.name}] CLUSTER RISK: Rejecting {symbol} (Already hold {asset})")
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
        
        # --- NANO OVERRIDE ---
        if self.balance < config.NANO_CAPITAL_THRESHOLD:
            # Allow full 20x (or whatever NANO_MAX_LEVERAGE is)
            max_allowed = self.balance * config.NANO_MAX_LEVERAGE
        else:
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
        Calculate maximum allowable risk per trade (USD) based on Volatility-Adjusted Kelly Criterion.
        Replaces arbitrary 5% hard caps with Probability-Based Sizing.
        """
        # 1. Get Win Rate & Reward:Risk
        win_rate = self.calculate_recent_win_rate()
        risk_reward = 2.0 # Conservative estimate for R:R
        
        # 2. Kelly Formula -> Optimal Fraction
        # f* = (p(b+1) - 1) / b
        # where p = win_rate, b = risk_reward
        kelly_fraction = ((win_rate * (risk_reward + 1)) - 1) / risk_reward
        
        # 3. Fractional Kelly (Safety)
        # Use Half-Kelly for safety (industry standard)
        fractional_kelly = kelly_fraction * 0.5
        
        # 4. Volatility Adjustment
        # If market is wild, reduce size.
        atr_ref = getattr(self, 'reference_atr', 0.0) or 0.0
        # If we can't find ATR, we assume 1.0 scalar. 
        # But we need current ATR... which isn't passed here. 
        # We assume the Caller (calc_position_size) applies vol_scalar LATER.
        # So here we return the BASE Kelly risk.
        
        # Sanity Bounds (Never risk <0% or >20% of equity per trade)
        safe_fraction = max(0.01, min(0.20, fractional_kelly))
        
        max_usd_risk = balance * safe_fraction
        
        if self.DEBUG:
            print(f"[{self.name}] 🧠 Kelly Risk: WR {win_rate:.2f}, Kelly {kelly_fraction:.2f}, Safe {safe_fraction:.2f} -> ${max_usd_risk:.2f}")
            
        return max_usd_risk

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
            # Executor sends positive Qty. open_position handles logic.
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
        def check_stacking_logic(self, symbol: str, current_price: float, direction: str, atr: float = None) -> Tuple[bool, str]:
            """
            Phase 18: Smart Stacking
            Only allow adding to position if:
            1. Winning (PnL > 0)
            2. Price moved significantly (Dynamic ATR-based Distance)
            3. Trend is still young (< 24h)
            4. Regime Alignment (Don't stack Longs in Bear Market)
            """
            if symbol not in self.positions:
                return True, "New Position"
                
            pos = self.positions[symbol]
            entry = pos['entry_price']
            qty = pos['quantity']
            existing_dir = pos.get('direction', 'BUY')
            stacks = pos.get('stack_count', 1)
            
            # 0. Direction Check (Sanity)
            if direction != existing_dir:
                return False, f"Opposite Direction ({direction} vs {existing_dir})"
                
            # 1. PnL Check (Only stack winners)
            pnl_pct = (current_price - entry) / entry
            if existing_dir == 'SELL': pnl_pct *= -1
            
            if pnl_pct <= 0:
                return False, f"Losing Position ({pnl_pct*100:.2f}%)"
                
            # 2. Distance Check (Standard + Volatility Scaling)
            # Standard Min Dist
            min_dist_pct = config.GOVERNOR_MIN_STACK_DIST
            
            # VOLATILITY SCALING (User Request)
            if atr and current_price > 0:
                # If Volatility is High, we need MORE distance to confirm trend
                # Normalize ATR pct: e.g. 1% ATR -> 1.0 multiplier
                atr_pct = atr / current_price
                vol_multiplier = max(1.0, atr_pct / 0.01) # Baseline 1% ATR
                min_dist_pct *= vol_multiplier
                # print(f"[{self.name}] 📏 Dynamic Stack Dist: {min_dist_pct*100:.2f}% (Vol Mult: {vol_multiplier:.2f}x)")

            current_dist = abs((current_price - entry) / entry)
            if current_dist < min_dist_pct:
                return False, f"Too Close (Dist {current_dist*100:.2f}% < {min_dist_pct*100:.2f}%)"
                
            # 3. Stack Limit
            max_stacks = config.REGIME_PERMISSIONS['SMALL']['max_stacks'] # Default
            if self.regime_controller:
                max_stacks = self.regime_controller.get_permissions().get('max_stacks', 0)
                
            # REGIME OVERRIDE: Cap Long Stacks in Bear Market
            # How to know regime here? We can check recent sentiment or pass it in.
            # Fallback: Use simple trend check or external flag if available.
            # For now, strict limit.
            if stacks >= max_stacks:
                return False, f"Max Stacks Reached ({stacks}/{max_stacks})"
                
            return True, "Stacking Approved"
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

