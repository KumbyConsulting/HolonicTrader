"""
ExecutorHolon - The Executor Agent (Phase 4)

This agent acts as the 'trade executor' of the AEHML core.
It executes trades only if the Disposition allows it and maintains
a local pseudo-blockchain ledger for audit purposes.

Key Features:
1. Pseudo-Blockchain Ledger (AuditLedger) with SHA-256 hashing
2. Disposition-based trade execution logic
"""

import hashlib
import json
import config
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, List, Optional, Dict, Tuple
import math

from HolonicTrader.holon_core import Holon, Disposition, Message


@dataclass
class TradeSignal:
    """
    Represents a trading signal.
    """
    symbol: str
    direction: Literal['BUY', 'SELL']
    size: float
    price: float
    conviction: float = 0.5
    metadata: Dict = field(default_factory=dict)


@dataclass
class TradeDecision:
    """
    Represents the result of a trade decision.
    """
    action: Literal['EXECUTE', 'HALT', 'REDUCE']
    original_signal: TradeSignal
    adjusted_size: float
    disposition: Disposition
    block_hash: str
    entropy_score: float = 0.0


class ExecutorHolon(Holon):
    """
    ExecutorHolon is the 'Executor' that decides whether to execute trades
    based on market regime and entropy levels. All decisions are logged
    to a tamper-evident pseudo-blockchain ledger.
    """

    @dataclass
    class LedgerBlock:
        """
        A single block in the audit ledger.
        """
        timestamp: str
        entropy_score: float
        regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION']
        action: Literal['EXECUTE', 'HALT', 'REDUCE']
        prev_hash: str
        hash: str = ""

        def compute_hash(self) -> str:
            """
            Compute SHA-256 hash of the block's contents (excluding current hash).
            """
            block_data = {
                'timestamp': self.timestamp,
                'entropy_score': self.entropy_score,
                'regime': self.regime,
                'action': self.action,
                'prev_hash': self.prev_hash
            }
            block_string = json.dumps(block_data, sort_keys=True)
            return hashlib.sha256(block_string.encode()).hexdigest()

    class AuditLedger:
        """
        A pseudo-blockchain ledger for maintaining an immutable audit trail
        of all trading decisions.
        
        Each block is chained to the previous via SHA-256 hashing.
        """

        def __init__(self):
            self._chain: List['ExecutorHolon.LedgerBlock'] = []

        @property
        def chain(self) -> List['ExecutorHolon.LedgerBlock']:
            """Return the chain as a read-only property."""
            return list(self._chain)

        def add_block(
            self,
            entropy_score: float,
            regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION'],
            action: Literal['EXECUTE', 'HALT', 'REDUCE']
        ) -> 'ExecutorHolon.LedgerBlock':
            """
            Add a new block to the ledger.
            
            Args:
                entropy_score: The entropy value at decision time
                regime: The market regime (ORDERED, CHAOTIC, TRANSITION)
                action: The action taken (EXECUTE, HALT, REDUCE)
                
            Returns:
                The newly created and added block
            """
            # Get previous hash (genesis block uses "0")
            prev_hash = self._chain[-1].hash if self._chain else "0"

            # Create new block
            block = ExecutorHolon.LedgerBlock(
                timestamp=datetime.now(timezone.utc).isoformat(),
                entropy_score=entropy_score,
                regime=regime,
                action=action,
                prev_hash=prev_hash
            )

            # Compute and set the hash
            block.hash = block.compute_hash()

            # Add to chain
            self._chain.append(block)

            return block

        def verify_chain(self) -> bool:
            """
            Verify the integrity of the entire chain.
            
            Returns:
                True if the chain is valid, False otherwise
            """
            if not self._chain:
                return True

            # Check genesis block
            if self._chain[0].prev_hash != "0":
                return False

            # Verify each block
            for i, block in enumerate(self._chain):
                # Verify block's own hash
                if block.hash != block.compute_hash():
                    return False

                # Verify chain linkage (skip genesis)
                if i > 0 and block.prev_hash != self._chain[i - 1].hash:
                    return False

            return True

        def __len__(self) -> int:
            return len(self._chain)

    def __init__(
        self, 
        name: str = "ExecutorAgent", 
        initial_capital: float = 10.0,
        use_compounding: bool = True,
        fixed_stake: float = 10.0,
        db_manager: Any = None,
        governor: Any = None,
        actuator: Any = None,
        gui_queue: Any = None # NEW: Dashboard Link
    ):
        """
        Initialize the ExecutorHolon with a neutral disposition and starting capital.
        
        Args:
            name: Agent name
            initial_capital: Starting balance in USD
            use_compounding: If True, uses % of portfolio. If False, uses fixed_stake.
            fixed_stake: Amount in USD to bet per trade if use_compounding is False.
            db_manager: Optional DatabaseManager instance for persistence.
            governor: Optional GovernorHolon instance for risk management.
            actuator: Optional ActuatorHolon instance for execution.
        """
        # Start with balanced disposition; will be adjusted per trade decision
        default_disposition = Disposition(autonomy=0.5, integration=0.5)
        super().__init__(name=name, disposition=default_disposition)
        
        # Initialize the audit ledger
        self.ledger = self.AuditLedger()
        
        # Persistence & Risk & Execution
        self.db_manager = db_manager
        self.governor = governor
        self.actuator = actuator
        self.gui_queue = gui_queue # Store reference
        
        # Portfolio Management
        self.initial_capital = initial_capital
        self.balance_usd = initial_capital
        
        # Multi-Asset Tracking
        self.held_assets = {}   # symbol -> quantity
        self.entry_prices = {}  # symbol -> entry_price (for SL/TP)
        self.entry_timestamps = {}  # symbol -> entry_timestamp (for position age)
        self.latest_prices = {} # symbol -> last_seen_price (for valuation)
        self.position_metadata = {} # symbol -> {'leverage': float, 'entry_price': float, 'entry_timestamp': str}
        
        # Stop-Loss / Take-Profit Parameters (Synced with config)
        self.stop_loss_pct = config.SCAVENGER_STOP_LOSS
        self.take_profit_pct = config.PREDATOR_TAKE_PROFIT
        
        # Sizing Strategy
        self.use_compounding = use_compounding
        self.fixed_stake = fixed_stake
        
        # Dashboard Details
        self.last_order_details = "NONE"
        
        # Load state from DB if available
        if self.db_manager:
            self._load_state()

    def _persist_portfolio(self):
        """Helper to save current balance and assets to DB."""
        if self.db_manager:
            self.db_manager.save_portfolio(self.balance_usd, self.held_assets, self.position_metadata)

    def _load_state(self):
        """Premium State Restoration: Reconstructs portfolio and records from DB."""
        # Load Portfolio
        portfolio = self.db_manager.get_portfolio()
        if portfolio:
            self.balance_usd = portfolio.get('balance_usd', self.initial_capital)
            self.held_assets = portfolio.get('held_assets', {})
            self.position_metadata = portfolio.get('position_metadata', {})
            
            # Reconstruct Entry Prices and sync metadata
            for sym, meta in self.position_metadata.items():
                if 'entry_price' in meta:
                    self.entry_prices[sym] = meta['entry_price']
                if 'entry_timestamp' in meta:
                    self.entry_timestamps[sym] = meta['entry_timestamp']
            
            print(f"[{self.name}] 🏦 Portfolio Restored: ${self.balance_usd:.2f} USD")
            active_list = [f"{s}({q:.4f})" for s, q in self.held_assets.items() if abs(q) > 0.00000001]
            if active_list:
                print(f"[{self.name}] 📦 Active Positions: {', '.join(active_list)}")
        
        # Load Last Block
        last_block = self.db_manager.get_last_block()
        if last_block:
            restored_block = self.LedgerBlock(
                timestamp=last_block['timestamp'],
                entropy_score=last_block['entropy_score'],
                regime=last_block['regime'],
                action=last_block['action'],
                prev_hash=last_block['prev_hash'],
                hash=last_block['hash']
            )
            self.ledger._chain.append(restored_block)
            print(f"[{self.name}] ⛓️ Ledger Tip Restored: {restored_block.hash[:8]}...")

    def reconcile_exchange_positions(self):
        """
        Fetch REAL positions from Exchange and sync them to Brain.
        HARD RESET: Wipes local state and re-imports entirely from exchange truth.
        """
        if not self.actuator: return
        
        print(f"[{self.name}] 🔄 Reconciling Exchange Positions (HARD RESET)...")
        try:
            positions = self.actuator.exchange.fetch_positions()
            
            # --- HARD RESET: Wipe all local positions first ---
            old_positions = dict(self.held_assets)
            self.held_assets.clear()
            self.entry_prices.clear()
            self.entry_timestamps.clear()
            self.position_metadata.clear()
            
            synced_count = 0
            for p in positions:
                size = float(p.get('contracts', 0))
                if size == 0: continue
                
                # 1. Map Exchange Symbol -> Internal Symbol
                exchange_sym = p['symbol'] # 'XRP/USD:USD'
                internal_sym = None
                
                # Try simple match first
                if exchange_sym in config.ALLOWED_ASSETS:
                    internal_sym = exchange_sym
                else:
                    # Reverse Lookup in KRAKEN_SYMBOL_MAP
                    for k, v in config.KRAKEN_SYMBOL_MAP.items():
                        if v == exchange_sym:
                            internal_sym = k
                            break
                            
                if not internal_sym:
                    # HEURISTIC: Try to guess (XRP/USD:USD -> XRP/USDT)
                    raw = exchange_sym.split('/')[0]
                    if raw.startswith('PF_'): raw = raw[3:6]
                    guess = f"{raw}/USDT"
                    if guess in config.ALLOWED_ASSETS:
                        internal_sym = guess
                        
                if not internal_sym:
                    print(f"[{self.name}] ⚠️ Unknown Position Found: {exchange_sym} ({size}). Skipping.")
                    continue
                
                # 2. Import this position from exchange
                direction = 'BUY' if p['side'] == 'long' else 'SELL'
                entry_price = float(p.get('entryPrice', 0.0))
                leverage = float(p.get('leverage', 1.0) or 1.0)
                
                # Check if this differs from what we had before
                old_qty = old_positions.get(internal_sym, 0.0)
                if abs(old_qty - size) > 0.0001 or old_qty == 0:
                    print(f"[{self.name}] 📥 Importing: {internal_sym} ({direction}) Size: {size} Entry: {entry_price}")
                
                self.held_assets[internal_sym] = size if direction == 'BUY' else -size
                self.entry_prices[internal_sym] = entry_price
                self.entry_timestamps[internal_sym] = datetime.now(timezone.utc).isoformat()
                
                # Reconstruct Metadata
                direction_mult = 1.0 if direction == 'BUY' else -1.0
                sl_price = entry_price * (1.0 - (config.PREDATOR_STOP_LOSS * direction_mult))
                tp_price = entry_price * (1.0 + (config.PREDATOR_TAKE_PROFIT * direction_mult))
                
                self.position_metadata[internal_sym] = {
                    'symbol': internal_sym,
                    'direction': direction,
                    'quantity': size,
                    'entry_price': entry_price,
                    'entry_timestamp': self.entry_timestamps[internal_sym],
                    'leverage': leverage,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'strategy': 'RECOVERED',
                    'stack_count': 1 # Default to 1 on fresh import (Can't know history unless mapped)
                }
                synced_count += 1
                
            # Report any positions that were in old_positions but NOT re-imported (i.e., ghosts)
            for old_sym, old_qty in old_positions.items():
                if abs(old_qty) > 0.00000001 and old_sym not in self.held_assets:
                    print(f"[{self.name}] 👻 GHOST CLEARED: {old_sym} ({old_qty}) - not on exchange")
            
            if synced_count > 0:
                print(f"[{self.name}] ✅ Imported {synced_count} positions from Exchange.")
            else:
                print(f"[{self.name}] ✅ No open positions on Exchange.")
                
            if self.db_manager:
                self.db_manager.save_portfolio(self.balance_usd, self.held_assets, self.position_metadata)
                
            # Notify Governor
            if self.governor:
                self.governor.sync_positions(self.held_assets, self.position_metadata)

        except Exception as e:
            print(f"[{self.name}] ❌ Reconciliation Failed: {e}")


    def sync_balance(self, confirmed_balance: float):
        """
        Force-update the internal balance to match Reality (Exchange).
        Overrides any stale DB state.
        """
        if abs(self.balance_usd - confirmed_balance) > 0.01:
            print(f"[{self.name}] 📉 Ledger Correction: Overwriting DB Balance ${self.balance_usd:.2f} with Real Capital ${confirmed_balance:.2f}")
            self.balance_usd = confirmed_balance
            self.initial_capital = confirmed_balance # update basis too
            self._persist_portfolio()
        else:
             print(f"[{self.name}] ✅ Ledger Balance Verified: ${self.balance_usd:.2f}")

    def gc_reconcile_positions(self) -> list:
        """
        Garbage Collector: Full bidirectional position sync with exchange.
        Returns list of ghost positions that were zeroed out.
        """
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        ghosts_found = []
        
        if verbose:
            print(f"[GC Monitor] 🔄 Running Position Reconciliation...")
        
        # Store current state to detect what changes
        before_assets = dict(self.held_assets)
        
        # Run the main reconciliation logic
        self.reconcile_exchange_positions()
        
        # Detect what was removed (ghost positions)
        for sym, qty in before_assets.items():
            if abs(qty) > 0.00000001:
                new_qty = self.held_assets.get(sym, 0.0)
                if abs(new_qty) < 0.00000001:
                    ghosts_found.append(sym)
        
        if verbose and ghosts_found:
            print(f"[GC Monitor] ✅ Position Reconciliation: {len(ghosts_found)} ghosts cleared: {ghosts_found}")
        elif verbose:
            print(f"[GC Monitor] ✅ Position Reconciliation: Clean - no discrepancies.")
        
        return ghosts_found


    def get_execution_summary(self) -> dict:
        """Returns a high-level summary of execution status and portfolio health."""
        equity = self.get_portfolio_value()
        margin_used = sum(
            (abs(qty) * self.entry_prices.get(sym, 0.0)) / self.position_metadata.get(sym, {}).get('leverage', 1.0)
            for sym, qty in self.held_assets.items() if abs(qty) > 1e-8
        )
        return {
            'balance': self.balance_usd,
            'equity': equity,
            'margin_used': margin_used,
            'margin_available': self.balance_usd, # Simplification: balance is essentially avail if we only subtract margin
            'active_positions': len([q for q in self.held_assets.values() if abs(q) > 1e-8]),
            'ledger_size': len(self.ledger)
        }



    def get_balance_details(self) -> Tuple[float, float]:
        """
        Returns (Total Equity, Free Margin).
        Used for Solvency Checks.
        """
        if self.actuator:
            # Equity = Total Net Worth (use specific equity method)
            equity = self.actuator.get_equity()
            # Free = Account Balance (Cash/Available Margin)
            free = self.actuator.get_account_balance()
            return equity, free
        else:
            return self.balance_usd, self.balance_usd

    def get_portfolio_value(self, current_price: float = 0.0) -> float:
        """Alias for Equity (Legacy Support)."""
        eq, _ = self.get_balance_details()
        return eq

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[Tuple[str, str]]:
        """
        Check if current price triggers Stop-Loss or Take-Profit for a specific symbol.
        Direction-aware (Long/Short).
        """
        entry_price = self.entry_prices.get(symbol)
        qty = self.held_assets.get(symbol, 0.0)
        
        if entry_price is None or abs(qty) < 0.00000001:
            return None
        
        # Determine Direction
        meta = self.position_metadata.get(symbol, {})
        direction = meta.get('direction', 'BUY')

        # Calculate PnL % 
        if direction == 'BUY':
            price_change_pct = (current_price - entry_price) / entry_price
        else: # SHORT (SELL)
            price_change_pct = (entry_price - current_price) / entry_price
        
        # Stop-Loss triggered
        if price_change_pct <= -self.stop_loss_pct:
            print(f"[{self.name}] {symbol} ({direction}) STOP-LOSS triggered at {price_change_pct*100:.2f}%")
            return 'STOP_LOSS'
        
        # Take-Profit triggered
        if price_change_pct >= self.take_profit_pct:
            print(f"[{self.name}] {symbol} ({direction}) TAKE-PROFIT triggered at {price_change_pct*100:.2f}%")
            return 'TAKE_PROFIT'
        
        return None

    def decide_trade(
        self,
        signal: TradeSignal,
        current_regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION'],
        entropy_score: float
    ) -> TradeDecision:
        """
        Decide whether to execute a trade based on market regime and entropy.
        
        Disposition Logic:
            - CHAOTIC: Autonomy=0.1, Integration=0.9 → HALT (reject trade)
            - ORDERED: Autonomy=0.9, Integration=0.1 → EXECUTE (full trade)
            - TRANSITION: Autonomy=0.5, Integration=0.5 → REDUCE (50% size)
        
        Constraint: The decision is ALWAYS hashed and saved to the ledger
        before returning.
        
        Args:
            signal: The trade signal to evaluate
            current_regime: The current market regime
            entropy_score: The current entropy score
            
        Returns:
            TradeDecision containing action, adjusted size, and block hash
        """
        action: Literal['EXECUTE', 'HALT', 'REDUCE']
        adjusted_size: float

        # Apply disposition logic based on continuous Sigmoid function
        # Autonomy = 1 / (1 + e^(k * (Entropy - Threshold)))
        # k=5 (steepness), Threshold=0.75 (calibrated for live data)
        # 
        # CALIBRATION NOTE (Phase 11):
        # - Original threshold: 2.0 (for backtest data with max entropy ~2.25)
        # - Live data max entropy: ~1.85
        # - Adjusted to 0.75 to enable HALT/REDUCE triggers        # - Live data max entropy: ~1.85
        # - Gaussian Noise: ~1.40
        # - Phase 34 Tuned Threshold: 1.1 (Midpoint of new Transition zone)
        # - Phase 35 Update: Raised to 1.5 to handle high-entropy assets (DOT, XTZ showing 1.8-1.9)
        # - UNLEASHED (Phase 36): Raised to 2.2 to basically disable the Halt. (Max Entropy ~2.3)
        k = 5.0
        threshold = 2.2  # UNLEASHED MODE: High tolerance for Chaos
        
        # Ranges:
        # Entropy < 1.0 (Ordered) -> Autonomy > 0.7
        # Entropy > 1.35 (Chaotic) -> Autonomy < 0.3
        
        autonomy = 1.0 / (1.0 + math.exp(k * (entropy_score - threshold)))
        
        # Integration is the inverse
        integration = 1.0 - autonomy
        
        self.disposition = Disposition(autonomy=autonomy, integration=integration)
        
        # Map continuous autonomy to discrete actions for Ledger/Protocol compliance
        # Autonomy > 0.6 -> EXECUTE (High Independence)
        # Autonomy < 0.4 -> HALT (High Safety)
        # 0.4 <= Autonomy <= 0.6 -> REDUCE (Balanced)
        
        if autonomy > 0.5:  # Relaxed from 0.6 to allow more entries
            action = 'EXECUTE'
            adjusted_size = signal.size
            
        elif autonomy < 0.05:  # Very relaxed - only HALT in extreme chaos
            action = 'HALT'
            adjusted_size = 0.0
            print(f"[{self.name}] ⚠️ HALT: Autonomy too low ({autonomy:.3f}), Entropy: {entropy_score:.2f}")
            
        else:
            # SOFT-HALT / REDUCE Range (0.05 - 0.5 autonomy)
            action = 'REDUCE'
            # Scale participation: 25% minimum, up to 100% near 0.5
            adjusted_size = signal.size * max(0.25, autonomy)

        # Check for sufficient funds/assets
        # Note: adjusted_size here is a multiplier (0.0 to 1.0) of the signal
        # The signal size is usually 1.0 (100% of intended move), but we interpret it as
        # "Target allocation of available execution power".
        # Real logic happens in execute_transaction, but we should update action if impossible.
        
        # For simplicity in this simulation, we keep the decision logic "pure" to the regime,
        # but the EXECUTION logic below handles the constraints.
        
        # CONSTRAINT: Always hash and save to ledger BEFORE returning
        block = self.ledger.add_block(
            entropy_score=entropy_score,
            regime=current_regime,
            action=action
        )
        
        # Persist Block
        if self.db_manager:
            self.db_manager.add_block(asdict(block))

        return TradeDecision(
            action=action,
            original_signal=signal,
            adjusted_size=adjusted_size,
            disposition=self.disposition,
            block_hash=block.hash,
            entropy_score=entropy_score
        )


    def record_external_decision(
        self,
        signal: TradeSignal,
        current_regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION'],
        entropy_score: float,
        action: Literal['EXECUTE', 'HALT', 'REDUCE']
    ) -> TradeDecision:
        """
        Record a decision made by an external agent (e.g., RL Agent) into the ledger.
        """
        # Set disposition based on the action (reverse engineer or just set based on regime?)
        # For consistency, let's just set a "delegated" disposition or match the action's typical one.
        
        if action == 'EXECUTE':
             self.disposition = Disposition(autonomy=0.9, integration=0.1)
             adjusted_size = signal.size
        elif action == 'HALT':
             self.disposition = Disposition(autonomy=0.1, integration=0.9)
             adjusted_size = 0.0
        else: # REDUCE
             self.disposition = Disposition(autonomy=0.5, integration=0.5)
             adjusted_size = signal.size * 0.5

        # Save to ledger
        block = self.ledger.add_block(
            entropy_score=entropy_score,
            regime=current_regime,
            action=action
        )
        
        # Persist Block
        if self.db_manager:
            self.db_manager.add_block(asdict(block))

        return TradeDecision(
            action=action,
            original_signal=signal,
            adjusted_size=adjusted_size,
            disposition=self.disposition,
            block_hash=block.hash,
            entropy_score=entropy_score
        )

    def execute_transaction(self, decision: TradeDecision, current_price: float) -> Optional[float]:
        """
        Premium Unified Execution Engine.
        Executes trade decisions against the portfolio and/or linked Actuator.
        Handles Long Entries, Long Exits, Short Entries, and Short Covers.
        
        Returns:
            Optional[float]: Realized PnL percentage if an exit occurred, else None.
        """
        symbol = decision.original_signal.symbol
        direction = decision.original_signal.direction
        action_type = decision.action
        pnl_to_return = None
        
        if action_type == 'HALT' or decision.adjusted_size <= 0:
            return None

        current_holding = self.held_assets.get(symbol, 0.0)
        
        # 1. CLASSIFY TRANSACTION
        # ---------------------------------------------------------
        is_long_entry = (direction == 'BUY' and current_holding >= -0.00000001)
        is_short_cover = (direction == 'BUY' and current_holding < -0.00000001)
        is_long_exit = (direction == 'SELL' and current_holding > 0.00000001)
        is_short_entry = (direction == 'SELL' and current_holding <= 0.00000001)

        # 2. GOVERNOR VALIDATION & SIZING (For Entries)
        # ---------------------------------------------------------
        
        # --- SPAM GUARD with TIMEOUT ---
        # Check if we already have a working order for this symbol
        # But allow override if the order is stale (older than 70s - Actuator timeout is ~60s)
        WORKING_ORDER_TIMEOUT = 70  # Reduced from 300s to prevent deadlocks
        
        if self.actuator:
            has_pending = False
            stale_order_indices = []
            current_time = time.time()
            
            for idx, o in enumerate(getattr(self.actuator, 'pending_orders', [])):
                if o.get('symbol') == symbol:
                    order_age = current_time - o.get('timestamp', current_time)
                    
                    if order_age > WORKING_ORDER_TIMEOUT:
                        # Order is stale, mark for removal
                        stale_order_indices.append(idx)
                        print(f"[{self.name}] ⏰ STALE WORKING ORDER: {symbol} ({order_age:.0f}s old). Clearing.")
                    else:
                        has_pending = True
                        break
            
            # Remove stale orders (in reverse to preserve indices)
            for idx in reversed(stale_order_indices):
                try:
                    stale_order = self.actuator.pending_orders.pop(idx)
                    print(f"[{self.name}] 🗑️ Removed stale order {stale_order.get('id', 'N/A')}")
                except:
                    pass
                        
            if has_pending:
                # --- EMERGENCY OVERRIDE FOR STOP LOSS ---
                signal_reason = decision.original_signal.metadata.get('reason', '')
                if signal_reason == 'STOP_LOSS':
                    print(f"[{self.name}] 🚨 EMERGENCY STOP LOSS: Cancelling conflicting working orders for {symbol} to force exit.")
                    
                    orders_to_cancel = [o for o in self.actuator.pending_orders if o.get('symbol') == symbol]
                    
                    for order_record in orders_to_cancel:
                        order_id = order_record.get('id')
                        try:
                            # Try to force cancel on exchange via Actuator's connection
                            if self.actuator and hasattr(self.actuator, 'exchange'):
                                self.actuator.exchange.cancel_order(order_id, symbol)
                                print(f"[{self.name}] 🗑️ Force Cancelled Order {order_id}")
                        except Exception as e:
                            print(f"[{self.name}] ⚠️ Force Cancel Failed for {order_id}: {e}")
                        
                        # Remove from pending list to unlock execution
                        if order_record in self.actuator.pending_orders:
                            self.actuator.pending_orders.remove(order_record)
                    
                    # Proceed with execution (has_pending check bypassed)
                    pass 
                else: 
                    print(f"[{self.name}] ⏳ Working Order exists for {symbol}. Skipping duplicate execution.")
                    return None
        # ------------------

        leverage = 1.0
        exec_qty = 0.0
        
        if is_long_entry or is_short_entry:
            # Treat adjusted_size as Absolute Quantity (Units of Asset)
            # This aligns with Governor's output in Phase 4
            requested_qty = decision.adjusted_size
            usd_to_spend = requested_qty * current_price
                
            if self.governor:
                self.governor.update_balance(self.get_portfolio_value())
                
                # Extract context from signal metadata
                meta_atr = decision.original_signal.metadata.get('atr')
                meta_conviction = decision.original_signal.metadata.get('ppo_conviction')
                
                is_approved, safe_qty, leverage = self.governor.receive_message(self, {
                    'type': 'VALIDATE_TRADE', 
                    'price': current_price, 
                    'symbol': symbol,
                    'direction': direction,
                    'atr': meta_atr,
                    'conviction': meta_conviction,
                    'crisis_score': decision.entropy_score
                })
                
                if not is_approved:
                    print(f"  [RISK] Governor REJECTED {direction} for {symbol}.")
                    return None
                
                # Cap the requested quantity by the Governor's safe limit
                exec_qty = min(requested_qty, safe_qty)
                
            else:
                # No Governor? Use the requested quantity directly
                exec_qty = requested_qty
                leverage = 1.0
            
            # --- SOLVENCY CHECK (Prevent Infinite Negative Balance) ---
            # Ensure the Margin Requirement doesn't exceed available Free Balance
            
            # PATCH 5: REAL-TIME MARGIN CHECK (Buying Power)
            if self.actuator:
                # Use Leveraging Power if available
                # We ask for buying power at default 5x leverage
                real_avail_power = self.actuator.get_buying_power(leverage=5.0)
                avail_capital = real_avail_power
            else:
                avail_capital = self.balance_usd * 5.0 # Simulating 5x in paper

            # Safety Buffer (Leave 1% for fees/slippage)
            safe_capital = avail_capital * 0.99
            
            # Margin Requirement for the new order (Notional Value)
            # When checking against Buying Power (Equity * Lev), we check against the full Notional Value of the trade
            notional_req = exec_qty * current_price
            
            if notional_req > safe_capital:
                # Cap to Max Buying Power
                max_affordable_notional = max(0.0, safe_capital)
                max_qty = max_affordable_notional / current_price
                
                # Check Min Order Value again after capping
                if (max_qty * current_price) < config.MIN_ORDER_VALUE:
                    print(f"  [SOLVENCY] ❌ Insufficient Buying Power. Req: ${config.MIN_ORDER_VALUE}, Power: ${max_affordable_notional:.2f} (Nav: ${safe_capital:.2f})")
                    return None
                    
                print(f"  [SOLVENCY] Capping Qty {exec_qty:.4f} -> {max_qty:.4f} (Power ${avail_capital:.2f})")
                exec_qty = max_qty

            if exec_qty < 0.00000001: return None
            
        elif is_long_exit or is_short_cover:
            # For exits, we use the specified size from the decision
            exec_qty = abs(current_holding) * decision.adjusted_size
            if action_type == 'EXIT': exec_qty = abs(current_holding)
            # Leverage is pulled from existing position metadata
            meta = self.position_metadata.get(symbol, {})
            leverage = meta.get('leverage', 1.0)

        # 3. INTERACT WITH ACTUATOR (REAL MARKET) OR SIMULATE
        # ---------------------------------------------------------
        fills = []
        if self.actuator:
            # Map logical signal to actuator direction
            # Long Entry: BUY
            # Short Cover: BUY
            # Long Exit: SELL
            # Short Entry: SELL
            
            # --- PATCH: PASS LEVERAGE ---
            # We must pass the leverage we calculated/retrieved to the Actuator
            # so it can set it on the exchange before
            # 4. EXECUTE VIA ACTUATOR
            # ------------------------------------------------
            # ORDER TYPE LOGIC:
            # - ENTRIES (Long/Short) -> MARKET (Priority: Speed/Certainty)
            # - EXITS (Close/Reduce) -> LIMIT (Priority: Rebates/Cost)
            #   (Unless it's a Stop Loss, which might need Market, but for now Limit w/ actuator retry logic)
            
            order_type = 'market' if (is_long_entry or is_short_entry) else 'limit'
            
            # Urgent if it's an Exit or Cover (We want to realize PnL, not fish for rebates endlessly)
            urgent_flag = (is_long_exit or is_short_cover)
            
            # Execute
            # Execute
            tx_id = self.actuator.place_order(
                symbol=symbol,
                direction=direction, # Actuator handles side map
                quantity=exec_qty,
                price=current_price,
                order_type=order_type,
                margin=True, 
                leverage=leverage,
                urgent=urgent_flag,
                reduce_only=(is_long_exit or is_short_cover)
            )
            
            if tx_id:
                print(f"[{self.name}] 📡 ORDER SENT {direction} {symbol}: {exec_qty} @ {current_price} ({order_type.upper()}). Verifying...")
                
                # --- HARD EXECUTION GATE (Request A/C) ---
                # Kill Optimistic Updates. Truth comes from Exchange.
                time.sleep(2.0) # Network propagation buffer
                
                # Verify Logic
                max_retries = 3
                confirmed_fill = None
                
                for attempt in range(max_retries):
                     order_status = self.actuator.fetch_order_status(tx_id, symbol)
                     if not order_status:
                         time.sleep(1.0)
                         continue
                         
                     status = order_status.get('status')
                     filled = float(order_status.get('filled', 0.0))
                     
                     if status == 'closed' or filled > 0:
                         # We have a fill!
                         confirmed_fill = order_status
                         break
                     elif status == 'canceled' or status == 'rejected':
                         print(f"[{self.name}] ❌ Order {tx_id} was {status}. Execution Failed.")
                         break
                     else:
                         # Open/New - Wait longer
                         time.sleep(1.0 + attempt)
                
                if not confirmed_fill:
                     print(f"[{self.name}] ⚠️ EXECUTION UNCONFIRMED: Order {tx_id} not filled after verification. Dropping from State.")
                     # FIX: Ensure we remove it from pending_orders so we can try again immediately
                     if self.actuator:
                         to_remove = [o for o in self.actuator.pending_orders if o.get('id') == tx_id]
                         for o in to_remove:
                             self.actuator.pending_orders.remove(o)
                     return None
                     
                # Use CONFIRMED data for updates
                final_fill_qty = float(confirmed_fill.get('filled', 0.0))
                remaining = float(confirmed_fill.get('remaining', 0.0))
                avg_price = float(confirmed_fill.get('average', current_price)) 
                # If average is None (some exchanges), use price or current_price
                if avg_price is None or avg_price == 0: avg_price = current_price
                
                # Only update fills with REAL data
                fills = [{
                    'symbol': symbol,
                    'direction': direction,
                    'filled_qty': final_fill_qty,
                    'price': avg_price,
                    'cost_usd': final_fill_qty * avg_price,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }]
                
                if final_fill_qty < (exec_qty * 0.9):
                     print(f"[{self.name}] ⚠️ PARTIAL FILL: Requested {exec_qty}, Got {final_fill_qty}. State Updated.")
            else:
                 # Tx Failed
                 return None

        # 4. PROCESS RESULTS & UPDATE STATE (Now utilizing VERIFIED fills)
        # ---------------------------------------------------------
        # Filter for RELEVANT fills only (Prevent Cross-Talk)
        relevant_fills = [f for f in fills if f.get('symbol') == symbol]
        
        if not relevant_fills: return None
        
        fill = relevant_fills[0]
        actual_qty = fill['filled_qty']
        
        if actual_qty == 0: return None # Happened on partial
        
        actual_price = fill['price']
        notional_value = fill['cost_usd']
        margin_impact = notional_value / leverage
        
        if is_long_entry:
            self.balance_usd -= margin_impact
            old_qty = self.held_assets.get(symbol, 0.0)
            
            # --- PATCH 4: REALITY CHECK (Fix the Math) ---
            # Ensure we are adding Quantity (Units), NOT Notional (USD)
            new_qty = old_qty + actual_qty
            
            # Sanity Guard
            if new_qty > 1_000_000 and actual_price > 1.0:
                 print(f"[{self.name}] ⚠️ CRITICAL MATH WARNING: Phantom Whale Detected? Qty: {new_qty}")
            # ---------------------------------------------
            
            # Weighted average entry price
            old_entry = self.entry_prices.get(symbol, actual_price)
            if old_qty > 0:
                self.entry_prices[symbol] = ((old_qty * old_entry) + (actual_qty * actual_price)) / new_qty
            else:
                 self.entry_prices[symbol] = actual_price
                 
            self.held_assets[symbol] = new_qty
            
            # Persist original timestamp if stacking, else new
            existing_ts = self.position_metadata.get(symbol, {}).get('entry_timestamp')
            final_ts = existing_ts if existing_ts else datetime.now(timezone.utc).isoformat()
            
            # Extract PPO Metadata if available
            ppo_state = decision.original_signal.metadata.get('ppo_state')
            ppo_conv = decision.original_signal.metadata.get('ppo_conviction')

            self.position_metadata[symbol] = {
                'leverage': leverage,
                'entry_price': self.entry_prices[symbol],
                'entry_timestamp': final_ts,
                'direction': 'BUY',
                'ppo_state': ppo_state,
                'ppo_conviction': ppo_conv,
                'stack_count': self.position_metadata.get(symbol, {}).get('stack_count', 0) + 1
            }
            print(f"[{self.name}] LONG ENTRY: {symbol} @ {actual_price} (Qty: {actual_qty:.4f}, NewTotal: {new_qty:.4f}, Margin: ${margin_impact:.2f})")
            
        elif is_short_entry:
            self.balance_usd -= margin_impact
            old_qty_abs = abs(self.held_assets.get(symbol, 0.0))
            
            # --- PATCH 4: REALITY CHECK (Fix the Math) ---
            # new_total_qty = old_qty + order_qty
            new_qty_abs = old_qty_abs + actual_qty
            
             # Sanity Guard
            if new_qty_abs > 1_000_000 and actual_price > 1.0:
                 print(f"[{self.name}] ⚠️ CRITICAL MATH WARNING: Phantom Whale Detected? Qty: {new_qty_abs}")
            # ---------------------------------------------

            # Weighted average entry price
            old_entry = self.entry_prices.get(symbol, actual_price)
            self.entry_prices[symbol] = ((old_qty_abs * old_entry) + (actual_qty * actual_price)) / new_qty_abs
            self.held_assets[symbol] = -new_qty_abs
            
            # Persist original timestamp if stacking, else new
            existing_ts = self.position_metadata.get(symbol, {}).get('entry_timestamp')
            final_ts = existing_ts if existing_ts else datetime.now(timezone.utc).isoformat()
            
            # Extract PPO Metadata if available
            ppo_state = decision.original_signal.metadata.get('ppo_state')
            ppo_conv = decision.original_signal.metadata.get('ppo_conviction')
            
            self.position_metadata[symbol] = {
                'leverage': leverage,
                'entry_price': self.entry_prices[symbol],
                'entry_timestamp': final_ts,
                'direction': 'SELL',
                'ppo_state': ppo_state,
                'ppo_conviction': ppo_conv,
                'stack_count': self.position_metadata.get(symbol, {}).get('stack_count', 0) + 1
            }
            print(f"[{self.name}] SHORT ENTRY: {symbol} @ {actual_price} (Qty: {actual_qty:.4f}, NewTotal: {new_qty_abs:.4f}, Margin: ${margin_impact:.2f})")

        elif is_long_exit:
            entry_p = self.entry_prices.get(symbol, actual_price)
            pnl_usd = (actual_price - entry_p) * actual_qty
            pnl_pct = (actual_price - entry_p) / entry_p if entry_p > 0 else 0
            margin_released = (actual_qty * entry_p) / leverage
            
            self.balance_usd += (margin_released + pnl_usd)
            self.held_assets[symbol] -= actual_qty
            pnl_to_return = pnl_pct
            print(f"[{self.name}] LONG EXIT: {symbol} @ {actual_price} (PnL: {pnl_pct*100:+.2f}%, ${pnl_usd:+.2f})")

        elif is_short_cover:
            entry_p = self.entry_prices.get(symbol, actual_price)
            pnl_usd = (entry_p - actual_price) * actual_qty
            pnl_pct = (entry_p - actual_price) / entry_p if entry_p > 0 else 0
            margin_released = (actual_qty * entry_p) / leverage
            
            self.balance_usd += (margin_released + pnl_usd)
            self.held_assets[symbol] += actual_qty
            pnl_to_return = pnl_pct
            print(f"[{self.name}] SHORT COVER: {symbol} @ {actual_price} (PnL: {pnl_pct*100:+.2f}%, ${pnl_usd:+.2f})")

        # Cleanup positions that are fully closed
        if abs(self.held_assets.get(symbol, 0.0)) < 0.00000001:
            self.held_assets[symbol] = 0.0
            if symbol in self.entry_prices: del self.entry_prices[symbol]
            if symbol in self.position_metadata: del self.position_metadata[symbol]

        # 5. POST-EXECUTION: LOGGING & SYNC
        # ---------------------------------------------------------
        self._persist_portfolio()
        
        # Save Trade to Ledger DB
        if self.db_manager:
            # We record entries with cost and 0 pnl, exits with pnl and 0 cost (relative to close)
            is_exit = is_long_exit or is_short_cover
            self.db_manager.save_trade({
                'symbol': symbol,
                'direction': direction,
                'quantity': actual_qty,
                'price': actual_price,
                'cost_usd': margin_impact if not is_exit else 0,
                'leverage': leverage,
                'notional_value': notional_value,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pnl': pnl_usd if is_exit else 0.0,
                'pnl_percent': pnl_pct if is_exit else 0.0,
                'unrealized_pnl': 0.0,
                'unrealized_pnl_percent': 0.0
            })

        # Notify Governor
        if self.governor:
            gov_dir = 'LONG' if is_long_entry else ('SHORT' if is_short_entry else ('EXIT' if is_long_exit else 'COVER'))
            self.governor.receive_message(self, {
                'type': 'POSITION_FILLED',
                'symbol': symbol,
                'direction': gov_dir,
                'price': actual_price,
                'quantity': actual_qty if (is_long_entry or is_short_cover) else -actual_qty
            })

        self.last_order_details = f"{direction} {actual_qty:.4f} {symbol} @ {actual_price:.2f}"
        
        # --- IMPROVEMENT: EMIT ORDER TO DASHBOARD ---
        if self.gui_queue:
            try:
                self.gui_queue.put({
                    'type': 'order',
                    'data': {
                        'Time': datetime.now().strftime('%H:%M:%S'),
                        'Symbol': symbol,
                        'Side': direction,
                        'Qty': f"{actual_qty:.4f}",
                        'Price': f"{actual_price:.4f}",
                        'Status': 'FILLED'
                    }
                })
            except: pass
        # --------------------------------------------
        
        return pnl_to_return

    def get_portfolio_value(self, current_price_ref: float = 0.0) -> float:
        """
        Calculate total portfolio value in USD based on all held assets (Leveraged Equity).
        Equity = Free Balance + Sum(Margin Used + Unrealized PnL)
        Direction-aware (Long/Short).
        """
        equity = self.balance_usd
        
        for sym, qty in self.held_assets.items():
            if abs(qty) < 0.00000001: continue
            
            current_price = self.latest_prices.get(sym, 0.0)
            entry_price = self.entry_prices.get(sym, 0.0)
            meta = self.position_metadata.get(sym, {})
            leverage = meta.get('leverage', 1.0)
            direction = meta.get('direction', 'BUY')
            
            if current_price > 0 and entry_price > 0:
                qty_abs = abs(qty)
                # Margin currently locked in this position
                margin_used = (qty_abs * entry_price) / leverage
                
                # Unrealized PnL
                if direction == 'BUY':
                    unrealized_pnl = (current_price - entry_price) * qty_abs
                else: # SELL (SHORT)
                    unrealized_pnl = (entry_price - current_price) * qty_abs
                
                equity += (margin_used + unrealized_pnl)
                
        return equity



    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Handle incoming messages.
        """
        if isinstance(content, Message):
            if content.type == 'EXECUTE':
                 pass # Logic to trigger execution via message
        else:
            pass

    def panic_close_all(self, current_prices: Dict[str, float]) -> List[str]:
        """
        🚨 PANIC BUTTON: Force close ALL positions immediately.
        Bypasses Governor, Risk Checks, and Disposition.
        Uses Actuator directly for maximum speed.
        """
        print(f"[{self.name}] 🚨🚨 PANIC PROTOCOL INITIATED 🚨🚨")
        results = []
        
        # Iterate over a copy of items since we'll modify the dict
        for symbol, qty in list(self.held_assets.items()):
            if abs(qty) < 0.00000001: continue
            
            price = current_prices.get(symbol, self.latest_prices.get(symbol, 0.0))
            if price <= 0:
                results.append(f"❌ {symbol}: No Price Data")
                continue
                
            direction = 'SELL' if qty > 0 else 'BUY' # Exit Long or Cover Short
            # Panic -> Market Order equivalent (Aggressive Limit)
            
            # Calculate PnL for DB Record
            entry_p = self.entry_prices.get(symbol, price)
            pnl_usd = 0.0
            pnl_pct = 0.0
            
            if direction == 'SELL': # Closing Long
                pnl_usd = (price - entry_p) * abs(qty)
                pnl_pct = (price - entry_p) / entry_p if entry_p > 0 else 0
            else: # Covering Short
                pnl_usd = (entry_p - price) * abs(qty) 
                pnl_pct = (entry_p - price) / entry_p if entry_p > 0 else 0

            print(f"[{self.name}] PANIC CLOSING {symbol} ({qty:.4f}) @ {price} | PnL: ${pnl_usd:.2f} ({pnl_pct*100:.2f}%)")
            
            # Direct Actuator Call
            if self.actuator:
                # Use max leverage for Panic Close? Or assume 1x? 
                # Panic close is usually reduce-only or implied. 
                # Ideally, we knew the leverage of the position we are closing.
                # Let's try to grab it from metadata.
                panic_lev = self.position_metadata.get(symbol, {}).get('leverage', 1.0)
                # Fix: Use place_order instead of place_limit_order, set reduce_only=True
                self.actuator.place_order(
                    symbol=symbol, 
                    direction=direction, 
                    quantity=abs(qty), 
                    price=price, 
                    order_type='limit',
                    margin=True, 
                    leverage=panic_lev,
                    reduce_only=True,
                    urgent=True
                )
                
                
            # SAFE STATE WIPE: Only if order was submitted (Actuator didn't crash)
            # Ideally we check result[id], but actuator returns None on hard failure
            # If we used Actuator, trust the fire-and-forget nature of panic for now, 
            # BUT at least ensure we reached the call.
            
            self.balance_usd += (abs(qty) * price) # Roughly returning capital
            
            # Save to Ledger DB
            if self.db_manager:
                self.db_manager.save_trade({
                    'symbol': symbol,
                    'direction': direction,
                    'quantity': abs(qty),
                    'price': price,
                    'cost_usd': 0, # Exit so cost is 0 (Margin Release logic)
                    'leverage': self.position_metadata.get(symbol, {}).get('leverage', 1.0),
                    'notional_value': abs(qty) * price,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'pnl': pnl_usd,
                    'pnl_percent': pnl_pct,
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_percent': 0.0,
                    'note': 'PANIC_CLOSE'
                })

            del self.held_assets[symbol]
            if symbol in self.entry_prices: del self.entry_prices[symbol]
            if symbol in self.position_metadata: del self.position_metadata[symbol]
            
            results.append(f"✅ {symbol} CLOSED (PnL ${pnl_usd:.2f})")
            
        self._persist_portfolio()
        return results

    def get_ledger_summary(self) -> dict:
        """
        Get a summary of the audit ledger.
        
        Returns:
            Dictionary with chain length, validity, and last block info
        """
        return {
            'total_blocks': len(self.ledger),
            'chain_valid': self.ledger.verify_chain(),
            'last_block': asdict(self.ledger.chain[-1]) if self.ledger.chain else None
        }
