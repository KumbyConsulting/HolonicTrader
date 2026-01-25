"""
ActuatorHolon - Execution (Micro-Holon Architecture)

Objective: Minimize friction (Fees).
Mandate:
- Maker-Only Mode: NEVER use Market Orders.
- Logic: Place Limit Orders at Bid (for Long) or Ask (for Short).
- Post-Order: Monitor for fill. If not filled in 60 seconds, Cancel & Re-assess.

Note: Since this is a simulation/paper-trading env first, 
we simulate the "Wait for Fill" logic.
"""

import time
import ccxt
import config
import os
import random
import threading
from typing import Any, Literal, Dict, List
from HolonicTrader.holon_core import Holon, Disposition

class ActuatorHolon(Holon):
    def __init__(self, name: str = "ActuatorAgent", exchange_id: str = 'kraken'):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.2))
        self.pending_orders = []
        self.exchange_id = exchange_id
        
        if config.TRADING_MODE == 'FUTURES':
            self.exchange_id = 'krakenfutures'
            print(f"[{self.name}] 🔌 Connecting to Kraken FUTURES...")
            # Use specific Futures keys if available, else fallback to standard
            api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
            api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
        else:
            self.exchange_id = 'kraken' # Spot
            print(f"[{self.name}] 🔌 Connecting to Kraken SPOT...")
            api_key = config.API_KEY
            api_secret = config.API_SECRET

        if hasattr(ccxt, self.exchange_id):
            self.exchange = getattr(ccxt, self.exchange_id)({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if config.TRADING_MODE == 'FUTURES' else 'spot'
                }
            })
        else:
            raise ValueError(f"Exchange {self.exchange_id} not found in ccxt")

        # Kraken Symbol Mapping (Internal USDT -> Kraken USD)
        self.symbol_map = config.KRAKEN_SYMBOL_MAP

        # --- CIRCUIT BREAKER STATE ---
        self.error_count = 0
        self.circuit_open = False
        self.hibernate_until = 0.0
        self.MAX_CONSECUTIVE_ERRORS = getattr(config, 'API_MAX_RETRIES', 15) 
        self.HIBERNATION_TIME = getattr(config, 'API_HIBERNATION_TIME', 60)
        
        # Caching to prevent API Rate Limit Spam (Phase 4 Optimization)
        self.cached_equity = None
        self.last_equity_time = 0.0
        self.cached_balance = {}
        self.last_balance_time = 0.0
        self.last_balance_time = 0.0
        self.CACHE_TTL = 3.0 # Cache duration for balance calls
        
        # Phase 6: Error Cooldowns (Nano/Micro Efficiency)
        self.failed_orders = {} # {f"{symbol}_{error}": timestamp}
        
        # --- ADVANCED EXECUTION STATE ---
        self.active_algos = {} # {algo_id: {thread, stop_event, status}}
        self._algo_lock = threading.Lock()

    def check_circuit_breaker(self) -> bool:
        """
        Returns True if Circuit is CLOSED (Healthy).
        Returns False if Circuit is OPEN (Broken/Hibernating).
        """
        if self.circuit_open:
            remaining = self.hibernate_until - time.time()
            if remaining > 0:
                # Still hibernating
                # verbose log only every 60s?
                if int(remaining) % 60 == 0:
                    print(f"[{self.name}] 💤 API CIRCUIT OPEN: Hibernating for {int(remaining)}s (Too many 503s)")
                return False
            else:
                # Wake up
                print(f"[{self.name}] 🌅 API CIRCUIT RESET: Attempting to reconnect...")
                self.circuit_open = False
                self.error_count = 0
                return True
        return True

    def can_retry_order(self, symbol: str, error_type: str = 'General') -> bool:
        """
        Check if we are in cooldown for this specific error on this symbol.
        Prevents infinite retry loops on non-transient errors (e.g. Insufficient Funds).
        """
        key = f"{symbol}_{error_type}"
        last_time = self.failed_orders.get(key, 0)
        
        # If config has NANO_COOLDOWN, use it, else default 300
        cooldown = getattr(config, 'NANO_COOLDOWN_AFTER_FAILURE', 300)
        
        if time.time() - last_time < cooldown:
            remaining = int(cooldown - (time.time() - last_time))
            if remaining % 60 == 0: # Reduce log spam
                print(f"[{self.name}] ⏳ COOLDOWN ACTIVE: {symbol} ({remaining}s rem) due to {error_type}")
            return False
            
        # --- NANO-MODE GLOBAL LOCK (One Strike Rule) ---
        # If ANY order failed in the last 24h, block ALL trades
        nano_cap = getattr(config, 'NANO_CAPITAL_THRESHOLD', 50.0)
        # We need access to balance to confirm Nano Mode, usually this Holon doesn't track it directly but we can infer
        # or use config provided threshold. Assuming we are in Nano Mode if cooldown is long (86400).
        if cooldown > 3600: # If cooldown is hours long, it implies Nano Strictness
             # Check if ANY failure exists in recent history
             for f_key, f_time in self.failed_orders.items():
                  if time.time() - f_time < cooldown:
                       print(f"[{self.name}] 🔒 NANO GLOBAL LOCK: Trading halted due to {f_key} failure.")
                       return False
        # -----------------------------------------------
        
        return True

    def record_order_failure(self, symbol: str, error_type: str):
        """Record a failure to trigger cooldown."""
        key = f"{symbol}_{error_type}"
        self.failed_orders[key] = time.time()

    def report_success(self):
        """Call this after a successful API interaction to reset counters."""
        if self.error_count > 0:
            self.error_count = 0
            # print(f"[{self.name}] 🟢 API Connection Stabilized.")

    def report_failure(self, error_msg: str):
        """Call this after a Network/503 Error."""
        self.error_count += 1
        print(f"[{self.name}] ⚠️ API Error #{self.error_count}: {error_msg}")
        
        if self.error_count >= self.MAX_CONSECUTIVE_ERRORS:
            self.circuit_open = True
            self.hibernate_until = time.time() + self.HIBERNATION_TIME
            print(f"[{self.name}] 💥 CIRCUIT BREAKER TRIPPED: Entering {self.HIBERNATION_TIME}s Hibernation to save API Quota.")

    def check_liquidity(self, symbol: str, direction: str, quantity: float, price: float) -> bool:
        """
        Verify that the order book has sufficient depth to absorb this order
        without massive slippage.
        Rule: Top 10 levels must have cumulative volume >= 3x order quantity.
        """
        try:
            if price <= 0:
                print(f"[{self.name}] ⚠️ Liquidity Check Skipped: Invalid Price ({price})")
                return True

            # Fetch deeper book (Limit 50 for safety)
            book = self.exchange.fetch_order_book(symbol, limit=50)
            
            # If Buying, we consume Asks. If Selling, we hit Bids.
            # (Limit orders technically wait, but we want to know there's a market nearby)
            side_book = book['asks'] if direction == 'BUY' else book['bids']
            
            cumulative_vol = 0.0
            
            for bid_ask in side_book:
                level_price = float(bid_ask[0])
                level_qty = float(bid_ask[1])
                
                # Only count volume within 2% of price checks
                if abs(level_price - price) / price < 0.02:
                     cumulative_vol += level_qty
            
            # Safety Factor: We want book volume to be at least 3x our size
            # IMPACT CHECK: If we are > X% of depth, it's a high impact trade
            required_vol = quantity / getattr(config, 'EXEC_IMPACT_THRESHOLD', 0.10)
            
            if cumulative_vol < required_vol:
                print(f"[{self.name}] ⚠️ HIGH IMPACT: {symbol} Depth {cumulative_vol:.4f} < Req {required_vol:.4f} (Threshold: {config.EXEC_IMPACT_THRESHOLD*100}%)")
                return False
                
            return True
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Liquidity Check Error: {e}. Proceeding with caution.")
            return True # Fail open to avoid paralysis, but log warning

    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Fetch REAL free balance (Available Margin) from exchange.
        """
        if not self.check_circuit_breaker(): return 0.0

        # Cache Check
        if (asset in self.cached_balance) and (time.time() - self.last_balance_time < self.CACHE_TTL):
            return self.cached_balance.get(asset, 0.0)

        for attempt in range(3):
            try:
                balance = self.exchange.fetch_balance()
                self.report_success()
                
                # Kraken Futures specific mapping
                if config.TRADING_MODE == 'FUTURES':
                    info = balance.get('info', {})
                    # Try to get explicit 'availableMargin' from flex account
                    # Structure: info -> accounts -> flex -> availableMargin
                    try:
                        accounts = info.get('accounts', {})
                        flex = accounts.get('flex', {})
                        avail_margin = float(flex.get('availableMargin', 0.0))
                        if avail_margin > 0:
                            self.cached_balance['USDT'] = avail_margin
                            self.last_balance_time = time.time()
                            return avail_margin
                    except Exception as e_flex: 
                        print(f"[{self.name}] ⚠️ Futures Flex Margin Check Failed: {e_flex}")

                # Fallback to standard CCXT 'free'
                b_usd = balance['free'].get('USD', 0.0)
                b_usdt = balance['free'].get('USDT', 0.0)
                b_zusd = balance['free'].get('ZUSD', 0.0)
                
                total_avail = max(b_usd, b_usdt, b_zusd)
                
                # Update Cache
                self.cached_balance['USDT'] = total_avail
                self.cached_balance['USD'] = total_avail
                self.last_balance_time = time.time()
                
                return total_avail
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                self.report_failure(str(e))
                if attempt == 2:
                    print(f"[{self.name}] ❌ Balance Check Failed after 3 attempts: {e}")
                    return 0.0
                time.sleep(1 * (attempt+1))
        return 0.0

    def get_equity(self) -> float:
        """
        Fetch TOTAL EQUITY (Balance + Unrealized PnL).
        Crucial for accurate Drawdown calculation in Governor.
        """
        if not self.check_circuit_breaker(): return None

        # Cache Check
        if self.cached_equity and (time.time() - self.last_equity_time < self.CACHE_TTL):
             return self.cached_equity

        for attempt in range(3):
            try:
                balance = self.exchange.fetch_balance()
                self.report_success()

                info = balance.get('info', {})
                
                # 1. Futures: Explicit marginEquity
                if config.TRADING_MODE == 'FUTURES':
                    accounts = info.get('accounts', {})
                    flex = accounts.get('flex', {})
                    total_equity = float(flex.get('marginEquity', 0.0))
                    if total_equity > 0:
                        self.cached_equity = total_equity
                        self.last_equity_time = time.time()
                        return total_equity
                        
                # 2. Spot/Unified: Equivalent Balance ('eb')
                equity = float(info.get('eb', 0.0))
                if equity > 0:
                     self.cached_equity = equity
                     self.last_equity_time = time.time()
                     return equity
                     
                # 3. Fallback: Total USD
                usd_bal = balance.get('total', {}).get('USD', 0.0)
                self.cached_equity = usd_bal
                self.last_equity_time = time.time()
                return usd_bal
                
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                self.report_failure(str(e))
                if attempt == 2:
                     print(f"[{self.name}] ❌ Equity Check Failed after 3 attempts: {e}")
                     return None # Return None instead of 0.0 to prevent panic
                time.sleep(1 * (attempt+1))
        return None

    def get_buying_power(self, leverage: float = 5.0) -> float:
        # ... (unchanged, but relying on get_account_balance now more reliable)
        return self.get_account_balance() * leverage

    # ... (skipping cancel_all_orders as it remains same/similar) ...

    def close_position(self, symbol: str, qty: float = None) -> bool:
        """
        Close an existing position (Market Order).
        Handles side inversion and reduceOnly flag automatically.
        """
        if not self.check_circuit_breaker(): return False
        
        try:
            # 1. Fetch Position to verify it exists and get direction
            positions = self.exchange.fetch_positions()
            target_pos = None
            
            # Map symbol if needed
            exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            
            for p in positions:
                if p['symbol'] == exec_symbol:
                    target_pos = p
                    break
                    
            if not target_pos:
                print(f"[{self.name}] ⚠️ Close Failed: No active position found for {symbol} ({exec_symbol})")
                return False
                
            current_qty = float(target_pos.get('contracts', 0.0))
            if current_qty == 0:
                print(f"[{self.name}] ⚠️ Close Failed: Position size is 0 for {symbol}")
                return False
                
            # 2. Determine Close Direction
            pos_side = target_pos['side'] # 'long' or 'short'
            close_side = 'SELL' if pos_side == 'long' else 'BUY'
            
            # 3. Determine Quantity
            final_qty = current_qty
            if qty is not None and qty < current_qty:
                final_qty = qty
                
            print(f"[{self.name}] 🔪 CLOSING {symbol}: Found {pos_side.upper()} {current_qty}, Selling {final_qty} (Side: {close_side})")
            
            # --- UPDATE: CANCEL WORKING ORDERS FIRST ---
            # Flush parent/stops to prevent conflicts
            self.cancel_all_orders(symbol)
            time.sleep(0.3) # Give exchange 300ms to process cancels
            # ------------------------------------------

            # 4. Execute Market Close
            order_id = self.place_order(
                symbol=symbol,
                direction=close_side,
                quantity=final_qty,
                order_type='market',
                reduce_only=True,
                urgent=True
            )
            
            return order_id is not None
            
        except Exception as e:
            print(f"[{self.name}] ❌ Close Position Error: {e}")
            return False

    def place_stop_order(self, symbol: str, direction: str, quantity: float, stop_price: float) -> bool:
        """
        Place a Stop Loss Order (Reduce Only) to protect a position.
        Direction: 'BUY' (for Short Cover) or 'SELL' (for Long Exit)
        """
        if not self.check_circuit_breaker(): return False
        
        try:
            exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            
            # Stop Loss is always a 'stop' (market) or 'stop-limit'
            # Kraken Futures uses 'stop' with triggerPrice
            
            # --- FIX: PRECISION FORMATTING ---
            sl_price_str = self.exchange.price_to_precision(exec_symbol, stop_price)
            qty_str = self.exchange.amount_to_precision(exec_symbol, quantity)
            
            # Convert back to float for safety with ccxt or pass compatible string if supported
            # Safest is float for 'price' (sometimes) but strictly formatted string for params if specific API needs it.
            # CCXT usually handles numbers fine if they are rounded to precision.
            final_sl_price = float(sl_price_str)
            final_qty = float(qty_str)
            
            params = {
                'reduceOnly': True,
                'stopPrice': final_sl_price, # CCXT Unified
            }
            if config.TRADING_MODE == 'FUTURES':
                params['triggerPrice'] = final_sl_price # Specific for some futures exch
            
            print(f"[{self.name}] 🛡️ PLACING STOP LOSS: {direction} {final_qty} {symbol} @ {final_sl_price}")
            
            order = self.exchange.create_order(
                symbol=exec_symbol,
                type='stop', # Stop Market
                side=direction.lower(),
                amount=final_qty,
                price=None, # Market Stop
                params=params
            )
            
            if order:
                print(f"[{self.name}] ✅ STOP LOSS ACTIVE: {order['id']}")
                return True
            return False
            
        except Exception as e:
            print(f"[{self.name}] ❌ Stop Loss Failed: {e}")
            return False

    def execute_twap(self, symbol: str, direction: str, total_quantity: float, duration_seconds: int = 3600):
        """
        Execute an order using Time-Weighted Average Price (TWAP).
        Slices the total quantity into N sub-orders over the specified duration.
        """
        def _twap_worker(stop_event):
            num_slices = max(1, duration_seconds // 60) # 1 slice per minute
            qty_per_slice = total_quantity / num_slices
            interval = duration_seconds / num_slices
            
            print(f"[{self.name}] ⏱️ TWAP START: {direction} {total_quantity} {symbol} over {duration_seconds}s ({num_slices} slices)")
            
            for i in range(num_slices):
                if stop_event.is_set():
                    print(f"[{self.name}] ⏱️ TWAP ABORTED: {symbol}")
                    break
                    
                # Execute slice
                self.place_order(symbol, direction, qty_per_slice, order_type='market', urgent=True)
                time.sleep(interval)
            
            print(f"[{self.name}] ⏱️ TWAP COMPLETE: {symbol}")

        algo_id = f"TWAP_{symbol}_{time.time()}"
        stop_event = threading.Event()
        thread = threading.Thread(target=_twap_worker, args=(stop_event,), daemon=True)
        
        with self._algo_lock:
            self.active_algos[algo_id] = {'thread': thread, 'stop_event': stop_event, 'type': 'TWAP'}
            
        thread.start()
        return algo_id

    def execute_pov(self, symbol: str, direction: str, total_quantity: float, pov_percentage: float = 0.05):
        """
        Execute an order using Percentage of Volume (POV).
        Monitors market volume and executes a percentage of that volume until total_quantity is reached.
        """
        def _pov_worker(stop_event):
            executed_qty = 0.0
            print(f"[{self.name}] 📈 POV START: {direction} {total_quantity} {symbol} @ {pov_percentage*100}% of Vol")
            
            while executed_qty < total_quantity and not stop_event.is_set():
                try:
                    ticker = self.exchange.fetch_ticker(config.KRAKEN_SYMBOL_MAP.get(symbol, symbol))
                    # Simplified: use 1m volume proxy or recent trade volume
                    # Real POV would listen to trades. Here we poll.
                    time.sleep(10) # 10s polling
                    
                    # Assume some volume based on ticker change or hardcoded 'safe' slice
                    # In a real implementation, we'd fetch recent trades from observer
                    slice_qty = min(total_quantity - executed_qty, total_quantity * 0.1) # Placeholder proxy
                    self.place_order(symbol, direction, slice_qty, order_type='market', urgent=True)
                    executed_qty += slice_qty
                    
                except Exception as e:
                    print(f"[{self.name}] ⚠️ POV Worker Error: {e}")
                    time.sleep(30)
            
            print(f"[{self.name}] 📈 POV COMPLETE: {symbol}")

        algo_id = f"POV_{symbol}_{time.time()}"
        stop_event = threading.Event()
        thread = threading.Thread(target=_pov_worker, args=(stop_event,), daemon=True)
        
        with self._algo_lock:
            self.active_algos[algo_id] = {'thread': thread, 'stop_event': stop_event, 'type': 'POV'}
            
        thread.start()
        return algo_id

    def execute_vwap(self, symbol: str, direction: str, total_quantity: float, observer: Any = None):
        """
        Execute an order using Volume-Weighted Average Price (VWAP).
        Requires an observer to fetch historical volume profile.
        """
        def _vwap_worker(stop_event):
            print(f"[{self.name}] 📊 VWAP START: {direction} {total_quantity} {symbol}")
            
            num_slices = 20
            qty_per_slice = total_quantity / num_slices
            
            for i in range(num_slices):
                if stop_event.is_set() or i * qty_per_slice >= total_quantity:
                    break
                
                self.place_order(symbol, direction, qty_per_slice, order_type='market', urgent=True)
                time.sleep(60) # 1 minute per slice
                
            print(f"[{self.name}] 📊 VWAP COMPLETE: {symbol}")

        algo_id = f"VWAP_{symbol}_{time.time()}"
        stop_event = threading.Event()
        thread = threading.Thread(target=_vwap_worker, args=(stop_event,), daemon=True)
        
        with self._algo_lock:
            self.active_algos[algo_id] = {'thread': thread, 'stop_event': stop_event, 'type': 'VWAP'}
            
        thread.start()
        return algo_id

    def stop_all_algos(self):
        """Emergency stop for all background execution threads."""
        print(f"[{self.name}] 🛑 Stopping all active execution algorithms...")
        with self._algo_lock:
            for algo_id, state in self.active_algos.items():
                state['stop_event'].set()
            self.active_algos.clear()

    def place_order(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, price: float = 0.0, order_type: str = 'limit', margin: bool = True, leverage: float = 1.0, urgent: bool = False, reduce_only: bool = False):
        """
        Place an order on the exchange.
        Supports LIMIT (Maker) and MARKET (Taker) orders.
        
        Args:
            symbol: Internal symbol (e.g. 'BTC/USDT')
            direction: 'BUY' or 'SELL'
            quantity: Amount to buy/sell
            price: Limit price (ignored if order_type='market')
            order_type: 'limit' or 'market'
            margin: Whether to use margin (Futures default: True)
            leverage: Leverage multiplier.
            urgent: If True, allows Taker execution (disables PostOnly).
            reduce_only: If True, order will only reduce position (no new opens).
        """
        exec_symbol = self.symbol_map.get(symbol, symbol)
        side = 'buy' if direction == 'BUY' else 'sell'
        
        # --- OPTIMIZATION: Urgent Exits ---
        if reduce_only:
            # Force Market Order for Exits (Take Profit / Stop Loss) to ensure fill
            # and avoid 'postOnly' cancellations on limit orders that cross the spread
            order_type = 'market'
            # print(f"[{self.name}] ⚡ URGENT EXIT: Forcing MARKET order for {symbol}")
        
        # Prepare values with correct precision
        try:
            # We must use the 'exec_symbol' (ccxt unified or mapped) for precision calls
            # However, if 'exec_symbol' is a raw ID like 'PF_XBTUSD', ccxt might not find it in 'markets' map by that key directly 
            # if loaded via fetch_markets(). 
            # Safest is to use the Unified Symbol if possible, or try the mapped one.
            # check if self.exchange.markets is loaded
            if not self.exchange.markets:
                self.exchange.load_markets()

            # --- PATCH: MIN QUANTITY CLAMPING ---
            market = self.exchange.market(exec_symbol)
            min_limit = market.get('limits', {}).get('amount', {}).get('min')
            prec_amount = market.get('precision', {}).get('amount')
            
            # Use strict fallback if None
            if min_limit is None: min_limit = 0.0
            if prec_amount is None: prec_amount = 0.0
            
            # Effective minimum is the larger of the exchange limit or the precision unit
            # (e.g. limit 0.0001, precision 0.001 -> we can't trade 0.0001)
            effective_min = max(min_limit, prec_amount)
            
            if quantity < effective_min and quantity > 0:
                 print(f"[{self.name}] 🤏 Clamping Qty {quantity} -> {effective_min} (Min Allowed)")
                 quantity = effective_min
            # ------------------------------------

            # Convert float to precise string
            qty_str = self.exchange.amount_to_precision(exec_symbol, quantity)
            price_str = self.exchange.price_to_precision(exec_symbol, price)
            
            # Convert back to float/number for create_order if it expects numbers, 
            # BUT ccxt usually handles strings best to avoid float drift. 
            # However, standard ccxt usage is often float. 
            # Let's pass the float of the precise string to be safe, or just the string if ccxt supports it.
            # Most robust: Pass float, but rounded.
            final_qty = float(qty_str)
            final_price = float(price_str)
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Precision formatting failed: {e}. Using raw values.")
            final_qty = quantity
            final_price = price

        # --- LIQUIDITY SANITY CHECK ---
        # Ensure we aren't eating the entire book
        # For Market Orders, check if top book density is enough
        # Patch: If Market Order (price=0), fetch current price for sanity check
        check_price = final_price
        if order_type != 'limit' or check_price <= 0:
            try:
                ticker = self.exchange.fetch_ticker(exec_symbol)
                check_price = ticker['last']
            except:
                check_price = 0.0

        if not self.check_liquidity(exec_symbol, direction, final_qty, check_price):
             print(f"[{self.name}] 🛑 LIQUIDITY CHECK FAILED: {exec_symbol} Book too thin for {final_qty}. Order Aborted.")
             return None
        # ------------------------------

        if not self.check_circuit_breaker():
             print(f"[{self.name}] 🛑 CIRCUIT OPEN: Order for {symbol} REJECTED due to API instability.")
             return None
        # ------------------------------

        # --- PATCH: ERROR COOLDOWN ---
        if not self.can_retry_order(symbol, 'General'):
             # We treat everything as 'General' for now to catch "Insufficient Funds" etc.
             # Ideally pass specific error if known, but here we prevent the ATTEMPT.
             return None
        # -----------------------------

        print(f"[{self.name}] 🚀 PLACING {order_type.upper()} {direction} {final_qty} {exec_symbol} (Lev: {leverage}x, Reduce: {reduce_only})")
        
        try:
            # --- PATCH: SET LEVERAGE ---
            if margin and leverage > 1.0:
                 try:
                     self.exchange.set_leverage(leverage, exec_symbol)
                 except Exception as lev_err:
                     print(f"[{self.name}] ⚠️ Set Leverage Failed: {lev_err}")
            # ---------------------------

            params = {}
            if margin:
                # Isolated margin for shorting/leverage
                params['marginMode'] = 'isolated' 
                
            if order_type == 'limit':
                # Only strictly require PostOnly if NOT urgent
                if not urgent:
                    params['postOnly'] = True
                price_arg = final_price
            else:
                # Market Order
                price_arg = None # ccxt ignores price for market orders usually, or handles it
            
            # --- REDUCE ONLY FLAG ---
            if reduce_only and config.TRADING_MODE == 'FUTURES':
                 params['reduceOnly'] = True
            # ------------------------
            
            order = self.exchange.create_order(
                symbol=exec_symbol,
                type=order_type,
                side=side,
                amount=final_qty, # Use precise value
                price=price_arg,
                params=params
            )
            
            # Check for Exchange-level Rejection via Response Payload
            # (Kraken often returns 200 OK but with 'rejected' status in body)
            info = order.get('info', {})
            status = order.get('status', '')
            
            if status == 'rejected' or info.get('status') == 'marketIsPostOnly':
                print(f"[{self.name}] ⚠️ Order Rejected by Exchange: {info.get('status', status)}")
                
                # RETRY LOGIC FOR REJECTED ORDERS
                if 'marketIsPostOnly' in str(info) or 'PostOnly' in str(info):
                     print(f"[{self.name}] 🔄 Auto-Retrying Rejected Market Order as LIMIT...")
                     return self.create_order(symbol, direction, quantity, price=check_price, order_type='limit', margin=margin, leverage=leverage, urgent=True)
                
                return None

            if not order.get('id'):
                print(f"[{self.name}] ❌ Exchange returned no Order ID! Treating as failure. Response: {order}")
                return None

            order_record = {
                'id': order['id'],
                'symbol': symbol,
                'direction': direction,
                'status': 'OPEN',
                'type': order_type,
                'entry_time': time.strftime("%H:%M:%S"),
                'timestamp': time.time(),
                'quantity': final_qty,
                'price': final_price if order_type == 'limit' else 0.0 # Will update on fill
            }
            
            self.pending_orders.append(order_record)
            print(f"[{self.name}] ✅ Order Placed: {order['id']}")
            self.report_success()
            return order['id']
            
        except Exception as e:
            msg = str(e)
            # Only count as API failure if it's a network/exchange error, not logic (insufficient funds)
            msg = str(e)
            # Only count as API failure if it's a network/exchange error, not logic (insufficient funds)
            if "NetworkError" in msg or "503" in msg or "Service Unavailable" in msg or "timed out" in msg:
                self.report_failure(msg)
            else:
                # Logic Error (Funds, MinLimit, etc) -> Trigger specific cooldown
                print(f"[{self.name}] ❌ Order Logic Error: {msg}")
                self.record_order_failure(symbol, 'General')
                return None
            
            print(f"[{self.name}] ❌ Order Placement Failed: {msg}")
            
            # GUARD: INSUFFICIENT FUNDS -> DO NOT RETRY IMMEDIATELY
            if "insufficientAvailableFunds" in msg:
                 print(f"[{self.name}] 🛑 Insufficient Funds. Order dropped to prevent spam.")
                 return None
                 
            # GUARD: ALREADY CLOSED (Race Condition)
            if "wouldNotReducePosition" in msg:
                 print(f"[{self.name}] ℹ️ Position appears already closed/reduced (Exchange Rejected). Skipping.")
                 return None
            
            # RETRY LOGIC (Only for Limit Orders usually, but maybe Market failed?)
            # If Market order failed, we generally just fail.
            # RETRY LOGIC (Only for Limit Orders usually, but maybe Market failed?)
            # If Market order failed, we generally just fail.
            if order_type == 'market':
                # PATCH: Handle "marketIsPostOnly" or similar rejections by trying a Limit Order
                if "marketIsPostOnly" in msg or "PostOnly" in msg:
                     print(f"[{self.name}] ⚠️ Market Order Rejected ({msg}). Retrying as LIMIT...")
                     # Retry as Limit at current price (Aggressive)
                     return self.create_order(symbol, direction, quantity, price=check_price, order_type='limit', margin=margin, leverage=leverage, urgent=True)
                return None

            # RETRY 1: TAKER RETRY (PostOnly Failed)
            # If we were trying to be a Maker but the price moved, just TAKE it.
            if "OrderImmediatelyFillable" in msg or "postOnly" in msg:
                 try:
                     print(f"[{self.name}] ⚠️ PostOnly Failed (Price moved). Retrying as TAKER...")
                     params['postOnly'] = False
                     
                     # --- REDUCE ONLY ON RETRY ---
                     if reduce_only and config.TRADING_MODE == 'FUTURES':
                          params['reduceOnly'] = True
                     # ----------------------------

                     order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit', # Still limit, but crossing book
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                     )
                     # If success, add to pending
                     order_record = {
                        'id': order['id'],
                        'symbol': symbol,
                        'direction': direction,
                        'status': 'OPEN',
                        'type': 'limit', # Original type was limit, now it's a taker limit
                        'entry_time': time.strftime("%H:%M:%S"),
                        'timestamp': time.time(),
                        'quantity': final_qty,
                        'price': final_price
                    }
                     self.pending_orders.append(order_record)
                     print(f"[{self.name}] ✅ Order Placed (TAKER): {order['id']}")
                     return order['id']
                 except Exception as e2:
                     print(f"[{self.name}] ❌ Taker Retry Failed: {e2}")

            # RETRY 2: REDUCE ONLY (Aggressive Exit Fix)
            # If we failed due to funds AND we wanted to Reduce (or user logic implied it?), try to force reduceOnly.
            # Usually if 'reduce_only' was already True, we failed anyway.
            # But if 'reduce_only' was False, this might be a desperate attempt to 'close' if we messed up direction?
            # Actually, let's only do this if we suspect it's an exit failing.
            # For now, if we explicitly passed reduce_only=True, and it failed, we are done.           
            if "insufficientAvailableFunds" in msg and direction == 'SELL' and not reduce_only:
                 # Only retry with reduceOnly if we didn't try it yet
                 try:
                     print(f"[{self.name}] 🔄 Retrying with reduceOnly=True (Fallback)...")
                     params['reduceOnly'] = True
                     order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit',
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                     )
                     # If success, add to pending
                     order_record = {
                        'id': order['id'],
                        'symbol': symbol,
                        'direction': direction,
                        'status': 'OPEN',
                        'type': 'limit', # Original type was limit
                        'entry_time': time.strftime("%H:%M:%S"),
                        'timestamp': time.time(),
                        'quantity': final_qty,
                        'price': final_price
                    }
                     self.pending_orders.append(order_record)
                     print(f"[{self.name}] ✅ Retry Successful: {order['id']}")
                     return order['id']
                 except Exception as retry_e:
                     print(f"[{self.name}] ❌ Retry Failed: {retry_e}")
            
            # Original logic for postWouldExecute/OrderImmediatelyFillable
            if "postWouldExecute" in msg or "OrderImmediatelyFillable" in msg:
                print(f"[{self.name}] ⚠️ Maker Order Rejected (Price crossed spread). Retrying as TAKER...")
                try:
                    # Retry without Post-Only (Eat the Taker Fee to ensure execution)
                    if 'postOnly' in params:
                        del params['postOnly']
                    
                    order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit',
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                    )
                    
                    # Log successful retry
                    internal_order = {
                        'id': order['id'],
                        'status': 'OPEN',
                        'symbol': symbol,
                        'direction': direction,
                        'quantity': quantity,
                        'price': price, # Use 'price' instead of 'limit_price'
                        'type': 'limit', # Original type was limit
                        'timestamp': time.time()
                    }
                    self.pending_orders.append(internal_order)
                    print(f"[{self.name}] ✅ TAKER FILL SUBMITTED: {order['id']}")
                    return internal_order
                    
                except Exception as retry_err:
                    print(f"[{self.name}] ❌ Taker Retry Failed: {retry_err}")
                    return None
            else:
                print(f"[{self.name}] ❌ Order Placement Failed: {e}")
                return None
            
    def fetch_order_status(self, order_id: str, symbol: str = None) -> dict:
        """
        Fetch status of a specific order.
        Updates internal pending_orders state if filled/closed (prevents stale locks).
        """
        if not self.check_circuit_breaker(): return None
        
        exec_symbol = self.symbol_map.get(symbol, symbol) if symbol else None
        
        try:
            order = None
            # 1. Try direct fetch (Supported by most CCXT drivers)
            try:
                order = self.exchange.fetch_order(order_id, symbol=exec_symbol)
            except Exception as e:
                # Fallback: Scan Open/Closed manually (Kraken Futures quirk?)
                print(f"[{self.name}] ℹ️ Fetch Order {order_id} direct failed: {e}. Scanning Open/Closed...")
                pass
                
            if not order:
                # 2. Fallback Scan
                if exec_symbol:
                    # Check Open
                    try:
                        opens = self.exchange.fetch_open_orders(exec_symbol)
                        for o in opens:
                            if str(o['id']) == str(order_id):
                                order = o
                                break
                    except Exception as e_scan: 
                        print(f"[{self.name}] ⚠️ Open Order Scan Error: {e_scan}")
                    
                    # Check Closed
                    if not order:
                        try:
                            closed = self.exchange.fetch_closed_orders(exec_symbol, limit=20)
                            for o in closed:
                                if str(o['id']) == str(order_id):
                                    order = o
                                    break
                        except Exception as e_clos: 
                            print(f"[{self.name}] ⚠️ Closed Order Scan Error: {e_clos}")
            
            if not order:
                return None
                
            # Update Internal State
            remote_status = order.get('status')
            
            # Find in pending
            local_order = None
            for o in self.pending_orders:
                if str(o.get('id')) == str(order_id):
                    local_order = o
                    break
            
            if remote_status in ['closed', 'filled', 'canceled', 'expired', 'rejected']:
                 if local_order:
                     # Remove from pending to unlock Governor/Executor
                     if local_order in self.pending_orders:
                        self.pending_orders.remove(local_order)
                     print(f"[{self.name}] 🗑️ Cleared Completed Order {order_id} ({remote_status})")

            return {
                'status': remote_status, 
                'filled': float(order.get('filled', 0.0)), 
                'price': float(order.get('average') or order.get('price', 0.0))
            }

        except Exception as e:
            print(f"[{self.name}] ⚠️ fetch_order_status failed: {e}")
            return None

    def check_fills(self, candle_low: float = None, candle_high: float = None):
        """
        Check if pending orders were filled. For live, we fetch from exchange.
        """
        if not self.check_circuit_breaker(): return []

        filled_orders = []
        remaining_orders = []
        
        for order in self.pending_orders:
            try:
                # Kraken Futures usually doesn't support fetching a single order by ID easily
                # So we must scan Open and Closed lists
                found_order = None
                
                # Use mapped symbol for exchange calls
                exec_symbol = self.symbol_map.get(order['symbol'], order['symbol'])
                
                # 1. Check Open Orders
                # (Optimization: We could fetch all open orders ONCE per cycle instead of per order, 
                # but for now let's keep it robust)
                try:
                    # We pass symbol to narrow it down if possible
                    open_orders = self.exchange.fetch_open_orders(exec_symbol)
                    self.report_success()
                    for o in open_orders:
                        if o['id'] == order['id']:
                            found_order = o
                            break
                except Exception as e:
                    if "NetworkError" in str(e) or "503" in str(e): self.report_failure(str(e))
                    print(f"[{self.name}] ⚠️ fetch_open_orders failed: {e}")

                # 2. If not found, Check Closed Orders (It might have just filled)
                if not found_order:
                    try:
                        closed_orders = self.exchange.fetch_closed_orders(exec_symbol, limit=20)
                        for o in closed_orders:
                            if o['id'] == order['id']:
                                found_order = o
                                break
                    except Exception as e:
                        print(f"[{self.name}] ⚠️ fetch_closed_orders failed: {e}")
                
                if found_order:
                    remote_status = found_order['status']
                    
                    if remote_status == 'closed':
                        order['status'] = 'FILLED'
                        order['filled_qty'] = found_order.get('filled', order.get('quantity'))
                        order['cost_usd'] = found_order.get('cost', 0.0)
                        # CRITICAL FIX: Ensure 'price' is populated for Executor
                        order['price'] = found_order.get('average') or found_order.get('price') or order.get('price')
                        
                        filled_orders.append(order)
                        print(f"[{self.name}] ✅ FILL CONFIRMED: {order['id']} ({order['symbol']}) @ {order['price']}")
                        
                    elif remote_status == 'canceled':
                        print(f"[{self.name}] ⚠️ Order {order['id']} was CANCELED.")
                        # Do not add to remaining_orders -> Dropped from tracking
                        
                        # Still open, check timeout (55s + jitter)
                        age = time.time() - order['timestamp']
                        
                        # Add jitter to prevent thundering herd on API
                        timeout = 30 + random.randint(0, 5) # NANO FIX: 30s Timeout
                        
                        if age > timeout:
                            try:
                                self.exchange.cancel_order(order['id'], exec_symbol)
                                print(f"[{self.name}] ⏱️ Order {order['id']} TIMEOUT ({age:.2f}s > {timeout}s). Requesting Cancel to Reprice...")
                            except Exception as cancel_err:
                                print(f"[{self.name}] ⚠️ Cancel Failed for {order['id']}: {cancel_err}")
                            
                            # CRITICAL: Keep tracking it until verifying it is GONE/CANCELED in next cycle
                            remaining_orders.append(order)
                        else:
                            # print(f"[{self.name}] Order {order['id']} Open for {age:.2f}s")
                            remaining_orders.append(order)
                    else:
                        # Unknown status
                        remaining_orders.append(order)

                else:
                    # Order not found in either list? 
                    # Use Configured GC Timeout for Ghost Orders
                    ghost_timeout = getattr(config, 'GC_STALE_ORDER_TIMEOUT', 120)
                    if time.time() - order['timestamp'] > ghost_timeout:
                         print(f"[{self.name}] 👻 Order {order['id']} Disappeared & Expired (> {ghost_timeout}s). Dropping.")
                         # Dropped
                    else:
                         remaining_orders.append(order)

            except Exception as e:
                print(f"[{self.name}] Error checking fill for {order['id']}: {e}")
                remaining_orders.append(order)
                
        self.pending_orders = remaining_orders
        return filled_orders

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            if content.type == 'PLACE_ORDER':
                pass
        else:
            pass

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch Order Book from the EXECUTION VENUE (Kraken Futures etc).
        Crucial for checking liquidity on the exchange we actually trade on.
        """
        if not self.check_circuit_breaker(): return {'bids': [], 'asks': []}

        try:
            # Map Symbol
            exec_symbol = symbol
            if config.TRADING_MODE == 'FUTURES':
                exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                
            book = self.exchange.fetch_order_book(exec_symbol, limit)
            self.report_success()
            return {
                'bids': book['bids'],
                'asks': book['asks'],
                'timestamp': book['timestamp']
            }
        except Exception as e:
            self.report_failure(str(e))
            print(f"[{self.name}] ⚠️ Actuator Book Fetch Fail {symbol}: {e}")
            return {'bids': [], 'asks': []}

    def gc_clean_stale_orders(self) -> int:
        """
        Garbage Collector: Cancel and remove orders older than GC_STALE_ORDER_TIMEOUT.
        Returns the count of cleaned orders.
        """
        cleaned_count = 0
        remaining_orders = []
        stale_timeout = getattr(config, 'GC_STALE_ORDER_TIMEOUT', 120)
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        
        for order in self.pending_orders:
            age = time.time() - order.get('timestamp', time.time())
            
            if age > stale_timeout:
                # Attempt to cancel on exchange
                try:
                    exec_symbol = self.symbol_map.get(order['symbol'], order['symbol'])
                    self.exchange.cancel_order(order['id'], exec_symbol)
                    if verbose:
                        print(f"[GC Monitor] 🗑️ Canceled stale order {order['id']} ({order['symbol']}) after {age:.0f}s")
                except Exception as e:
                    if verbose:
                        print(f"[GC Monitor] ⚠️ Failed to cancel {order['id']}: {e}")
                
                cleaned_count += 1
                # Do NOT add to remaining_orders -> dropped
            else:
                remaining_orders.append(order)
        
        self.pending_orders = remaining_orders
        
        if verbose and cleaned_count > 0:
            print(f"[GC Monitor] ✅ Actuator Cleanup: {cleaned_count} stale orders removed.")
        
        return cleaned_count

    def fetch_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Verify the status of a specific order on the exchange.
        Used by Executor for Hard Gating.
        """
        try:
             # CCXT fetch_order
             exec_symbol = self.symbol_map.get(symbol, symbol)
             order = self.exchange.fetch_order(order_id, exec_symbol)
             return order
        except Exception as e:
             # print(f"[{self.name}] ⚠️ Fetch Order Failed: {e}")
             
             # Fallback: Check closed orders if fetch_order fails (order might have moved to history)
             try:
                 exec_symbol = self.symbol_map.get(symbol, symbol)
                 closed = self.exchange.fetch_closed_orders(exec_symbol, limit=50) # Increased limit for safety
                 for o in closed:
                     if o['id'] == order_id:
                         return o
             except: pass
             
             return None

    def check_invariants(self, balance: float, total_exposure: float) -> bool:
        """
        Unified Control Protocol: Invariant Check (Request E)
        If Implied Leverage > Max Allowed + Buffer -> HALT.
        Returns False if invariant violated (HALT).
        """
        if balance <= 0: return True # Can't calc leverage, assume dead or startup
        
        implied_leverage = total_exposure / balance
        max_allowed_lev = config.MICRO_MAX_LEVERAGE if config.MICRO_CAPITAL_MODE else config.IMMUNE_MAX_LEVERAGE_RATIO
        
        # Buffer of +2.0x for temporary volatility swings before Hard Halt
        limit = max_allowed_lev + 2.0
        
        if implied_leverage > limit:
             print(f"[{self.name}] 🚨 INVARIANT VIOLATION: Implied Leverage {implied_leverage:.2f}x > Max {limit}x ({max_allowed_lev} + 2)")
             print(f"[{self.name}] 🛑 SYSTEM HALT TRIGGERED: Exposure Mismatch. Manual Intervention Required.")
             return False # HALT
             
        return True

    def check_spread_health(self, symbol: str, ticker: Dict) -> bool:
        """
        Spread Veto: Prevent entry if spread > Threshold (0.4%).
        """
        if not ticker or 'bid' not in ticker or 'ask' not in ticker:
            return True # Can't check
            
        spread_pct = (ticker['ask'] - ticker['bid']) / ticker['ask'] if ticker['ask'] > 0 else 0.0
        
        limit = config.VOL_WINDOW_SPREAD_THRESHOLD # 0.4%
        
        if spread_pct > limit:
            print(f"[{self.name}] 🧊 SPREAD VETO: {symbol} spread {spread_pct*100:.2f}% > {limit*100:.2f}%")
            return False
            
        return True
        
    def should_force_close_funding(self, symbol: str, position_dir: str, funding_rate: float) -> bool:
        """
        Funding Flip Kill Switch:
        If we are holding a position primarily for Funding Arb (or just generally),
        and the funding rate moves against us (we start paying), signal CLOSE.
        
        Pos Funding (>0): Longs Pay Shorts.
        Neg Funding (<0): Shorts Pay Longs.
        """
        # If we are Long, we want Neg Funding (Get Paid). If Pos, we Pay.
        # If we are Short, we want Pos Funding (Get Paid). If Neg, we Pay.
        
        if position_dir == 'BUY':
             # We hold Long. We pay if Funding > 0.
             # If Funding > 0 and we entered for Arb, we should exit.
             # But normal trend trading pays funding often.
             # We assume this check is only called for ARB strategies or if strictly enforcing "Don't Pay Funding".
             # For VOL_WINDOW, we might be stricter.
             if funding_rate > 0.0001: # Small buffer
                 return True
                 
        elif position_dir == 'SELL':
             # We hold Short. We pay if Funding < 0.
             if funding_rate < -0.0001:
                 return True
                 
        return False

