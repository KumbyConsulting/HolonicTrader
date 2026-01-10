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
from typing import Any, Literal, Dict
from HolonicTrader.holon_core import Holon, Disposition, Message

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
            safety_ratio = 3.0
            
            if cumulative_vol < (quantity * safety_ratio):
                print(f"[{self.name}] ⚠️ THIN BOOK: {symbol} Top 10 Vol {cumulative_vol:.4f} < Req {quantity * safety_ratio:.4f}")
                return False
                
            return True
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Liquidity Check Error: {e}. Proceeding with caution.")
            return True # Fail open to avoid paralysis, but log warning

    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Fetch REAL free balance from exchange.
        """
        for attempt in range(3):
            try:
                balance = self.exchange.fetch_balance()
                
                # Check for Unified Margin 'total' or 'free'
                # Kraken Futures usually puts USD in 'free'
                b_usd = balance['free'].get('USD', 0.0)
                b_usdt = balance['free'].get('USDT', 0.0)
                b_zusd = balance['free'].get('ZUSD', 0.0)
                
                total_avail = max(b_usd, b_usdt, b_zusd)
                # print(f"[{self.name}] 💰 Real Balance: ${total_avail:.2f}")
                return total_avail
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
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
        for attempt in range(3):
            try:
                balance = self.exchange.fetch_balance()
                info = balance.get('info', {})
                
                # 1. Futures: Explicit marginEquity
                if config.TRADING_MODE == 'FUTURES':
                    accounts = info.get('accounts', {})
                    flex = accounts.get('flex', {})
                    total_equity = float(flex.get('marginEquity', 0.0))
                    if total_equity > 0:
                        return total_equity
                        
                # 2. Spot/Unified: Equivalent Balance ('eb')
                equity = float(info.get('eb', 0.0))
                if equity > 0:
                     return equity
                     
                # 3. Fallback: Total USD
                return balance.get('total', {}).get('USD', 0.0)
                
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                     print(f"[{self.name}] ❌ Equity Check Failed after 3 attempts: {e}")
                     return None # Return None instead of 0.0 to prevent panic
                time.sleep(1 * (attempt+1))
        return None

    def get_buying_power(self, leverage: float = 5.0) -> float:
        """
        Fetch Effective Buying Power (Equity * Leverage).
        Uses Kraken's 'eb' (Equivalent Balance) or 'tb' (Trade Balance).
        """
        for attempt in range(3):
            try:
                balance = self.exchange.fetch_balance()
                info = balance.get('info', {})
                
                # 1. Try Equivalent Balance (Equity) - Spot/Unified
                equity = float(info.get('eb', 0.0))
                
                # 2. Try Futures 'marginEquity' (Common in Kraken Futures API)
                if equity <= 0 and config.TRADING_MODE == 'FUTURES':
                    # Handle Kraken Futures 'flex' account structure
                    accounts = info.get('accounts', {})
                    flex = accounts.get('flex', {})
                    
                    # Priority: availableMargin (Free to trade) -> marginEquity (Total Net Worth)
                    # We use availableMargin to avoid rejecting orders due to tied up funds
                    avail_margin = float(flex.get('availableMargin', 0.0))
                    margin_equity = float(flex.get('marginEquity', 0.0))
                    
                    if avail_margin > 0:
                        equity = avail_margin
                        # NOTE: availableMargin is already " Buying Power / Leverage " ? 
                        # No, usually it's the equity available for initial margin.
                        # Buying Power = availableMargin * Leverage.
                    elif margin_equity > 0:
                         equity = margin_equity
                    else:
                        # Fallback to total USD if flex is empty (e.g. cash only)
                        equity = balance.get('total', {}).get('USD', 0.0)
                    
                # 3. Fallback to Trade Balance
                if equity <= 0:
                    equity = float(info.get('tb', 0.0))
                    
                # 4. Fallback to Free Balance
                if equity <= 0:
                    return self.get_account_balance()
                    
                # Buying Power = Equity * Leverage
                if config.TRADING_MODE == 'FUTURES':
                    # For Futures, let's trust the configured leverage cap
                    return equity * leverage
                else:
                    return equity * leverage
                
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ❌ Buying Power Check Failed: {e}")
                    return self.get_account_balance()
                time.sleep(1 * (attempt+1))
        return self.get_account_balance()

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
            return order['id']
            
        except Exception as e:
            msg = str(e)
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
            if order_type == 'market':
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
            
    def check_fills(self, candle_low: float = None, candle_high: float = None):
        """
        Check if pending orders were filled. For live, we fetch from exchange.
        """
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
                    for o in open_orders:
                        if o['id'] == order['id']:
                            found_order = o
                            break
                except Exception as e:
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
                        timeout = 55 + random.randint(-5, 5)
                        
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
        try:
            # Map Symbol
            exec_symbol = symbol
            if config.TRADING_MODE == 'FUTURES':
                exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                
            book = self.exchange.fetch_order_book(exec_symbol, limit)
            return {
                'bids': book['bids'],
                'asks': book['asks'],
                'timestamp': book['timestamp']
            }
        except Exception as e:
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
