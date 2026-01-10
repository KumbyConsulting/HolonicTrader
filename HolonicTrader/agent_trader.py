import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from scipy.stats import linregress
from rich.live import Live
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.console import Group
import sys
import os

# Initialize Console for Real Terminal (sys.__stdout__)
# We bypass sys.stdout (QueueLogger) so the Table doesn't spam the GUI/LogFile
console = Console(file=sys.__stdout__, force_terminal=True, width=120)

from HolonicTrader.holon_core import Holon, Disposition, Message
from HolonicTrader.agent_executor import TradeSignal
from performance_tracker import get_performance_data
import config

class TraderHolon(Holon):
    """
    TraderHolon (Supra-Holon)
    The central coordinator that orchestrates the trading lifecycle using a 
    concurrency-first architecture (Phase 28: Warp Velocity).
    """
    
    def __init__(self, name: str = "TraderNexus", sub_holons: Dict[str, Holon] = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        self.sub_holons = sub_holons if sub_holons else {}
        self.market_state = {'price': 0.0, 'regime': 'UNKNOWN', 'entropy': 0.0, 'signal': None}
        self.gui_queue = None
        self.gui_stop_event = None
        self.last_ppo_conviction = 0.5
        
        # Scout State
        self.scout_last_run = 0.0
        self.active_session_whitelist = config.ACTIVE_WATCHLIST.copy() # Start with Hot List
        self._load_whitelist_from_disk()
        
        self.verbose_logging = True # Request C: Enable transparency logs

    def _load_whitelist_from_disk(self):
        import json
        try:
            path = os.path.join(os.getcwd(), 'scout_whitelist.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    saved_list = json.load(f)
                    self.active_session_whitelist = list(set(self.active_session_whitelist + saved_list))
                    print(f"[{self.name}] 📂 Loaded {len(self.active_session_whitelist)} assets from local scout sync.")
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to load scout whitelist: {e}")

    def _sync_whitelist_to_disk(self):
        import json
        try:
            path = os.path.join(os.getcwd(), 'scout_whitelist.json')
            with open(path, 'w') as f:
                json.dump(self.active_session_whitelist, f)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to sync scout whitelist: {e}")

    def _sync_scout_status_to_disk(self, results: dict):
        import json
        try:
            path = os.path.join(os.getcwd(), 'scout_status.json')
            # Add timestamp
            data = {
                'timestamp': time.time(),
                'results': results
            }
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to sync scout status: {e}")

    def _run_scout_cycle(self):
        """
        The Slow Loop: Scans the Cold List for opportunities to promote.
        """
        if time.time() - self.scout_last_run < config.SCOUT_CYCLE_INTERVAL:
            return

        observer = self.sub_holons.get('observer')
        oracle = self.sub_holons.get('oracle')
        if not observer or not oracle: return

        print(f"[{self.name}] 🔭 SCOUT CYCLE STARTING... Scanning {len(config.SCOUT_CANDIDATES)} candidates.")
        
        # 1. Efficient Batch Fetch (1 API Call)
        tickers = observer.fetch_tickers_batch(config.SCOUT_CANDIDATES)
        
        promoted_count = 0
        scout_results = {}
        
        for symbol, ticker_data in tickers.items():
            # 2. Profile
            personality = oracle.profile_asset_class(symbol, ticker_data)
            scout_results[symbol] = personality
            
            if symbol in self.active_session_whitelist: continue # Already active
            
            if personality in ['ROCKET', 'ANCHOR']:
                print(f"[{self.name}] 🚀 SCOUT PROMOTION: {symbol} identified as {personality}. Adding to Active Loop.")
                self.active_session_whitelist.append(symbol)
                promoted_count += 1
        
        # Sync Status for GUI
        self._sync_scout_status_to_disk(scout_results)
        
        if promoted_count > 0:
            self._sync_whitelist_to_disk()
            
        self.scout_last_run = time.time()
        self.last_ppo_reward = 0.0
        self.gc_cycle_counter = 0  # GC Monitor cycle counter

    def register_agent(self, role: str, agent: Holon):
        self.sub_holons[role] = agent
        print(f"[{self.name}] Registered {role}: {agent.name}")

    def perform_health_check(self):
        observer = self.sub_holons.get('observer')
        if observer:
            try:
                status = observer.receive_message(self, {'type': 'GET_STATUS'})
                if not (isinstance(status, dict) and status.get('status') == 'OK'):
                    observer.receive_message(self, {'type': 'FORCE_FETCH'})
            except Exception as e:
                print(f"[{self.name}] ❌ OBSERVER HEALTH FAIL: {e}")

    def run_cycle(self):
        self.perform_health_check()
        
        # --- PATCH 1: THE KILL SWITCH (Connect the Brakes) ---
        monitor = self.sub_holons.get('monitor')
        if monitor:
            is_healthy, risk_msg = monitor.check_vital_signs()
            if not is_healthy:
                print(f"[{self.name}] 🛑 CRITICAL HEALTH LOCKDOWN: {risk_msg}")
                print(f"[{self.name}] 💤 HIBERNATING for 4 hours to cool down...")
                time.sleep(14400) # 4 Hour Hard Sleep
                return [] # Skip cycle
                
        interval = getattr(self, '_active_interval', 60)
        print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval}s) ---") 
        
        cycle_report = []
        entropies = []
        cycle_data_cache = {}

        oracle = self.sub_holons.get('oracle')
        observer = self.sub_holons.get('observer')
        executor = self.sub_holons.get('executor')
        governor = self.sub_holons.get('governor')
        ppo = self.sub_holons.get('ppo')
        guardian = self.sub_holons.get('guardian')
        monitor = self.sub_holons.get('monitor')
        sentiment = self.sub_holons.get('sentiment')
        overwatch = self.sub_holons.get('overwatch') # <--- Get Overwatch Ref
        regime_controller = self.sub_holons.get('regime') # <--- Phase 7

        # --- PHASE -2: REGIME STATE UPDATE ---
        if regime_controller and executor:
            equity = executor.get_portfolio_value(0.0)
            health_metrics = {
                'trade_completed': False,  # Updated by exit logic
                'solvency_rejection': False,  # Could be tracked from Governor
                'gc_correction': False,
                'slippage': 0.0,
            }
            regime_controller.update_state(equity, health_metrics)
            
            # Log current regime
            regime_status = regime_controller.get_status_summary()
            print(f"[{self.name}] 📊 Regime: {regime_status['regime']} | Health: {regime_status['health_score']:.2f} | Peak: ${regime_status['peak_equity']:.2f}")

        # --- PHASE -1: OVERWATCH AUDIT (The Sentry) ---
        if overwatch:
            overwatch.perform_audit()
        self._run_scout_cycle()

        # --- PHASE 0: PARALLEL PRE-FLIGHT (GMB Sync) ---
        sent_score = 0.0
        if sentiment and oracle:
            sent_score = sentiment.fetch_and_analyze()
            oracle.set_crisis_score(getattr(sentiment, 'crisis_score', 0.0))
        else:
            sent_score = 0.0

        if oracle and observer:
            # OPTIMIZATION: Emotional Feedback Loop (Amygdala)
            fear = 0.0
            greed = 1.0
            if monitor:
                fear = monitor.metrics.get('current_drawdown', 0.0)
            if governor:
                greed = getattr(governor, 'risk_multiplier', 1.0)
            oracle.set_emotional_bias(fear, greed)
            
            # OPTIMIZATION: Parallel Batch Fetch via Observer
            target_assets = self.active_session_whitelist
            batch_data = observer.fetch_market_data_batch(target_assets, limit=100)
            
            for sym, data in batch_data.items():
                cycle_data_cache[sym] = data
                # Warmup Oracle (Kalman) - Fast compute, can run sequentially or parallel if needed
                # For 10-20 assets, sequential compute is negligible compared to IO
                try:
                    oracle.get_kalman_estimate(sym, data)
                except: pass
            
            # Pass Sentiment Score to Oracle
            global_bias = oracle.get_market_bias(sentiment_score=sent_score)
            print(f"[{self.name}] 📊 GLOBAL BIAS: {global_bias:.2f} (Sentiment: {sent_score:+.2f})")

        # --- PHASE 1: PARALLEL ANALYSIS PASS ---
        analysis_results = []
        with ThreadPoolExecutor(max_workers=config.TRADER_MAX_WORKERS) as t_pool:
            futures = [t_pool.submit(self._analyze_asset, s, cycle_data_cache.get(s)) for s in self.active_session_whitelist]
            try:
                for f in as_completed(futures, timeout=30):
                    try:
                        res = f.result()
                        if res: analysis_results.append(res)
                    except Exception as e:
                        print(f"[{self.name}] ⚠️ Analysis Logic Error: {e}")
            except TimeoutError:
                 print(f"[{self.name}] ⚠️ Analysis Cycle Timed Out (proceeding with partial results)")

        analysis_results.sort(key=lambda x: x['symbol'])

        # --- PHASE 2: SEQUENTIAL EXECUTION PASS ---
        for res in analysis_results:
            symbol, data, current_price = res['symbol'], res['data'], res['price']
            row_data, indicators = res['row_data'], res['indicators']
            entropy_val, regime = res['entropy_val'], res['regime']
            
            if entropy_val > 0: entropies.append(entropy_val)

            try:
                if executor: executor.latest_prices[symbol] = current_price
                if executor and governor: 
                    # --- PATCH: SOLVENCY UPDATE ---
                    e_tot, e_free = executor.get_balance_details()
                    governor.set_live_balance(e_tot, e_free)
                    
                    # --- PHASE 7: CONSOLIDATION ENGINE ---
                    # Run intelligent consolidation (replaces simple MICRO logic)
                    to_close = governor.run_consolidation_engine(
                        executor.latest_prices,
                        position_metadata=executor.position_metadata
                    )
                    for c_sym in to_close:
                        print(f"[{self.name}] 🧹 EXECUTING CONSOLIDATION CLOSE: {c_sym}")
                        direction = executor.position_metadata.get(c_sym, {}).get('direction', 'BUY')
                        is_long = direction == 'BUY'
                        close_qty = abs(executor.held_assets.get(c_sym, 0.0))
                        close_price = executor.latest_prices.get(c_sym, 0.0)
                        
                        # DEBUG: Inspect quantity
                        real_holding = executor.held_assets.get(c_sym, "MISSING")
                        print(f"[{self.name}] 🔍 CONSOLIDATION DEBUG: Sym={c_sym}, Held={real_holding}, CloseQty={close_qty:.8f}")
                        
                        # Construct proper TradeSignal and TradeDecision for Executor
                        from HolonicTrader.agent_executor import TradeSignal, TradeDecision
                        from HolonicTrader.holon_core import Disposition
                        
                        close_signal = TradeSignal(
                            symbol=c_sym,
                            direction='SELL' if is_long else 'BUY',
                            size=1.0, # FIXED: Use 1.0 (100%) percent multiplier
                            price=close_price,
                            conviction=1.0,
                            metadata={'reason': 'CONSOLIDATION', 'reduce_only': True}
                        )
                        close_decision = TradeDecision(
                            action='EXECUTE',
                            original_signal=close_signal,
                            adjusted_size=1.0, # FIXED: Use 1.0 (100%) percent multiplier
                            disposition=Disposition(autonomy=1.0, integration=1.0),
                            block_hash='CONSOLIDATION',
                            entropy_score=0.0
                        )
                        
                        # Execute the close
                        executor.execute_transaction(close_decision, close_price)
                        
                        # Re-sync Governor after each close
                        governor.sync_positions(executor.held_assets, executor.position_metadata)
                    # -----------------------------------
                    
                    # --- PATCH: IMMUNE SYSTEM ACTIVATION ---
                    if monitor:
                        # Feed the Immune System (Check Drawdown/Solvency)
                        perf = get_performance_data()
                        is_healthy = monitor.update_health(
                            executor_summary=executor.get_execution_summary(),
                            performance_data=perf
                        )
                        if not is_healthy:
                            print(f"[{self.name}] ☠️ IMMUNE SYSTEM TRIGGERED: HALTING CYCLE.")
                            # Optional: Panic Close?
                            # executor.panic_close_all(executor.latest_prices)
                            return [] # Abort Cycle
                    # ------------------------------

                # --- PHASE 7: TRANSITION FREEZE ---
                # Block new entries during regime transition
                if regime_controller and regime_controller.is_transition_pending():
                    print(f"[{self.name}] ⏸️ Regime Transition Pending. Entries PAUSED.")
                    entry_sig = None  # Override any signal

                # A. Handle Entry
                entry_sig = res.get('entry_signal') if not (regime_controller and regime_controller.is_transition_pending()) else None
                if entry_sig and executor and governor and oracle:
                    pnl_tracker = get_performance_data()
                    atr_ref = indicators['tr'].rolling(14).mean().rolling(14).mean().iloc[-1]
                    atr_ratio = min(2.0, indicators['atr'] / atr_ref) if atr_ref > 0 else 1.0
                    gov_health = governor.get_portfolio_health()
                    
                    ppo_state = np.array([
                        {'ORDERED': 0.0, 'TRANSITION': 0.5, 'CHAOTIC': 1.0}.get(regime, 0.5),
                        entropy_val, pnl_tracker.get('win_rate', 0.5), atr_ratio, 
                        gov_health['drawdown_pct'], gov_health['margin_utilization']
                    ], dtype=np.float32)

                    conviction = ppo.get_conviction(ppo_state) if ppo else 0.5
                    self.last_ppo_conviction = conviction
                    entry_sig.metadata = {'ppo_state': ppo_state.tolist(), 'ppo_conviction': conviction, 'atr': indicators['atr']}

                    approved, safe_qty, leverage = governor.calc_position_size(
                        symbol, current_price, indicators['atr'], atr_ref, conviction, 
                        direction=entry_sig.direction, sentiment_score=sent_score
                    )

                    if approved and safe_qty > 0:
                        entry_sig.size = safe_qty
                        decision = executor.decide_trade(entry_sig, regime, entropy_val)
                        if decision.action != 'HALT':
                            print(f"[{self.name}] 🎯 EXECUTING ENTRY: {symbol} (Qty: {safe_qty:.4f}, Lev: {leverage}x)")
                            executor.execute_transaction(decision, current_price)
                            
                            # --- PATCH: NOTIFY GOVERNOR (Update Timestamps/Stacks) ---
                            # Critical for Cooldown and Stack Distance Logic
                            if governor:
                                direction = entry_sig.direction
                                governor.open_position(symbol, direction, current_price, safe_qty)
                            # ---------------------------------------------------------
                            
                            # Safe Telegram Notification
                            telegram = self.sub_holons.get('telegram')
                            if telegram and hasattr(telegram, 'send_message'):
                                msg = f"🚀 **ENTRY** {symbol}\nPrice: {current_price}\nSize: {entry_sig.size:.4f}"
                                telegram.send_message(msg)
                                
                            row_data['Action'] = f"BUY ({res['metabolism']})"
                            
                            # Log specific reason
                            reason_tag = entry_sig.metadata.get('reason', 'TREND')
                            if entry_sig.metadata.get('is_whale'):
                                row_data['Action'] = f"WHALE BUY 🐋"
                            else:
                                row_data['Action'] = f"BUY ({reason_tag})"
                        else:
                            print(f"[{self.name}] 🛑 ENTRY HALTED: {symbol} (Executor HALT)")
                            row_data['Action'] = "BUY (HALT)"
                    else:
                        # Log the rejection reason
                        if not approved:
                            row_data['Action'] = "BUY (GOV REJECT)"
                            # Request C: Transparency
                            # If Governor rejected, it usually prints why (DEBUG).
                            # We can reinforce it here:
                            print(f"[{self.name}] 🛡️ Governor Vetoed {symbol}: Risk/Exposure Limits.")
                        elif safe_qty <= 0:
                            row_data['Action'] = "BUY (NO QTY)"

                else:
                    # Request C: No Signal - Healthy Silence
                    # If we don't have an entry signal, and we don't have a position...
                    if not executor.held_assets.get(symbol):
                        # Filter spam by only logging "interesting" silences (e.g. if Price is moving)
                        # or just log periodically? 
                        # Let's log if Regime implies we *should* exist but don't.
                        # Actually, user wants "No Entry: Market Regime = Low Energy (Expected)"
                        # We can log this clearly for the Dashboard Log.
                         if entropy_val < 0.8 and regime == 'ORDERED':
                              # We are in a good regime but no signal? 
                              pass
                         else:
                              # logging every cycle for every asset is too much.
                              # Just rely on row_data['Action'] = "WAIT" default?
                              pass
                         
                         # Explicitly set Action for GUI clarity
                         if not row_data.get('Action'):
                             row_data['Action'] = "WAIT"
                             if self.verbose_logging: # Only if verbose
                                 print(f"[{self.name}] 💤 No Entry {symbol}: Regime={regime} (Expected)")

                # B. Handle Exit
                guardian_exit = res.get('guardian_exit')
                
                # --- PHASE 40: THESIS VALIDATION (Proof of Holding) ---
                thesis_valid = True
                qty_held = executor.held_assets.get(symbol, 0.0) if executor else 0.0
                
                if abs(qty_held) > 0.00000001 and hasattr(oracle, 'verify_holding_physics') and executor:
                    ex_meta = executor.position_metadata.get(symbol, {})
                    direction = ex_meta.get('direction', 'BUY')
                    thesis_valid = oracle.verify_holding_physics(symbol, direction)
                    
                thesis_exit = None
                if not thesis_valid:
                    print(f"[{self.name}] 🚫 THESIS INVALIDATED for {symbol}. Exiting.")
                    # FIX: Dynamic Exit Direction
                    exit_dir = 'BUY' if direction == 'SELL' else 'SELL'
                    thesis_exit = TradeSignal(symbol, exit_dir, 1.0, current_price)
                # -----------------------------------------------------

                hard_exit_type = executor.check_stop_loss_take_profit(symbol, current_price) if executor else None
                
                final_exit = None
                reason = "IDLE"
                if thesis_exit:
                    final_exit = thesis_exit
                    reason = "Thesis"
                elif hard_exit_type:
                    # FIX: Dynamic Exit Direction
                    ex_meta = executor.position_metadata.get(symbol, {})
                    direction = ex_meta.get('direction', 'BUY')
                    exit_dir = 'BUY' if direction == 'SELL' else 'SELL'
                    final_exit = TradeSignal(symbol, exit_dir, 1.0, current_price)
                    reason = hard_exit_type
                elif guardian_exit:
                    final_exit = guardian_exit
                    reason = "Strat"

                if final_exit and executor:
                    # FIX: Infinite Exit Loop
                    # Guardian sends size=1.0 (dummy). We must override with ACTUAL held quantity to close fully.
                    # Correct: Signal size should range 0.0-1.0 (Percentage of holding to exit)
                    # Guardian sends 1.0 by default. Executor handles logic: exec_qty = holding * size.
                    # Do NOT overwrite with actual quantity or you get squared exits (Qty * Qty).
                    pass

                    # Capture metadata BEFORE execution deletes it!
                    meta = executor.position_metadata.get(symbol, {}).copy()
                    
                    decision = executor.decide_trade(final_exit, regime, entropy_val)
                    pnl_res = executor.execute_transaction(decision, current_price)
                    
                    if pnl_res is not None:
                        if guardian: guardian.record_exit(symbol, time.time())
                        if ppo and meta:
                            if meta.get('ppo_state') is not None and meta.get('ppo_conviction') is not None:
                                # --- PPO REWARD REFACTOR (Hybrid Velocity/Pain) ---
                                # Previous: reward = pnl_res - (drawdown * 2.0) [Death Spiral]
                                # New: Asymmetric Loss Aversion + Time Decay Efficiency
                                
                                pnl_pct = pnl_res # pnl_res is percentage
                                is_win = pnl_pct > 0
                                
                                # 1. Base Utility
                                # Scale small % (0.01) to recognizable scalar (0.1)
                                reward = pnl_pct * 10.0
                                
                                # 2. Asymmetric Punishment (Loss Aversion)
                                if not is_win:
                                    reward *= 2.5 # Pain Factor
                                    
                                # 3. Time Duration
                                from datetime import datetime
                                entry_ts_iso = meta.get('entry_timestamp')
                                duration_mins = 1.0 # Default minimum
                                if entry_ts_iso:
                                    try:
                                        # Handle ISO parsing manually if needed, but pd.to_datetime is robust
                                        t_entry = pd.to_datetime(entry_ts_iso)
                                        t_exit = pd.Timestamp.now(tz='UTC')
                                        duration_mins = (t_exit - t_entry).total_seconds() / 60.0
                                    except: pass
                                    
                                duration_mins = max(1.0, duration_mins)
                                
                                # 4. Time Decay / Velocity
                                # log1p(duration) -> log(2) ~ 0.69, log(61) ~ 4.1
                                time_factor = np.log1p(duration_mins)
                                if time_factor < 1.0: time_factor = 1.0 # Floor at 1
                                
                                if is_win:
                                    reward = reward / time_factor # Fast wins > Slow wins
                                else:
                                    reward = reward * time_factor # Long losses > Fast losses (Double Pain)

                                # === MISSION PSYCHOLOGY (Operation Centurion) ===
                                # 1. Calculate Progress
                                current_eq = executor.get_portfolio_value(0.0) if executor else config.INITIAL_CAPITAL
                                mission_progress = (current_eq - config.INITIAL_CAPITAL) / (config.MISSION_TARGET - config.INITIAL_CAPITAL)
                                mission_progress = max(0.0, min(1.0, mission_progress)) # Clamp 0-1

                                # 2. Progress Booster (Mid-Game Motivation)
                                # Reward strengthens as we get closer to the goal.
                                # e.g. at 50% progress, wins are 1.5x sweeter.
                                if reward > 0:
                                    reward *= (1.0 + mission_progress)

                                # 3. Proximity Defense (End-Game Anxiety)
                                # If we are > 80% to the goal, we become terrified of losing.
                                # Losses are penalized 2x harder.
                                if reward < 0 and mission_progress > 0.80:
                                    print(f"[{self.name}] 🛡️ PROXIMITY DEFENSE ACTIVE: Double Penalty applied.")
                                    reward *= 2.0
                                    
                                self.last_ppo_reward = reward
                                print(f"[{self.name}] 🧠 PPO REWARD: {reward:.4f} (PnL {pnl_pct*100:.2f}%, {duration_mins:.0f}m)")
                                
                                # Convert list back to numpy if needed
                                state = np.array(meta['ppo_state']) 
                                ppo.remember(state, meta['ppo_conviction'], reward, 0.0, 0.0, True)

                        # Safe Telegram Notification
                        telegram = self.sub_holons.get('telegram')
                        if telegram and hasattr(telegram, 'send_message'):
                            msg = f"📉 **EXIT** {symbol}\nPrice: {current_price}\nPnL: {pnl_res*100:+.2f}% ({reason})"
                            telegram.send_message(msg)

                        row_data['Action'] = f"SELL ({reason})"

            except Exception as e:
                print(f"[{self.name}] ❌ Error processing {symbol}: {e}")

            # --- LIQUIDITY & HEALTH MONITOR ---
            # If we hold a position, check its liquidity health on the EXECUTION VENUE
            qty_held = executor.held_assets.get(symbol, 0.0) if executor else 0.0
            actuator = self.sub_holons.get('executor', {}).actuator if executor else None
            
            # Note: Executor holds the Actuator reference
            
            if abs(qty_held) > 0.00000001 and guardian and actuator:
                try:
                    # Fetch live book from KRAKEN FUTURES (via Actuator)
                    book = actuator.fetch_order_book(symbol)
                    
                    # Determine Exit Direction for check
                    exit_dir = 'SELL' if qty_held > 0 else 'BUY'
                    liq_status = guardian.check_liquidity_health(symbol, exit_dir, abs(qty_held), book)
                    
                    if liq_status != "HEALTHY":
                        warn_msg = f"⚠️ LIQUIDITY WARNING for {symbol}: {liq_status}"
                        print(f"[{self.name}] {warn_msg}")
                        row_data['Note'] = liq_status
                        
                except Exception as e:
                    # print(f"[{self.name}] LiqCheck error: {e}")
                    pass

            cycle_report.append(row_data)

        # --- PHASE 3: AGGREGATE & UI ---
        if entropies and self.sub_holons.get('entropy'):
            avg_e = sum(entropies) / len(entropies)
            self.market_state['entropy'] = avg_e
            self.market_state['regime'] = self.sub_holons['entropy'].determine_regime(avg_e)

        # Removed redundant _print_summary call
        if monitor and executor: 
            exec_summary = executor.get_execution_summary()
            is_solvent = monitor.update_health(exec_summary, get_performance_data())
            
            if not is_solvent:
                # TRIGGER LIQUIDATION
                print(f"[{self.name}] 📞 MARGIN CALL RECEIVED. LIQUIDATING...")
                executor.panic_close_all(executor.latest_prices)
                self.last_ppo_reward = -100.0 # Severe Penalty for Liquidation
                # Maybe pause for a bit?
                time.sleep(5)

        # Run Overwatch Audit
        if 'overwatch' in self.sub_holons:
            try: self.sub_holons['overwatch'].perform_audit()
            except Exception as e: print(f"[{self.name}] ⚠️ Overwatch Error: {e}")

        self.publish_agent_status()
        return cycle_report

    def _analyze_asset(self, symbol: str, data: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        observer = self.sub_holons.get('observer')
        if data is None and observer:
            try: data = observer.fetch_market_data(limit=100, symbol=symbol)
            except: return None
        if data is None: return None

        row_data = {'Symbol': symbol, 'Price': f"{data['close'].iloc[-1]:.4f}", 'Regime': '?', 'Action': 'HOLD', 'PnL': '-', 'Note': ''}
        current_price = data['close'].iloc[-1]
        
        # --- PATCH: MULTI-TIMEFRAME & POLYMARKET CONTEXT ---
        # 1. Calculate Minutes into Candle (15m)
        last_ts = data['timestamp'].iloc[-1]
        current_time = pd.Timestamp.now(tz=timezone.utc if getattr(last_ts, 'tzinfo', None) else None)
        delta_mins = (current_time - last_ts).total_seconds() / 60.0
        
        # 2. Get 1h Trend (Macro)
        macro_trend = 'NEUTRAL'
        if observer:
            try:
                # Assuming observer has cached 1h data or fetches it fast
                df_1h = observer.fetch_market_data(timeframe='1h', limit=50, symbol=symbol)
                if not df_1h.empty:
                    # User Request: Stable Trend Filter using 20 EMA
                    ema_trend = df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                    macro_trend = 'BULLISH' if df_1h['close'].iloc[-1] > ema_trend else 'BEARISH'
            except: pass
        
        structure_ctx = {
            'minutes_into_candle': delta_mins,
            'macro_trend': macro_trend
        }
        

        
        entropy_agent, oracle = self.sub_holons.get('entropy'), self.sub_holons.get('oracle')
        guardian, governor = self.sub_holons.get('guardian'), self.sub_holons.get('governor')
        executor = self.sub_holons.get('executor')
        
        if entropy_agent:
            entropy_val = entropy_agent.calculate_shannon_entropy(data['returns'])
            regime = entropy_agent.determine_regime(entropy_val)
            row_data['Entropy'], row_data['Regime'] = f"{entropy_val:.3f}", regime
            
            # --- INTEGRATION: Pass Entropy to Oracle via Context ---
            structure_ctx['entropy_val'] = entropy_val
            structure_ctx['entropy_regime'] = regime
            # -----------------------------------------------------

        # Structure Scan (Fractals)
        if oracle:
             base_ctx = oracle.get_structural_context(symbol, data, current_price) if hasattr(oracle, 'get_structural_context') else {}
             structure_ctx.update(base_ctx)
             row_data['Struct'] = structure_ctx.get('structure_mode', '?')

        # Indicators
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        row_data['RSI'] = f"{(100 - (100 / (1 + (gain / loss))).iloc[-1]):.1f}"

        rolling_mean, rolling_std = data['close'].rolling(20).mean(), data['close'].rolling(20).std()
        bb_vals = {'upper': (rolling_mean + 2*rolling_std).iloc[-1], 'middle': rolling_mean.iloc[-1], 'lower': (rolling_mean - 2*rolling_std).iloc[-1]}
        
        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        obv = (np.sign(data['close'].diff()).fillna(0) * data['volume']).cumsum()
        obv_slope, _, _, _, _ = linregress(np.arange(14), obv.iloc[-14:].values)

        metabolism = 'PREDATOR' if executor and executor.get_portfolio_value(current_price) > config.SCAVENGER_THRESHOLD else 'SCAVENGER'
        
        entry_sig = None
        if (not governor or governor.is_trade_allowed(symbol, current_price)) and oracle:
            last_exit = guardian.last_exit_times.get(symbol) if guardian else None
            last_exit = guardian.last_exit_times.get(symbol) if guardian else None
            if not (last_exit and (time.time() - last_exit) < (config.STRATEGY_POST_EXIT_COOLDOWN_CANDLES * 3600)):
                # --- PROJECT AHAB: DATA PREP ---
                book_data = {}
                funding_rate = 0.0
                if observer:
                    try:
                        book_data = observer.fetch_order_book(symbol)
                        funding_rate = observer.fetch_funding_rate(symbol)
                    except: pass
                # -------------------------------
                
                entry_sig = oracle.analyze_for_entry(
                    symbol, data, bb_vals, obv_slope, metabolism, 
                    structure_ctx=structure_ctx,
                    book_data=book_data,
                    funding_rate=funding_rate
                )

        guardian_exit = None
        entry_p = executor.entry_prices.get(symbol, 0.0) if executor else 0.0
        if entry_p > 0 and guardian:
            direction = executor.position_metadata.get(symbol, {}).get('direction', 'BUY')
            age_h = 0.0
            if executor.entry_timestamps.get(symbol):
                from datetime import datetime, timezone
                try: age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(executor.entry_timestamps[symbol])).total_seconds() / 3600
                except: pass
            guardian_exit = guardian.analyze_for_exit(symbol, current_price, entry_p, bb_vals, atr, metabolism, age_h, direction)
            pnl_pct = (current_price - entry_p) / entry_p if direction == 'BUY' else (entry_p - current_price) / entry_p
            row_data['PnL'] = f"{pnl_pct*100:+.2f}%"

        # Enrichment for Dashboard
        probes = oracle.last_probes.get(symbol, {'lstm': 0.5, 'xgb': 0.5}) if oracle else {'lstm': 0.5, 'xgb': 0.5}
        row_data['LSTM'] = f"{probes['lstm']:.2f}"
        row_data['XGB'] = f"{probes['xgb']:.2f}"

        return {
            'symbol': symbol, 'data': data, 'price': current_price, 'row_data': row_data,
            'entropy_val': entropy_val, 'regime': regime, 'metabolism': metabolism,
            'entry_signal': entry_sig, 'guardian_exit': guardian_exit,
            'indicators': {'bb_vals': bb_vals, 'obv_slope': obv_slope, 'atr': atr, 'tr': tr}
        }

    def _create_summary_layout(self, cycle_report: List[Dict]):
        oracle = self.sub_holons.get('oracle')
        sentiment = self.sub_holons.get('sentiment')
        bias = oracle.get_market_bias(sentiment_score=sentiment.current_sentiment_score if sentiment else 0.0) if oracle else 0.5
        
        # 1. Market Status Panel
        bias_color = "green" if bias >= config.GMB_THRESHOLD else ("yellow" if bias >= 0.4 else "red")
        status_text = f"[bold {bias_color}]GLOBAL MARKET BIAS: {bias:.2f}[/bold {bias_color}] | " \
                      f"Status: [{'bold green' if bias >= config.GMB_THRESHOLD else 'bold red'}]" \
                      f"{'BULLISH' if bias >= config.GMB_THRESHOLD else 'CAUTIOUS'}[/]"
        
        header = Panel(status_text, title=f"[{self.name}] Live Dashboard", border_style="blue", expand=False)

        # 2. Detail Table
        table = Table(title="Asset Register", box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Price", style="white")
        table.add_column("Regime", style="magenta")
        table.add_column("Entropy", justify="right")
        table.add_column("Brains (LSTM/XGB)", justify="center")
        table.add_column("Action", style="bold")
        table.add_column("PnL", justify="right", style="green")

        for row in cycle_report:
            probes = oracle.last_probes.get(row['Symbol'], {'lstm': 0.5, 'xgb': 0.5}) if oracle else {'lstm': 0.5, 'xgb': 0.5}
            
            # Colorize Action
            action = row['Action']
            act_style = "dim"
            if "BUY" in action: act_style = "bold green"
            elif "SELL" in action: act_style = "bold red"
            elif "HOLD" in action: act_style = "dim white"
            
            # Colorize Regime
            reg_style = "white"
            if row['Regime'] == 'CHAOTIC': reg_style = "red"
            elif row['Regime'] == 'ORDERED': reg_style = "green"
            
            table.add_row(
                row['Symbol'],
                row['Price'],
                f"[{reg_style}]{row['Regime']}[/{reg_style}]",
                row.get('Entropy', 'N/A'),
                f"{probes['lstm']:.2f} / {probes['xgb']:.2f}",
                f"[{act_style}]{action}[/{act_style}]",
                row['PnL']
            )
            
        # Combine into group (or just return a group/layout)
        from rich.console import Group
        return Group(header, table)

    def publish_agent_status(self):
        if not self.gui_queue: return
        gov, executor = self.sub_holons.get('governor'), self.sub_holons.get('executor')
        oracle = self.sub_holons.get('oracle')
        perf = get_performance_data()
        # Real-time Valuation for Asset Allocation
        latest_prices = executor.latest_prices if executor else {}
        holdings = {'CASH': gov.balance if gov else 0.0}
        total_exp = 0.0
        
        if gov:
            for s, p in gov.positions.items():
                curr_p = latest_prices.get(s, p['entry_price'])
                val = p['quantity'] * curr_p
                holdings[s] = val
                total_exp += val

        portfolio_val = executor.get_portfolio_value(0.0) if executor else 1.0
        
        self.gui_queue.put({
            'type': 'agent_status',
            'data': {
                'gov_state': f"{gov.get_metabolism_state() if gov else 'OFFLINE'}",
                'gov_alloc': f"{config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%",
                'gov_lev': f"{config.PREDATOR_LEVERAGE}x",
                'gov_trends': str(len(gov.positions)) if gov else "0",
                'gov_micro': f"{'ACTIVE' if config.MICRO_CAPITAL_MODE else 'OFF'}", # NEW: Wiring Fix
                'regime': self.market_state['regime'],
                'entropy': f"{self.market_state['entropy']:.4f}",
                'strat_model': 'Warp-V4 (Hybrid)',
                'kalman_active': 'True' if oracle and oracle.kalman_filters else 'False',
                'ppo_conv': f"{self.last_ppo_conviction:.2f}",
                'ppo_reward': f"{self.last_ppo_reward:.2f}",
                'sentiment_score': f"{self.sub_holons['sentiment'].current_sentiment_score:.2f}" if 'sentiment' in self.sub_holons else "0.00",
                'lstm_prob': f"{oracle.get_health().get('last_lstm', 0.5):.2f}",
                'xgb_prob': f"{oracle.get_health().get('last_xgb', 0.5):.2f}",
                'last_order': executor.last_order_details if executor else 'NONE',
                'win_rate': f"{perf.get('win_rate', 0.0):.1f}%",
                'pnl': f"${perf.get('total_pnl', 0.0):.2f}",
                'omega': f"{perf.get('omega_ratio', 0.0):.2f}",
                'exposure': f"${total_exp:.2f}",
                'margin': f"${executor.get_execution_summary()['margin_used']:.2f}" if executor else "$0.00",
                'actual_lev': f"{total_exp/portfolio_val:.2f}x",
                'holdings': holdings,
                # === Dashboard Wiring Fix ===
                'balance': gov.balance if gov else 0.0,
                'equity': portfolio_val,
                'entry_prices': {s: p['entry_price'] for s, p in (gov.positions.items() if gov else {})},
                'current_prices': latest_prices,
                'pending_count': len(executor.actuator.pending_orders) if executor and getattr(executor, 'actuator', None) else 0,
                'news_feed': self.sub_holons['sentiment'].latest_news if 'sentiment' in self.sub_holons else []
            }
        })

    def start_live_loop(self, interval_seconds: int = 60):
        self._active_interval = interval_seconds
        
        # User requested to remove terminal table and reduce noise
        # with Live(console=console, screen=False, refresh_per_second=4) as live:
            
        while True:
            if self.gui_stop_event and self.gui_stop_event.is_set(): break
            
            # --- COMMAND QUEUE PROCESSING ---
            if hasattr(self, 'command_queue') and self.command_queue and not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                    if cmd['type'] == 'update_config':
                        print(f"[{self.name}] ⚙️ Applying Runtime Config: {cmd['data']}")
                        # Apply updates
                        data = cmd['data']
                        if 'max_allocation' in data:
                            config.GOVERNOR_MAX_MARGIN_PCT = float(data['max_allocation'])
                        if 'leverage_cap' in data:
                            config.PREDATOR_LEVERAGE = float(data['leverage_cap'])
                        if 'micro_mode' in data:
                            config.MICRO_CAPITAL_MODE = bool(data['micro_mode'])
                        
                        # Apply to Sub-Holdons if needed
                        gov = self.sub_holons.get('governor')
                        if gov:
                             gov.max_allocation = config.GOVERNOR_MAX_MARGIN_PCT
                    
                    elif cmd['type'] == 'panic_close':
                        print(f"[{self.name}] 🚨 PANIC SIGNAL RECEIVED via Queue")
                        ex = self.sub_holons.get('executor')
                        if ex: ex.panic_close_all(ex.latest_prices)
                        
                except Exception as e:
                    print(f"[{self.name}] Command Error: {e}")
            # --------------------------------
            
            start = time.time()
            try: 
                # Reduced Log Noise: Commented out cycle start print
                # print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval_seconds}s) ---") 
                
                report = self.run_cycle()
                
                # GC Monitor: Run every N cycles
                self.gc_cycle_counter += 1
                gc_interval = getattr(config, 'GC_INTERVAL_CYCLES', 5)
                if self.gc_cycle_counter >= gc_interval:
                    self.run_gc_cycle()
                    self.gc_cycle_counter = 0
                
                if self.gui_queue: self.gui_queue.put({'type': 'summary', 'data': report})
                
                # Disable Terminal Table Update
                # layout = self._create_summary_layout(report)
                # live.update(layout)
                
            except Exception as e:
                print(f"[{self.name}] ☠️ Cycle Error: {e}")
                time.sleep(30)
            
            wait = max(0, interval_seconds - (time.time() - start))
            for _ in range(int(wait * 2)):
                if self.gui_stop_event and self.gui_stop_event.is_set(): break
                time.sleep(0.5)

    def run_gc_cycle(self):
        """
        Garbage Collector Monitor: Run cleanup across all components.
        Called periodically every GC_INTERVAL_CYCLES.
        """
        gc_interval = getattr(config, 'GC_INTERVAL_CYCLES', 5)
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        
        if verbose:
            print(f"\n[GC Monitor] 🧹 Starting Garbage Collection Cycle...")
        
        actuator = self.sub_holons.get('actuator')
        executor = self.sub_holons.get('executor')
        governor = self.sub_holons.get('governor')
        
        # 1. Actuator: Clean stale pending orders
        stale_orders = 0
        if actuator and hasattr(actuator, 'gc_clean_stale_orders'):
            stale_orders = actuator.gc_clean_stale_orders()
        
        # 2. Executor: Reconcile positions with exchange
        ghosts = []
        if executor and hasattr(executor, 'gc_reconcile_positions'):
            ghosts = executor.gc_reconcile_positions()
        
        # 3. Governor: Sync with Executor
        mismatches = []
        if governor and hasattr(governor, 'gc_sync_with_executor'):
            mismatches = governor.gc_sync_with_executor(executor)
        
        if verbose:
            print(f"[GC Monitor] ✅ GC Complete: {stale_orders} stale orders, {len(ghosts)} ghosts, {len(mismatches)} gov mismatches.")

    def receive_message(self, sender, content): pass
    def _adapt_to_regime(self, regime): pass
