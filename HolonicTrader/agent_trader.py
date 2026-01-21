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
from HolonicTrader.agent_executor import TradeSignal, TradeDecision
from performance_tracker import get_performance_data
import config
from HolonicTrader.agent_trinity import TrinityStrategy # NEW: Phase 46

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
        self.scout_results = {} # Cache for scout findings
        self.scout_active_list = [] # Persistent Scout Findings
        self.active_session_whitelist = config.ACTIVE_WATCHLIST.copy() # Start with Hot List
        self._load_whitelist_from_disk()
        
        # Evolution Engine Watcher
        self.last_genome_mtime = 0
        
        self.verbose_logging = True # Request C: Enable transparency logs
        
        # Phase 46: Trinity Strategy
        self.trinity = TrinityStrategy()

        self.cycle_counter = 0 # General cycle counter (Phase 46 Fix)

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

    def _scan_for_genome_updates(self):
        """Monitor for new strategies from the Evolution Engine."""
        try:
            path = os.path.join(os.getcwd(), 'live_genome.json')
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime > self.last_genome_mtime:
                    # New Genome detected
                    try:
                        import json
                        with open(path, 'r') as f:
                            data = json.load(f)
                            equity = data.get('final_equity', 0)
                            roi = data.get('roi', 0)
                            
                        # Only log if we have seen a previous version (avoid startup spam if older)
                        # Or just log every time it changes
                        if self.last_genome_mtime > 0:
                            print(f"[{self.name}] 🧬 SYSTEM UPGRADE: New Genome Active! (Sim ROI: {roi*100:.1f}%, Eq: ${equity:.2f})")
                        
                        self.last_genome_mtime = mtime
                    except Exception as e:
                        print(f"[{self.name}] ⚠️ Genome Read Error: {e}")
        except Exception as e:
            pass

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
            
            if symbol in self.scout_active_list: continue # Already active
            
            if personality in ['ROCKET', 'ANCHOR']:
                print(f"[{self.name}] 🚀 SCOUT PROMOTION: {symbol} identified as {personality}. Adding to Active Loop.")
                self.scout_active_list.append(symbol)
                promoted_count += 1
        
        # Sync Status for GUI
        self._sync_scout_status_to_disk(scout_results)
        
        if promoted_count > 0:
            self._sync_whitelist_to_disk()
            
        self.scout_last_run = time.time()
        self.last_ppo_reward = 0.0
        self.scout_last_run = time.time()
        self.last_ppo_reward = 0.0
        self.gc_cycle_counter = 0  # GC Monitor cycle counter
        self.cycle_counter = 0 # General cycle counter (Phase 46 Fix)

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
        self.cycle_counter += 1
        self.perform_health_check()
        
        executor = self.sub_holons.get('executor')
        monitor = self.sub_holons.get('monitor')
        governor = self.sub_holons.get('governor') # Moved up for Balance Sync Patch
        
        # --- PATCH 2: SAFETY-FIRST LOOP ORDERING (Resilience Update) ---
        # 1. Check Connectivity & Equity Status
        current_equity = None
        blind_mode = False
        
        if executor and executor.actuator:
             # Circuit Breaker Check is implicit in Actuator calls
             current_equity = executor.actuator.get_equity()
             
             # --- PATCH: CRITICAL BALANCE SYNC ---
             # We must sync BOTH Equity and Free Margin to Governor/Executor
             # to prevent "Insufficient Funds" errors due to drift.
             free_margin = executor.actuator.get_account_balance()
             
             if current_equity is not None and free_margin is not None:
                 if governor:
                     governor.set_live_balance(current_equity, free_margin)
                 if executor:
                     executor.sync_balance(current_equity) # Updates DB and Internal
                     
                 # Also update Monitor if present
                 if monitor:
                     monitor.metrics['current_equity'] = current_equity
             # ------------------------------------
             
             if current_equity is None:
                 print(f"[{self.name}] ⚠️ BLIND MODE: Cannot fetch Equity (API/Network Issue). Skipping Entry Logic.")
                 blind_mode = True
                 # If we are blind, we generally shouldn't trade, but maybe we can manage exits?
                 # Safest is to skip entries but allow exits if Order Book allows.
                 # But if Actuator is down, we can't do anything.
                 if getattr(executor.actuator, 'circuit_open', False):
                     print(f"[{self.name}] 💤 API CIRCUIT OPEN. Skipping Cycle.")
                     time.sleep(10) 
                     return []
        
        # 2. IMMEDIATE HEALTH CHECK (The Fever Check)
        if monitor and current_equity:
            is_healthy, risk_msg = monitor.perform_live_check(current_equity)
            if not is_healthy:
                print(f"[{self.name}] 🛑 CRITICAL HEALTH LOCKDOWN: {risk_msg}")
                print(f"[{self.name}] 💤 HIBERNATING for 4 hours to cool down...")
                time.sleep(14400) # 4 Hour Hard Sleep
                return [] # Skip cycle
        
        # 3. Old Check (Backup)
        if monitor:
             is_healthy_old, _ = monitor.check_vital_signs()
             if not is_healthy_old:
                 print(f"[{self.name}] 🛑 HEALTH LOCKDOWN (Persistent State). Skipping.")
                 return []
        # -------------------------------------------------------------
        
        interval = getattr(self, '_active_interval', 60)
        print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval}s) ---") 
        
        cycle_report = []
        entropies = []
        cycle_data_cache = {}

        oracle = self.sub_holons.get('oracle')
        observer = self.sub_holons.get('observer')
        # executor fetched above
        governor = self.sub_holons.get('governor')
        ppo = self.sub_holons.get('ppo')
        guardian = self.sub_holons.get('guardian')
        # monitor fetched above
        sentiment = self.sub_holons.get('sentiment')
        overwatch = self.sub_holons.get('overwatch') 
        regime_controller = self.sub_holons.get('regime')

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
            
        # Evolution Watcher
        self._scan_for_genome_updates()
        
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
            
            # Pass Sentiment Score to Oracle & Get Bias
            oracle.set_emotional_bias(fear, greed)
            global_bias = oracle.get_market_bias(sentiment_score=sent_score)
            
            # --- PHASE 46: TRINITY ASSET ROTATION ---
            # 1. Determine Market Regime (from History/Entropy)
            m_regime = self.market_state.get('regime', 'TRANSITION')
            btc_trend = 'BULL' if global_bias >= 0.50 else 'BEAR'
            
            # 2. Get Targets from Trinity Strategy
            trinity_targets = self.trinity.get_allocation_target(m_regime, btc_trend)
            
            # 3. Update Whitelist (Dynamic Focus)
            # Only trade what the strategy wants + Any open positions (to manage exits)
            open_positions = []
            if executor: open_positions = list(executor.held_assets.keys())
            
            # Merge and Dedup (Trinity + OpenPositions + ScoutRockets)
            target_list = list(trinity_targets.keys()) + open_positions + self.scout_active_list
            self.active_session_whitelist = list(set(target_list))
            
            print(f"[{self.name}] 🧘 Trinity Rotation: {m_regime} | {btc_trend} -> Focus: {list(trinity_targets.keys())} (+{len(self.scout_active_list)} Scout Items)")
            
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
            

            
            # --- PACK HUNT DATA PREP ---
            # Calculate 24h % Change for all assets to find Alpha/Beta dispersion
            pack_changes = []
            self.session_ticker_data = {} # Store for _analyze_asset lookup
            
            for sym, d in batch_data.items():
                if len(d) >= 90: # Need approx 24h data (96 bars of 15m)
                    try:
                        # Use first available bar if < 96, else -96
                        start_idx = -96 if len(d) >= 96 else 0
                        start_p = d['close'].iloc[start_idx]
                        end_p = d['close'].iloc[-1]
                        if start_p > 0:
                            pct_change = ((end_p - start_p) / start_p) * 100.0
                            pack_changes.append(pct_change)
                            self.session_ticker_data[sym] = {'percentage': pct_change}
                    except: pass
            
            if pack_changes:
                pack_arr = np.array(pack_changes)
                self.session_pack_stats = {
                    'mean': float(np.mean(pack_arr)),
                    'std': float(np.std(pack_arr))
                }
                # print(f"[{self.name}] 🐺 PACK STATS: Mean {self.session_pack_stats['mean']:.2f}% | Std {self.session_pack_stats['std']:.2f}%")
            else:
                self.session_pack_stats = {'mean': 0.0, 'std': 1.0}
            # ---------------------------

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
            tda_score, tda_status = res.get('tda_score', 0.5), res.get('tda_status', 'STABLE')
            
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
                    
                    # --- PATCH: DRAWDOWN EMERGENCY LIQUIDATION ---
                    # If Drawdown Lock is active, aggressively cut losing positions
                    if governor.drawdown_lock:
                         for pos_sym, pos_data in governor.positions.items():
                             if pos_sym in to_close: continue
                             
                             entry_p = pos_data.get('entry_price', 0)
                             curr_p = executor.latest_prices.get(pos_sym, entry_p)
                             direction = pos_data.get('direction', 'BUY')
                             
                             if entry_p > 0:
                                 if direction == 'BUY':
                                     pnl = (curr_p - entry_p) / entry_p
                                 else:
                                     pnl = (entry_p - curr_p) / entry_p
                                     
                                 if pnl < -0.01: # -1% Loss Threshold (Quick Cut)
                                     print(f"[{self.name}] 🚨 DRAWDOWN EMERGENCY: Liquidating Loser {pos_sym} (PnL {pnl:.2%}) due to Lock.")
                                     to_close.append(pos_sym)
                    # ---------------------------------------------
                    
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
                        # Imports moved to global scope
                        
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
                
                # --- PATCH: HARD TOPOLOGY VETO ---
                if tda_status == 'CRITICAL' and entry_sig:
                     print(f"[{self.name}] 🚨 TOPOLOGY HARD VETO: Structure Collapse detected for {symbol}. Blocking Entry.")
                     entry_sig = None
                # ---------------------------------
                
                # BLIND MODE GUARD: No Entries if we count not verify Equity
                if blind_mode: entry_sig = None
                
                if entry_sig and executor and governor and oracle:
                    pnl_tracker = get_performance_data()
                    atr_ref = indicators['tr'].rolling(14).mean().rolling(14).mean().iloc[-1]
                    atr_ratio = min(2.0, indicators['atr'] / atr_ref) if atr_ref > 0 else 1.0
                    gov_health = governor.get_portfolio_health()
                    
                    # AEHML 2.0: PPO State Expansion (Now 8-Dim)
                    # [Regime_ID, Entropy, WinRate, ATR_Ratio, Drawdown, Margin, TDA_Score, RCMWPE_Regime]
                    
                    # Ensure we have the new metrics from row_data context or analysis loop
                    tda_score_val = tda_score if 'tda_score' in locals() else 0.5 
                    if tda_status == 'CRITICAL': tda_score_val = 0.0 # Force low score for crash
                    
                    # RCMWPE Regime: 0=Safe, 1=Complex/Transitions
                    # Approximation: Use Entropy Regime ID for now, or assume 0.5 if unknown
                    
                    ppo_state = np.array([
                        {'ORDERED': 0.0, 'TRANSITION': 0.5, 'CHAOTIC': 1.0}.get(regime, 0.5), # Legacy Regime
                        entropy_val, 
                        pnl_tracker.get('win_rate', 0.5), 
                        atr_ratio, 
                        gov_health['drawdown_pct'], 
                        gov_health['margin_utilization'],
                        tda_score_val, # NEW
                        0.5 # Placeholder for RCMWPE specific feature
                    ], dtype=np.float32)

                    conviction = ppo.get_conviction(ppo_state) if ppo else 0.5
                    self.last_ppo_conviction = conviction
                    entry_sig.metadata = {'ppo_state': ppo_state.tolist(), 'ppo_conviction': conviction, 'atr': indicators['atr']}

                    # --- NOISE REDUCTION: Signal Deduplication ---
                    # Prevent spamming "EXECUTING ENTRY" or "GOVERNOR VETO" for the same signal 15x times
                    # Checks: Symbol, Direction, and Time (< 5 mins since last attempt)
                    
                    sig_key = f"{symbol}_{entry_sig.direction}"
                    last_attempt = getattr(self, 'last_signal_attempts', {}).get(sig_key, 0)
                    now_ts = time.time()
                    
                    # If tried < 2 mins ago, skip silently (unless price moved > 1%)
                    # We initialize the dict in __init__ ideally, but using getattr is safe patch
                    if not hasattr(self, 'last_signal_attempts'): self.last_signal_attempts = {}
                    
                    if (now_ts - last_attempt) < 120: # 2 Minute Cooldown on Retries
                         # Silent Skip
                         entry_sig = None
                    else:
                         # --- DEBOUNCE CHECK (Optimization #1) ---
                         # Check if price has moved enough from LAST ENTRY to justify disturbing the Governor
                         # This prevents "Stack Too Close" log spam
                         should_skip_governor = False
                         if governor:
                             pos_data = governor.positions.get(symbol)
                             if pos_data:
                                 last_entry_price = pos_data.get('entry_price', 0.0)
                                 # Calculate distance
                                 if last_entry_price > 0:
                                     dist_pct = abs(current_price - last_entry_price) / last_entry_price
                                     # Use the same threshold as Governor uses for stacking
                                     min_dist = getattr(config, 'GOVERNOR_MIN_STACK_DIST', 0.002) 
                                     
                                     if dist_pct < min_dist:
                                         # Too close. Governor WILL reject this.
                                         # So we skip calling it to save logs.
                                         # entry_sig = None # Don't kill signal, just mark to skip? No, kill it.
                                         entry_sig = None
                                         # Only log sparsely (e.g. if we haven't logged this in a while?)
                                         # actually, silently skipping is the goal of debounce.
                                         # But maybe debug print if verbose?
                                         # print(f"[{self.name}] 🤫 Debouncing {symbol}: Price change {dist_pct:.2%} < {min_dist:.2%}")
                                         pass

                         pass

                    if entry_sig:
                        is_whale = entry_sig.metadata.get('is_whale', False)
                        
                        approved, safe_qty, leverage = governor.calc_position_size(
                            symbol, current_price, indicators['atr'], atr_ref, conviction, 
                            direction=entry_sig.direction, sentiment_score=sent_score,
                            whale_confirmed=is_whale
                        )
                    else:
                        approved = False
                        safe_qty = 0
                        leverage = 1.0

                    if approved and safe_qty > 0:
                        entry_sig.size = safe_qty
                        decision = executor.decide_trade(entry_sig, regime, entropy_val)
                        if decision.action != 'HALT':
                            print(f"[{self.name}] 🎯 EXECUTING ENTRY: {symbol} (Qty: {safe_qty:.4f}, Lev: {leverage}x)")
                            
                            # Update Debounce Timestamp (We tried!)
                            self.last_signal_attempts[sig_key] = time.time()
                            
                            pnl_res = executor.execute_transaction(decision, current_price)
                            
                            if pnl_res is not None:
                                # --- PATCH: NOTIFY GOVERNOR (Update Timestamps/Stacks) ---
                                # Critical for Cooldown and Stack Distance Logic
                                if governor:
                                    direction = entry_sig.direction
                                    governor.open_position(symbol, direction, current_price, safe_qty)
                                # ---------------------------------------------------------
                                
                                # Safe Telegram Notification (Overwatch)
                                overwatch = self.sub_holons.get('overwatch')
                                if overwatch and hasattr(overwatch, 'send_telegram_alert'):
                                    msg = f"🚀 **ENTRY** {symbol}\nPrice: {current_price}\nSize: {entry_sig.size:.4f}"
                                    overwatch.send_telegram_alert(msg)

                                row_data['Action'] = f"BUY ({res['metabolism']})"
                                
                                # Log specific reason
                                reason_tag = entry_sig.metadata.get('reason', 'TREND')
                                if entry_sig.metadata.get('is_whale'):
                                    row_data['Action'] = f"WHALE BUY 🐋"
                                else:
                                    row_data['Action'] = f"BUY ({reason_tag})"
                            else:
                                print(f"[{self.name}] ⚠️ ENTRY ABORTED: {symbol} (Execution Failed/Unconfirmed). Governor NOT updated.")
                                row_data['Action'] = "BUY (FAILED)"
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

                # hard_exit_type removed (Redundant)
                
                final_exit = None
                reason = "IDLE"
                if thesis_exit:
                    final_exit = thesis_exit
                    reason = "Thesis"
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
                                    # STRATEGY ALIGNMENT: Scalp-to-Pyramid
                                    # If short duration (< 60m), reward velocity (Scalp).
                                    # If long duration (> 60m), do NOT penalize (Pyramid/Trend).
                                    if duration_mins < 60:
                                        reward = reward / time_factor # Fast wins > Slow scalps
                                    else:
                                        reward = reward # Pure PnL for Trend Following (Don't punish patience)
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

                        # Safe Telegram Notification (Overwatch)
                        overwatch = self.sub_holons.get('overwatch')
                        if overwatch and hasattr(overwatch, 'send_telegram_alert'):
                            msg = f"📉 **EXIT** {symbol}\nPrice: {current_price}\nPnL: {pnl_res*100:+.2f}% ({reason})"
                            overwatch.send_telegram_alert(msg)

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
        if self.sub_holons.get('overwatch'):
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
        
        # SAFETY CHECK: Ensure columns exist
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        if not all(col in data.columns for col in required_cols):
             print(f"[{self.name}] ⚠️ Data Validation Error for {symbol}. Missing Columns. Keys: {data.columns.tolist()}")
             return None

        row_data = {'Symbol': symbol, 'Price': f"{data['close'].iloc[-1]:.4f}", 'Regime': '?', 'Action': 'HOLD', 'PnL': '-', 'Note': ''}
        current_price = data['close'].iloc[-1]
        
        # PROFILING LOG (User Request)
        if symbol in ['SOL/USDT', 'XRP/USDT', 'BTC/USDT', 'XTZ/USDT', 'TBTC/USDT']:
             # Quick peek at Scout personality if available
             scout_res = getattr(self, 'scout_results', {})
             pers = scout_res.get(symbol, "Unknown")
             print(f"[{self.name}] 🕵️ PROFILING {symbol}: Price ${current_price:.2f} | Personality: {pers} | Rows: {len(data)}")
        
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
        topology = self.sub_holons.get('topology') # <--- AEHML 2.0
        
        # 3. Calculate Entropy & Regime
        entropy_val = 0.0
        regime = 'TRANSITION'
        tda_score = 0.5
        tda_status = 'STABLE'

        if entropy_agent:
            # Note: Using calculated returns from indicators usually, but here calculating fresh?
            # Let's assume data['close'] is Series.
            # Efficiency: Calculate returns once
            returns = data['close'].pct_change().dropna()
            entropy_val = entropy_agent.calculate_shannon_entropy(returns)
            regime = entropy_agent.determine_regime(entropy_val)
            row_data['Entropy'], row_data['Regime'] = f"{entropy_val:.3f}", regime
            
            # --- INTEGRATION: Pass Entropy to Oracle via Context ---
            structure_ctx['entropy_val'] = entropy_val
            structure_ctx['entropy_regime'] = regime
            # -----------------------------------------------------

        # AEHML 2.0: Topological Check
        if topology:
            tda_res = topology.analyze_structure(data)
            tda_score = tda_res.get('score', 0.5)
            tda_status = tda_res.get('status', 'STABLE')
            
            # If topology is collapsing, override regime display to warn user
            if tda_status == 'CRITICAL':
                row_data['Regime'] = f"CRASH WARNING (TDA {tda_score:.2f})"
                structure_ctx['tda_critical'] = True

        # Structure Scan (CTKS Integration)
        structure = self.sub_holons.get('structure')
        if structure:
             ctx = structure.get_structural_context(symbol, observer)
             structure_ctx.update(ctx)
             row_data['Struct'] = f"{ctx.get('sls_zone', 'N')}" # Display Zone
        elif oracle:
             # Fallback (Legacy)
             base_ctx = oracle.get_structural_context(symbol, data, current_price) if hasattr(oracle, 'get_structural_context') else {}
             structure_ctx.update(base_ctx)

        # Indicators (RUST ACCELERATED)
        # Common Indicators (calculated globally to fix UnboundLocalError)
        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)

        try:
            import holonic_speed
            
            closes_list = data['close'].values.tolist()
            highs_list = data['high'].values.tolist()
            lows_list = data['low'].values.tolist()
            
            rsi_series = holonic_speed.calculate_rsi(closes_list, 14)
            data['rsi'] = rsi_series[-len(data):]
            rsi_val = rsi_series[-1]
            
            bb_u, bb_m, bb_l = holonic_speed.calculate_bollinger_bands(closes_list, 20, 2.0)
            bb_vals = {'upper': bb_u[-1], 'middle': bb_m[-1], 'lower': bb_l[-1]}
            
            atr_series = holonic_speed.calculate_atr(highs_list, lows_list, closes_list, 14)
            atr = atr_series[-1]
            
            row_data['RSI'] = f"{rsi_val:.1f}"

        except ImportError:
            # Fallback to Pandas (Legacy)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            row_data['RSI'] = f"{(100 - (100 / (1 + (gain / loss))).iloc[-1]):.1f}"

            rolling_mean, rolling_std = data['close'].rolling(20).mean(), data['close'].rolling(20).std()
            bb_vals = {'upper': (rolling_mean + 2*rolling_std).iloc[-1], 'middle': rolling_mean.iloc[-1], 'lower': (rolling_mean - 2*rolling_std).iloc[-1]}
            
            atr = tr.rolling(14).mean().iloc[-1]
        
        obv = (np.sign(data['close'].diff()).fillna(0) * data['volume']).cumsum()
        obv_slope, _, _, _, _ = linregress(np.arange(14), obv.iloc[-14:].values)

        metabolism = 'PREDATOR' if executor and executor.get_portfolio_value(current_price) > config.SCAVENGER_THRESHOLD else 'SCAVENGER'
        
        entry_sig = None
        if (not governor or governor.is_trade_allowed(symbol, current_price)) and oracle:
            pass # Continue to logic below
        else:
             if governor and not governor.is_trade_allowed(symbol, current_price):
                 print(f"[{self.name}] 🛑 GOVERNOR PRE-CHECK VETO for {symbol}: Trade Disallowed (Cooldown/Stack/Funds).")
                 
        if (not governor or governor.is_trade_allowed(symbol, current_price)) and oracle:
            last_exit = guardian.last_exit_times.get(symbol) if guardian else None
            
            # --- PROJECT AHAB: DATA PREP ---
            book_data = {}
            funding_rate = 0.0
            if observer:
                try:
                    book_data = observer.fetch_order_book(symbol)
                    funding_rate = observer.fetch_funding_rate(symbol)
                except: pass
            
            # --- INTEGRATION: WHALE HOLON ---
            whale = self.sub_holons.get('whale')
            is_whale_signal = False
            if whale:
                # Calculate Daily Volume (Approx) for Dynamic Thresholds
                # We have 100 candles of 15m (~25h). Sum(Close * Volume) for last 96 is strict 24h.
                daily_vol_usd = 0.0
                try:
                    recent = data.iloc[-96:] if len(data) >= 96 else data
                    daily_vol_usd = (recent['close'] * recent['volume']).sum()
                except: 
                    daily_vol_usd = 0.0

                # Check for "Whale-Scalper" Setup (Bid Wall)
                is_whale_signal = whale.check_bid_wall(symbol, book_data, daily_vol=daily_vol_usd)
                if is_whale_signal:
                    print(f"[{self.name}] 🐋 WHALE SIGNAL DETECTED: {symbol} (Bid Wall)")
            # --------------------------------
            
            # --- UPDATE INTERVAL DYNAMICALLY ---
            # Check Regime for VOL_WINDOW speed
            if regime == 'VOL_WINDOW':
                 self._active_interval = config.VOL_WINDOW_CYCLE_INTERVAL # 15s
            else:
                 self._active_interval = config.DEFAULT_CYCLE_INTERVAL if hasattr(config, 'DEFAULT_CYCLE_INTERVAL') else 60
            # -----------------------------------

            entry_sig = oracle.analyze_for_entry(
                symbol, data, bb_vals, obv_slope, metabolism, 
                structure_ctx=structure_ctx,
                book_data=book_data,
                ticker_data=getattr(self, 'session_ticker_data', {}).get(symbol, {}),
                pack_stats=getattr(self, 'session_pack_stats', {}),
                funding_rate=funding_rate,
                observer=observer
            )
            
            # Inject Whale Signal into Oracle Metadata if Oracle missed it or to reinforce
            if entry_sig:
                 if is_whale_signal:
                     entry_sig.metadata['is_whale'] = True
                     entry_sig.metadata['reason'] = 'WHALE_SCALPER'
            elif is_whale_signal:
                 # Force Entry if Whale Detected and Oracle didn't veto?
                 pass


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

        # 3D Holospace Data Injection
        row_data['_entropy'] = entropy_val
        row_data['_tda'] = tda_score
        row_data['_price'] = current_price
        row_data['_vol'] = atr if 'atr' in locals() else 0.0

        return {
            'symbol': symbol, 'data': data, 'price': current_price, 'row_data': row_data,
            'entropy_val': entropy_val, 'regime': regime, 'metabolism': metabolism,
            'entry_signal': entry_sig, 'guardian_exit': guardian_exit,
            'tda_score': tda_score, 'tda_status': tda_status,
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
                'gov_micro': f"{'ACTIVE' if config.MICRO_CAPITAL_MODE else 'OFF'}",
                'risk_budget': f"${gov.risk_budget:.2f}" if gov else "$0.00",
                'fortress_balance': f"${gov.fortress_balance:.2f}" if gov else "$300.00",
                'solvency_status': 'SOLVENT' if (gov and gov.balance >= config.MIN_ORDER_VALUE) else 'INSOLVENT', # Explicit Warning
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
                'news_feed': self.sub_holons['sentiment'].latest_news if 'sentiment' in self.sub_holons else [],
                # === Regime/Health Data ===
                'health_score': self.sub_holons['regime'].get_status_summary().get('health_score', 0.0) if 'regime' in self.sub_holons else 0.0,
                'promo_progress': self.sub_holons['regime'].get_status_summary().get('promotion_progress', 0.0) if 'regime' in self.sub_holons else 0.0,
                # === Consolidation Radar ===
                # === Consolidation Radar ===
                'scout_data': [{'symbol': s, 'score': 0.95, 'reason': p} for s, p in self.scout_results.items()],
                'consolidation_data': [
                    {'symbol': r[0], 'score': float(r[1]), 'reason': r[2]} 
                    for r in (oracle.get_consolidation_rankings()[:10] if oracle and hasattr(oracle, 'get_consolidation_rankings') else [])
                    if isinstance(r, (list, tuple)) and len(r) >= 3
                ]
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
                # --- CHECK FOR GENOME UPDATES (WINNING BRAIN) ---
                self._scan_for_genome_updates()
                # -----------------------------------------------

                # Reduced Log Noise: Commented out cycle start print
                # print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval_seconds}s) ---") 
                
                # --- PHASE X: TRAILING STOPS ---
                # Manage stops before running new signals
                ex = self.sub_holons.get('executor')
                if ex: self._manage_trailing_stops(ex)
                # -------------------------------

                report = self.run_cycle()
                
                # GC Monitor: Run every N cycles
                self.gc_cycle_counter += 1
                gc_interval = getattr(config, 'GC_INTERVAL_CYCLES', 5)
                if self.gc_cycle_counter >= gc_interval:
                    self.run_gc_cycle()
                    self.gc_cycle_counter = 0
                
                if self.gui_queue: self.gui_queue.put({'type': 'summary', 'data': report})
                
                # --- PHASE 22: PPO REWARD HOSE ---
                # Feed the Monolith with the results of this cycle (Action=Previous Conviction, Reward=Equity Delta)
                self._feed_the_monolith(report)
                # ---------------------------------
                
                # Disable Terminal Table Update
                # layout = self._create_summary_layout(report)
                # live.update(layout)
                
            except Exception as e:
                import traceback
                print(f"[{self.name}] ☠️ Cycle Error: {e}")
                traceback.print_exc()
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

    def _feed_the_monolith(self, report):
        """
        Calculate Step Reward and feed the PPO Brain.
        Reward = (Realized PnL + Unrealized Delta) / Volatility
        """
        governor = self.sub_holons.get('governor')
        if not governor or not hasattr(governor, 'ppo'): return
        
        # 1. Calculate Reward
        # We need equity change since last cycle.
        current_equity = governor.balance
        if not hasattr(self, 'last_ppo_equity'): self.last_ppo_equity = current_equity
        
        equity_delta = current_equity - self.last_ppo_equity
        self.last_ppo_equity = current_equity
        
        # Normalize reward (e.g. $0.10 gain on $10 account = 1%)
        # Scale up because PPO likes ~1.0 range
        raw_reward = (equity_delta / config.INITIAL_CAPITAL) * 100.0
        
        # simple clipping to avoid exploding gradients from wild swings
        reward = max(-5.0, min(5.0, raw_reward))
        
        # 2. Get State (Simplified for now)
        # [WinRate, Drawdown, MarginUsed, ... ]
        state = [
            governor.db_manager.get_win_rate() if governor.db_manager else 0.5,
            governor.drawdown_pct,
            governor.margin_utilization,
            0.0, 0.0, 0.0 # placeholders
        ]
        
        # 3. Remember (We assume Action was 0.5 for now, finding exact action requires tracking)
        # ideally we track what action PPO suggested at start of cycle.
        # For now, we just train it to associate State -> Reward.
        # This is a loose approximation to prime the memory.
        import numpy as np
        governor.ppo.remember(
            state=np.array(state, dtype=np.float32), 
            action=0.5, # Placeholder action
            reward=reward, 
            prob=0.5, 
            val=0.5, 
            done=False
        )
        
        # 4. Learn periodically
        if (self.cycle_counter % 10) == 0:
            a_loss, c_loss = governor.ppo.learn()
            if abs(a_loss) > 0:
                print(f"[{self.name}] 🧠 PPO LEARN: Rewards={reward:.4f} | Loss A={a_loss:.4f} C={c_loss:.4f}")

    def _scan_for_genome_updates(self):
        """
        Phase 46 + Ensemble: Checks for new Evolution Result & Hall of Fame.
        Hot-swaps parameters and Ensemble Strategies.
        """
        import os
        import json
        
        # 1. LIVE GENOME (Single Best) - Parameter Tuning
        path_genome = os.path.join(os.getcwd(), 'live_genome.json')
        if os.path.exists(path_genome):
            try:
                mtime = os.path.getmtime(path_genome)
                if not hasattr(self, 'last_genome_mtime'): self.last_genome_mtime = 0

                if mtime > self.last_genome_mtime:
                    self.last_genome_mtime = mtime
                    with open(path_genome, 'r') as f:
                        data = json.load(f)
                    
                    genome = data.get('genome', {})
                    if genome:
                        print(f"[{self.name}] 🧬 DETECTED NEW EVOLVED BRAIN (Fitness: {data.get('fitness', 0):.2f})")
                        
                        # Update Config (Global Defaults)
                        config.STRATEGY_RSI_OVERSOLD = float(genome.get('rsi_buy', config.STRATEGY_RSI_OVERSOLD))
                        config.STRATEGY_RSI_OVERBOUGHT = float(genome.get('rsi_sell', config.STRATEGY_RSI_OVERBOUGHT))
                        config.SATELLITE_STOP_LOSS = float(genome.get('stop_loss', config.SATELLITE_STOP_LOSS))
                        config.SATELLITE_TAKE_PROFIT_1 = float(genome.get('take_profit', config.SATELLITE_TAKE_PROFIT_1))
                        
                        print(f"[{self.name}] ✅ Brain Transplant Successful. Parameters Active.")
            except Exception as e:
                print(f"[{self.name}] ⚠️ Genome Read Error: {e}")

        # 2. HALL OF FAME (Ensemble) - Logic Update
        path_hof = os.path.join(os.getcwd(), 'hall_of_fame.json')
        if os.path.exists(path_hof):
            try:
                mtime = os.path.getmtime(path_hof)
                if not hasattr(self, 'last_hof_mtime'): self.last_hof_mtime = 0
                
                if mtime > self.last_hof_mtime:
                    self.last_hof_mtime = mtime
                    
                    oracle = self.sub_holons.get('oracle')
                    if oracle and hasattr(oracle, 'load_ensemble'):
                        oracle.load_ensemble(path_hof)
                        print(f"[{self.name}] 🎭 ENSEMBLE DEPLOYED: Hall of Fame Loaded into Oracle.")
            except Exception as e:
                print(f"[{self.name}] ⚠️ HOF Read Error: {e}")

    def _manage_trailing_stops(self, executor):
        """
        PHASE X: Active Trailing Stop Manager.
        Scans open positions. If PnL > Activation Threshold (e.g. 1.5R),
        Moves Stop Loss towards price to lock in gains.
        """
        if not executor or not executor.actuator: return
        
        # 1. Get Activation Parameters
        # TODO: Pull from Live Genome?
        # Default: Activate at 1.5% profit, Trail by 1.0% distance
        activation_pct = 0.015 
        trail_dist_pct = 0.010
        
        for symbol, qty in executor.held_assets.items():
            if abs(qty) < 0.00000001: continue
            
            # Get Current Price
            curr_price = executor.latest_prices.get(symbol, 0.0)
            if curr_price <= 0: continue
            
            # Get Entry Data
            meta = executor.position_metadata.get(symbol, {})
            entry_price = meta.get('entry_price', curr_price)
            if entry_price <= 0: continue
            
            direction = meta.get('direction', 'BUY')
            
            # Calculate PnL %
            if direction == 'BUY':
                pnl_pct = (curr_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - curr_price) / entry_price
                
            # Check Activation
            if pnl_pct > activation_pct:
                # We are in profit zone!
                
                # Check Expected Stop Price
                # Long: Price * (1 - trail)
                # Short: Price * (1 + trail)
                
                if direction == 'BUY':
                    new_stop_price = curr_price * (1.0 - trail_dist_pct)
                else:
                    new_stop_price = curr_price * (1.0 + trail_dist_pct)
                    
                # Find Existing Stop Order
                # We look for pending stop orders for this symbol
                existing_stop_id = None
                existing_stop_price = 0.0
                
                for order in executor.actuator.pending_orders:
                    if order.get('symbol') == symbol and order.get('type') == 'stop-market':
                         existing_stop_id = order.get('id')
                         existing_stop_price = float(order.get('stop_price', 0.0))
                         break
                         
                # Decide Update
                should_update = False
                if existing_stop_id:
                    if direction == 'BUY':
                        # Move UP only
                        if new_stop_price > (existing_stop_price * 1.001): # 0.1% buffer
                            should_update = True
                    else:
                        # Move DOWN only
                        if new_stop_price < (existing_stop_price * 0.999):
                            should_update = True
                else:
                    # No stop? Create one!
                    should_update = True
                    
                if should_update:
                    print(f"[{self.name}] 🥅 TRAILING STOP UPDATE: {symbol} PnL {pnl_pct*100:.1f}%. Moving Stop -> {new_stop_price:.2f}")
                    
                    # Cancel Old
                    if existing_stop_id:
                        executor.actuator.cancel_order(existing_stop_id, symbol)
                        
                    # Place New
                    # Stop Direction is Opposite to Position
                    stop_dir = 'SELL' if direction == 'BUY' else 'BUY'
                    executor.actuator.place_stop_order(symbol, stop_dir, abs(qty), new_stop_price)
