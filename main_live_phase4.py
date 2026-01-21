"""
HolonicTrader - LIVE Execution Entry Point (Phase 4)
"""

import config
from HolonicTrader.agent_trader import TraderHolon
from HolonicTrader.holon_core import Disposition
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_diagnostic import DiagnosticHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_monitor import MonitorHolon
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon
from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.agent_ppo import PPOHolon
from HolonicTrader.agent_sentiment import SentimentHolon
from HolonicTrader.agent_overwatch import OverwatchHolon # <--- NEW: The Sentry
from HolonicTrader.agent_whale import WhaleHolon # <--- NEW: The Harpoon
from HolonicTrader.agent_structure import CTKSStrategicHolon # <--- NEW: The Institution

from database_manager import DatabaseManager

from queue import Queue
import threading
import sys
from datetime import datetime
import re

class QueueLogger:
    """Redirects stdout to a Queue for GUI display, plus file logging."""
    def __init__(self, filename, log_queue=None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log_queue = log_queue
        self.log = open(filename, "a", encoding='utf-8')
        self.lock = threading.Lock()
    
    def write(self, message):
        with self.lock:
            # Timestamp logic
            final_msg = message
            if message.strip():
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                if not message.startswith("[20"): 
                    final_msg = f"{timestamp}{message}"
            
            # 1. Print to Real Terminal (Hidden in GUI mode usually, but good for debug)
            # self.terminal.write(final_msg)
            
            # 2. Write to File
            try:
                self.log.write(final_msg)
                self.log.flush()
            except Exception:
                pass # Prevent logging errors from crashing the bot
            
            # 3. Push to Queue (if exists)
            if self.log_queue:
                try:
                    # Strip ANSI codes for GUI display using regex
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    clean_msg = ansi_escape.sub('', final_msg)
                    
                    self.log_queue.put({
                        'type': 'log',
                        'message': clean_msg
                    }, block=False)
                except Exception:
                    pass # Queue full or closed

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass

def main_live(status_queue: Queue = None, stop_event: threading.Event = None, interval_seconds: int = 60, command_queue: Queue = None, disable_telegram: bool = False):
    print("==========================================")
    print("   HOLONIC TRADER - LIVE ENVIRONMENT      ")
    print("==========================================")
    
    # 0. Initialize Database
    db = DatabaseManager()

    # 0b. System Diagnostics
    diagnostic = DiagnosticHolon()
    if not diagnostic.run_system_check(db):
        print(">> 🛑 SYSTEM CHECK FAILED. HALTING STARTUP.")
        return

    # 0c. Capital Synchronization (Live & Paper)
    if not config.PAPER_TRADING:
        try:
            import ccxt
            print(f">> 🔄 Syncing Capital from Kraken ({config.TRADING_MODE})...")
        
            if config.TRADING_MODE == 'FUTURES':
                exchange_class = ccxt.krakenfutures
                # Use specific Futures keys if available
                api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
                api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
            else:
                exchange_class = ccxt.kraken
                api_key = config.API_KEY
                api_secret = config.API_SECRET
                
            exchange = exchange_class({'apiKey': api_key, 'secret': api_secret})
            bal = exchange.fetch_balance()
            info = bal.get('info', {})
            
            real_equity = 0.0
            
            if config.TRADING_MODE == 'FUTURES':
                # Futures Equity Check (Multi-Collateral 'flex')
                accounts = info.get('accounts', {})
                flex = accounts.get('flex', {})
                real_equity = float(flex.get('marginEquity', 0.0))
                
                if real_equity <= 0:
                     # Fallback to cash USD if no margin account
                     real_equity = bal.get('total', {}).get('USD', 0.0)
            else:
                # Spot Equity Check
                real_equity = float(info.get('eb', 0.0))
                if real_equity <= 0: real_equity = float(info.get('tb', 0.0))
                if real_equity <= 0: 
                     real_equity = bal['free'].get('USD', 0.0) + bal['free'].get('USDT', 0.0)
            
            if real_equity > 5.0: # Sanity check
                 print(f">> 💰 SYNC SUCCESS: Real Equity ${real_equity:.2f}")
                 config.INITIAL_CAPITAL = real_equity
                 print(f"   -> Set INITIAL_CAPITAL = ${config.INITIAL_CAPITAL:.2f}")
            else:
                 print(f">> ⚠️ Exchange Balance too low (${real_equity:.2f}), using Config Default (${config.INITIAL_CAPITAL}).")
    
        except Exception as e:
            print(f">> ⚠️ Capital Sync Failed: {e}. Using Config Defaults.")

    
    from HolonicTrader.agent_topology import TopologyHolon # <--- NEW: Structure Brain

    # 1. Instantiate Core Agents
    observer = ObserverHolon(exchange_id='kucoin')
    entropy = EntropyHolon()
    topology = TopologyHolon() # <--- AEHML 2.0
    oracle = EntryOracleHolon()
    guardian = ExitGuardianHolon()
    monitor = MonitorHolon(principal=config.PRINCIPAL)
    ppo = PPOHolon()
    sentiment = SentimentHolon() 
    whale = WhaleHolon() 
    structure = CTKSStrategicHolon() 
    
    # 2. Instantiate Execution Stack
    governor = GovernorHolon(initial_balance=config.INITIAL_CAPITAL, db_manager=db)
    
    actuator = None
    if not config.PAPER_TRADING:
        print(">>> 🚨 LIVE TRADING ENABLED - REAL MARKET EXECUTION 🚨 <<<")
        actuator = ActuatorHolon()
    else:
        print(">>> 📊 PAPER TRADING MODE ACTIVE - SIMULATED EXECUTION <<<")
        
    executor = ExecutorHolon(
        initial_capital=config.INITIAL_CAPITAL,
        governor=governor,
        actuator=actuator,
        db_manager=db,
        gui_queue=status_queue # NEW: Dashboard Link
    )
    
    # 2b. Sync Governor & Exchange
    executor.reconcile_exchange_positions() 
    
    # --- OPTIMIZED BALANCE SYNC (Phase 16) ---
    if actuator:
        # Try to get live balance from exchange
        live_bal = actuator.get_account_balance()
        if live_bal and live_bal > 0:
            executor.sync_balance(live_bal)
        else:
            # Fallback to DB state
            executor.sync_balance(executor.balance_usd)
    else:
        # Paper Trading: Trust the DB (restored in executor.__init__) over hardcoded config
        executor.sync_balance(executor.balance_usd)
    # -----------------------------------------

    governor.sync_positions(executor.held_assets, executor.position_metadata)

    # 2c. Overwatch (The Sentry: Telegram + NLP)
    overwatch = None
    if not disable_telegram:
        try:
            overwatch = OverwatchHolon()
        except:
            print(">> [Warning] Overwatch failed to start. Telegram disabled.")
    
    # 2d. Regime Controller (Phase 7: Capital Regime Management)
    from HolonicTrader.agent_regime import RegimeController
    regime_controller = RegimeController()
    governor.regime_controller = regime_controller  # Link to Governor

    # 3. Instantiate Trader
    trader = TraderHolon("TraderNexus", sub_holons={
        'observer': observer,
        'entropy': entropy,
        'topology': topology, # <--- Added to Nexus
        'oracle': oracle,
        'guardian': guardian,
        'monitor': monitor,
        'governor': governor,
        'executor': executor,
        'ppo': ppo,
        'sentiment': sentiment,
        'overwatch': overwatch,
        'regime': regime_controller,
        'whale': whale,
        'structure': structure 
    })

    trader_ref_linked = False
    try:
        # 4. Link Overwatch to Trader (Circular Dependency Resolution)
        if overwatch:
            overwatch.trader = trader
            trader_ref_linked = True
            print(">> [System] Overwatch Linked to TraderNexus.")

        # 4b. Integrate Stop Signals & Queue
        trader.gui_queue = status_queue
        trader.gui_stop_event = stop_event
        trader.command_queue = command_queue # <--- NEW: Command Link
        
        # 5. Start Loop
        print(">> Initializing System Components...")
        trader.start_live_loop(interval_seconds=interval_seconds)
        
    except KeyboardInterrupt:
        print("\n>> STOP REQUEST RECEIVED. SHUTTING DOWN...")
    except Exception as e:
        print(f"\n>> FATAL MAIN LOOP ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Proper Resource Cleanup
        print(">> Cleaning up resources...")
        try:
            if overwatch:
                overwatch.stop()
        except Exception:
             pass
             
        # Explicit State Save
        try:
             if 'executor' in locals() and executor:
                 executor.save_state()
        except Exception as e:
             print(f"Error saving state: {e}")

        try:
            db.close()
        except Exception:
            pass
        print(">> SYSTEM SHUTDOWN COMPLETE.")

def run_bot(stop_event, status_queue, config_dict=None, command_queue=None, disable_telegram=False):
    """Wrapper for GUI Thread"""
    # Setup Logger
    log_file = f"live_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = QueueLogger(log_file, log_queue=status_queue)
    
    try:
        # Update Config from GUI if provided
        if config_dict:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Applying Dashboard Config...")
            
            # Map GUI symbols to config
            gui_symbol = config_dict.get('symbol')
            if gui_symbol and gui_symbol not in config.ALLOWED_ASSETS:
                # Add the selected symbol to the universe if it's not there
                config.ALLOWED_ASSETS.append(gui_symbol)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Added {gui_symbol} to Asset Universe.")
                
            # Ensure uniqueness
            config.ALLOWED_ASSETS = list(set(config.ALLOWED_ASSETS))
            
            # Dynamic leverage and allocation
            config.GOVERNOR_MAX_MARGIN_PCT = float(config_dict.get('max_allocation', config.GOVERNOR_MAX_MARGIN_PCT))
            config.PREDATOR_LEVERAGE = float(config_dict.get('leverage_cap', config.PREDATOR_LEVERAGE))
            
            # Dynamic Micro Mode
            if 'micro_mode' in config_dict:
                config.MICRO_CAPITAL_MODE = bool(config_dict['micro_mode'])
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Config Applied: Allocation {config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%, Leverage {config.PREDATOR_LEVERAGE}x, MicroMode: {config.MICRO_CAPITAL_MODE}")
            
        # 1. Start Loop (Check if GUI provided a specific interval, else default to 60)
        interval = config_dict.get('loop_interval', 60) if config_dict else 60
        main_live(status_queue, stop_event, interval_seconds=interval, command_queue=command_queue, disable_telegram=disable_telegram)
    except Exception as e:
        print(f"Bot Crashed: {e}")

if __name__ == "__main__":
    # Standalone Mode
    log_file = f"live_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = QueueLogger(log_file)
    main_live()
