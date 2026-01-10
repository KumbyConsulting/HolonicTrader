"""
OverwatchHolon - The "Sentry" (Phase 45)

"I see all, I speak for all."

Responsibilities:
1.  **State Aggregation**: Collects health/status from Governor, Executor, Sentiment.
2.  **Narrative Engine**: Deterministically converts stats into human-readable Situation Reports (SitReps).
3.  **Communication**: Manages the Telegram Bot and pushes updates to the Dashboard.
"""

import threading
import asyncio
import time
from typing import Any, Dict, List
from enum import Enum
import random
import config

from HolonicTrader.holon_core import Holon, Disposition, Message

# Telegram Imports (Robust)
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Overwatch: python-telegram-bot not installed. Telegram features DISABLED.")

class SystemState(Enum):
    CRITICAL = "CRITICAL"   # High Danger (Drawdown, Crisis)
    CAUTION = "CAUTION"     # Heightened Risk (High Volatility, Negative Sentiment)
    NOMINAL = "NOMINAL"     # Standard Operation
    OPTIMAL = "OPTIMAL"     # Good Conditions (Profit & Positive Sentiment)

class NarrativeEngine:
    """
    Deterministic NLP Engine.
    Converts raw metrics into "SitReps" (Situation Reports).
    """
    def __init__(self):
        self.adjectives = {
            'positive': ['Robust', 'Healthy', 'Strong', 'Promising', 'Stable'],
            'negative': ['Fragile', 'Shaky', 'Volatile', 'Uncertain', 'Choppy'],
            'neutral': ['Steady', 'Quiet', 'Flat', 'Normal']
        }

    def _generate_mission_bar(self, ctx) -> str:
        current = ctx.get('equity', 0.0)
        start = config.INITIAL_CAPITAL
        target = config.MISSION_TARGET
        
        # Avoid division by zero
        denom = target - start if target > start else 1.0
        pct = (current - start) / denom
        pct = max(0.0, min(1.0, pct)) # Clamp
        
        # Bar: [|||||.....]
        bars = int(pct * 10)
        progress_bar = "█" * bars + "░" * (10 - bars)
        
        return f"🎯 **{config.MISSION_NAME}**: [{progress_bar}] {pct*100:.1f}% (${current:.2f}/${target:.2f})"

    def generate_sitrep(self, context: Dict[str, Any]) -> str:
        """
        Generate the situation report.
        Context requires: state, metabolism, leverage, sentiment_score, crisis_score, active_positions
        """
        system_state = context.get('state', SystemState.NOMINAL)
        sentiment_score = context.get('sentiment_score', 0.0)
        
        # 0. Mission Status Line
        mission_line = self._generate_mission_bar(context)
        
        # 1. Select Adjective based on Sentiment
        if sentiment_score > 0.3:
            mood = random.choice(self.adjectives['positive'])
        elif sentiment_score < -0.3:
            mood = random.choice(self.adjectives['negative'])
        else:
            mood = random.choice(self.adjectives['neutral'])

        # 2. Select Template based on System State
        if system_state == SystemState.CRITICAL:
            report = self._template_critical(context, mood)
        elif system_state == SystemState.CAUTION:
            report = self._template_caution(context, mood)
        elif system_state == SystemState.OPTIMAL:
            report = self._template_optimal(context, mood)
        else:
            report = self._template_nominal(context, mood)
            
        return f"{mission_line}\n\n{report}"

    def _template_critical(self, ctx, mood) -> str:
        reason = ctx.get('critical_reason', 'Unknown Threat')
        return (
            f"🚨 **CRITICAL ALERT**\n"
            f"System has entered defensive posture due to **{reason}**.\n"
            f"• Metabolism: HIBERNATE\n"
            f"• Action: Halting new entries, monitoring exits closely.\n"
            f"• Sentiment: {mood} ({ctx.get('sentiment_score', 0):.2f})"
        )

    def _template_caution(self, ctx, mood) -> str:
        return (
            f"⚠️ **CAUTION**\n"
            f"Market conditions are **{mood}**. Elevated risk detected.\n"
            f"• Metabolism: {ctx.get('metabolism', 'UNKNOWN')}\n"
            f"• Leverage: Reduced to {ctx.get('leverage_cap', '1.0')}x\n"
            f"• Positions: Holding {ctx.get('position_count', 0)} active trades.\n"
            f"• Note: Strict entry filters are active."
        )

    def _template_nominal(self, ctx, mood) -> str:
        return (
            f"✅ **SYSTEM NOMINAL**\n"
            f"Operations are proceeding normally. Market is **{mood}**.\n"
            f"• Metabolism: {ctx.get('metabolism', 'UNKNOWN')}\n"
            f"• Active Positions: {ctx.get('position_count', 0)}\n"
            f"• Daily PnL: {ctx.get('pnl_str', '$0.00')}"
        )

    def _template_optimal(self, ctx, mood) -> str:
        return (
            f"🚀 **OPTIMAL CONDITIONS**\n"
            f"Systems green. Market is **{mood}** and profitable.\n"
            f"• Performance: {ctx.get('pnl_str', '$0.00')} today.\n"
            f"• Metabolism: PREDATOR (Aggressive)\n"
            f"• Sentiment: High Confidence ({ctx.get('sentiment_score', 0):.2f})"
        )

class OverwatchHolon(Holon):
    def __init__(self, check_interval: int = 60, trader_ref=None):
        # High Integration (0.9), Low Autonomy (0.1) - The "Servant" Monitor
        super().__init__(name="OverwatchHolon", disposition=Disposition(autonomy=0.1, integration=0.9))
        
        self.trader = trader_ref
        self.narrative_engine = NarrativeEngine()
        self.check_interval = check_interval
        
        # State Cache
        self.latest_sitrep = "System Initializing..."
        self.latest_state = SystemState.NOMINAL
        
        # Telegram Config
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.app = None
        self.loop = None
        self.bot_thread = None
        self.stop_event = threading.Event()
        
        if TELEGRAM_AVAILABLE and config.TELEGRAM_ENABLED and config.TELEGRAM_BOT_TOKEN:
            self._setup_telegram()
        else:
            print(f"[{self.name}] Telegram connection skipped.")

    def _setup_telegram(self):
        """Initialize the Telegram Bot Application."""
        try:
            self.app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
            
            # Register Handlers
            self.app.add_handler(CommandHandler("start", self._cmd_start))
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("report", self._cmd_report)) # Verbose SitRep
            self.app.add_handler(CommandHandler("panic", self._cmd_panic))
            
            print(f"[{self.name}] ✅ Telegram Bot Ready (Overwatch Mode)")
            
            # Start Background Thread
            self.bot_thread = threading.Thread(target=self._run_bot_loop, daemon=True)
            self.bot_thread.start()
            
        except Exception as e:
            print(f"[{self.name}] ❌ Telegram Init Failed: {e}")

    def _run_bot_loop(self):
        """Asyncio loop for Telegram Polling."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        while not self.stop_event.is_set():
            try:
                # print(f"[{self.name}] 📞 Polling Telegram...")
                self.app.run_polling(stop_signals=None, close_loop=False, timeout=10)
            except Exception as e:
                print(f"[{self.name}] ⚠️ Telegram Polling Error: {e}. Retrying...")
                time.sleep(5)
            
            if self.stop_event.is_set():
                break

    def perform_audit(self):
        """
        The main 'Sentry' logic.
        Aggregates state -> Generates SitRep -> Broadcasts.
        """
        if not self.trader:
            print(f"[{self.name}] ⚠️ Audit Skipped: Trader Reference Missing!")
            return

        # 1. Gather Intelligence
        gov = self.trader.sub_holons.get('governor')
        sent = self.trader.sub_holons.get('sentiment')
        exec_agent = self.trader.sub_holons.get('executor')
        
        if not (gov and exec_agent):
            # System not fully ready
            return

        # 2. Determine State
        current_equity = exec_agent.get_portfolio_value(0.0)
        
        metrics = {
             'equity': current_equity, # <--- Added for Mission Bar
            'metabolism': gov.get_metabolism_state(),
            'position_count': len(gov.positions),
            'sentiment_score': sent.current_sentiment_score if sent else 0.0,
            'crisis_score': sent.crisis_score if sent else 0.0,
            'drawdown_pct': gov.drawdown_pct,
            'leverage_cap': config.PREDATOR_LEVERAGE if gov.get_metabolism_state() == 'PREDATOR' else config.SCAVENGER_LEVERAGE
        }
        
        # Calculate PnL string
        from performance_tracker import get_performance_data
        perf = get_performance_data()
        pnl_val = perf.get('total_pnl', 0.0)
        metrics['pnl_str'] = f"${pnl_val:+.2f}" 

        # State Logic
        state = SystemState.NOMINAL
        reason = ""
        
        if metrics['drawdown_pct'] > 0.05 or metrics['crisis_score'] > 0.8:
            state = SystemState.CRITICAL
            reason = "High Drawdown" if metrics['drawdown_pct'] > 0.05 else "Geopolitical Crisis"
        elif metrics['sentiment_score'] < -0.4 or metrics['drawdown_pct'] > 0.02:
            state = SystemState.CAUTION
        elif metrics['sentiment_score'] > 0.4:
            state = SystemState.OPTIMAL
            
        metrics['state'] = state
        metrics['critical_reason'] = reason # Logic for Critical
        
        self.latest_state = state
        
        # 3. Generate Narrative
        self.latest_sitrep = self.narrative_engine.generate_sitrep(metrics)
        
        # 4. Push to Dashboard
        if self.trader.gui_queue:
            self.trader.gui_queue.put({
                'type': 'overwatch_update',
                'state': state.value,
                'sitrep': self.latest_sitrep
            })
            
        # 5. Broadcast (Optional: Only on State Change or Critical)
        # For now, we don't spam Telegram every 60s. We specific commands or Critical transitions.
        if state == SystemState.CRITICAL and getattr(self, '_last_broadcast_state', None) != SystemState.CRITICAL:
            self.send_telegram_alert(self.latest_sitrep)
            
        self._last_broadcast_state = state

    # --- Telegram Commands ---
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("👁️ **Overwatch Online**.\nI am monitoring the system.")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self.latest_sitrep, parse_mode='Markdown')

    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Detailed stats + Narrative
        if not self.trader: return
        gov = self.trader.sub_holons.get('governor')
        
        msg = (
            f"{self.latest_sitrep}\n\n"
            f"**Technical Details:**\n"
            f"• Balance: ${gov.balance:.2f}\n"
            f"• Avail Margin: ${gov.available_balance:.2f}\n"
            f"• Drawdown: {gov.drawdown_pct*100:.2f}%\n"
            f"• State: {self.latest_state.value}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def _cmd_panic(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🚨 **PANIC SIGNAL RECEIVED** 🚨\nForwarding to Executor for immediate liquidation.")
        # Trigger Executor Panic
        if self.trader:
            executor = self.trader.sub_holons.get('executor')
            if executor:
                res = executor.panic_close_all(executor.latest_prices)
                await update.message.reply_text(f"🛑 Result:\n{res}")

    def send_telegram_alert(self, msg: str):
        """Thread-safe send."""
        if not (self.app and self.chat_id and self.loop): return
        try:
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.app.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode='Markdown'),
                    self.loop
                )
        except Exception as e:
            print(f"[{self.name}] ❌ Global Alert Failed: {e}")

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming Message objects (alerts)."""
        # If we receive a CRITICAL message, we broadcast immediately.
        pass

    def stop(self):
        self.stop_event.set()
