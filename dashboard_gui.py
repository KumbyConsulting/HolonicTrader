import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
from datetime import datetime
import time

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# Import the bot runners
from main_live_phase4 import run_bot
from run_backtest import run_backtest

# === IMPROVEMENT 1: DARK MODE COLORS ===
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_card': '#16213e',
    'bg_input': '#0f3460',
    'accent': '#e94560',
    'accent_green': '#00e676',
    'accent_red': '#ff5252',
    'accent_yellow': '#ffd740',
    'accent_blue': '#40c4ff',
    'text_primary': '#ffffff',
    'text_secondary': '#a0a0a0',
    'border': '#333355'
}

class HolonicDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("HolonicTrader Command Center")
        self.root.geometry("1500x950")
        
        # === IMPROVEMENT 1: Apply Dark Mode ===
        self.root.configure(bg=COLORS['bg_dark'])
        self._setup_dark_theme()
        
        # Threading vars
        self.gui_queue = queue.Queue()
        self.command_queue = queue.Queue() # Wrapper for thread commands
        self.gui_stop_event = threading.Event()
        self.bot_thread = None
        self.is_running_live = False
        self.is_running_backtest = False
        
        # PPO Tracking
        self.last_ppo_conviction = 0.5
        self.last_ppo_reward = 0.0
        
        # === IMPROVEMENT 2: Real-Time Equity Tracking ===
        self.equity_history = []
        self.max_equity_points = 100
        
        # === IMPROVEMENT 9: Order History ===
        self.order_history = []
        self.max_orders = 50
        
        # === IMPROVEMENT 6: GC Monitor Status ===
        self.gc_last_run = "Never"
        self.gc_items_cleaned = 0
        
        # === IMPROVEMENT 8: Exchange Balance ===
        self.exchange_balance = 0.0

        # === IMPROVEMENT 10: Scout Data ===
        self.scout_status = {}
        self.scout_last_read = 0.0
        
        # Configuration Vars
        self.conf_symbol = tk.StringVar(value="XRP/USDT")
        self.conf_timeframe = tk.StringVar(value="1h")
        self.conf_alloc = tk.DoubleVar(value=0.10)
        self.conf_leverage = tk.DoubleVar(value=5.0)

        self._setup_ui()
        
        # Start the polling loop
        self.root.after(100, self.process_queue)

    # === IMPROVEMENT 1: DARK THEME SETUP ===
    def _setup_dark_theme(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure dark theme for all widgets
        style.configure(".", 
            background=COLORS['bg_dark'], 
            foreground=COLORS['text_primary'],
            fieldbackground=COLORS['bg_input'],
            font=("Segoe UI", 10))
        
        style.configure("TFrame", background=COLORS['bg_dark'])
        style.configure("TLabel", background=COLORS['bg_dark'], foreground=COLORS['text_primary'])
        style.configure("TButton", 
            background=COLORS['bg_card'], 
            foreground=COLORS['text_primary'],
            borderwidth=1,
            focuscolor=COLORS['accent'])
        style.map("TButton",
            background=[('active', COLORS['accent']), ('pressed', COLORS['accent_red'])])
        
        style.configure("TNotebook", background=COLORS['bg_dark'], borderwidth=0)
        style.configure("TNotebook.Tab", 
            background=COLORS['bg_card'], 
            foreground=COLORS['text_primary'],
            padding=[15, 8])
        style.map("TNotebook.Tab",
            background=[('selected', COLORS['accent'])],
            foreground=[('selected', COLORS['text_primary'])])
        
        style.configure("TLabelframe", 
            background=COLORS['bg_card'], 
            foreground=COLORS['accent'],
            borderwidth=2,
            relief="solid")
        style.configure("TLabelframe.Label", 
            background=COLORS['bg_card'], 
            foreground=COLORS['accent'],
            font=("Segoe UI", 11, "bold"))
        
        style.configure("Treeview",
            background=COLORS['bg_card'],
            foreground=COLORS['text_primary'],
            fieldbackground=COLORS['bg_card'],

            borderwidth=0,
            rowheight=25, # Taller rows
            font=("Consolas", 10)) # Readable font
        style.configure("Treeview.Heading",
            background=COLORS['bg_input'],
            foreground=COLORS['accent'],
            font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
            background=[('selected', COLORS['accent'])])
        
        style.configure("TProgressbar",
            background=COLORS['accent_green'],
            troughcolor=COLORS['bg_input'])
        
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground=COLORS['accent'])
        style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Data.TLabel", font=("Consolas", 11), foreground=COLORS['accent_blue'])
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"))
        
        # Danger button style
        style.configure("Danger.TButton",
            background=COLORS['accent_red'],
            foreground=COLORS['text_primary'],
            font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton",
            background=[('active', '#ff0000')])

    def _setup_ui(self):
        # --- Header with Balance Display (IMPROVEMENT 8) ---
        header_frame = ttk.Frame(self.root, padding="10 10 10 0")
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text="🚀 AEHML HOLONIC TRADER", style="Header.TLabel").pack(side=tk.LEFT)
        
        # === IMPROVEMENT 8: Exchange Balance Display ===
        balance_frame = ttk.Frame(header_frame)
        balance_frame.pack(side=tk.LEFT, padx=50)
        ttk.Label(balance_frame, text="💰 Balance:", style="SubHeader.TLabel").pack(side=tk.LEFT)
        self.balance_label = ttk.Label(balance_frame, text="$0.00", style="Data.TLabel")
        self.balance_label.pack(side=tk.LEFT, padx=5)
        
        self.status_var = tk.StringVar(value="SYSTEM READY")
        self.status_label = ttk.Label(header_frame, textvariable=self.status_var, style="Status.TLabel", foreground=COLORS['accent_blue'])
        self.status_label.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, pady=5)

        # --- Main Notebook ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tabs (IMPROVEMENT 9: Added Order History Tab)
        self.tab_live = ttk.Frame(self.notebook, padding=10)
        self.tab_scout = ttk.Frame(self.notebook, padding=10) # 🔭 Scout Tab
        self.tab_agents = ttk.Frame(self.notebook, padding=10)
        self.tab_overwatch = ttk.Frame(self.notebook, padding=10) # 👁️ Overwatch Tab (NEW)
        self.tab_orders = ttk.Frame(self.notebook, padding=10)
        self.tab_news = ttk.Frame(self.notebook, padding=10) # 📰 New Tab
        self.tab_config = ttk.Frame(self.notebook, padding=10)
        self.tab_backtest = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_live, text="  📊 Live Operations  ")
        self.notebook.add(self.tab_overwatch, text="  👁️ Overwatch  ") # <--- NEW
        self.notebook.add(self.tab_scout, text="  🔭 Scout Radar  ")
        self.notebook.add(self.tab_agents, text="  🤖 Holon Status  ")
        self.notebook.add(self.tab_news, text="  📰 News & Trends  ")
        self.notebook.add(self.tab_orders, text="  📋 Order History  ")
        self.notebook.add(self.tab_config, text="  ⚙️ Configuration  ")
        self.notebook.add(self.tab_backtest, text="  📈 Backtesting  ")
        
        self._setup_live_tab()
        self._setup_overwatch_tab() # <--- NEW
        self._setup_scout_tab()
        self._setup_agents_tab()
        self._setup_news_tab() # 📰 Setup Method
        self._setup_orders_tab()
        self._setup_config_tab()
        self._setup_backtest_tab()

    # ========================== TAB 1: LIVE ==========================
    def _setup_live_tab(self):
        # Top Controls with Panic Button (IMPROVEMENT 7)
        ctl_frame = ttk.Frame(self.tab_live)
        ctl_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(ctl_frame, text="▶ START LIVE BOT", command=self.start_live_bot)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(ctl_frame, text="⏹ STOP BOT", command=self.stop_bot, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # === IMPROVEMENT 7: PANIC BUTTON ===
        self.panic_btn = ttk.Button(ctl_frame, text="🚨 PANIC CLOSE ALL", command=self.panic_close_all, style="Danger.TButton")
        self.panic_btn.pack(side=tk.RIGHT, padx=5)
        
        # === IMPROVEMENT 10: Log Export Button ===
        self.export_btn = ttk.Button(ctl_frame, text="💾 Export Log", command=self.export_log)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Main Grid
        grid_frame = ttk.Frame(self.tab_live)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # === IMPROVEMENT 11: Risk & Regime Panel (Top of Grid) ===
        regime_frame = ttk.LabelFrame(grid_frame, text="🛡️ Capital Regime & Risk Control", padding=5)
        regime_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Risk Grid
        r_grid = ttk.Frame(regime_frame)
        r_grid.pack(fill=tk.X)
        
        # 1. Regime Status
        ttk.Label(r_grid, text="CURRENT REGIME:", style="SubHeader.TLabel").grid(row=0, column=0, padx=5, sticky="w")
        self.regime_label = ttk.Label(r_grid, text="MICRO ($0 - $50)", style="Accent.TLabel", font=("Segoe UI", 12, "bold"))
        self.regime_label.grid(row=0, column=1, padx=5, sticky="w")
        
        # 2. System Health
        ttk.Label(r_grid, text="SYSTEM HEALTH:", style="SubHeader.TLabel").grid(row=0, column=2, padx=15, sticky="w")
        self.health_progress = ttk.Progressbar(r_grid, length=150, mode='determinate')
        self.health_progress.grid(row=0, column=3, padx=5, sticky="w")
        self.health_label = ttk.Label(r_grid, text="0.00", style="Data.TLabel")
        self.health_label.grid(row=0, column=4, padx=5, sticky="w")
        
        # 3. Promotion Timer
        ttk.Label(r_grid, text="PROMOTION IN:", style="SubHeader.TLabel").grid(row=0, column=5, padx=15, sticky="w")
        self.promo_label = ttk.Label(r_grid, text="--:--:--", style="Data.TLabel")
        self.promo_label.grid(row=0, column=6, padx=5, sticky="w")
        
        # Left Col: Metrics & Table
        left_col = ttk.Frame(grid_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # > Market Table (Treeview) with Color-Coded PnL (IMPROVEMENT 3)
        tbl_frame = ttk.LabelFrame(left_col, text="Market Overview", padding=5)
        tbl_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        cols = ("Symbol", "Price", "Regime", "Entropy", "Struct", "RSI", "LSTM", "XGB", "PnL", "Action")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', height=6) # Reduced height for Radar
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns
        self.tree.column("Symbol", width=60)
        self.tree.column("Price", width=70)
        self.tree.column("Regime", width=60)
        self.tree.column("Entropy", width=50)
        self.tree.column("Struct", width=60)
        self.tree.column("RSI", width=40)
        self.tree.column("LSTM", width=45)
        self.tree.column("XGB", width=45)
        self.tree.column("PnL", width=60)
        self.tree.column("Action", width=80)
        
        for col in cols:
            self.tree.heading(col, text=col)
        
        # === IMPROVEMENT 3: Color Tags for PnL ===
        self.tree.tag_configure('profit', foreground=COLORS['accent_green'])
        self.tree.tag_configure('loss', foreground=COLORS['accent_red'])
        self.tree.tag_configure('neutral', foreground=COLORS['text_secondary'])
        self.tree.tag_configure('whale', foreground='#00ffff', background='#003333') # Cyan for Whale
        
        # === IMPROVEMENT 12: Consolidation Radar (Bottom Left) ===
        radar_frame = ttk.LabelFrame(left_col, text="💀 Consolidation Radar (Kill List)", padding=5)
        radar_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        r_cols = ("Rank", "Symbol", "Score", "PnL", "Age", "EffectAge", "Status")
        self.radar_tree = ttk.Treeview(radar_frame, columns=r_cols, show='headings', height=5)
        self.radar_tree.pack(fill=tk.BOTH, expand=True)
        
        self.radar_tree.column("Rank", width=40)
        self.radar_tree.column("Symbol", width=80)
        self.radar_tree.column("Score", width=60)
        self.radar_tree.column("PnL", width=70)
        self.radar_tree.column("Age", width=60)
        self.radar_tree.column("EffectAge", width=70)
        self.radar_tree.column("Status", width=80)
        
        for col in r_cols:
            self.radar_tree.heading(col, text=col)
            
        self.radar_tree.tag_configure('keep', foreground=COLORS['accent_green'])
        self.radar_tree.tag_configure('close', foreground=COLORS['accent_red'])
        self.radar_tree.tag_configure('force', foreground=COLORS['accent'], background='#331111')
        
        # === IMPROVEMENT 4: Position Cards (Hidden/Miniaturized or kept?) -> Kept below radar?
        # Maybe move to right col or float? Let's keep it compacted.
        pos_frame = ttk.LabelFrame(left_col, text="Exchange Positions", padding=5)
        pos_frame.pack(fill=tk.X, pady=5)
        self.positions_container = ttk.Frame(pos_frame)
        self.positions_container.pack(fill=tk.X)
        self.position_labels = {}
        
        # Right Col: Logs + Equity Chart
        
        # Right Col: Logs + Equity Chart
        right_col = ttk.Frame(grid_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # === IMPROVEMENT 2: Real-Time Equity Chart ===
        equity_frame = ttk.LabelFrame(right_col, text="Live Equity Curve", padding=5)
        equity_frame.pack(fill=tk.BOTH, expand=False, pady=5) # EXPAND=FALSE (Fixed Height)
        
        self.fig_equity = Figure(figsize=(4, 2.5), dpi=90, facecolor=COLORS['bg_card'])
        self.ax_equity = self.fig_equity.add_subplot(111)
        self.ax_equity.set_facecolor(COLORS['bg_dark'])
        self.ax_equity.tick_params(colors=COLORS['text_secondary'])
        self.ax_equity.spines['bottom'].set_color(COLORS['border'])
        self.ax_equity.spines['left'].set_color(COLORS['border'])
        self.canvas_equity = FigureCanvasTkAgg(self.fig_equity, master=equity_frame)
        self.canvas_equity.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._init_equity_chart()
        
        # Scout Radar moved to self.tab_scout
        
        # Logs
        log_frame = ttk.LabelFrame(right_col, text="Activity Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5) # EXPAND=TRUE (Take all remaining space)
        
        self.log_tree = ttk.Treeview(log_frame, columns=("Time", "Level", "Agent", "Message"), show='headings', height=15) # Start Taller
        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_tree.yview)
        log_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_tree.configure(yscrollcommand=log_vsb.set)
        
        # Columns
        self.log_tree.column("Time", width=70, anchor="center")
        self.log_tree.column("Level", width=70, anchor="center")
        self.log_tree.column("Agent", width=120, anchor="w")
        self.log_tree.column("Message", width=400, anchor="w")
        
        # Headings
        self.log_tree.heading("Time", text="Time")
        self.log_tree.heading("Level", text="Level")
        self.log_tree.heading("Agent", text="Agent")
        self.log_tree.heading("Message", text="Message")
        
        # Premium Styling Tags
        self.log_tree.tag_configure('POSITIVE', foreground=COLORS['accent_green'])
        self.log_tree.tag_configure('NEGATIVE', foreground=COLORS['accent_red'])
        self.log_tree.tag_configure('WARNING', foreground=COLORS['accent_yellow'])
        self.log_tree.tag_configure('INFO', foreground=COLORS['text_primary'])
        self.log_tree.tag_configure('DIM', foreground=COLORS['text_secondary'])

        # === IMPROVEMENT 5: Notifications/Alerts Panel ===
        alert_frame = ttk.LabelFrame(right_col, text="🔔 Alerts", padding=5)
        alert_frame.pack(fill=tk.X, pady=5)
        self.alert_text = tk.Text(alert_frame, height=3, font=("Consolas", 9),
            bg=COLORS['bg_input'], fg=COLORS['accent_yellow'], wrap=tk.WORD)
        self.alert_text.pack(fill=tk.X)
        self.alert_text.insert(tk.END, "No alerts yet.\n")
        self.alert_text.config(state=tk.DISABLED)

    # ========================== TAB 2: SCOUT ==========================
    def _setup_scout_tab(self):
        # Layout: Full screen radar
        self.create_scout_panel(self.tab_scout).pack(fill=tk.BOTH, expand=True)



    def _init_equity_chart(self):
        self.ax_equity.clear()
        self.ax_equity.set_title("Equity", color=COLORS['text_primary'], fontsize=10)
        self.ax_equity.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', 
            color=COLORS['text_secondary'], transform=self.ax_equity.transAxes)
        self.canvas_equity.draw()

    # ========================== TAB 2: AGENTS ==========================
    def _setup_agents_tab(self):
        container = ttk.Frame(self.tab_agents)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Governor (Risk)
        gov_frame = ttk.LabelFrame(container, text="🛡️ Governor Holon (Risk Management)", padding=15)
        gov_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.gov_status = self._metric(gov_frame, "State:", "ACTIVE", 0)
        self.gov_alloc = self._metric(gov_frame, "Max Allocation:", "10.0%", 1)
        self.gov_lev = self._metric(gov_frame, "Leverage Cap:", "5.0x", 2)
        self.gov_trends = self._metric(gov_frame, "Active Trends:", "0", 3)
        self.gov_micro = self._metric(gov_frame, "Micro Mode:", "UNKNOWN", 4) # New Metric
        
        # Actuator (Execution)
        act_frame = ttk.LabelFrame(container, text="⚙️ Actuator Holon (Execution)", padding=15)
        act_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.act_last_ord = self._metric(act_frame, "Last Order:", "NONE", 0)
        self.act_pending = self._metric(act_frame, "Pending Orders:", "0", 1)
        
        # === IMPROVEMENT 6: GC Monitor Status ===
        gc_frame = ttk.LabelFrame(container, text="🧹 Garbage Collector", padding=15)
        gc_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        self.gc_last_run_label = self._metric(gc_frame, "Last Run:", "Never", 0)
        self.gc_cleaned_label = self._metric(gc_frame, "Items Cleaned:", "0", 1)
        
        # Brains
        brain_frame = ttk.LabelFrame(container, text="🧠 Brain Holons (Strategy & RL)", padding=15)
        brain_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.ag_regime = self._metric(brain_frame, "Market Regime:", "UNKNOWN", 0)
        self.ag_entropy = self._metric(brain_frame, "Entropy Score:", "0.0000", 1)
        self.ag_model = self._metric(brain_frame, "Strategy Matrix:", "UNIFIED (P+S)", 2) # Updated Label
        self.ag_kalman = self._metric(brain_frame, "Kalman Filter:", "ACTIVE", 3)
        self.ag_ppo_conv = self._metric(brain_frame, "PPO Conviction:", "-", 4)
        self.ag_ppo_reward = self._metric(brain_frame, "PPO Reward:", "-", 5)
        self.ag_lstm_prob = self._metric(brain_frame, "LSTM Prob:", "-", 6)
        self.ag_xgb_prob = self._metric(brain_frame, "XGB Prob:", "-", 7)
        
        # Performance
        perf_frame = ttk.LabelFrame(container, text="📈 Session Performance", padding=15)
        perf_frame.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)
        
        self.perf_winrate = self._metric(perf_frame, "Win Rate:", "-", 0)
        self.perf_pnl = self._metric(perf_frame, "Realized PnL:", "-", 1)
        self.perf_omega = self._metric(perf_frame, "Omega Ratio:", "-", 2)
        
        # Risk Metrics
        phase12_frame = ttk.LabelFrame(container, text="🛡️ Risk Management", padding=15)
        phase12_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        
        self.p12_exposure = self._metric(phase12_frame, "Total Exposure:", "$0.00", 0)
        self.p12_margin = self._metric(phase12_frame, "Used Margin:", "$0.00", 1)
        self.p12_actual_lev = self._metric(phase12_frame, "Actual Leverage:", "0.00x", 2)
        
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)

    # === IMPROVEMENT 9: ORDER HISTORY TAB ===
    # === IMPROVEMENT 10: Pinterest-Style News Feed ===
    def _setup_news_tab(self):
        # 1. Scrollable Canvas
        self.news_canvas = tk.Canvas(self.tab_news, bg='#0b0e14', highlightthickness=0)
        self.news_scrollbar = ttk.Scrollbar(self.tab_news, orient="vertical", command=self.news_canvas.yview)
        
        # 2. Container Frame inside Canvas
        self.news_container = ttk.Frame(self.news_canvas, style="Card.TFrame")
        self.news_window = self.news_canvas.create_window((0, 0), window=self.news_container, anchor="nw")
        
        self.news_canvas.configure(yscrollcommand=self.news_scrollbar.set)
        
        # Layout Scroll
        self.news_canvas.pack(side="left", fill="both", expand=True)
        self.news_scrollbar.pack(side="right", fill="y")
        
        # 3. Masonry Columns (3 Columns)
        self.news_cols = []
        for i in range(3):
            col_frame = ttk.Frame(self.news_container, style="Card.TFrame") # Transparent/Dark bg
            col_frame.grid(row=0, column=i, sticky="nw", padx=5, pady=5)
            self.news_cols.append(col_frame)
            
        # Hook resize to adjust scroll region
        self.news_container.bind("<Configure>", lambda e: self.news_canvas.configure(scrollregion=self.news_canvas.bbox("all")))
        self.tab_news.bind("<Configure>", self._on_news_resize)

        # Header Badge
        header_lbl = ttk.Label(self.news_container, text="🌍 Global Macro & Crypto Pulse", font=('Segoe UI', 14, 'bold'), foreground='#a4b1cd')
        header_lbl.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # Shift columns down to row 1
        for col in self.news_cols:
            col.grid(row=1)
            
        self.rendered_news_hashes = set()

    def _on_news_resize(self, event):
        # Responsive Width
        canvas_width = event.width
        self.news_canvas.itemconfig(self.news_window, width=canvas_width)

    def _update_news_feed(self, news_items):
        import webbrowser
        
        if not news_items: return
    
        # Simple caching to avoid redraw flicker
        # Only render if we have new items or empty
        # For efficiency, we just clear and redraw top 50 if list changed significantly
        # Or simplistic: Clear all if top item is different
        
        if not news_items: return
        top_hash = hash(news_items[0]['title'])
        if top_hash in self.rendered_news_hashes:
            return # Already rendered top item, assume list is similar for now
            
        # CLEAR existing (Optimized: destroy children of cols)
        for col in self.news_cols:
            for widget in col.winfo_children():
                widget.destroy()
        self.rendered_news_hashes.clear()
        self.rendered_news_hashes.add(top_hash)

        # Distribute items
        for i, item in enumerate(news_items):
            col_idx = i % 3
            parent = self.news_cols[col_idx]
            
            # Card Frame
            score = item.get('sentiment', 0.0)
            is_crisis = item.get('is_crisis', False)
            
            border_color = '#2e3b52' # Neutral Grey
            if is_crisis: border_color = '#d63031' # Red Crisis
            elif score > 0.2: border_color = '#00b894' # Green Bull
            elif score < -0.2: border_color = '#ff7675' # Red Bear
            
            # Use specific styles if possible, else standard Frame with border trick
            # Tkinter Frame border trick: Frame(bg=border) -> Frame (bg=inner, padding=1)
            card_border = tk.Frame(parent, bg=border_color, padx=1, pady=1)
            card_border.pack(fill='x', pady=5, padx=2)
            
            card_inner = tk.Frame(card_border, bg='#1e2742')
            card_inner.pack(fill='both')
            
            # Content
            # Source Pill
            source_lbl = tk.Label(card_inner, text=item.get('source', 'Unknown').upper(), font=('Segoe UI', 8), fg='#a4b1cd', bg='#1e2742', anchor='w')
            source_lbl.pack(fill='x', padx=10, pady=(8,0))
            
            # Title
            title_lbl = tk.Label(card_inner, text=item.get('title', 'No Title'), font=('Segoe UI', 10, 'bold'), fg='white', bg='#1e2742', wraplength=250, justify='left', anchor='w')
            title_lbl.pack(fill='x', padx=10, pady=5)
            
            # Sentiment Bar
            bar_color = border_color
            sent_bar = tk.Frame(card_inner, bg=bar_color, height=3)
            sent_bar.pack(fill='x', side='bottom')
            
            # Interaction
            def open_url(link=item.get('link')):
                if link: webbrowser.open(link)
                
            # Bind Click to everything in card
            for w in [card_border, card_inner, source_lbl, title_lbl, sent_bar]:
                w.bind("<Button-1>", lambda e, l=item.get('link'): open_url(l))
                w.bind("<Enter>", lambda e, c=card_border: c.configure(bg='white')) # Hover highlight border
                w.bind("<Leave>", lambda e, c=card_border, b=border_color: c.configure(bg=b))

    def _setup_orders_tab(self):
        frame = ttk.Frame(self.tab_orders)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Recent Order History", style="Header.TLabel").pack(anchor="w", pady=10)
        
        cols = ("Time", "Symbol", "Side", "Qty", "Price", "Status")
        self.order_tree = ttk.Treeview(frame, columns=cols, show='headings', height=20)
        self.order_tree.pack(fill=tk.BOTH, expand=True)
        
        self.order_tree.column("Time", width=150)
        self.order_tree.column("Symbol", width=100)
        self.order_tree.column("Side", width=80)
        self.order_tree.column("Qty", width=100)
        self.order_tree.column("Price", width=100)
        self.order_tree.column("Status", width=100)
        
        for col in cols:
            self.order_tree.heading(col, text=col)
        
        self.order_tree.tag_configure('buy', foreground=COLORS['accent_green'])
        self.order_tree.tag_configure('sell', foreground=COLORS['accent_red'])
        self.order_tree.tag_configure('filled', foreground=COLORS['accent_green'])
        self.order_tree.tag_configure('canceled', foreground=COLORS['accent_red'])

    # ========================== TAB OVERWATCH (NEW) ==========================
    def _setup_overwatch_tab(self):
        # 1. Header: System DEFCON Status
        status_frame = ttk.Frame(self.tab_overwatch)
        status_frame.pack(fill=tk.X, pady=10, padx=20)
        
        ttk.Label(status_frame, text="Current System State:", style="Header.TLabel").pack(side=tk.LEFT)
        self.ov_status_lbl = ttk.Label(status_frame, text="INITIALIZING...", font=("Segoe UI", 20, "bold"), foreground=COLORS['text_secondary'])
        self.ov_status_lbl.pack(side=tk.LEFT, padx=20)
        
        # 2. Main Narrative (SitRep)
        center_frame = ttk.LabelFrame(self.tab_overwatch, text="📝 Captain's Log (SitRep)", padding=10)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ov_text = scrolledtext.ScrolledText(center_frame, font=("Consolas", 11), bg=COLORS['bg_input'], fg=COLORS['text_primary'], height=15)
        self.ov_text.pack(fill=tk.BOTH, expand=True)
        self.ov_text.insert(tk.END, "Waiting for Overwatch Holon...")
        self.ov_text.config(state=tk.DISABLED)
        
        # 3. Connection Status
        footer = ttk.Frame(self.tab_overwatch)
        footer.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(footer, text="Connected to Telegram: YES (Check Console for ID)", foreground=COLORS['accent_blue']).pack(side=tk.RIGHT)

    # ========================== TAB 4: CONFIG ==========================
    def _setup_config_tab(self):
        f = ttk.Frame(self.tab_config)
        f.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)
        
        ttk.Label(f, text="System Configuration", style="Header.TLabel").pack(anchor="w", pady=10)
        
        # Symbol
        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=8)
        ttk.Label(r1, text="Trading Pair:", width=20).pack(side=tk.LEFT)
        self.symbol_cb = ttk.Combobox(r1, textvariable=self.conf_symbol, 
            values=["XRP/USDT", "ADA/USDT", "SOL/USDT", "DOGE/USDT", "BTC/USDT", "ETH/USDT", "LINK/USDT", "LTC/USDT"])
        self.symbol_cb.pack(side=tk.LEFT)
        self.symbol_cb.current(0)
        
        # Allocation Slider
        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=8)
        ttk.Label(r2, text="Max Allocation %:", width=20).pack(side=tk.LEFT)
        s = ttk.Scale(r2, from_=0.01, to=1.0, variable=self.conf_alloc, orient=tk.HORIZONTAL, length=200)
        s.pack(side=tk.LEFT)
        self.alloc_lbl = ttk.Label(r2, text="0.10")
        self.alloc_lbl.pack(side=tk.LEFT, padx=5)
        def update_alloc_lbl(val):
            self.alloc_lbl.config(text=f"{float(val):.2f}")
        s.configure(command=update_alloc_lbl)
        
        # Leverage
        r3 = ttk.Frame(f); r3.pack(fill=tk.X, pady=8)
        ttk.Label(r3, text="Leverage Cap (x):", width=20).pack(side=tk.LEFT)
        ttk.Entry(r3, textvariable=self.conf_leverage, width=10).pack(side=tk.LEFT)
        
        # Micro Mode (NEW)
        r5 = ttk.Frame(f); r5.pack(fill=tk.X, pady=8)
        ttk.Label(r5, text="Micro Account Mode:", width=20).pack(side=tk.LEFT)
        self.conf_micro_mode = tk.BooleanVar(value=True) # Default from Config
        ttk.Checkbutton(r5, text="Force Micro Protections (No Stacking)", variable=self.conf_micro_mode).pack(side=tk.LEFT)

        # Timeframe
        r4 = ttk.Frame(f); r4.pack(fill=tk.X, pady=8)
        ttk.Label(f, text="Timeframe:", width=20).pack(side=tk.LEFT)
        self.tf_cb = ttk.Combobox(r4, textvariable=self.conf_timeframe, values=["1m", "5m", "15m", "1h", "4h", "1d"])
        self.tf_cb.pack(side=tk.LEFT)
        self.tf_cb.current(3)
        
        # Save Button
        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="💾 APPLY & SAVE CONFIG", command=self.save_live_config).pack(side=tk.LEFT)
        self.cfg_status_lbl = ttk.Label(btn_frame, text="", foreground=COLORS['accent_green'])
        self.cfg_status_lbl.pack(side=tk.LEFT, padx=10)

    def save_live_config(self):
        """Sends update to bot and persists to file."""
        try:
            # 1. Build Config Dict
            new_cfg = {
                'max_allocation': self.conf_alloc.get(),
                'leverage_cap': self.conf_leverage.get(),
                'micro_mode': self.conf_micro_mode.get()
            }
            
            # 2. Send to Live Bot via Queue
            if self.is_running_live:
                self.command_queue.put({'type': 'update_config', 'data': new_cfg})
                self.cfg_status_lbl.config(text="✅ Applied to Running Bot")
            else:
                self.cfg_status_lbl.config(text="✅ Saved (Will apply on Start)")
                
            # 3. Persist to config.py (Regex Replace)
            # This is a bit hacky but safe for simple constants
            import re
            with open('config.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace GOVERNOR_MAX_MARGIN_PCT
            content = re.sub(r'GOVERNOR_MAX_MARGIN_PCT = [\d\.]+', f'GOVERNOR_MAX_MARGIN_PCT = {new_cfg["max_allocation"]}', content)
            # Replace PREDATOR_LEVERAGE
            content = re.sub(r'PREDATOR_LEVERAGE = [\d\.]+', f'PREDATOR_LEVERAGE = {new_cfg["leverage_cap"]}', content)
            # Replace MICRO_CAPITAL_MODE
            content = re.sub(r'MICRO_CAPITAL_MODE = (True|False)', f'MICRO_CAPITAL_MODE = {new_cfg["micro_mode"]}', content)
            
            with open('config.py', 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Update UI Display instantly
            self.gov_alloc.config(text=f"{new_cfg['max_allocation']*100:.1f}%")
            self.gov_lev.config(text=f"{new_cfg['leverage_cap']}x")
            
            mm_text = "ACTIVE" if new_cfg['micro_mode'] else "INACTIVE"
            mm_color = COLORS['accent_green'] if new_cfg['micro_mode'] else COLORS['text_secondary']
            try: self.gov_micro.config(text=mm_text, foreground=mm_color)
            except: pass
            
        except Exception as e:
            messagebox.showerror("Config Error", str(e))

    # ========================== TAB 5: BACKTEST ==========================
    def _setup_backtest_tab(self):
        control_frame = ttk.Frame(self.tab_backtest, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        self.bt_start_btn = ttk.Button(control_frame, text="RUN SIMULATION", command=self.start_backtest)
        self.bt_start_btn.pack(side=tk.LEFT)
        self.bt_progress = ttk.Progressbar(control_frame, mode='determinate', length=300)
        self.bt_progress.pack(side=tk.LEFT, padx=20)

        res_frame = ttk.Frame(self.tab_backtest, padding=10)
        res_frame.pack(side=tk.TOP, fill=tk.X)
        self.bt_roi = self._metric(res_frame, "ROI:", "0.00%", 0)
        self.bt_pnl = self._metric(res_frame, "PnL:", "$0.00", 1)
        
        self.chart_frame = ttk.LabelFrame(self.tab_backtest, text="Equity Curve")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._setup_chart()

    # ========================== LOGIC ==========================
    def _metric(self, parent, text, default, row):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(f, text=text, width=18, anchor="w").pack(side=tk.LEFT)
        l = ttk.Label(f, text=default, style="Data.TLabel")
        l.pack(side=tk.LEFT)
        return l

    def _setup_chart(self):
        self.fig = Figure(figsize=(5, 6), dpi=100, facecolor=COLORS['bg_card'])
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax)
        
        for ax in [self.ax, self.ax2]:
            ax.set_facecolor(COLORS['bg_dark'])
            ax.tick_params(colors=COLORS['text_secondary'])
        
        self.ax.set_title("Equity Curve", color=COLORS['text_primary'])
        self.ax.grid(True, color=COLORS['border'], alpha=0.3)
        self.ax2.set_title("Drawdown %", color=COLORS['text_primary'])
        self.ax2.grid(True, color=COLORS['border'], alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # === IMPROVEMENT 2: Update Live Equity Chart ===
    def update_equity_chart(self, equity_value):
        self.equity_history.append((datetime.now(), equity_value))
        if len(self.equity_history) > self.max_equity_points:
            self.equity_history.pop(0)
        
        if len(self.equity_history) < 2:
            return
            
        times = [h[0] for h in self.equity_history]
        values = [h[1] for h in self.equity_history]
        
        self.ax_equity.clear()
        self.ax_equity.set_facecolor(COLORS['bg_dark'])
        self.ax_equity.plot(times, values, color=COLORS['accent_green'], linewidth=2)
        self.ax_equity.fill_between(times, values, alpha=0.2, color=COLORS['accent_green'])
        self.ax_equity.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        self.ax_equity.set_title(f"Equity: ${values[-1]:.2f}", color=COLORS['text_primary'], fontsize=10)
        self.ax_equity.grid(True, color=COLORS['border'], alpha=0.3)
        
        # Format x-axis
        self.ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.fig_equity.autofmt_xdate()
        
        self.canvas_equity.draw()

    def create_scout_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="🔭 Scout Radar (Tiered Watchlist)", padding=10)
        
        # Grid of Cards style or Text List? Text List is safer for TKinter thread issues
        self.scout_text = tk.Text(frame, height=12, width=40, font=("Consolas", 10), bg=COLORS['bg_input'], fg=COLORS['text_primary'], relief="flat")
        self.scout_text.pack(fill='both', expand=True)
        
        # Legend
        legend = ttk.Label(frame, text="🚀=Rocket (Hype/Vol) | ⚓=Anchor (Stable) | 💀=Dead", font=("Segoe UI", 8), foreground=COLORS['text_secondary'])
        legend.pack(anchor='w', pady=2)
        
        return frame

    def _update_scout_radar(self):
        """Reads scout_status.json and updates the Scout Panel."""
        import json
        import os
        
        now = time.time()
        if now - self.scout_last_read < 2.0: return # Throttle 2s
        
        try:
            path = os.path.join(os.getcwd(), 'scout_status.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.scout_status = data.get('results', {})
                    self.scout_last_read = now
                    
                    # render
                    self.scout_text.delete(1.0, tk.END)
                    
                    # Sort: Rocket -> Anchor -> Dead
                    def sort_key(item):
                        k, v = item
                        if v == 'ROCKET': return 0
                        if v == 'ANCHOR': return 1
                        return 2
                        
                    sorted_items = sorted(self.scout_status.items(), key=sort_key)
                    
                    for symbol, personality in sorted_items:
                        icon = "💀"
                        color = COLORS['text_secondary']
                        
                        if personality == 'ROCKET':
                            icon = "🚀"
                            color = COLORS['accent_red']
                        elif personality == 'ANCHOR':
                            icon = "⚓"
                            color = COLORS['accent_blue']
                            
                        # Insert with color tag
                        tag_name = f"tag_{personality}"
                        self.scout_text.tag_config(tag_name, foreground=color)
                        self.scout_text.insert(tk.END, f"{icon} {symbol:<10} [{personality}]\n", tag_name)
                        
        except Exception as e:
            # print(f"Scout Read Error: {e}") 
            pass

    # === IMPROVEMENT 4: Update Position Cards ===
    def update_position_cards(self, holdings, entry_prices, current_prices):
        for widget in self.positions_container.winfo_children():
            widget.destroy()
        
        if not holdings:
            ttk.Label(self.positions_container, text="No open positions", 
                foreground=COLORS['text_secondary']).pack()
            return
        
        for sym, qty in holdings.items():
            if abs(qty) < 0.00000001:
                continue
                
            card = ttk.Frame(self.positions_container, style="TFrame")
            card.pack(fill=tk.X, pady=2)
            
            entry = entry_prices.get(sym, 0.0)
            current = current_prices.get(sym, entry)
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            pnl_color = COLORS['accent_green'] if pnl_pct >= 0 else COLORS['accent_red']
            
            ttk.Label(card, text=f"{sym}", width=12).pack(side=tk.LEFT)
            ttk.Label(card, text=f"Qty: {qty:.4f}", width=12).pack(side=tk.LEFT)
            ttk.Label(card, text=f"Entry: ${entry:.2f}", width=12).pack(side=tk.LEFT)
            ttk.Label(card, text=f"PnL: {pnl_pct:+.2f}%", foreground=pnl_color).pack(side=tk.LEFT)

    def update_chart(self, history):
        if not history: return
        dates = [h[0] for h in history]
        values = [h[1] for h in history]
        
        self.ax.clear()
        self.ax.set_facecolor(COLORS['bg_dark'])
        self.ax.plot(dates, values, color=COLORS['accent_blue'], label='Equity')
        self.ax.set_title("Equity Curve", color=COLORS['text_primary'])
        self.ax.grid(True, color=COLORS['border'], alpha=0.3)
        self.ax.legend()
        
        import pandas as pd
        s = pd.Series(values)
        cummax = s.cummax()
        drawdown = (s - cummax) / cummax
        
        self.ax2.clear()
        self.ax2.set_facecolor(COLORS['bg_dark'])
        self.ax2.fill_between(dates, drawdown, color=COLORS['accent_red'], alpha=0.3, label='Drawdown')
        self.ax2.set_title("Drawdown %", color=COLORS['text_primary'])
        self.ax2.grid(True, color=COLORS['border'], alpha=0.3)
        
        self.fig.autofmt_xdate()
        self.canvas.draw()

    # === IMPROVEMENT 5: Add Alert ===
    def add_alert(self, message, level="info"):
        self.alert_text.config(state=tk.NORMAL)
        
        # Limit to 10 alerts
        lines = self.alert_text.get("1.0", tk.END).strip().split('\n')
        if len(lines) >= 10:
            self.alert_text.delete("1.0", "2.0")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "success": "✅"}.get(level, "•")
        self.alert_text.insert(tk.END, f"{icon} [{timestamp}] {message}\n")
        self.alert_text.see(tk.END)
        self.alert_text.config(state=tk.DISABLED)

    def start_live_bot(self):
        if self.is_running_live: return
        
        cfg = {
            'symbol': self.conf_symbol.get(),
            'timeframe': self.conf_timeframe.get(),
            'max_allocation': self.conf_alloc.get(),
            'leverage_cap': self.conf_leverage.get(),
            'micro_mode': self.conf_micro_mode.get()
        }
        
        self.gui_stop_event.clear()
        self.gui_stop_event.clear()
        self.bot_thread = threading.Thread(target=run_bot, args=(self.gui_stop_event, self.gui_queue, cfg, self.command_queue))
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        self.is_running_live = True
        self.status_var.set("🟢 LIVE TRADING ACTIVE")
        self.status_label.config(foreground=COLORS['accent_green'])
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.gov_alloc.config(text=f"{cfg['max_allocation']*100:.1f}%")
        self.gov_lev.config(text=f"{cfg['leverage_cap']}x")
        
        mm_text = "ACTIVE" if cfg['micro_mode'] else "INACTIVE"
        mm_color = COLORS['accent_green'] if cfg['micro_mode'] else COLORS['text_secondary']
        self.gov_micro.config(text=mm_text, foreground=mm_color)
        
        self.add_alert("Live trading started", "success")

    # === IMPROVEMENT 13: Graceful Shutdown ===
    def stop_bot(self):
        if not self.is_running_live: return
        self.gui_stop_event.set()
        self.status_var.set("⏳ STOPPING...")
        self.status_label.config(foreground=COLORS['accent_yellow'])
        
        # Wait for thread with timeout
        def wait_and_finalize():
            if self.bot_thread:
                self.bot_thread.join(timeout=5.0)
            self.root.after(0, self._finalize_stop)
        
        threading.Thread(target=wait_and_finalize, daemon=True).start()
        
    def _finalize_stop(self):
        self.is_running_live = False
        self.status_var.set("🔴 STOPPED")
        self.status_label.config(foreground=COLORS['accent_red'])
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.add_alert("Bot stopped", "info")

    # === IMPROVEMENT 7: PANIC CLOSE ALL ===
    def panic_close_all(self):
        if not self.is_running_live:
            messagebox.showwarning("Panic", "Bot is not running.")
            return
        
        confirm = messagebox.askyesno("🚨 PANIC CLOSE ALL",
            "This will immediately close ALL open positions.\n\nAre you sure?",
            icon='warning')
        
        if confirm:
            self.command_queue.put({'type': 'panic_close'}) # Fix: Use Command Queue
            self.add_alert("PANIC CLOSE ALL triggered!", "error")

    # === IMPROVEMENT 10: Export Log ===
    def export_log(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
            initialfile=f"holonic_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("Time | Level | Agent | Message\n")
                f.write("-" * 80 + "\n")
                for item in self.log_tree.get_children():
                    vals = self.log_tree.item(item)['values']
                    f.write(f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]}\n")
            self.add_alert(f"Log exported to {filepath}", "success")

    # === IMPROVEMENT 11: Log Size Limit ===
    def log(self, msg):
        # Format: "[AgentName] Message Content..."
        ts = datetime.now().strftime("%H:%M:%S")
        
        # 1. Parse Agent Name
        agent = "SYSTEM"
        content = msg
        if msg.startswith("["):
            parts = msg.split("]", 1)
            if len(parts) > 1:
                agent = parts[0][1:] # Strip [
                content = parts[1].strip()
        
        # 2. Determine Level/Tag
        tag = 'INFO'
        level = 'INFO'
        
        upper_msg = content.upper()
        if "ERROR" in upper_msg or "FAIL" in upper_msg or "REJECT" in upper_msg or "CRITICAL" in upper_msg:
            tag = 'NEGATIVE'
            level = 'ERROR'
        elif "WARNING" in upper_msg or "RISK" in upper_msg or "DRAWDOWN" in upper_msg:
            tag = 'WARNING'
            level = 'WARN'
        elif "SUCCESS" in upper_msg or "PROFIT" in upper_msg or "BUY" in upper_msg or "FILLED" in upper_msg:
            tag = 'POSITIVE'
            level = 'OK'
        elif "DIM" in msg:
            tag = 'DIM'
        
        # 3. Insert into Treeview (Top)
        # self.log_tree.insert("", 0, values=(ts, level, agent, content), tags=(tag,))
        self.log_tree.insert("", tk.END, values=(ts, level, agent, content), tags=(tag,))
        self.log_tree.yview_moveto(1)
        
        # === IMPROVEMENT 11 & 12: PARSE LOGS FOR UI UPDATES ===
        try:
            # A. Regime Status
            # Log format: "📊 Regime: MICRO | Health: 0.00 | Peak: $49.10"
            if "Regime:" in content and "Health:" in content:
                parts = content.split('|')
                regime = parts[0].split(':')[1].strip()
                health = float(parts[1].split(':')[1].strip())
                
                # Update Labels
                self.regime_label.config(text=regime)
                
                # Color Coding
                fg = COLORS['accent_yellow'] # Default Micro
                if "SMALL" in regime: fg = COLORS['accent_blue']
                if "MEDIUM" in regime: fg = COLORS['accent_green']
                self.regime_label.config(foreground=fg)
                
                # Health Bar
                self.health_progress['value'] = health * 100
                self.health_label.config(text=f"{health:.2f}")
                
                # Promotion Timer (Mock logic for now, or parse if available)
                # If health >= 0.95, show "ELIGIBLE"
                if health >= 0.95:
                    self.promo_label.config(text="ELIGIBLE (Wait 72h)", foreground=COLORS['accent_green'])
                else:
                    self.promo_label.config(text="BUILDING...", foreground=COLORS['text_secondary'])
                    
            # B. Consolidation Radar - Start Block
            if "[ConsolidationEngine] Ranking:" in content:
                # Clear previous list
                for item in self.radar_tree.get_children():
                    self.radar_tree.delete(item)
                    
            # C. Consolidation Radar - Items
            # Log format: "1. XRP/USDT     score=0.53 (PnL:+0.0%) → KEEP"
            # Or with AgeEff: "1. SYM score=... (AgeEff:...) status"
            if content.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
                # Very simple parser: split by spaces
                # Example: "1.", "XRP/USDT", "score=0.53", "(PnL:+0.0%)", "→", "KEEP"
                tokens = content.split()
                if len(tokens) >= 5 and "score=" in tokens[2]:
                    rank = tokens[0].strip('.')
                    sym = tokens[1]
                    score = tokens[2].split('=')[1]
                    
                    # Try to find PnL and Status
                    pnl = "-"
                    status = "???"
                    age_eff = "-"
                    
                    for t in tokens:
                        if "(PnL:" in t:
                            pnl = t.replace("(PnL:", "").replace(")", "")
                        if "KEEP" in t: status = "KEEP"
                        if "CLOSE" in t: status = "CLOSE"
                        if "(AgeEff:" in t:
                             age_eff = t.replace("(AgeEff:", "").replace(")", "")
                             
                    # Determine Tag
                    tag = 'keep'
                    if status == "CLOSE": tag = 'close'
                    if "FORCED" in content: tag = 'force'
                    
                    self.radar_tree.insert("", tk.END, values=(
                        rank, sym, score, pnl, "-", age_eff, status
                    ), tags=(tag,))
                    
        except Exception as e:
            pass # Silent fail on parse errs
        
        # 4. Limit Rows (Keep last 500)
        children = self.log_tree.get_children()
        if len(children) > 500:
            self.log_tree.delete(children[0])


    # === IMPROVEMENT 12: Better Error Handling ===
    def process_queue(self):
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                try:
                    self._handle_queue_message(msg)
                except Exception as e:
                    print(f"[Dashboard] Error handling message: {e}")
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[Dashboard] Queue error: {e}")
        finally:
            self._update_scout_radar() # Check for Scout updates
            self.root.after(100, self.process_queue)
    
    def _handle_queue_message(self, msg):
        mtype = msg.get("type", "")
        
        if mtype == 'log':
            self.log(msg.get('message', ''))
            
        elif mtype == 'summary':
            data = msg.get('data', [])
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            for row in data:
                pnl_str = row.get('PnL', '-')
                struct_mode = row.get('Struct', '-')
                
                tag = 'neutral'
                if '+' in str(pnl_str):
                    tag = 'profit'
                elif '-' in str(pnl_str) and pnl_str != '-':
                    tag = 'loss'
                elif 'BREAKOUT' in struct_mode:
                    tag = 'profit' # Re-use Green for Breakout
                elif 'BREAKDOWN' in struct_mode:
                    tag = 'loss' # Re-use Red for Breakdown
                
                # Check for Whale
                action_text = row.get('Action', '')
                if 'WHALE' in str(action_text):
                    tag = 'whale'
                
                values = (
                    row.get('Symbol'),
                    row.get('Price'),
                    row.get('Regime', '?'),
                    row.get('Entropy', '0.00'),
                    row.get('Struct', '-'), # NEW
                    row.get('RSI', '-'),
                    row.get('LSTM', '0.50'),
                    row.get('XGB', '0.50'),
                    pnl_str,
                    row.get('Action')
                )
                self.tree.insert('', tk.END, values=values, tags=(tag,))
                
        elif mtype == "backtest_result":
            res = msg.get("data", {})
            self.bt_roi.config(text=f"{res.get('roi', 0):.2f}%")
            self.bt_pnl.config(text=f"${res.get('pnl', 0):.2f}")
            self.bt_progress['value'] = 100
            if 'history' in res:
                self.update_chart(res['history'])
                 
        elif mtype == 'agent_status':
            data = msg.get('data', {})
            # Update Agent Tab Labels
            self.gov_status.config(text=data.get('gov_state', 'ERR'))
            self.gov_alloc.config(text=data.get('gov_alloc', '-'))
            self.gov_lev.config(text=data.get('gov_lev', '-'))
            self.gov_trends.config(text=data.get('gov_trends', '0'))
            
            self.ag_regime.config(text=data.get('regime', '?'))
            self.ag_entropy.config(text=data.get('entropy', '0.0'))
            self.ag_model.config(text=data.get('strat_model', '-'))
            self.ag_kalman.config(text=data.get('kalman_active', '-'))
            self.ag_ppo_conv.config(text=data.get('ppo_conv', '0.50'))
            self.ag_ppo_reward.config(text=data.get('ppo_reward', '0.00'))
            self.ag_lstm_prob.config(text=data.get('lstm_prob', '0.50'))
            self.ag_xgb_prob.config(text=data.get('xgb_prob', '0.50'))
            
            self.act_last_ord.config(text=data.get('last_order', 'NONE'))
            self.act_pending.config(text=str(data.get('pending_count', '0')))
            
            # Update Micro Mode Status
            mm_status = data.get('gov_micro', 'UNKNOWN')
            mm_color = COLORS['accent_green'] if mm_status == 'ACTIVE' else COLORS['text_secondary']
            self.gov_micro.config(text=mm_status, foreground=mm_color)
            
            # Update News Feed
            self._update_news_feed(data.get('news_feed', []))
            
            self.perf_winrate.config(text=data.get('win_rate', '-'))
            self.perf_pnl.config(text=data.get('pnl', '-'))
            self.perf_omega.config(text=data.get('omega', '-'))
            
            self.p12_exposure.config(text=data.get('exposure', '$0.00'))
            self.p12_margin.config(text=data.get('margin', '$0.00'))
            self.p12_actual_lev.config(text=data.get('actual_lev', '0.00x'))
            
            # === IMPROVEMENT 8: Update Balance ===
            balance = data.get('balance')
            if balance:
                self.balance_label.config(text=f"${float(balance):.2f}")
            
            # === IMPROVEMENT 2: Update Equity Chart ===
            equity = data.get('equity')
            if equity:
                self.update_equity_chart(float(equity))
            
            # === IMPROVEMENT 4: Update Position Cards ===
            holdings = data.get('holdings')
            entry_prices = data.get('entry_prices', {})
            current_prices = data.get('current_prices', {})
            if holdings:
                self.update_position_cards(holdings, entry_prices, current_prices)
                
        elif mtype == 'order':
            order = msg.get('data', {})
            self._add_order_to_history(order)
            
        elif mtype == 'gc_status':
            data = msg.get('data', {})
            self.gc_last_run_label.config(text=data.get('last_run', 'Never'))
            self.gc_cleaned_label.config(text=str(data.get('cleaned', 0)))
            
        elif mtype == 'alert':
            self.add_alert(msg.get('message', ''), msg.get('level', 'info'))
            
        elif mtype == 'overwatch_update':
            state = msg.get('state', 'NOMINAL')
            sitrep = msg.get('sitrep', '')
            
            # Update Label Color
            color = COLORS['accent_green']
            if state == 'CRITICAL': color = COLORS['accent_red']
            elif state == 'CAUTION': color = COLORS['accent_yellow']
            elif state == 'OPTIMAL': color = COLORS['accent_blue']
            
            self.ov_status_lbl.config(text=state, foreground=color)
            
            # Update Text
            self.ov_text.config(state=tk.NORMAL)
            self.ov_text.delete(1.0, tk.END)
            self.ov_text.insert(tk.END, sitrep)
            self.ov_text.config(state=tk.DISABLED)

    # === IMPROVEMENT 9: Add Order to History ===
    def _add_order_to_history(self, order):
        if len(self.order_history) >= self.max_orders:
            oldest = self.order_tree.get_children()[0]
            self.order_tree.delete(oldest)
            self.order_history.pop(0)
        
        side = order.get('side', 'BUY')
        status = order.get('status', 'PENDING')
        tag = 'buy' if side.upper() == 'BUY' else 'sell'
        
        values = (
            order.get('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            order.get('symbol', '-'),
            side,
            f"{order.get('qty', 0):.4f}",
            f"${order.get('price', 0):.2f}",
            status
        )
        self.order_tree.insert('', 0, values=values, tags=(tag,))
        self.order_history.append(order)
            
    def start_backtest(self):
        if self.is_running_backtest: return
        self.bt_progress['value'] = 0
        self.is_running_backtest = True
        
        symbol = self.conf_symbol.get()
        
        def run_and_reset():
            run_backtest(self.gui_queue, symbol=symbol)
            self.is_running_backtest = False
        
        t = threading.Thread(target=run_and_reset)
        t.daemon = True
        t.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = HolonicDashboard(root)
    root.mainloop()
