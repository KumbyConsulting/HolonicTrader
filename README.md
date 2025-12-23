# HolonicTrader: AEHML Framework Proof of Concept

**Autonomous Entropy-Holonic Machine Learning (AEHML) Framework**  
A production-ready cryptocurrency trading system demonstrating holonic architecture, entropy-based regime detection, and adaptive autonomous agents.

[![Status](https://img.shields.io/badge/status-live-success)](https://github.com/TinzyWinzy/HolonicTrader)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 What is AEHML?

**AEHML (Autonomous Entropy-Holonic Machine Learning)** is a novel framework combining:
- **Holonic Architecture**: Self-organizing agents with dual autonomy/integration properties
- **Entropy Analysis**: Market regime detection using Shannon entropy
- **Adaptive Learning**: Deep Q-Learning and LSTM for pattern recognition
- **Immutable Ledger**: Blockchain-inspired audit trail for all decisions

**HolonicTrader** is the first production implementation, proving AEHML's viability in real-world financial markets.

---

## 🏗️ Architecture

### Holonic Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    TraderHolon (Nexus)                  │
│                  Supra-Holon Orchestrator               │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  ObserverHolon │  │EntropyHolon │  │ StrategyHolon   │
│  Data Fetcher  │  │Regime Detect│  │Signal Generator │
└────────────────┘  └─────────────┘  └─────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ GovernorHolon  │  │ExecutorHolon│  │ ActuatorHolon   │
│Risk Management │  │Trade Decision│  │Order Execution  │
└────────────────┘  └─────────────┘  └─────────────────┘
                            │
                    ┌───────▼────────┐
                    │  DQN Holon     │
                    │ Deep Q-Learning│
                    └────────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|:---|:---|:---|
| **ObserverHolon** | Market data acquisition | Hybrid local/live data, multi-asset support |
| **EntropyHolon** | Regime classification | Shannon entropy, calibrated thresholds (0.67/0.80) |
| **StrategyHolon** | Signal generation | RSI, Bollinger Bands, OBV, LSTM confirmation |
| **GovernorHolon** | Risk management | Position sizing, leverage control, pre-trade validation |
| **ExecutorHolon** | Trade execution | Sigmoid-based autonomy, immutable ledger |
| **ActuatorHolon** | Order placement | Limit orders, maker-only execution |
| **DQN Holon** | Reinforcement learning | Risk-adjusted rewards, experience replay |

---

## 🧠 Entropy-Based Decision Making

### Market Regime Detection

```python
Shannon Entropy = -Σ(p(x) * log(p(x)))

Thresholds (Calibrated on 9,956 live samples):
├─ ORDERED:     entropy < 0.67  (53% of market conditions)
├─ TRANSITION:  0.67 ≤ entropy ≤ 0.80  (32%)
└─ CHAOTIC:     entropy > 0.80  (15%)
```

### Adaptive Autonomy

```python
Autonomy = 1 / (1 + e^(5 * (entropy - 0.75)))

Decision Mapping:
├─ Autonomy > 0.6  → EXECUTE (full trade)
├─ 0.4 ≤ Autonomy ≤ 0.6 → REDUCE (50% size)
└─ Autonomy < 0.4  → HALT (reject trade)
```

**Result**: System dynamically adjusts risk based on market uncertainty.

---

## 📊 Performance Highlights

### Live Trading Results

| Metric | Value | Notes |
|:---|---:|:---|
| **Total Trades** | 1,311 | Multi-asset (ADA, BTC, DOGE, SUI, XRP) |
| **Win Rate** | 71.2% | Validated on live data |
| **PnL Coverage** | 100% | All trades tracked (Phase 11) |
| **Regime Detection** | 53/32/15 | ORDERED/TRANSITION/CHAOTIC |
| **Risk Management** | 47% | HALT/REDUCE triggers active |

### System Health

- **Database**: 9,956 ledger entries, verified integrity
- **Models**: LSTM (trend prediction), DQN (policy learning)
- **Uptime**: Continuous operation with GUI monitoring
- **Entropy Calibration**: Live-tuned thresholds (Phase 10)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
Virtual environment
KuCoin/Binance API keys (for live trading)
```

### Installation

```bash
# Clone repository
git clone https://github.com/TinzyWinzy/HolonicTrader.git
cd HolonicTrader

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.py`:
```python
# Exchange API (for live trading)
EXCHANGE_ID = 'kucoin'
API_KEY = 'your_api_key'
API_SECRET = 'your_api_secret'

# Trading parameters
INITIAL_CAPITAL = 10.0
ALLOWED_ASSETS = ['ADA/USDT', 'BTC/USDT', 'DOGE/USDT', 'SUI/USDT', 'XRP/USDT']
```

### Run Backtest

```bash
python run_backtest.py
```

### Run Live Trading (Paper Mode)

```bash
# With GUI
python dashboard_gui.py

# Headless
python main_live_phase4.py
```

---

## 🔬 Key Innovations

### 1. Holonic Architecture
- **Self-Organization**: Agents autonomously coordinate without central control
- **Dual Nature**: Each agent balances autonomy (independence) and integration (cooperation)
- **Emergent Behavior**: System-level intelligence from agent interactions

### 2. Entropy-Driven Adaptation
- **Real-Time Calibration**: Thresholds adjusted to live market conditions
- **Dynamic Risk**: Autonomy scales with market uncertainty
- **Regime Awareness**: Different strategies for ORDERED vs CHAOTIC markets

### 3. Immutable Audit Trail
- **Blockchain-Inspired**: SHA-256 chained ledger blocks
- **Full Transparency**: Every decision logged and verifiable
- **Tamper-Evident**: Chain integrity validation

### 4. Multi-Strategy Execution
- **SCAVENGER Mode**: Mean reversion (4h max hold, +2% target)
- **PREDATOR Mode**: Momentum following (8h max hold, +3% target)
- **Metabolic Switching**: Capital-based strategy selection

---

## 📁 Project Structure

```
HolonicTrader/
├── HolonicTrader/           # Core agent implementations
│   ├── holon_core.py        # Base Holon class
│   ├── agent_trader.py      # Supra-Holon orchestrator
│   ├── agent_observer.py    # Data fetching
│   ├── agent_entropy.py     # Regime detection
│   ├── agent_strategy.py    # Signal generation
│   ├── agent_governor.py    # Risk management
│   ├── agent_executor.py    # Trade execution
│   ├── agent_actuator.py    # Order placement
│   └── agent_dqn.py         # Deep Q-Learning
├── market_data/             # Historical price data
├── config.py                # System configuration
├── database_manager.py      # SQLite persistence
├── main_live_phase4.py      # Live trading entry point
├── dashboard_gui.py         # GUI control panel
├── run_backtest.py          # Backtesting framework
└── holonic_trader.db        # State database
```

---

## 🧪 Validation & Testing

### System Health Check
```bash
python system_health_check.py
```

### Performance Analysis
```bash
python performance_analysis.py
```

### Ledger Validation
```bash
python validate_ledger_logic.py
```

### Entropy Calibration
```bash
python analyze_live_entropy.py
```

---

## 📈 Recent Improvements (Phase 10-11)

### Phase 10: Entropy Recalibration
- ✅ Analyzed 9,956 live trading samples
- ✅ Adjusted thresholds from 1.96/2.10 → 0.67/0.80
- ✅ Achieved 53/32/15 regime distribution

### Phase 11: PnL & Exit Strategy
- ✅ 100% PnL tracking (was 4%)
- ✅ Time-based exits (4h/8h limits)
- ✅ Tighter profit targets (+2%/+3%)
- ✅ Fixed sigmoid threshold (2.0 → 0.75)
- ✅ HALT/REDUCE triggers now active (47% combined)

---

## 🎓 Academic Foundation

Based on the **AEHML Framework White Paper**, this implementation demonstrates:

1. **Holonic Principles**: Arthur Koestler's concept of holons applied to ML agents
2. **Entropy Theory**: Claude Shannon's information theory for market analysis
3. **Adaptive Systems**: Self-organizing, self-regulating autonomous agents
4. **Blockchain Integration**: Immutable decision logging for transparency

**Reference**: See `archive/Academic_White_Paper_on_AEHML_Framework-1.pdf`

---

## 🛠️ Technology Stack

- **Python 3.10+**: Core language
- **TensorFlow/Keras**: LSTM neural networks
- **NumPy**: Deep Q-Learning implementation
- **CCXT**: Exchange connectivity
- **SQLite**: State persistence
- **Tkinter**: GUI dashboard
- **Pandas**: Data manipulation

---

## 📊 Monitoring & Observability

### Real-Time Dashboard
- Live portfolio value
- Regime detection status
- Agent health metrics
- Trade execution log
- Performance analytics

### Database Schema
```sql
ledger          -- Immutable decision log (9,956 entries)
trades          -- Execution history with PnL
portfolio       -- Current holdings and balance
rl_experiences  -- DQN training data
```

---

## 🔮 Future Enhancements

- [ ] Multi-exchange support (Binance, Coinbase)
- [ ] Advanced RL algorithms (PPO, A3C)
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization (Markowitz)
- [ ] Real-time risk metrics (VaR, Sharpe)
- [ ] Web-based dashboard
- [ ] Backtesting framework enhancements

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New holonic agents (e.g., SentimentHolon)
- Alternative entropy measures (Rényi, Tsallis)
- Strategy improvements
- Performance optimizations
- Documentation enhancements

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **AEHML Framework**: Original theoretical foundation
- **Arthur Koestler**: Holonic systems theory
- **Claude Shannon**: Information theory and entropy
- **Community**: Open-source contributors and testers

---

## 📞 Contact

**Project**: [HolonicTrader](https://github.com/TinzyWinzy/HolonicTrader)  
**Issues**: [GitHub Issues](https://github.com/TinzyWinzy/HolonicTrader/issues)

---

**⚠️ Disclaimer**: This is a proof-of-concept implementation for research and educational purposes. Cryptocurrency trading involves substantial risk. Always test thoroughly in paper trading mode before live deployment.
