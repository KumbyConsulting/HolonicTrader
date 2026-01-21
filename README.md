# HolonicTrader: AEHML Framework Implementation (V4.0)
 
**Autonomous Entropy-Holonic Machine Learning (AEHML) Framework**  
A production-hardened high-frequency trading system demonstrating holonic architecture, entropy-based regime detection, and "Immune System" homeostasis. This project serves as a proof-of-concept for the AEHML framework, utilizing the high-volatility cryptocurrency market as an ideal testing ground for autonomous adaptive agents.

[![Status](https://img.shields.io/badge/status-live%20V4.0-success)](https://github.com/TinzyWinzy/HolonicTrader)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 What is AEHML?

**AEHML (Autonomous Entropy-Holonic Machine Learning)** is a novel framework combining:
- **Holonic Architecture**: Self-organizing agents with dual autonomy/integration properties.
- **Entropy Analysis**: Market regime detection using Shannon entropy to switch between strategies (Scavenger vs. Predator).
- **Sovereign Strategy**: A PPO (Proximal Policy Optimization) brain orchestrates global risk by interpreting market entropy against portfolio health.
- **Immutable Ledger**: Blockchain-inspired audit trail for all decisions.

**HolonicTrader** demonstrates AEHML's viability in real-world financial markets. The crypto market's extreme volatility provides the perfect stress test for the framework's core capability: **adaptive homeostasis**.

> **Reference**: See [Academic_White_Paper_on_AEHML_Framework-1.pdf](./Academic_White_Paper_on_AEHML_Framework-1.pdf) for the theoretical foundation.

---

## 🏗️ Architecture (Hybrid Python/Rust)

The system uses a hybrid architecture where high-level logic (Holons) runs in Python, while computationally intensive tasks (clustering, veto logic) are offloaded to a high-performance **Rust Engine** (`holonic_speed`).

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
│  ObserverHolon │  │EntropyHolon │  │  MonitorHolon   │
│  Data/Structure│  │Regime Detect│  │Homeostasis (Immune)
└───────┬────────┘  └──────┬──────┘  └────────┬────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
              [EntryOracle (Monolith-V4)]
              (LSTM + XGBoost + Whale Detection)
                           │
        ┌──────────────────▼──────────────────┐
┌───────▼────────┐  ┌─────────────┐  ┌────────▼────────┐
│ GovernorHolon  │  │ExecutorHolon│  │ ActuatorHolon   │
│Risk (Rust Core)│  │Decision/PnL │  │Order Placement  │
└────────────────┘  └─────────────┘  └─────────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|:---|:---|:---|
| **Rust Engine** | **Performance** | Offloads correlation checks (`governor_check_cluster_risk`) and math to compiled Rust (`holonic_speed`). |
| **StructureBoss** | Market Physics | Identifies Support, Resistance, and Pivot points using fractal analysis. |
| **MonitorHolon** | Homeostasis | **Kill Switch** for drawdown control and "Fever" checks. |
| **EntropyHolon** | Regime Switch | Shannon entropy classifies market as "Ordered" (Trend) or "Chaotic" (Mean Reversion). |
| **EntryOracle** | Trend Prediction | **Monolith-V4**: Ensemble of LSTM, XGBoost, and Whale Detection logic. |
| **GovernorHolon** | Risk Management | Kelly Criterion, Volatility Scaling, and **Adaptive Vetoes** (Pivot/Crisis). |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Rust Toolchain** (cargo) - *Required for compiling the engine*
- **Virtual Environment** (highly recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TinzyWinzy/HolonicTrader.git
   cd HolonicTrader
   ```

2. **Setup Python Environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Compile Rust Engine**
   The project includes a Rust extension module. Ensure `cargo` is installed.
   ```bash
   cd rust_engine
   matorin build --release  # Or standard cargo build if configured with PyO3
   # Move the resulting .pyd/.so to the root directory
   mv target/release/holonic_speed.dll ../holonic_speed.pyd # Window example
   cd ..
   ```

4. **Run the Dashboard**
   ```bash
   python dashboard_gui.py
   ```

---

## 🔬 Key Innovations

### 1. Adaptive Immune System 💉
The system features a biological-inspired "Immune System" that monitors portfolio health. If "Fever" (Drawdown) is detected, it triggers:
- **Crisis Vetoes**: Bans new risk.
- **Nucleus Shutdown**: Hard sleep if thresholds are breached.
- **Trend Lock**: Enforces alignment with Global Market Bias.

### 2. Whale Shadow Strategy 🐋
Utilizes On-Balance Volume (OBV) divergence and price fractals to detect institutional accumulation ("Whale Shadows") and distribution ("Bid Walls"), allowing the bot to "ride the wave" of smart money.

### 3. Granular Correlation Guard 🔗
Instead of generic asset classes, the Governor uses a granular family map (e.g., `BITCOIN` vs `ETHEREUM`, `SOLANA` vs `MOVE_L1`) to allow diversified portfolios while preventing concentration risk in correlated assets.

---

## 📁 Project Structure

```
HolonicTrader/
├── HolonicTrader/           # Core Python Logic
├── rust_engine/             # High-performance Rust Core
├── dashboard_gui.py         # Control Panel (Tkinter)
├── run_evolution_loop.py    # Evolutionary Strategy Engine
├── veto_analytics.py        # Veto Analysis Tool
├── requirements.txt         # Python Dependencies
├── .gitignore               # Build artifacts ignore list
└── Academic_White_Paper...  # AEHML Framework Documentation
```

---

## 📊 Disclaimers & License

**License**: MIT  
**Disclaimer**: This software is an experimental implementation of the AEHML framework. Cryptocurrency trading involves significant financial risk. The authors are not responsible for financial losses. **Use at your own risk.**

---
*Built with ❤️ by the AEHML Research Team*
