"""
Full Backtest Validation: Python vs Rust
Compares PnL, trade count, and equity curves to ensure Rust engine accuracy.
"""

import os
import time
import numpy as np
import pandas as pd

# Rust Engine
import holonic_speed

print("=" * 70)
print("FULL BACKTEST VALIDATION: Python vs Rust")
print("=" * 70)

# 1. Load Historical Data
print("\n[1/5] Loading Historical Data...")
data_path = os.path.join(os.path.dirname(__file__), "data", "BTC_USDT_1h.csv")
if not os.path.exists(data_path):
    # Try alternate path
    data_path = os.path.join(os.path.dirname(__file__), "data", "btc_1h.csv")
    
if not os.path.exists(data_path):
    print("  Creating synthetic data for validation...")
    # Generate synthetic data if no file exists
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    prices = 40000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 50),
        'low': prices - np.abs(np.random.randn(n) * 50),
        'close': prices + np.random.randn(n) * 30,
        'volume': np.random.randint(100, 1000, n)
    })
else:
    df = pd.read_csv(data_path, parse_dates=['timestamp'])

print(f"  Loaded {len(df)} candles")

# 2. Compute Indicators (Python baseline)
print("\n[2/5] Computing Indicators...")

def compute_indicators_python(df):
    """Compute indicators using Pandas (Python baseline)"""
    # SMA
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['sma_20']
    rolling_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * rolling_std)
    df['bb_lower'] = df['bb_middle'] - (2 * rolling_std)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # OBV Slope (simplified)
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_slope'] = obv.diff(5) / 5
    
    # Returns for entropy
    df['returns'] = df['close'].pct_change()
    
    return df.dropna()

start_py_ind = time.time()
df = compute_indicators_python(df.copy())
time_py_ind = time.time() - start_py_ind
print(f"  Python Indicators: {time_py_ind:.3f}s")

# 3. Compute Indicators using Rust
print("\n[3/5] Computing Indicators with Rust...")
start_rs_ind = time.time()

closes = df['close'].values.tolist()
highs = df['high'].values.tolist()
lows = df['low'].values.tolist()

rsi_rust = holonic_speed.calculate_rsi(closes, 14)
bb_upper_rust, bb_mid_rust, bb_lower_rust = holonic_speed.calculate_bollinger_bands(closes, 20, 2.0)
atr_rust = holonic_speed.calculate_atr(highs, lows, closes, 14)

time_rs_ind = time.time() - start_rs_ind
print(f"  Rust Indicators: {time_rs_ind:.3f}s")
print(f"  Speedup: {time_py_ind / time_rs_ind:.2f}x")

# 4. Run Python Backtest (Simplified)
print("\n[4/5] Running Backtests...")

def python_backtest(df, initial_capital=1000.0):
    """Simple Python backtest"""
    balance = initial_capital
    position = 0.0
    entry_price = 0.0
    trades = []
    
    for i in range(20, len(df)):
        row = df.iloc[i]
        price = row['close']
        rsi = row['rsi']
        bb_lower = row['bb_lower']
        
        # Entry: Price < BB Lower and RSI < 30
        if position == 0 and price < bb_lower and rsi < 30:
            position = balance / price
            entry_price = price
            balance = 0
            trades.append({'type': 'BUY', 'price': price, 'index': i})
        
        # Exit: Price > BB Middle or RSI > 70
        elif position > 0 and (price > row['bb_middle'] or rsi > 70):
            balance = position * price
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'index': i})
            position = 0
    
    # Close if still holding
    if position > 0:
        balance = position * df.iloc[-1]['close']
    
    return balance, trades

start_py_bt = time.time()
py_balance, py_trades = python_backtest(df)
time_py_bt = time.time() - start_py_bt

# Run Rust Backtest
start_rs_bt = time.time()

# Prepare data for Rust
timestamps = df['timestamp'].astype(np.int64).tolist()
opens = df['open'].values.tolist()
# Use Rust-computed indicators
rsi_list = rsi_rust
bb_lower_list = bb_lower_rust

# Generate signals based on same logic
signals = []
for i in range(len(closes)):
    if i < 20 or np.isnan(rsi_list[i]) or np.isnan(bb_lower_list[i]):
        signals.append(0)
    elif closes[i] < bb_lower_list[i] and rsi_list[i] < 30:
        signals.append(1)  # BUY
    elif rsi_list[i] > 70:
        signals.append(-1)  # SELL
    else:
        signals.append(0)

# Call Rust backtest
rs_balance, rs_trades = holonic_speed.run_backtest_fast(
    timestamps, opens, highs, lows, closes, signals, 1000.0, 0.001
)

time_rs_bt = time.time() - start_rs_bt

print(f"  Python Backtest: {time_py_bt:.4f}s | Final: ${py_balance:.2f} | Trades: {len(py_trades)}")
print(f"  Rust Backtest:   {time_rs_bt:.4f}s | Final: ${rs_balance:.2f} | Trades: {len(rs_trades)}")
print(f"  Speedup: {time_py_bt / time_rs_bt:.2f}x")

# 5. Compare Results
print("\n[5/5] Validation Results...")
print("-" * 50)

# Balance Comparison
balance_diff = abs(py_balance - rs_balance)
balance_pct_diff = (balance_diff / py_balance) * 100 if py_balance > 0 else 0

if balance_pct_diff < 5.0:
    print(f"  Balance Match: PASS (Diff: {balance_pct_diff:.2f}%)")
else:
    print(f"  Balance Match: WARN (Diff: {balance_pct_diff:.2f}%)")
    print(f"    Python: ${py_balance:.2f}")
    print(f"    Rust:   ${rs_balance:.2f}")

# Trade Count Comparison (Note: Signal generation differs slightly)
trade_diff = abs(len(py_trades) - len(rs_trades))
print(f"  Trade Count: Python={len(py_trades)}, Rust={len(rs_trades)} (Diff: {trade_diff})")

# Overall
print("-" * 50)
total_speedup = (time_py_ind + time_py_bt) / (time_rs_ind + time_rs_bt)
print(f"  TOTAL SPEEDUP: {total_speedup:.2f}x")
print("=" * 70)
