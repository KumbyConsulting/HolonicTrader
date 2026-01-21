
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sandbox.playground import Playground
from sandbox.strategies.evo_strategy import EvoStrategy
from sandbox.strategies.base import Signal

def test_funding_rates():
    print("\n🧪 TEST 1: FUNDING RATES")
    print("-" * 30)
    
    # Init Sim
    sim = Playground(symbol='BTC/USDT', initial_capital=10000.0, verbose=False, leverage=1.0)
    
    # Mock Data: 12 Hours of boring flat price
    # Funding triggers every 8 hours. Time delta > 8h.
    start = pd.Timestamp('2024-01-01 00:00')
    dates = pd.date_range(start, periods=12, freq='1h')
    
    data = []
    for d in dates:
        data.append({
            'timestamp': d, 
            'open': 50000, 'high': 50000, 'low': 50000, 'close': 50000, 
            'volume': 100, 
            'rsi': 20 if d == start else 50, 
            'atr': 100,
            'bb_upper': 60000, 'bb_lower': 40000, 'bb_middle': 50000
        })
        
    df = pd.DataFrame(data)
    # df['rsi'] manually set to trigger entry at index 0 (rsi=20)
    
    sim.df = df
    
    # Strategy: Buy always if RSI < 30
    strat_gene = {'rsi_buy': 30, 'rsi_sell': 90, 'leverage_cap': 10.0}
    sim.inject_strategy(EvoStrategy(strat_gene))
    
    # Run
    # Step 0: Buy. RSI=20.
    # Step 1-7: Hold.
    # Step 8: Funding Check. (8 hours passed).
    
    sim.run()
    
    # ANALYSIS
    # We bought at $50000. Size ~2% Risk? Or 100%?
    # EvoStrategy defaults to position sizing based on risk/stop.
    # We didn't set Stop Loss in gene? Default is 0?
    # Actually Playground 2% risk rule checks signal.stop_loss.
    # If EvoStrat sets stop loss?
    # Let's force full size via high risk tolerance for test?
    # Actually, easier to check IF funding happened.
    
    is_funding_paid = False
    init_cap = 10000.0
    final_cap_approx = sim.capital
    
    # Did we lose money despite flat price?
    loss = init_cap - final_cap_approx
    
    # We paid fees on entry.
    # Entry Trade:
    if len(sim.trades) > 0:
        entry = sim.trades[0]
        fee = 50000 * entry['qty'] * sim.fee_rate
        print(f"  Entry Fee Paid: ${fee:.2f}")
        
    # Funding?
    # Notional = Qty * 50000
    # Funding = Notional * 0.0001
    
    print(f"  Final Capital: ${sim.capital:.2f}")
    
    # Check if Capital < (Init - Fee).
    # If yes, Funding was paid.
    
    expected_non_funding = init_cap - fee
    if sim.capital < expected_non_funding - 0.01: # Epsilon
        print(f"  ✅ FUNDING DEDUCTED. Diff: ${expected_non_funding - sim.capital:.4f}")
    else:
        print(f"  ❌ NO FUNDING DETECTED.")


class TestStrategy(EvoStrategy):
    def on_candle(self, slice_df, indicators, portfolio_state, secondary_slice_df=None):
        if portfolio_state['inventory'] == 0:
            if indicators['rsi'] <= 30:
                 # Force FULL SIZE for physics stress test
                 return Signal('BUY', size=1.0, reason="Test Entry", stop_loss=None)
        return Signal('HOLD')

def test_liquidation():
    print("\n🧪 TEST 2: LIQUIDATION PHYSICS")
    print("-" * 30)
    
    # Init Sim: HIGH LEVERAGE (50x). Low Capital ($1000).
    sim = Playground(symbol='BTC/USDT', initial_capital=1000.0, verbose=True, leverage=50.0)
    
    dates = pd.date_range('2024-01-01 00:00', periods=5, freq='1h')
    data = [
        {'timestamp': dates[0], 'close': 50000, 'rsi': 20, 'atr': 100}, # Buy
        {'timestamp': dates[1], 'close': 50000, 'rsi': 50, 'atr': 100}, # Hold
        {'timestamp': dates[2], 'close': 49250, 'rsi': 50, 'atr': 100}, # -1.5% Drop (Trigger Liq)
        {'timestamp': dates[3], 'close': 49000, 'rsi': 50, 'atr': 100}, # -2.0%
        {'timestamp': dates[4], 'close': 55000, 'rsi': 50, 'atr': 100}, # Recovery (Too late)
    ]
    
    # Hydrate required columns
    for d in data:
        d['open'] = d['close']
        d['high'] = d['close']
        d['low'] = d['close']
        d['volume'] = 100
        d['bb_upper'] = d['close']*1.1
        d['bb_lower'] = d['close']*0.9
        d['bb_middle'] = d['close']
        
    sim.df = pd.DataFrame(data)
    
    # Force Gene to buy full size using TestStrategy
    strat_gene = {'rsi_buy': 30, 'rsi_sell': 90, 'leverage_cap': 50.0} 
    sim.inject_strategy(TestStrategy(strat_gene))
    
    sim.run()
    
    print("-" * 20)
    if sim.capital == 0.0:
        print("  ✅ LIQUIDATION CONFIRMED (Capital = 0.0)")
        if sim.trades:
            liq_trade = sim.trades[-1]
            print(f"  Last Trade: {liq_trade['type']} @ {liq_trade['price']}")
    else:
        print(f"  ❌ FAILED TO LIQUIDATE. Capital: ${sim.capital:.2f}")

if __name__ == "__main__":
    test_funding_rates()
    test_liquidation()
