from sandbox.playground import Playground
from sandbox.strategies.example import RSIScalper

if __name__ == "__main__":
    print("🏝️ WELCOME TO THE HOLONIC SANDBOX 🏝️")
    
    # 1. Initialize Engine
    # We use ETH for volatility testing
    sim = Playground(symbol='ETH/USDT', initial_capital=10000.0)
    
    # 2. Load Data
    # Fetches 2000 hours of history
    sim.load_data(limit=2000)
    
    # 3. Inject Strategy
    # Import Trend Follow
    from sandbox.strategies.trend_follow import TrendFollowStrategy
    
    strat = TrendFollowStrategy()
    sim.inject_strategy(strat)
    
    # 4. Configure Reality
    sim.fee_rate = 0.0006 # 0.06% VIP Tier
    sim.slippage = 0.0002 # Tight spreads on ETH
    
    # 5. Run
    sim.run()
