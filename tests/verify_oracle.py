import holonic_speed
import numpy as np

print("VERIFYING RUST ORACLE ENGINE")
print("=" * 60)

# Create test data
closes = list(np.linspace(100, 95, 30))  # Downtrend
rsi = list(np.linspace(50, 25, 30))       # RSI falling to oversold
bb_lower = list(np.linspace(98, 96, 30))  # Lower BB
bb_upper = list(np.linspace(102, 104, 30))
obv_slope = 0.5  # Positive OBV = accumulation

# Test 1: Scavenger Entry (Mean Reversion)
print("[1/3] Scavenger Entry Signal")
direction, confidence, reason = holonic_speed.oracle_analyze_for_entry(
    closes, rsi, bb_lower, bb_upper, obv_slope, "SCAVENGER"
)
print(f"  Direction: {direction}, Confidence: {confidence:.2f}")
print(f"  Reason: {reason}")
if direction == "BUY":
    print("  Logic: CORRECT (Oversold + Accumulation)")
else:
    print("  Logic: CHECK (Expected BUY signal)")

# Test 2: Predator Entry (Trend Following)
print("\n[2/3] Predator Entry Signal")
closes_up = list(np.linspace(100, 110, 30))  # Uptrend
rsi_strong = list(np.linspace(40, 60, 30))
bb_upper_break = list(np.linspace(102, 105, 30))  # Price above upper
bb_lower_up = list(np.linspace(98, 100, 30))

direction, confidence, reason = holonic_speed.oracle_analyze_for_entry(
    closes_up, rsi_strong, bb_lower_up, bb_upper_break, 0.5, "PREDATOR"
)
print(f"  Direction: {direction}, Confidence: {confidence:.2f}")
print(f"  Reason: {reason}")

# Test 3: Market Bias Calculation
print("\n[3/3] Market Bias Calculation")
btc_returns = [0.01, -0.005, 0.02, 0.015, -0.01] * 4  # Mixed returns
sentiment = 0.3
bias = holonic_speed.oracle_calculate_market_bias(btc_returns, sentiment)
print(f"  Bias: {bias:.3f}")
if -1.0 <= bias <= 1.0:
    print("  Logic: CORRECT (Bias within range)")
else:
    print("  Logic: FAILED")

print("\n" + "="*60)
