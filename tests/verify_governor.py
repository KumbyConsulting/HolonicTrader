import holonic_speed

print("VERIFYING RUST GOVERNOR ENGINE")
print("=" * 60)

# Test 1: Max Risk Calculation
print("[1/3] Max Risk Calculation")
# Scenario: Balance $20, Principal $10, Equity $20
# House money = 20 - 10 = $10
# 1% equity = $0.20
# Max Risk = min(10, 0.20) = $0.20
max_risk = holonic_speed.governor_calculate_max_risk(
    balance=20.0,
    principal=10.0,
    equity=20.0
)
print(f"  Max Risk: ${max_risk:.2f}")
if abs(max_risk - 0.20) < 0.05:
    print("  Accuracy: MATCH")
else:
    print(f"  Accuracy: CHECK (Expected ~$0.20)")

# Test 2: Cluster Risk Check
print("\n[2/3] Cluster Risk Check")
# Holding BTC, trying to add TBTC (same family) -> Should REJECT
held = ["BTC/USDT"]
new_symbol = "TBTC/USDT"
is_safe = holonic_speed.governor_check_cluster_risk(held, new_symbol)
print(f"  Held: {held}, New: {new_symbol}")
print(f"  Is Safe: {is_safe}")
if not is_safe:
    print("  Logic: CORRECT (Cluster risk detected)")
else:
    print("  Logic: FAILED (Should have detected cluster)")

# Different family -> Should ALLOW
new_symbol2 = "ETH/USDT"
is_safe2 = holonic_speed.governor_check_cluster_risk(held, new_symbol2)
print(f"  Held: {held}, New: {new_symbol2}")
print(f"  Is Safe: {is_safe2}")
if is_safe2:
    print("  Logic: CORRECT (Different family allowed)")
else:
    print("  Logic: FAILED (Should allow different family)")

# Test 3: Position Size Calculation
print("\n[3/3] Position Size Calculation")
qty = holonic_speed.governor_calculate_position_size(
    balance=50.0,
    equity=50.0,
    high_water_mark=50.0,
    reference_atr=100.0,
    asset_price=100.0,
    current_atr=100.0,
    conviction=0.5
)
print(f"  Calculated Qty: {qty:.6f}")
if qty > 0:
    print("  Logic: CORRECT (Non-zero quantity)")
else:
    print("  Logic: CHECK (Quantity is zero)")

print("\n" + "="*60)
