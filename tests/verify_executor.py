import holonic_speed

print("VERIFYING RUST EXECUTOR ENGINE")
print("=" * 60)

# Test 1: ORDERED regime -> EXECUTE
print("[1/3] ORDERED Regime")
action, size = holonic_speed.executor_decide_trade(
    signal_size=1.0,
    entropy_score=1.5,
    regime="ORDERED"
)
print(f"  Action: {action}, Size: {size:.2f}")
if action == "EXECUTE" and size > 0:
    print("  Logic: CORRECT")
else:
    print("  Logic: FAILED")

# Test 2: CHAOTIC regime -> HALT
print("\n[2/3] CHAOTIC Regime")
action, size = holonic_speed.executor_decide_trade(
    signal_size=1.0,
    entropy_score=2.5,
    regime="CHAOTIC"
)
print(f"  Action: {action}, Size: {size:.2f}")
if action == "HALT" and size == 0:
    print("  Logic: CORRECT")
else:
    print("  Logic: FAILED")

# Test 3: TRANSITION regime -> REDUCE
print("\n[3/3] TRANSITION Regime")
action, size = holonic_speed.executor_decide_trade(
    signal_size=1.0,
    entropy_score=1.5,
    regime="TRANSITION"
)
print(f"  Action: {action}, Size: {size:.2f}")
if action == "REDUCE" and 0 < size < 1.0:
    print("  Logic: CORRECT")
else:
    print("  Logic: FAILED")

print("\n" + "="*60)
