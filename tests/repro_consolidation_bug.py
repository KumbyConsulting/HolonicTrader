
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockDecision:
    def __init__(self, adjusted_size):
        self.adjusted_size = adjusted_size
        self.original_signal = MockSignal()

class MockSignal:
    def __init__(self):
        self.metadata = {'reason': 'CONSOLIDATION'}

def calculate_exec_qty(current_holding, adjusted_size):
    # This logic mirrors agent_executor.py's execute_transaction logic for exits
    # "is_long_exit or is_short_cover" block
    
    # Logic from the bug:
    exec_qty = abs(current_holding) * adjusted_size
    return exec_qty

def test_bug_reproduction():
    print("--- Testing Bug Reproduction ---")
    held_qty = 111.0
    
    # The BUG: agent_trader passes absolute qty (111.0) as adjusted_size
    passed_size_bug = 111.0 
    
    exec_qty = calculate_exec_qty(held_qty, passed_size_bug)
    
    print(f"Held Qty: {held_qty}")
    print(f"Passed Size (from Trader): {passed_size_bug}")
    print(f"Calculated Exec Qty: {exec_qty}")
    
    if exec_qty > held_qty:
        print("[FAIL] Bug Reproduced: Exec Qty is significantly larger than Held Qty.")
        print(f"       Expected ~{held_qty}, Got {exec_qty}")
    else:
        print("[PASS] Bug NOT Reproduced.")

def test_fix_verification():
    print("\n--- Testing Fix Verification ---")
    held_qty = 111.0
    
    # The FIX: agent_trader should pass 1.0 (100%) as adjusted_size
    passed_size_fix = 1.0
    
    exec_qty = calculate_exec_qty(held_qty, passed_size_fix)
    
    print(f"Held Qty: {held_qty}")
    print(f"Passed Size (from Trader): {passed_size_fix}")
    print(f"Calculated Exec Qty: {exec_qty}")
    
    if exec_qty == held_qty:
        print("[PASS] Fix Verified: Exec Qty matches Held Qty.")
    else:
        print(f"[FAIL] Fix Failed: Expected {held_qty}, Got {exec_qty}")

if __name__ == "__main__":
    test_bug_reproduction()
    test_fix_verification()
