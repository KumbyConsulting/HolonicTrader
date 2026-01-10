import sys
import os
import time
from unittest.mock import MagicMock

# Allow import from local
sys.path.append(os.getcwd())

from HolonicTrader.agent_actuator import ActuatorHolon

def test_infinite_fill_bug():
    print("--- Testing Actuator Fill Logic ---")
    
    # 1. Setup Actuator
    actuator = ActuatorHolon()
    actuator.exchange = MagicMock()
    
    # 2. Add a Pending Order
    order_id = "test_order_1"
    actuator.pending_orders.append({
        'id': order_id,
        'symbol': 'BTC/USDT',
        'status': 'OPEN',
        'timestamp': time.time()
    })
    
    # 3. Simulate Exchange returning CLOSED (Filled) order
    mock_order_data = {
        'id': order_id,
        'status': 'closed',
        'filled': 1.0,
        'cost': 50000.0,
        'average': 50000.0,
        'price': 50000.0
    }
    
    # Mock return for fetch_open_orders (Empty)
    actuator.exchange.fetch_open_orders.return_value = []
    # Mock return for fetch_closed_orders (Contains our order)
    actuator.exchange.fetch_closed_orders.return_value = [mock_order_data]
    
    # 4. Run check_fills
    print("Running check_fills()...")
    filled = actuator.check_fills()
    
    # 5. Verify Results
    print(f"Filled Orders Returned: {len(filled)}")
    print(f"Pending Orders Remaining: {len(actuator.pending_orders)}")
    
    if len(filled) == 1 and len(actuator.pending_orders) == 0:
        print("✅ SUCCESS: Order processed and removed from pending.")
    elif len(actuator.pending_orders) > 0:
        print("❌ FAIL: Order remains in pending list (Infinite Loop detected).")
    else:
        print("❌ FAIL: Unexpected state.")

if __name__ == "__main__":
    test_infinite_fill_bug()
