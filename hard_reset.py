import json
import os
import sqlite3

DB_PATH = "holonic_trader.db"
HOF_PATH = "hall_of_fame.json"
INITIAL_CAPITAL = 300.0

def hard_reset():
    print("🚨 STARTING HARD RESET PROTOCOL 🚨")
    
    # 1. Clear Hall of Fame
    if os.path.exists(HOF_PATH):
        try:
            with open(HOF_PATH, 'w') as f:
                json.dump([], f)
            print(f"✅ Hall of Fame Cleared ({HOF_PATH})")
        except Exception as e:
             print(f"❌ Failed to clear Hall of Fame: {e}")
    else:
        print(f"⚠️ Hall of Fame not found at {HOF_PATH}")

    # 2. Reset Database
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # WIPE TRADES
            c.execute("DELETE FROM trades")
            print("✅ Trades Table Wiped.")
            
            # WIPE AUDIT LOG
            try:
                c.execute("DELETE FROM audit_log")
                print("✅ Audit Log Wiped.")
            except sqlite3.OperationalError:
                print("⚠️ Audit Log table not found (skipping).")
                
            # RESET PORTFOLIO
            # Check schema first? Or just delete and insert.
            c.execute("DELETE FROM portfolio")
            
            # Insert Fresh Row
            # Assuming schema: user_id (TEXT), balance_usd (REAL), held_assets (TEXT), position_metadata (TEXT), fortress_balance (REAL)
            # We don't know the exact column order, so let's try to inspect or just assume standard.
            # Actually, let's just update if it exists, or insert.
            # Safe bet: DELETE all, then INSERT default.
            
            # We need to know the columns. Let's list them.
            c.execute("PRAGMA table_info(portfolio)")
            cols = [row[1] for row in c.fetchall()]
            print(f"ℹ️ Portfolio Columns: {cols}")
            
            # Construct dictionary for dynamic insert
            data = {}
            if 'balance_usd' in cols: data['balance_usd'] = INITIAL_CAPITAL
            if 'fortress_balance' in cols: data['fortress_balance'] = 100.0 # Default Iron Bank
            if 'held_assets' in cols: data['held_assets'] = "{}"
            if 'position_metadata' in cols: data['position_metadata'] = "{}"
            if 'user_id' in cols: data['user_id'] = "default_user"
            if 'allocation_ratios' in cols: data['allocation_ratios'] = "{}"
            
            # Dynamic Insert
            col_names = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            values = list(data.values())
            
            sql = f"INSERT INTO portfolio ({col_names}) VALUES ({placeholders})"
            c.execute(sql, values)
            
            conn.commit()
            print(f"✅ Portfolio Reset to ${INITIAL_CAPITAL}")
            conn.close()
            
        except Exception as e:
            print(f"❌ Database Reset Failed: {e}")
    else:
        print(f"⚠️ Database not found at {DB_PATH}")

    print("🚀 RESET COMPLETE. PLEASE RESTART THE BOT.")

if __name__ == "__main__":
    hard_reset()
