
import sqlite3
import json

def inspect_portfolio():
    try:
        conn = sqlite3.connect('holonic_trader.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM portfolio WHERE id = 1")
        row = cursor.fetchone()
        
        if row:
            print("--- Portfolio Table Contents ---")
            for key in row.keys():
                val = row[key]
                if key in ['held_assets', 'position_metadata']:
                    try:
                        val = json.loads(val)
                    except:
                        pass
                print(f"{key}: {val}")
        else:
            print("Portfolio table is empty or id=1 not found.")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_portfolio()
