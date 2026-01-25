import time
import os
import re
from datetime import datetime

LOG_FILE = "live_trading_session_20260121_231115.log" # Update dynamically or find latest
SEARCH_DIR = "."

def find_latest_log():
    files = [f for f in os.listdir(SEARCH_DIR) if f.startswith("live_trading_session") and f.endswith(".log")]
    if not files: return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def monitor():
    latest_log = find_latest_log()
    if not latest_log:
        print("No log file found.")
        return

    print(f"Watching {latest_log} for Governor Activity...")
    print("="*60)
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        # Seek to end
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.1)
                continue
                
            if "[GovernorAgent]" in line:
                # Highlight Vetoes
                if "Stack Too Close" in line:
                    print(f"\033[93m{line.strip()}\033[0m") # Yellow
                elif "REJECT" in line or "VETO" in line or "Block" in line:
                    print(f"\033[91m{line.strip()}\033[0m") # Red
                elif "APPROVED" in line or "OPENED" in line:
                    print(f"\033[92m{line.strip()}\033[0m") # Green
                else:
                    print(line.strip())

if __name__ == "__main__":
    monitor()
