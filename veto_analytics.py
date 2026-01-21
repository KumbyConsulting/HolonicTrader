import re
import glob
import os
from collections import Counter
from datetime import datetime

def analyze_vetoes(log_file=None):
    """
    Parses the latest log file for VETO events and generates a report.
    """
    if not log_file:
        # Find latest log
        list_of_files = glob.glob('live_trading_session_*.log')
        if not list_of_files:
            print("No log files found.")
            return
        log_file = max(list_of_files, key=os.path.getctime)
    
    print(f"📊 ANALYZING VETOES IN: {log_file}")
    
    veto_counts = Counter()
    asset_counts = Counter()
    total_signals = 0
    total_vetoes = 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Count Signals
            if "BUY SIGNAL" in line or "SELL SIGNAL" in line:
                total_signals += 1
            
            # Count Vetoes
            if "VETO" in line:
                total_vetoes += 1
                
                # Extract Veto Type
                # Patterns: "🛡️ PIVOT VETO", "☢️ CRISIS VETO", "🔗 CORRELATION VETO", "🛑 GOVERNOR PRE-CHECK VETO"
                veto_type = "UNKNOWN"
                if "PIVOT VETO" in line: veto_type = "PIVOT"
                elif "CRISIS VETO" in line: veto_type = "CRISIS"
                elif "CORRELATION VETO" in line: veto_type = "CORRELATION"
                elif "GOVERNOR PRE-CHECK" in line: veto_type = "GOVERNOR_PRE"
                elif "Governor Vetoed" in line: veto_type = "GOVERNOR_RISK"
                elif "LOW ENERGY" in line: veto_type = "LOW_ENERGY"
                elif "FAIR WEATHER" in line: veto_type = "FAIR_WEATHER"
                elif "DRAWDOWN LOCK" in line: veto_type = "DRAWDOWN_LOCK"
                
                veto_counts[veto_type] += 1
                
                # Extract Asset
                # Pattern: for BTC/USDT or : BTC/USDT
                match = re.search(r'([A-Z]+/[A-Z]+)', line)
                if match:
                    asset = match.group(1)
                    asset_counts[asset] += 1
    
    # === REPORT ===
    print("\n" + "="*40)
    print(f"🛡️  VETO ANALYTICS DASHBOARD")
    print("="*40)
    print(f"Total Signals Detected: {total_signals}")
    print(f"Total Vetoes Triggered: {total_vetoes}")
    if total_signals > 0:
        print(f"Global Veto Rate:       {(total_vetoes/total_signals)*100:.1f}%")
    
    print("\n🚫 VETOES BY TYPE:")
    for v_type, count in veto_counts.most_common():
        print(f"  • {v_type:<15} : {count} ({count/total_vetoes*100:.1f}%)")
        
    print("\n🎯 MOST BLOCKED ASSETS:")
    for asset, count in asset_counts.most_common(5):
        print(f"  • {asset:<15} : {count}")
        
    print("\n" + "="*40)

if __name__ == "__main__":
    analyze_vetoes()
