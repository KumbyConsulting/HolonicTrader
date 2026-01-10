
import re
import sys
import os
from datetime import datetime
from collections import defaultdict

def analyze_log(log_path):
    if not os.path.exists(log_path):
        print(f"Error: File {log_path} not found.")
        return

    print(f"Analyzing {log_path}...")
    
    stats = {
        'entries': 0,
        'exits': 0,
        'errors': 0,
        'warnings': 0,
        'drawdown_alerts': 0,
        'pnl_events': [],
        'positions': {},
        'errors_list': []
    }

    entry_pattern = re.compile(r"\[ExecutorAgent\] (LONG|SHORT) ENTRY: (\w+/\w+) @ ([\d\.]+)")
    exit_pattern = re.compile(r"\[ExecutorAgent\] (LONG EXIT|SHORT COVER): (\w+/\w+) @ ([\d\.]+) \(PnL: ([+\-]?[\d\.]+)%, \$([+\-]?[\d\.]+)\)")
    error_pattern = re.compile(r"\[.*?\] (❌|⚠️|Error|Failed|Exception)(.*)")
    drawdown_pattern = re.compile(r"DRAWDOWN.*|Drawdown.*")
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check Entries
            m_entry = entry_pattern.search(line)
            if m_entry:
                stats['entries'] += 1
                direction, symbol, price = m_entry.groups()
                stats['positions'][symbol] = stats['positions'].get(symbol, 0) + 1
            
            # Check Exits
            m_exit = exit_pattern.search(line)
            if m_exit:
                stats['exits'] += 1
                type_, symbol, price, pnl_pct, pnl_usd = m_exit.groups()
                stats['pnl_events'].append({
                    'symbol': symbol,
                    'type': type_,
                    'price': price,
                    'pnl_pct': float(pnl_pct),
                    'pnl_usd': float(pnl_usd),
                    'time': line[:21]
                })
            
            # Check Errors
            m_err = error_pattern.search(line)
            if m_err:
                stats['errors'] += 1
                stats['errors_list'].append(line.strip())

            # Check Drawdown
            if "Drawdown" in line and "HALT" in line:
                stats['drawdown_alerts'] += 1

    # Generate Report
    report_lines = []
    report_lines.append(f"# Log Analysis Report: {os.path.basename(log_path)}")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_lines.append("## 📊 Summary Statistics")
    report_lines.append(f"- **Total Entries:** {stats['entries']}")
    report_lines.append(f"- **Total Exits:** {stats['exits']}")
    report_lines.append(f"- **Errors/Warnings:** {stats['errors']}")
    report_lines.append(f"- **Drawdown Halts:** {stats['drawdown_alerts']}\n")
    
    report_lines.append("## 💰 Performance (Realized)")
    if stats['pnl_events']:
        total_pnl = sum(x['pnl_usd'] for x in stats['pnl_events'])
        wins = len([x for x in stats['pnl_events'] if x['pnl_usd'] > 0])
        losses = len([x for x in stats['pnl_events'] if x['pnl_usd'] <= 0])
        win_rate = (wins / len(stats['pnl_events'])) * 100 if stats['pnl_events'] else 0
        
        report_lines.append(f"**Net PnL:** ${total_pnl:.2f}")
        report_lines.append(f"**Win Rate:** {win_rate:.1f}% ({wins}W / {losses}L)\n")
        
        report_lines.append("| Time | Symbol | Type | PnL $ | PnL % |")
        report_lines.append("|---|---|---|---|---|")
        for e in stats['pnl_events']:
            report_lines.append(f"| {e['time']} | {e['symbol']} | {e['type']} | ${e['pnl_usd']:+.2f} | {e['pnl_pct']:+.2f}% |")
    else:
        report_lines.append("No realized trades found in this session.")
    
    report_lines.append("\n## 🚨 Critical Errors & Warnings")
    if stats['errors_list']:
        for i, err in enumerate(stats['errors_list'][-20:]): # Last 20 errors
            report_lines.append(f"{i+1}. `{err}`")
        if len(stats['errors_list']) > 20:
            report_lines.append(f"... (and {len(stats['errors_list']) - 20} more)")
    else:
        report_lines.append("None.")

    # Write Report
    report_path = log_path + "_analysis.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"\nAnalysis Complete. Report saved to: {report_path}")
    print("\n".join(report_lines))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to latest log if available
        logs = [f for f in os.listdir('.') if f.endswith('.log') and 'live_trading' in f]
        if logs:
            latest = max(logs, key=os.path.getmtime)
            analyze_log(latest)
        else:
            print("Usage: python log_analyzer.py <log_file>")
    else:
        analyze_log(sys.argv[1])
