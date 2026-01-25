import re
import sys
import os
from datetime import datetime
from typing import List, Dict, Optional
import collections

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LogEvent:
    def __init__(self, timestamp_str: str, agent: str, message: str):
        self.raw_timestamp = timestamp_str
        self.agent = agent.strip('[]')
        self.message = message.strip()
        # Parse timestamp if possible (format: 2026-01-23 07:23:44)
        try:
            self.timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            self.timestamp = None

    def __repr__(self):
        return f"[{self.raw_timestamp}] [{self.agent}] {self.message}"

class TransactionContext:
    """Represents the lifecycle of a potential trade (Signal -> Decision -> Execution)."""
    def __init__(self, symbol: str, signal_time: datetime, direction: str):
        self.symbol = symbol
        self.start_time = signal_time
        self.direction = direction
        self.signal_details = ""
        self.vetoes = []
        self.execution = None
        self.status = "PENDING" # PENDING, VETOED, EXECUTED, CONFIRMED

    def add_veto(self, reason: str):
        self.vetoes.append(reason)
        # Prioritize "Hard" Vetoes over "Soft" ones for status
        if "Disallowed" in reason or "REJECTED" in reason:
            self.status = "VETOED"
        elif self.status != "VETOED":
            self.status = "BLOCKED" # Soft veto like cooldown

    def set_executed(self, details: str):
        self.execution = details
        self.status = "EXECUTED"

class LogAnalyzer:
    def __init__(self):
        self.events: List[LogEvent] = []
        self.transactions: List[TransactionContext] = []
        self.active_contexts: Dict[str, TransactionContext] = {} # Symbol -> Context
        self.system_health = []
        self.stats = collections.defaultdict(int)
        
        # Regex Patterns
        self.log_pattern = re.compile(r'\[(.*?)\] \[(.*?)\] (.*)')
        
        # Signal Pattern: [EntryOracle] 🚀 LINK/USDT BUY SIGNAL ...
        self.sig_pattern = re.compile(r'🚀 (.*?) (BUY|SELL) SIGNAL')
        
        # Veto Patterns
        self.veto_patterns = [
            (re.compile(r'Stack Too Close for (.*):'), "Stack Too Close"),
            (re.compile(r'Cooldown Active for (.*)'), "Cooldown Active"),
            (re.compile(r'GOVERNOR PRE-CHECK VETO for (.*):'), "Governor Veto"),
            (re.compile(r'FILTER: Overextended'), "RSI Overextended"),
            (re.compile(r'THESIS INVALIDATED for (.*)'), "Thesis Invalidated")
        ]
        
        # Execution Pattern
        self.exec_pattern = re.compile(r'EXECUTING ENTRY: (.*?) \(')
        self.fill_pattern = re.compile(r'(LONG|SHORT) (ENTRY|EXIT): (.*?) @')

    def parse_file(self, filepath: str):
        print(f"{Colors.HEADER}🔍 Parsing Log File: {filepath}{Colors.ENDC}")
        if not os.path.exists(filepath):
            print(f"{Colors.FAIL}Error: File not found.{Colors.ENDC}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = self.log_pattern.match(line)
                if match:
                    ts, agent, msg = match.groups()
                    event = LogEvent(ts, agent, msg)
                    self.events.append(event)
                    self._process_event(event)

    def _process_event(self, event: LogEvent):
        # 1. Detect Signals
        sig_match = self.sig_pattern.search(event.message)
        if sig_match:
            symbol, direction = sig_match.groups()
            self._open_context(symbol, event.timestamp, direction, event.message)
            self.stats['TOTAL_SIGNALS'] += 1
            return

        # 2. Detect Vetoes
        for pattern, reason in self.veto_patterns:
            if pattern.search(event.message):
                # Extract symbol if possible, or infer from context
                # Regex usually captures symbol as group 1
                match = pattern.search(event.message)
                symbol = match.group(1) if match.groups() else None
                
                # Clean symbol string (sometimes has trailing chars)
                if symbol: 
                    symbol = symbol.split(' ')[0].strip('():')
                    self._add_veto(symbol, reason, event.message)
                    self.stats[f'VETO_{reason.upper().replace(" ", "_")}'] += 1
                return

        # 3. Detect Execution Intent
        exec_match = self.exec_pattern.search(event.message)
        if exec_match:
            symbol = exec_match.group(1)
            self._confirm_execution(symbol, event.message)
            self.stats['EXECUTIONS_ATTEMPTED'] += 1
            return

        # 4. Detect Fills (Paper or Real)
        fill_match = self.fill_pattern.search(event.message)
        if fill_match:
            side, type_, symbol = fill_match.groups()
            self.stats['FILLS_CONFIRMED'] += 1
            # Special handling for Exits
            if type_ == "EXIT":
                self.stats['EXITS'] += 1

    def _open_context(self, symbol: str, timestamp: datetime, direction: str, details: str):
        # Close existing context for this symbol if any (assuming fast cycles)
        if symbol in self.active_contexts:
            # Mark previous as timed out or complete?
            pass 
        
        ctx = TransactionContext(symbol, timestamp, direction)
        ctx.signal_details = details
        self.active_contexts[symbol] = ctx
        self.transactions.append(ctx)

    def _add_veto(self, symbol: str, reason: str, full_msg: str):
        # Link veto to the active context OR create a "Ghost" context (Veto without recent signal)
        if symbol in self.active_contexts:
            self.active_contexts[symbol].add_veto(full_msg)
        else:
            # Ghost Veto (Governor checking during loop without Oracle Signal)
            pass

    def _confirm_execution(self, symbol: str, details: str):
        if symbol in self.active_contexts:
            self.active_contexts[symbol].set_executed(details)
            # Close context after execution? Keep open for fill?
            # For now, simplistic correlation
            del self.active_contexts[symbol]

    def generate_report(self, output_path: str = "session_analysis.md"):
        print(f"\n{Colors.CYAN}Generating Report...{Colors.ENDC}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🕵️ Log Analysis Report\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## 📊 Session Statistics\n")
            f.write("| Metric | Count |\n")
            f.write("| :--- | :---: |\n")
            for k, v in self.stats.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")
            
            f.write("## 🎬 The Story of the Session\n")
            
            # Group transactions by time
            for tx in self.transactions:
                icon = "⚪"
                if tx.status == "EXECUTED": icon = "🟢"
                elif tx.status == "VETOED": icon = "🔴"
                elif tx.status == "BLOCKED": icon = "🟠"
                
                f.write(f"### {icon} {tx.start_time.strftime('%H:%M:%S')} - {tx.symbol} ({tx.direction})\n")
                f.write(f"- **Signal:** `{tx.signal_details}`\n")
                
                if tx.vetoes:
                    f.write(f"- **Result:** {tx.status}\n")
                    f.write("- **Obstacles:**\n")
                    unique_vetoes = list(set(tx.vetoes)) # Dedup
                    for v in unique_vetoes[:3]: # Limit spam
                        f.write(f"  - `{v}`\n")
                    if len(unique_vetoes) > 3:
                        f.write(f"  - ... and {len(unique_vetoes)-3} more.\n")
                
                if tx.execution:
                    f.write(f"- **✅ EXECUTION:** `{tx.execution}`\n")
                
                f.write("\n")

        print(f"{Colors.GREEN}✅ Report Saved: {output_path}{Colors.ENDC}")
        self._print_cli_summary()

    def _print_cli_summary(self):
        print(f"\n{Colors.HEADER}=== ANALYSIS SUMMARY ==={Colors.ENDC}")
        print(f"Signals Detected: {self.stats['TOTAL_SIGNALS']}")
        print(f"Executions:       {self.stats['EXECUTIONS_ATTEMPTED']}")
        print(f"Fills:            {self.stats['FILLS_CONFIRMED']}")
        print("-" * 30)
        print(f"{Colors.WARNING}Top Vetoes:{Colors.ENDC}")
        # Sort veto stats
        vetoes = {k:v for k,v in self.stats.items() if k.startswith("VETO_")}
        for k, v in sorted(vetoes.items(), key=lambda item: item[1], reverse=True):
            print(f"  {k.replace('VETO_', '')}: {v}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python log_analyzer.py <logfile>")
        return # Dev mode override: Use default log if none provided?
        
    logfile = sys.argv[1]
    analyzer = LogAnalyzer()
    analyzer.parse_file(logfile)
    analyzer.generate_report()

if __name__ == "__main__":
    main()
