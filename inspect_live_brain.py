import json
import os
import config

def inspect():
    print("🧠 LIVE BRAIN INSPECTOR")
    print("="*60)
    
    path = 'live_genome.json'
    if not os.path.exists(path):
        print("❌ No Live Genome Found. System using Static Config.")
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        genome = data['genome']
        equity = data['final_equity']
        roi = data['roi']
        
        print(f"✅ LIVE GENOME ACTIVE (Equity Benchmark: ${equity:.2f} | ROI: {roi*100:.1f}%)")
        print("-" * 60)
        print(f"{'PARAMETER':<25} | {'STATIC (Config)':<20} | {'LIVE (Evolved)':<20}")
        print("-" * 60)
        
        # Compare Key Stats
        params = [
            ('RSI Limit (Buy)', config.SATELLITE_ENTRY_RSI_CAP, genome.get('rsi_buy', 'N/A')),
            ('RVOL Thresh', config.SATELLITE_RVOL_THRESHOLD, genome.get('sat_rvol', 'N/A')),
            ('BBW Expansion', config.SATELLITE_BBW_EXPANSION_THRESHOLD, genome.get('sat_bb_expand', 'N/A')),
            ('Stop Loss', config.SATELLITE_STOP_LOSS, genome.get('satellite_stop_loss', genome.get('stop_loss', 'N/A'))),
            ('Take Profit', config.SATELLITE_TAKE_PROFIT_1, genome.get('satellite_take_profit', genome.get('take_profit', 'N/A'))),
        ]
        
        for name, static, live in params:
            # Format numbers
            s_val = f"{static:.4f}" if isinstance(static, (int, float)) else str(static)
            l_val = f"{live:.4f}" if isinstance(live, (int, float)) else str(live)
            
            # Highlight differences
            diff = "match" if s_val == l_val else ">>> MODIFY <<<"
            print(f"{name:<25} | {s_val:<20} | {l_val:<20} {diff if diff != 'match' else ''}")
            
        print("-" * 60)
        print("Conclusion: Agents are currently trading with EVOLVED parameters.")
        
    except Exception as e:
        print(f"Error reading genome: {e}")

if __name__ == "__main__":
    inspect()
