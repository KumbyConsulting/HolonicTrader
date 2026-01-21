
import sys
import os
# Adjust path to find the HolonicTrader package if run from root or scripts
sys.path.append(os.path.join(os.getcwd(), 'HolonicTrader'))

from HolonicTrader.agent_trinity import TrinityStrategy
import config

def verify_trinity():
    print("=== Verifying Trinity Strategy Logic ===")
    
    trinity = TrinityStrategy()
    print(f"Predators: {trinity.predators}")
    print(f"Scavengers: {trinity.scavengers}")
    
    # CASE 1: MICRO Regime (Allowed: BTC, ETH, XRP, DOGE, ADA)
    print("\n--- TEST: MICRO Regime ---")
    
    # 1.1 Chaotic (Should pick best Scavenger allowed in Micro)
    # Scav List: XRP, DOGE, ADA, XTZ, PAXG
    # Allowed: BTC, ETH, XRP, DOGE, ADA
    # Best Match: XRP (First allowed)
    alloc = trinity.get_allocation_target(market_regime='CHAOTIC', current_capital_regime='MICRO')
    print(f"CHAOTIC Target: {alloc} (Expect Allowed Scavenger)")
    
    # 1.2 Ordered Bull (Should pick best Predator allowed in Micro)
    # Pred List: ETH, BTC, SOL, AVAX, LINK
    # Allowed: [Same]
    # Best Match: ETH
    alloc = trinity.get_allocation_target(market_regime='ORDERED', btc_trend='BULL', current_capital_regime='MICRO')
    print(f"ORDERED BULL Target: {alloc} (Expect Allowed Predator)")

    # CASE 2: MEDIUM Regime (Allowed: All)
    print("\n--- TEST: MEDIUM Regime ---")
    
    # 2.1 Chaotic (Should pick best Scavenger overall)
    # Best Match: XRP (First in list)
    alloc = trinity.get_allocation_target(market_regime='CHAOTIC', current_capital_regime='MEDIUM')
    print(f"CHAOTIC Target: {alloc} (Expect Top Scavenger)")
    
    # 2.2 Ordered Bull (Should pick best Predator overall)
    # Best Match: ETH or SOL
    alloc = trinity.get_allocation_target(market_regime='ORDERED', btc_trend='BULL', current_capital_regime='MEDIUM')
    print(f"ORDERED BULL Target: {alloc} (Expect Top Predator)")

if __name__ == "__main__":
    verify_trinity()
