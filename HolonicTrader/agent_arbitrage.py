"""
ArbitrageHolon - The "Yield Hunter" Brain (Phase 40)

Specialized in:
1. Spatial Spread Monitoring (KuCoin vs. Kraken).
2. Funding Rate Yield Detection (Basis Trading).
3. Cross-Exchange Conviction Injection.
"""

import time
from typing import Any, Dict, List, Optional
from HolonicTrader.holon_core import Holon, Disposition
import config

class ArbitrageHolon(Holon):
    def __init__(self, name: str = "ArbHunter"):
        # High Autonomy (decides what is a 'good' arb), High Integration (feeds into Oracle)
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.9))
        self.last_sync_ts = 0
        self.price_spreads = {}  # {symbol: spread_pct}
        self.funding_yields = {} # {symbol: annual_yield_pct}
        self.kucoin_observer = None
        self.kraken_observer = None

    def perform_sync(self, symbols: List[str]):
        """Fetch data from both observers and update internal arb state."""
        if not self.kucoin_observer or not self.kraken_observer:
            return

        # 1. Fetch Tickers in parallel
        # Note: Observer handles its own batch fetching and caching
        kucoin_tickers = self.kucoin_observer.fetch_tickers_batch(symbols)
        
        # Kraken needs mapped symbols
        kraken_symbols = [config.KRAKEN_SYMBOL_MAP.get(s, s) for s in symbols]
        kraken_tickers = self.kraken_observer.fetch_tickers_batch(kraken_symbols)

        for sym in symbols:
            # KuCoin Price
            k_ticker = kucoin_tickers.get(sym)
            ku_price = float(k_ticker.get('last', 0.0)) if k_ticker else 0.0
            
            # Kraken Price
            kr_sym = config.KRAKEN_SYMBOL_MAP.get(sym, sym)
            kr_ticker = kraken_tickers.get(kr_sym)
            kr_price = float(kr_ticker.get('last', 0.0)) if kr_ticker else 0.0
            
            if ku_price > 0 and kr_price > 0:
                self.calculate_spatial_spread(sym, ku_price, kr_price)
                
            # Funding (Kraken only)
            funding = self.kraken_observer.fetch_funding_rate(sym)
            if funding != 0:
                self.analyze_funding_yield(sym, funding)
        
    def calculate_spatial_spread(self, symbol: str, kucoin_price: float, kraken_price: float) -> float:
        """Calculate the percentage spread between two prices."""
        if not kucoin_price or not kraken_price:
            return 0.0
        
        # Spread = (Target - Source) / Source
        spread = (kraken_price - kucoin_price) / kucoin_price
        self.price_spreads[symbol] = spread
        
        # If spread exceeds threshold, report it
        threshold = getattr(config, 'ARB_SPATIAL_THRESHOLD', 0.005) # Default 0.5%
        if abs(spread) >= threshold:
            print(f"[{self.name}] ⚖️ SPATIAL ARB DETECTED | {symbol}: {spread*100:.2f}% (KuC: {kucoin_price} vs Kra: {kraken_price})")
            
        return spread

    def analyze_funding_yield(self, symbol: str, funding_rate: float) -> float:
        """Convert 8-hour funding rate to projected annualized yield (EFFECTIVE APY)."""
        # CCXT Kraken Futures returns rate as decimal (e.g. 0.0001 = 0.01%)
        # But if we see rates like 0.25, it's often 0.25% (0.0025 decimal).
        # We apply a scaling correction if the raw rate is suspiciously high.
        corrected_rate = funding_rate
        if abs(funding_rate) > 0.01: # > 1% per 8h is extreme for most pairs
             corrected_rate = funding_rate / 100.0
             
        # FIX: Effective APY Calculation (Compound)
        # Annualized = (1 + rate)^1095 - 1 (compounded 3 times/day * 365 days)
        # OLD (Wrong): rate * 3 * 365 = -273.8%
        # NEW (Correct): (1 + rate)^1095 - 1 = ~-93.5% for -0.25% per 8h
        try:
            apy = ((1 + corrected_rate) ** 1095 - 1) * 100
        except OverflowError:
            # Fallback for extreme rates
            apy = 500.0 if corrected_rate > 0 else -500.0
        
        # SANITY FILTER: APY > 500% is likely a data error or extreme anomaly
        if abs(apy) > 500.0:
            apy = 500.0 if apy > 0 else -500.0
            
        self.funding_yields[symbol] = apy
        
        threshold = getattr(config, 'ARB_FUNDING_THRESHOLD', 0.0001) # Default 0.01% per 8h
        if abs(corrected_rate) >= threshold:
             status = "YIELD" if corrected_rate > 0 else "SHORTING_COST"
             print(f"[{self.name}] 💰 FUNDING {status} | {symbol}: {apy:.1f}% APY (Rate: {corrected_rate*100:.4f}% per 8h)")
             
        return apy


    def get_arb_conviction(self, symbol: str) -> float:
        """
        Returns a conviction boost [-1.0, 1.0] based on arbitrage opportunities.
        """
        boost = 0.0
        
        # Funding Yield Logic (Basis Trade)
        # If Funding is Negative (-0.25%), Shorts PAY Longs.
        # Yield for LONG = abs(NegativeRate).
        # Yield for SHORT = NegativeRate (Cost).
        
        funding_apy = self.funding_yields.get(symbol, 0.0)
        spread = self.price_spreads.get(symbol, 0.0) # (Kraken - KuCoin) / KuCoin
        
        # Scenario A: LONG (Farm Negative Funding)
        # We want to Long if Yield is massive, even if price is slightly high.
        # But if Price is TOO high (Spread > 1%), it might revert against us.
        if funding_apy > 50.0: # Huge Positive Yield (Longs earn > 50% APY)
             # But check if we are buying the top?
             # If Kraken > KuCoin by 2%, reversion risk is -2%.
             # If Yield is 270% APY -> 0.74% per day.
             # We need to hold for 3 days to breakeven on a 2% spread reversion.
             # If Spread is small (< 1%), it's a Gold Nugget.
             if spread < 0.01:
                 boost += 0.3 # Strong Long Boost (Yield Farm)
                 
        # Scenario B: SHORT (Spatial Arb)
        # If Kraken >> KuCoin (Spread > 1%) AND Funding is Positive (Longs Pay Shorts) or Low Cost
        if spread > 0.01:
             # Price is 1% premium. We want to SHORT.
             # Check cost. If Funding is -270%, cost is 0.75% daily.
             # Reversion (1%) > Cost (0.75%). Profitable if it closes in < 24h.
             # If Funding is Positive (>0), we get PAID to Short. Double Win.
             if funding_apy > -50.0: # Cost is less than 50% APY (or we earn yield)
                 boost -= 0.3 # Strong Short Boost
                 
        return boost

    def get_active_signal(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Generates a direct trade signal if a 'Gold Nugget' arb exists.
        Returns: Dict with 'direction', 'confidence', 'reason'
        
        TUNED: More aggressive thresholds per user request.
        """
        funding_apy = self.funding_yields.get(symbol, 0.0)
        spread = self.price_spreads.get(symbol, 0.0)
        
        # 1. BASIS YIELD TRADE (Long Side)
        # TUNED: Lowered from 100% to 50% APY threshold
        # Tiered confidence: Higher yield = higher conviction
        if funding_apy > 50.0 and spread < 0.01:  # Was: >100% and <0.5%
            confidence = min(0.95, 0.6 + (funding_apy / 500.0))  # Scale with yield
            return {
                'direction': 'BUY',
                'confidence': confidence,
                'reason': f"BASIS_YIELD (APY {funding_apy:.0f}%)"
            }
            
        # 2. SPATIAL REVERSION (Short Side)
        # TUNED: Lowered from 1.5% to 0.5% spread threshold
        # Even small spreads are actionable if funding is favorable
        if spread > 0.005 and funding_apy > -100.0:  # Was: >1.5% and >-50%
            confidence = min(0.95, 0.6 + (abs(spread) * 10))  # Scale with spread
            return {
                'direction': 'SELL',
                'confidence': confidence,
                'reason': f"SPATIAL_ARB (Spread {spread*100:.2f}%)"
            }
        
        # 3. NEGATIVE SPREAD LONG (Discount Buy)
        # NEW: If Kraken is CHEAPER than KuCoin, BUY on Kraken
        if spread < -0.005 and funding_apy > -100.0:
            confidence = min(0.95, 0.6 + (abs(spread) * 10))
            return {
                'direction': 'BUY',
                'confidence': confidence,
                'reason': f"DISCOUNT_ARB (Spread {spread*100:.2f}%)"
            }
            
        return None

    def receive_message(self, sender: Any, content: Any) -> None:
        """Process price updates from Observer."""
        pass
