import sys
import os
import time

# Allow import from local
sys.path.append(os.getcwd())

from HolonicTrader.agent_sentiment import SentimentHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
import config

def test_sentiment_flow():
    print("--- Testing Sentiment Holon ---")
    
    # 1. Instantiate
    print("1. Instantiating Agent...")
    try:
        agent = SentimentHolon()
        print("✅ Agent Instantiated.")
    except Exception as e:
        print(f"❌ Instantiation Failed: {e}")
        return

    # 2. Test Fetching (Network)
    print("\n2. Testing RSS Fetching (Real Network Call)...")
    try:
        # Force fetch even if cache thinks it's fresh (bypass time check manually if needed, 
        # but fetch_and_analyze does check time. Let's reset last_update.)
        agent.last_update = 0 
        
        score = agent.fetch_and_analyze()
        print(f"✅ Fetch Complete. Score: {score}")
        print(f"   News Cache Size: {len(agent.news_cache)}")
        if agent.news_cache:
            print(f"   Sample Headline: {agent.news_cache[0]}")
    except Exception as e:
        print(f"❌ Fetching Failed: {e}")

    # 3. Test Oracle Integration
    print("\n3. Testing Oracle Bias Blending...")
    try:
        oracle = EntryOracleHolon()
        # Mock Oracle Trends (50/50 split)
        oracle.symbol_trends = {
            'BTC': True, 'ETH': True, 'SOL': False, 'XRP': False
        }
        
        # Base Technical Bias should be 0.5
        tech_bias = oracle.get_market_bias(sentiment_score=0.0)
        print(f"   Technical Bias (Should be 0.5): {tech_bias:.2f}")
        
        # Test Bullish Sentiment Impact
        bull_bias = oracle.get_market_bias(sentiment_score=1.0) # Max Greed
        print(f"   Bias with +1.0 Sentiment (Target ~0.65): {bull_bias:.2f}")
        
        # Test Bearish Sentiment Impact
        bear_bias = oracle.get_market_bias(sentiment_score=-1.0) # Max Fear
        print(f"   Bias with -1.0 Sentiment (Target ~0.35): {bear_bias:.2f}")
        
        if bull_bias > tech_bias and bear_bias < tech_bias:
            print("✅ Oracle Bias Blending Working Correctly.")
        else:
            print("❌ Oracle Bias Logic Error.")

    except Exception as e:
        print(f"❌ Oracle Test Failed: {e}")

if __name__ == "__main__":
    test_sentiment_flow()
