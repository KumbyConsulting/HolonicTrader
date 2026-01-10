import os
import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Any, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from HolonicTrader.holon_core import Holon, Disposition, Message

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import config

class ObserverHolon(Holon):
    """
    ObserverHolon is responsible for acquiring market data from exchanges
    and processing it for other agents (like the Entropy Agent).
    """

    def __init__(self, exchange_id: str = 'kucoin', symbol: str = 'BTC/USDT'):
        # Initialize with default highly autonomous and integrated disposition for now
        # or minimal, depending on system design. Using balanced values here.
        default_disposition = Disposition(autonomy=0.5, integration=0.5)
        super().__init__(name=f"Observer_{exchange_id}_{symbol}", disposition=default_disposition)
        
        self.symbol = symbol
        self.exchange_id = exchange_id
        
        # Initialize exchange with rate limiting and larger pool size
        if hasattr(ccxt, exchange_id):
            # Create a custom session with a larger connection pool
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=config.CCXT_POOL_SIZE, 
                pool_maxsize=config.CCXT_POOL_SIZE
            )
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': config.CCXT_RATE_LIMIT,
                'session': session
            })
        else:
            raise ValueError(f"Exchange {exchange_id} not found in ccxt")

        # Map for local history files
        self.data_dir = os.path.join(os.getcwd(), 'market_data')
        
        # Data Cache
        self._cache = {} # symbol -> DataFrame
        self._ticker_cache = {}
        self._last_ticker_fetch = 0.0

    import time

    def fetch_tickers_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Optimized Scout Fetch: Gets 24hr stats for MULTIPLE symbols in ONE API call.
        Implements TTL Cache to prevent rate limit bans.
        """
        now = time.time()
        # 1. Check Cache
        if self._ticker_cache and (now - self._last_ticker_fetch < config.SCOUT_CACHE_TTL):
            return self._ticker_cache

        # 2. Fetch Live
        # 2. Fetch Live with Retry
        for attempt in range(3):
            try:
                if self.exchange.has['fetchTickers']:
                    # Filter strictly for requested symbols to save bandwidth (if exchange supports partial)
                    tickers = self.exchange.fetch_tickers(symbols)
                    
                    # 3. Update Cache
                    self._ticker_cache = tickers
                    self._last_ticker_fetch = now
                    return tickers
                else:
                    print(f"[{self.name}] ⚠️ Exchange does not support fetchTickers!")
                    return {}
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ⚠️ Ticker Batch Fetch Error after 3 attempts: {e}")
                    return {}
                time.sleep(1 * (attempt + 1))
        return {}

    def _get_local_filename(self, symbol: str, timeframe: str = '1h') -> str:
        """Map symbol to its local CSV filename (Supports 1h and 15m)."""
        suffix = f"_{timeframe}"
        
        # 1. Cleaner Logic: Construct filename directly first
        clean_symbol = symbol.replace('/', '').replace(':', '')
        filename = f"{clean_symbol}{suffix}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        # 2. Check if file exists. If so, return it.
        if os.path.exists(filepath):
            return filepath
            
        # 3. Legacy Fallback (for 1h files named 'BTCUSD_1h.csv' instead of 'BTCUSDT_1h.csv')
        # Only needed if direct construction failed
        legacy_map = {
            'BTC/USDT': f'BTCUSD{suffix}.csv',
            'BTC/USD': f'BTCUSD{suffix}.csv',
        }
        
        legacy_name = legacy_map.get(symbol)
        if legacy_name:
             return os.path.join(self.data_dir, legacy_name)
             
        return filepath # Return built path anyway so we know what was expected

    def load_local_history(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load historical data from market_data directory (Cached)."""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        filepath = self._get_local_filename(symbol, timeframe)
        
        if not os.path.exists(filepath):
            # print(f"[{self.name}] No local history for {symbol} at {filepath}") # Reduce noise
            return pd.DataFrame()
            
        print(f"[{self.name}] Loading local history for {symbol} from {filepath} (DISK READ)")
        df = self.load_data_from_csv(filepath)
        self._cache[cache_key] = df
        return df

    def fetch_market_data(self, timeframe: str = None, limit: int = 500, symbol: str = None) -> pd.DataFrame:
        """
        Fetches Hybrid Market Data: Local History + CCXT Live Sync.
        """
        target_timeframe = timeframe if timeframe else config.TIMEFRAME
        target_symbol = symbol if symbol else self.symbol
        
        # 1. Load Local History
        df_local = self.load_local_history(target_symbol, target_timeframe)
        
        # 2. Fetch Live Sync (CCXT)
        if not self.exchange.has['fetchOHLCV']:
             # If no CCXT support, return local or empty
             return df_local

        df_live = pd.DataFrame()
        
        for attempt in range(3):
            try:
                # If we have local data, we fetch since last timestamp
                if not df_local.empty:
                    last_ts = int(df_local['timestamp'].iloc[-1].timestamp() * 1000)
                    current_ts = int(time.time() * 1000)
                    
                    # Gap Analysis
                    gap_ms = current_ts - last_ts
                    gap_hours = gap_ms / (1000 * 3600)
                    
                    if gap_ms < 0: # Future data check
                         # print(f"[{self.name}] Local history is in future. Skipping live sync.")
                         return df_local

                    # DELTA FETCHING LOGIC
                    # If gap is small (< 5 hours), just fetch the tip.
                    # Otherwise, fetch full history (up to 1000).
                    fetch_limit = 5 if gap_hours < 5 else 1000
                    
                    ohlcv_live = self.exchange.fetch_ohlcv(target_symbol, target_timeframe, since=last_ts, limit=fetch_limit)
                else:
                    # Startup / Fresh: Fetch full history
                    ohlcv_live = self.exchange.fetch_ohlcv(target_symbol, target_timeframe, limit=limit)
                
                if ohlcv_live:
                    df_temp = pd.DataFrame(ohlcv_live, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                    df_live = df_temp
                break # Success

            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                print(f"[{self.name}] ⏳ RATE LIMIT HIT for {target_symbol}. Cooling down 5s...")
                time.sleep(5) # Smart Backoff
                if attempt == 2: print(f"[{self.name}] ❌ Max Retries (Rate Limit) for {target_symbol}")

            except Exception as e:
                print(f"[{self.name}] Sync Attempt {attempt+1}/3 failed for {target_symbol}: {e}")
                time.sleep(2) # Standard Backoff
        
        try:
            # Combine
            if not df_live.empty:
                if not df_local.empty:
                    df = pd.concat([df_local, df_live]).drop_duplicates(subset='timestamp').reset_index(drop=True)
                else:
                    df = df_live
            else:
                 df = df_local # Fallback
                
        except Exception as e:
            print(f"[{self.name}] Data Merge error for {target_symbol}: {e}")
            df = df_local # Fallback to local only

        # 3. Process Returns
        if not df.empty:
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            df.dropna(inplace=True)
        
        return df

    def fetch_market_data_batch(self, symbols: List[str], timeframe: str = None, limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Parallelized Batch Fetch for OHLCV data.
        Distributes requests across threads to minimize latency.
        """
        target_timeframe = timeframe if timeframe else config.TIMEFRAME
        results = {}
        with ThreadPoolExecutor(max_workers=config.CCXT_POOL_SIZE) as executor:
            future_to_symbol = {}
            for symbol in symbols:
                # STAGGERED POLLING: Sleep 0.2s between submissions to prevent 429 bursts
                time.sleep(0.2) 
                future = executor.submit(self.fetch_market_data, target_timeframe, limit, symbol)
                future_to_symbol[future] = symbol
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Batch Fetch Failed for {symbol}: {e}")
                    
        return results

    def get_latest_price(self, symbol: str = None) -> float:
        """
        Returns the current market price (last close).
        """
        target_symbol = symbol if symbol else self.symbol
        for attempt in range(3):
            try:
                ticker = self.exchange.fetch_ticker(target_symbol)
                return float(ticker['last'])
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ⚠️ Price Fetch Error {target_symbol}: {e}")
                    return 0.0
                time.sleep(1 * (attempt + 1))
        return 0.0

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch current order book depth (Bids/Asks).
        Returns {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
        """
        for attempt in range(3):
            try:
                # Map symbol if needed
                req_symbol = symbol
                # FIX: Check if we are incorrectly mapping Kraken Futures symbols for KuCoin
                if config.TRADING_MODE == 'FUTURES':
                    # If this observer is KuCoin, we CANNOT use Kraken symbols (e.g. BTC/USD:USD)
                    if 'kucoin' in self.exchange.id:
                        # KuCoin Futures often uses XBTUSDTM or similar, but for Spot/Simulated checking
                        # we likely want the standard symbol 'BTC/USDT' or 'BTC-USDT'
                        # Assuming 'symbol' passed in IS 'BTC/USDT' (internal format).
                        # We just rely on CCXT's unified symbol handling for KuCoin which usually works with 'BTC/USDT'.
                        req_symbol = symbol 
                    else:
                        # If this IS Kraken (e.g. Executor referencing it), use the map.
                        req_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                    
                book = self.exchange.fetch_order_book(req_symbol, limit)
                return {
                    'bids': book['bids'],
                    'asks': book['asks'],
                    'timestamp': book['timestamp']
                }
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ⚠️ OrderBook Fetch Fail {symbol}: {e}")
                    return {'bids': [], 'asks': []}
                time.sleep(1 * (attempt + 1))
        return {'bids': [], 'asks': []}

    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch Current Funding Rate for a symbol.
        Used for Short Squeeze detection (Negative Funding = Shorts paying Longs).
        Returns the funding rate as a decimal (e.g., 0.0001 = 0.01%).
        """
        if config.TRADING_MODE != 'FUTURES':
            return 0.0

        # Simple Cache (Funding rates strictly change every 1-4-8h depending on exchange)
        # We can cache for ~15 mins safely.
        cache_key = f"funding_{symbol}"
        now = time.time()
        
        # Initialize specialized cache if missing
        if not hasattr(self, '_funding_cache'):
             self._funding_cache = {}
        
        if cache_key in self._funding_cache:
             entry = self._funding_cache[cache_key]
             if now - entry['ts'] < 900: # 15 min TTL
                 return entry['rate']

        for attempt in range(3):
            try:
                # CCXT Unified
                # Check if exchange supports it
                if self.exchange.has['fetchFundingRate']:
                     exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                     data = self.exchange.fetch_funding_rate(exec_symbol)
                     rate = float(data.get('fundingRate', 0.0))
                     
                     # Update Cache
                     self._funding_cache[cache_key] = {'rate': rate, 'ts': now}
                     return rate
                else:
                     return 0.0
                     
            except Exception as e:
                # print(f"[{self.name}] ⚠️ Funding Rate Fetch Fail: {e}")
                time.sleep(1)
        
        return 0.0

    def receive_message(self, sender: Any, content: Any) -> Any:
        """
        Handle incoming messages for health checks or data requests.
        """
        # Unwrap Holon Message if needed
        if isinstance(content, Message):
            msg_type = content.type
            payload = content.payload
        elif isinstance(content, dict):
             msg_type = content.get('type')
             payload = content
        else:
             return None

        if msg_type == 'GET_STATUS':
            # Report health status
            return {
                'status': 'OK',
                'last_fetch': datetime.now().isoformat(),
                'primary_symbol': self.symbol
            }
            
        elif msg_type == 'FORCE_FETCH':
            symbol = payload.get('symbol') if isinstance(payload, dict) else None
            print(f"[{self.name}] Received FORCE_FETCH for {symbol or 'ALL'}")
            return True
            
        return None

    def load_data_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load historical data from a CSV file.
        Expects columns: timestamp, open, high, low, close, volume
        Calculates returns automatically.
        """
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate Log Returns
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Drop NaNs
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            print(f"[{self.name}] Error loading CSV: {e}")
            return pd.DataFrame()
