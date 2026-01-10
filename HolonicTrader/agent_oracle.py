"""
EntryOracleHolon - The "Offense" Brain (Phase 16)

Specialized in:
1. Pattern Recognition (LSTM)
2. Global Market Bias (GMB) Calculation
3. Entry Signal Generation (Scavenger/Predator)
"""

import pandas as pd
import numpy as np
import traceback
from typing import Any, Dict, List, Optional, Literal
from .agent_executor import TradeSignal as GlobalTradeSignal # Fix: Renamed to avoid scope collision
import os
try:
    import joblib
except ImportError:
    joblib = None

try:
    import tensorflow
    import tensorflow as tf
except ImportError:
    tensorflow = None
    tf = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import openvino as ov
except ImportError:
    ov = None

from typing import Any, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
from .kalman import KalmanFilter1D
import config
import threading

class EntryOracleHolon(Holon):
    def __init__(self, name: str = "EntryOracle", xgb_model=None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.6))
        
        # Parameters
        self.rsi_period = 14
        self._lock = threading.Lock()
        
        # AI Brains
        self.model = None       # LSTM
        self.scaler = None      # Scaler for LSTM
        self.xgb_model = xgb_model   # XGBoost - ALLOW INJECTION
        self.ov_compiled_model = None # OpenVINO
        
        # State Memory
        self.kalman_filters = {} # {symbol: KalmanFilter1D}
        self.kalman_last_ts = {} # {symbol: timestamp}
        self.symbol_trends = {}  # {symbol: bool (is_bullish)}
        self.last_probes = {}    # {symbol: {'lstm': prob, 'xgb': prob}}
        self.last_macro_state = {} # {symbol: 'BULLISH'/'BEARISH'} for log damping
        self.market_state = {}   # {correlation_matrix: df, entropy: float}
        self.crisis_score = 0.0  # Macro Crisis Score (0.0 - 1.0)
        
        # Optimization: Inference Cache
        self._inference_cache = {} # {f"{symbol}_{ts}": prob}

        # Emergence: Emotional State
        self.emotional_state = {} # {'fear': 0.0, 'greed': 1.0}
        
        # Load Brains
        self.load_brains()
        
    def set_market_state(self, correlation_matrix: pd.DataFrame = None, entropy: float = None):
        """Receive live market physics data from Trader/Observer."""
        with self._lock:
            if correlation_matrix is not None:
                self.market_state['correlation'] = correlation_matrix
                self.market_state['entropy'] = entropy
        
    def set_crisis_score(self, score: float):
        """Update macro crisis score from SentimentHolon."""
        self.crisis_score = score
        
    def set_expert_model(self, model):
        """Inject a specific XGBoost model (for Walk-Forward Optimization)."""
        with self._lock:
            self.xgb_model = model
            print(f"[{self.name}] 🧠 New XGBoost Brain Injected.")

    def set_emotional_bias(self, fear: float, greed: float):
        """
        Receive Emotional Feedback from the Neural Network (Trader).
        Fear = Drawdown (0.0 - 1.0)
        Greed = Risk Multiplier (0.5 - 2.0)
        """
        with self._lock:
            self.emotional_state = {'fear': fear, 'greed': greed}
            
    def apply_asset_personality(self, symbol: str, signal: Any) -> Any:
        """
        Apply Asset-Specific Rules (The Physics Layer).
        Modifies or Vetos signals based on asset class.
        """
        if not signal: return None
        
        # --- EMOTIONAL OVERRIDE (Amygdala) ---
        fear = self.emotional_state.get('fear', 0.0)
        greed = self.emotional_state.get('greed', 1.0)
        
        # FEAR: If Drawdown > 10% (Fear > 0.1), Block weak signals
        # FEAR: If Drawdown > 10% (Fear > 0.1), Block weak signals
        # UNLEASHED: Relaxed scaling. Max inhibition is +0.3 (Req 0.8)
        if fear > 0.10:
             # Old: 0.6 + fear (Blocked everything if fear > 0.4)
             # New: Base 0.5 + (Fear * 0.4). Maxes at 0.9.
             required_conviction = min(0.9, 0.5 + (fear * 0.4))
             
             if signal.direction == 'BUY' and signal.conviction < required_conviction:
                 print(f"[{self.name}] 😨 FEAR VETO: {symbol} Conviction {signal.conviction:.2f} < Req {required_conviction:.2f} (Fear {fear:.2f})")
                 return None
                 
        # GREED: If Risk Multiplier > 1.2, Boost conviction (Confidence)
        if greed > 1.2:
            signal.conviction = min(1.0, signal.conviction * 1.1)
            
        # 1. BTC: Dead Market Filter
        if symbol == 'BTC/USDT':
            meta = signal.metadata
            atr = meta.get('atr', 0)
            avg_atr = meta.get('avg_atr', atr) # Fallback
            if avg_atr > 0 and atr < (avg_atr * config.PERSONALITY_BTC_ATR_FILTER):
                print(f"[{self.name}] ☠️ BTC FILTER: Market Dead (ATR {atr:.2f} < 50% Avg). Signal IGNORED.")
                return None
                
        # 2. DOGE: Fakeout Filter (RVOL)
        elif symbol == 'DOGE/USDT':
            # This is partly handled in Satellite logic, but as a safety net for standard signals:
            rvol = signal.metadata.get('rvol', 1.0)
            if rvol < config.PERSONALITY_DOGE_RVOL:
                print(f"[{self.name}] 🐕 DOGE FILTER: Potential Fakeout (RVOL {rvol:.1f} < {config.PERSONALITY_DOGE_RVOL}). IGNORED.")
                return None
                
        # 3. SOL: Momentum Only
        elif symbol == 'SOL/USDT':
            rsi = signal.metadata.get('rsi', 50.0)
            if signal.direction == 'BUY' and rsi < config.PERSONALITY_SOL_RSI_LONG:
                print(f"[{self.name}] 🟣 SOL FILTER: Too Weak for Long (RSI {rsi:.1f} < {config.PERSONALITY_SOL_RSI_LONG})")
                return None
            elif signal.direction == 'SELL' and rsi > config.PERSONALITY_SOL_RSI_SHORT:
                print(f"[{self.name}] 🟣 SOL FILTER: Too Strong for Short (RSI {rsi:.1f} > {config.PERSONALITY_SOL_RSI_SHORT})")
                return None
                
        # 4. XRP: Whole Number Front-running
        elif symbol == 'XRP/USDT':
            # Add TP instruction to metadata
            # For Phase 4 simple execution, we just log it. Real execution needs smarter order types.
            signal.metadata['special_instruction'] = 'FRONT_RUN_WHOLE_NUMBERS'
            
        # 5. FAIR WEATHER PROTOCOL (Global Bias Veto)
        # Block ALL Satellite Longs if Global Bias is weak (< 0.45)
        # Core assets (BTC/ETH) are strong enough to buck the trend.
        if signal.direction == 'BUY' and symbol in config.SATELLITE_ASSETS:
            gmb = self.get_market_bias()
            if gmb < 0.45:
                print(f"[{self.name}] ☁️ FAIR WEATHER VETO: {symbol} Long blocked (Bias {gmb:.2f} < 0.45)")
                return None
                
        # 6. CRISIS PROTOCOL (Macro Strategy)
        # Assuming self.crisis_score is updated by TraderHolon
        if self.crisis_score > 0.5:
            # A. FLIGHT TO SAFETY (Boost Gold/BTC)
            if symbol in config.CRISIS_SAFE_HAVENS and signal.direction == 'BUY':
                signal.conviction = min(1.0, signal.conviction * 1.2) # +20% Boost
                signal.metadata['crisis_boost'] = True
                print(f"[{self.name}] ☢️ CRISIS BOOST: {symbol} Conviction increased to {signal.conviction:.2f}")
            
            # B. RISK OFF (Block Meme Longs)
            elif symbol in config.CRISIS_RISK_ASSETS and signal.direction == 'BUY':
                print(f"[{self.name}] ☢️ CRISIS VETO: {symbol} Long blocked (Crisis Score {self.crisis_score:.2f})")
                return None
        
        # 6b. WHALE TRACKING (Volume Physics)
        if signal.metadata.get('is_whale', False):
            signal.conviction = min(1.0, signal.conviction * 1.25) # +25% Boost
            print(f"[{self.name}] 🐋 WHALE BOOST: {symbol} Riding the wave! Conviction -> {signal.conviction:.2f}")

        # 7. PHYSICS LAYER (Global Validation)
        return self.apply_market_physics(symbol, signal)

    def apply_market_physics(self, symbol: str, signal: Any) -> Any:
        """
        The 'Physics Layer': Validates signals against laws of Correlation, Entropy, and Energy (Volume).
        """
        if not signal: return None
        
        # A. ENTROPY PROOF (Proof of Order)
        # We need the Entropy Agent's assessment. 
        # Ideally, we query it. For now, we assume if we are in this method, 
        # the market is 'tradeable' or we calculate it locally if crucial.
        # *Optimization: We will defer strictly to Entropy Agent's regime check in Trader loop
        # but here we can check specific asset entropy if data available.*
        
        # B. VOLUME TRUTH (Energy)
        rvol = signal.metadata.get('rvol', 1.0)
        if rvol < config.PHYSICS_MIN_RVOL:
            # Degrade confidence or Veto
            # Relaxed Soft Veto: Only blocks if GMB is VERY weak (<0.35)
            gmb = self.get_market_bias()
            if gmb < 0.35:  # Relaxed from 0.6 - only veto in very weak markets
                print(f"[{self.name}] 🔋 LOW ENERGY VETO: {symbol} RVOL {rvol:.1f} < 1.1 & Very Weak Bias ({gmb:.2f})")
                return None
                
        # C. PACK LOGIC (Correlation)
        # "Don't fight the Alpha."
        corr_matrix = self.market_state.get('correlation')
        if corr_matrix is not None and not corr_matrix.empty and 'BTC/USDT' in corr_matrix.columns:
            # Check Correlation to BTC
            btc_corr = corr_matrix['BTC/USDT'].get(symbol, 0.5)
            
            # Check BTC Trend (Proxy via Market Bias or explicit check. Using Bias < 0.5 as Bearish)
            gmb = self.get_market_bias()
            
            # RULE: If Correlated (>0.75) AND Leader is Weak (<0.5) -> VETO LONG
            # (Configurable Thresholds)
            if signal.direction == 'BUY':
                if btc_corr > config.PHYSICS_CORRELATION_THRESHOLD and gmb < 0.5:
                    print(f"[{self.name}] 🐺 PACK VETO: {symbol} Correlated ({btc_corr:.2f}) & Market Weak ({gmb:.2f})")
                    return None
            
            # RULE: If Inverse Correlated (<-0.75) AND Leader is Strong -> VETO LONG? 
            # (Usually we want to buy inverse assets when market is weak, so this is fine.)
        
        return signal

    def verify_holding_physics(self, symbol: str, direction: str, current_price: float = None) -> bool:
        """
        Proof of Holding: Re-verify the thesis for an open position.
        Returns check_result (True = KEEP HOLDING, False = THESIS INVALID / EXIT).
        """
        # 1. ENTROPY CHECK (Chaos Veto)
        #Ideally we query Entropy Agent, but assuming we have access or use a proxy
        # For now, we return True as placeholder or implement local check if needed.
        # *Optimization*: In Phase 5 we link Agent State. 
        # Here we check Global Bias as a proxy for "Market Environment".
        
        # 2. PACK LOGIC (Correlation Veto)
        # If we are Long, and Global Bias drops to Bearish (<0.4), and we are a Satellite...
        if direction == 'BUY' and symbol in config.SATELLITE_ASSETS:
             gmb = self.get_market_bias()
             if gmb < 0.35: # Strong Bearish Turn
                 print(f"[{self.name}] 📉 THESIS FAILED: {symbol} Long held but Global Bias collapsed to {gmb:.2f}")
                 return False

        # 3. FALLING KNIFE CHECK (Structure)
        # If we have a cached structure context or price > support?
        # This requires price data.
        # For Phase 5b, we rely on PREDATOR STOP LOSS for "price" exits.
        # But we can add a logical exit if "Market Personality" changes.
                 
        return True

    def audit_asset_profile(self, symbol: str, data: Any) -> Dict[str, Any]:
        """
        Diagnostic: Check asset health against personality rules WITHOUT needing a signal.
        """
        status = "HEALTHY"
        details = []
        
        # Calculate Metrics
        atr_period = 14
        if len(data) < atr_period + 1:
            return {'status': 'INSUFFICIENT_DATA', 'metrics': {}}
            
        # ATR / Volatility
        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean().iloc[-1]
        avg_atr = tr.rolling(30).mean().iloc[-1] # 30-period baseline
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # RVOL
        vol_sma = data['volume'].rolling(20).mean().iloc[-1]
        rvol = data['volume'].iloc[-1] / vol_sma if vol_sma > 0 else 0.0
        
        metrics = {
            'ATR': f"{atr:.4f}",
            'AvgATR': f"{avg_atr:.4f}",
            'RSI': f"{rsi:.1f}",
            'RVOL': f"{rvol:.2f}",
            'Price': f"{data['close'].iloc[-1]:.4f}"
        }
        
        # Check Rules
        if symbol == 'BTC/USDT':
            if avg_atr > 0 and atr < (avg_atr * config.PERSONALITY_BTC_ATR_FILTER):
                status = "DEAD_MARKET"
                details.append(f"ATR {atr:.4f} < {config.PERSONALITY_BTC_ATR_FILTER*100}% of Avg")
    

                
        elif symbol == 'DOGE/USDT':
            if rvol < config.PERSONALITY_DOGE_RVOL:
                status = "FAKEOUT_RISK"
                details.append(f"RVOL {rvol:.1f} < {config.PERSONALITY_DOGE_RVOL}")
                
        elif symbol == 'SOL/USDT':
            if rsi < config.PERSONALITY_SOL_RSI_LONG:
                details.append(f"Weak Momentum (RSI {rsi:.1f} < {config.PERSONALITY_SOL_RSI_LONG})") # Just a note, not necessarily unhealthy if Shorting
            if rsi > config.PERSONALITY_SOL_RSI_SHORT:
                 details.append(f"Strong Momentum (RSI {rsi:.1f} > {config.PERSONALITY_SOL_RSI_SHORT})")

        # Global Bias Check
        gmb = self.get_market_bias()
        if symbol in config.SATELLITE_ASSETS and gmb < 0.45:
             status = "VETOED (Fair Weather)"
             details.append(f"GMB {gmb:.2f} < 0.45")
             
        return {
            'symbol': symbol,
            'status': status,
            'details': ", ".join(details) if details else "None",
            'metrics': metrics
        }

    def analyze_satellite_entry(self, symbol: str, df_1h: pd.DataFrame, observer: Any) -> Any:
        from .agent_executor import TradeSignal
        
        # 🔑 KEY 1: TIMEFRAME ALIGNMENT (Trend)
        # 1H Check
        ema200_1h = df_1h['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        price = df_1h['close'].iloc[-1]
        
        trend_1h = 'BULL' if price > ema200_1h else 'BEAR'
        
        # 15m Check (Fetch fresh data)
        df_15m = observer.fetch_market_data(timeframe='15m', limit=100, symbol=symbol)
        if df_15m.empty or len(df_15m) < 50: return None
        
        ema50_15m = df_15m['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        price_15m = df_15m['close'].iloc[-1]
        
        trend_15m = 'BULL' if price_15m > ema50_15m else 'BEAR'
        
        # Alignment Veto
        if trend_1h != trend_15m: return None
        
        direction = 'BUY' if trend_1h == 'BULL' else 'SELL'
        
        # 🔑 KEY 2: VOLATILITY SQUEEZE (Timing)
        # Bollinger Bands (20, 2) on 15m
        sma20 = df_15m['close'].rolling(20).mean()
        std20 = df_15m['close'].rolling(20).std()
        upper = sma20 + (std20 * 2)
        lower = sma20 - (std20 * 2)
        
        bbw = (upper - lower) / sma20
        
        # Expansion Check (Current BBW vs Previous BBW)
        bbw_current = bbw.iloc[-1]
        bbw_prevent = bbw.iloc[-2]
        expansion_pct = (bbw_current - bbw_prevent) / bbw_prevent if bbw_prevent > 0 else 0
        
        if expansion_pct < config.SATELLITE_BBW_EXPANSION_THRESHOLD: return None
        
        # Breakout Check
        if direction == 'BUY' and price_15m <= upper.iloc[-1]: return None
        if direction == 'SELL' and price_15m >= lower.iloc[-1]: return None
        
        # 🔑 KEY 3: VOLUME CONFIRMATION (Truth)
        # RVOL Calculation
        current_vol = df_15m['volume'].iloc[-1]
        avg_vol = df_15m['volume'].rolling(20).mean().iloc[-2] # Preceding 20 avg
        rvol = current_vol / avg_vol if avg_vol > 0 else 0
        
        threshold = config.SATELLITE_DOGE_RVOL_THRESHOLD if 'DOGE' in symbol else config.SATELLITE_RVOL_THRESHOLD
        
        if rvol < threshold: return None
        
        # 🚀 ALL KEYS TURNED - FIRE
        self._safe_print(f"[{self.name}] 🚀 SATELLITE ENTRY: {symbol} {direction} (1H/15m Align, BBW Exp {expansion_pct:.1%}, RVOL {rvol:.1f})")
        
        sig = TradeSignal(symbol=symbol, direction=direction, size=1.0, price=price_15m)
        sig.metadata = {'strategy': 'SATELLITE', 'atr': 0.0} # ATR filled later if needed
        return self.apply_asset_personality(symbol, sig)

    def _safe_print(self, msg: str):
        """Thread-safe printing to avoid log corruption."""
        with self._lock:
            print(msg)

    def load_brains(self):
        """Load AI brains (LSTM and XGBoost)."""
        # LSTM Paths
        model_path = 'lstm_model.keras'
        scaler_path = 'scaler.pkl'
        # XGBoost Path
        xgb_path = 'xgboost_model.json'
        
        # 1. Load LSTM
        if os.path.exists(model_path) and os.path.exists(scaler_path) and tf is not None and joblib is not None:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self._safe_print(f"[{self.name}] LSTM Brain loaded successfully.")
            except Exception as e:
                self._safe_print(f"[{self.name}] Error loading LSTM: {e}")
        
        # 2. Load XGBoost
        if os.path.exists(xgb_path) and xgb is not None:
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                self._safe_print(f"[{self.name}] XGBoost Brain loaded successfully.")
            except Exception as e:
                self._safe_print(f"[{self.name}] Error loading XGBoost: {e}")

        if self.model is None and self.xgb_model is None:
            self._safe_print(f"[{self.name}] All brains missing or deps failed. Running heuristic mode.")

        # 3. OpenVINO Integration (Speed Optimization)
        if self.model is not None and ov is not None and config.USE_OPENVINO:
            try:
                core = ov.Core()
                # Convert Keras model to OpenVINO IR
                ov_model = ov.convert_model(self.model)
                device = "GPU" if config.USE_INTEL_GPU else "CPU"
                self.ov_compiled_model = core.compile_model(ov_model, device)
                self._safe_print(f"[{self.name}] OpenVINO LSTM Backend initialized on {device}.")
            except Exception as e:
                self._safe_print(f"[{self.name}] OpenVINO Setup failed: {e}. Falling back to native TensorFlow.")

    def predict_trend_lstm(self, prices: pd.Series, symbol: str) -> float:
        if self.model is None or self.scaler is None or len(prices) < 60 or tf is None:
            return 0.53

        # 1. Check Cache
        last_ts = prices.index[-1]
        cache_key = f"{symbol}_{last_ts}" # Using Last Price Timestamp
        if cache_key in self._inference_cache:
            return self._inference_cache[cache_key]

        try:
            data = prices.values[-60:].reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            x_input = scaled_data.reshape(1, 60, 1)
            
            res = 0.5
            
            # High-Performance OpenVINO Inference if available
            if self.ov_compiled_model:
                out = self.ov_compiled_model(x_input)[0]
                res = float(out[0][0])
            else:
                # High-Performance Functional Call (Avoids Retracing)
                x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
                prob_tensor = self.model(x_tensor, training=False)
                res = prob_tensor.numpy()[0][0]
                
            result = float(res)

            # 2. Update Cache
            # Simple eviction rule
            if len(self._inference_cache) > 2000:
                self._inference_cache.clear()
            self._inference_cache[cache_key] = result
            
            return result
        except Exception as e:
            self._safe_print(f"[{self.name}] Prediction Error: {e}")
            return 0.5

    def predict_trend_xgboost(self, features: dict) -> float:
        """Predict trend probability using XGBoost."""
        if self.xgb_model is None or xgb is None:
            return 0.55
        try:
            # Prepare DMatrix
            df_feat = pd.DataFrame([features])
            dmatrix = xgb.DMatrix(df_feat)
            prob = self.xgb_model.predict(dmatrix)[0]
            return float(prob)
        except Exception as e:
            self._safe_print(f"[{self.name}] XGBoost Prediction Error: {e}")
            return 0.5
    def get_kalman_estimate(self, symbol: str, window_data: pd.DataFrame) -> float:
        prices = window_data['close']
        log_prices = np.log(prices)
        current_ts = window_data['timestamp'].iloc[-1]
        
        if symbol not in self.kalman_filters:
            self.kalman_filters[symbol] = KalmanFilter1D(process_noise=0.0001, measurement_noise=0.001)
            self.kalman_last_ts[symbol] = None
            for i in range(len(log_prices)):
                self.kalman_filters[symbol].update(log_prices.iloc[i])
                self.kalman_last_ts[symbol] = window_data['timestamp'].iloc[i]
        else:
            if current_ts != self.kalman_last_ts.get(symbol):
                self.kalman_filters[symbol].update(log_prices.iloc[-1])
                self.kalman_last_ts[symbol] = current_ts
                
        kalman_price = float(np.exp(self.kalman_filters[symbol].x))
        self.symbol_trends[symbol] = prices.iloc[-1] > kalman_price
        return kalman_price

    def get_market_bias(self, sentiment_score: float = 0.0) -> float:
        if not self.symbol_trends:
            return 0.5
            
        # Hardening: If we only have a few symbols, return neutral 0.5
        # This prevents the first 1-2 bearish symbols from setting GMB to 0.0
        if len(self.symbol_trends) < (len(config.ALLOWED_ASSETS) / 4):
            return 0.5
            
        bullish_count = sum(1 for trend in self.symbol_trends.values() if trend)
        technical_bias = bullish_count / len(self.symbol_trends)
        
        # Blend with Sentiment (Configurable Weight)
        # Scale Sentiment (-1 to 1) to Bias (0 to 1) -> (S + 1) / 2
        sentiment_bias_norm = (sentiment_score + 1.0) / 2.0
        
        if sentiment_score != 0.0:
            final_bias = (technical_bias * (1 - config.SENTIMENT_WEIGHT)) + (sentiment_bias_norm * config.SENTIMENT_WEIGHT)
            # Log significant divergence
            if abs(final_bias - technical_bias) > 0.1:
                self._safe_print(f"[{self.name}] 📰 Sentiment Modified Bias: {technical_bias:.2f} -> {final_bias:.2f} (Score: {sentiment_score:.2f})")
            return final_bias
            
        return technical_bias

    # === PROJECT AHAB HELPER METHODS ===
    def detect_accumulation(self, window_data: pd.DataFrame, atr: float, avg_atr: float) -> bool:
        """
        Stealth Accumulation: High Volume + Low Volatility.
        Whales absorbing supply without moving price.
        """
        if len(window_data) < 20: return False
        
        # 1. Volume Spike (Quietly High)
        vol_avg = window_data['volume'].rolling(20).mean().iloc[-1]
        rvol = window_data['volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        
        # 2. Volatility Compression
        # If ATR is decreasing or below average
        is_compressed = atr < (avg_atr * config.WHALE_ACCUMULATION_ATR_FACTOR) if avg_atr > 0 else False
        
        # 3. Price Contained (Tight Range)
        # Check last 3 candles range < 1%
        last_3 = window_data['close'].iloc[-3:]
        range_pct = (last_3.max() - last_3.min()) / last_3.mean()
        is_tight = range_pct < 0.01
        
        return rvol > config.WHALE_ACCUMULATION_RVOL and (is_compressed or is_tight)

    def detect_whale_defense(self, structure_ctx: Dict[str, Any], window_data: pd.DataFrame) -> bool:
        """
        Whale Defense: Price hits Support + Volume Spike + Rejection Wick.
        """
        if not structure_ctx: return False
        
        # 1. Context: At Support
        dist_sup = structure_ctx.get('dist_to_sup_pct', 0.0)
        # Allow slight undercut (bear trap) or near miss
        at_support = -0.005 <= dist_sup <= 0.005 
        
        if not at_support: return False
        
        # 2. Volume Spike
        vol_avg = window_data['volume'].rolling(20).mean().iloc[-1]
        rvol = window_data['volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        
        if rvol < config.WHALE_DEFENSE_RVOL: return False
        
        # 3. Rejection Wick (Bullish Hammer / Pinbar)
        # Check last candle
        row = window_data.iloc[-1]
        body = abs(row['close'] - row['open'])
        lower_wick = min(row['close'], row['open']) - row['low']
        
        # Wick must be significant (e.g., > body)
        return lower_wick > body

    def calculate_book_pressure(self, book_data: Dict[str, Any]) -> float:
        """
        Order Book Imbalance: Ratio of Bid Volume to Ask Volume.
        Returns: Ratio (e.g. 2.0 = 2x Bids vs Asks)
        """
        if not book_data or not book_data.get('bids') or not book_data.get('asks'):
            return 1.0
            
        # Sum top 20 levels volume
        # Bids: [[price, qty], ...]
        bid_vol = sum([b[1] for b in book_data['bids'][:20]])
        ask_vol = sum([a[1] for a in book_data['asks'][:20]])
        
        if ask_vol == 0: return 99.0 # Infinite support
        
        return bid_vol / ask_vol

    def detect_short_squeeze(self, funding_rate: float, trend: str) -> bool:
        """
        Short Squeeze: Negative Funding (Shorts Paying) + Bullish/Neutral Trend.
        """
        if funding_rate >= 0: return False # Positive funding = Longs paying
        
        # Check if significantly negative
        is_negative = funding_rate < config.WHALE_FUNDING_SQUEEZE_THRESHOLD
        
        # Squeeze usually happens when shorts are trapped in a non-bearish market
        is_trapped = trend != 'BEARISH'
        
        return is_negative and is_trapped

    def analyze_for_entry(
        self, 
        symbol: str,
        window_data: pd.DataFrame, 
        bb_vals: dict, 
        obv_slope: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR'],
        structure_ctx: Dict[str, Any] = None,
        book_data: Dict[str, Any] = None,
        funding_rate: float = 0.0
    ):
        # from .agent_executor import TradeSignal # MOVED TO GLOBAL
        is_whale = False # Default initialization

        # --- PATCH 3: THE TREND LOCK (Respect the Bias) ---
        global_bias = self.get_market_bias()
        
        # Rule A: If Global_Bias >= 0.80 (Max Bullish), HARD BAN on all SHORT signals.
        can_short = global_bias < 0.80
        
        # Rule B: If Global_Bias <= 0.20 (Max Bearish), HARD BAN on all LONG signals.
        can_long = global_bias > 0.20
        
        # --- PATCH 4: STRUCTURAL TARGETING (Fractal Flows) ---
        if structure_ctx:
            # 1. Broken Support Check (Falling Knife)
            # Only veto if significantly below support (> 0.2%) to allow for "Reclaiming Support" plays.
            dist_sup = structure_ctx.get('dist_to_sup_pct', 0.0) # usually negative if below
            if structure_ctx.get('structure_mode') == 'BREAKDOWN_DOWN':
                if dist_sup < -0.002: # More than 0.2% below support
                    # self._safe_print(f"[{self.name}] 🧱 STRUCTURAL VETO {symbol}: Price < Support ({dist_sup*100:.2f}%). (Falling Knife)")
                    # can_long = False # DISABLED: User Requested "Take All Opportunities"
                    pass
                
            # 2. Key Level Resistance Check (Buying the Ceiling)
            # Only allow if 'Whale' is present or if we have at least 0.15% room (Scalpable)
            dist_res = structure_ctx.get('dist_to_res_pct', 1.0)
            if 0.0 < dist_res < 0.0015 and not is_whale: # Reduced from 0.3% to 0.15%
                # self._safe_print(f"[{self.name}] 🧱 STRUCTURAL VETO {symbol}: Too close to Resistance ({dist_res*100:.2f}%) without Whale backup.")
                # can_long = False # DISABLED: User Requested "Take All Opportunities"
                pass
        
        # Optimization: Early exit if trapped (though we need analysis to know direction...
        # unless we pass 'allowed_directions' down? Or just filter at the end.)
        
        # --- PATCH 5: MULTI-TIMEFRAME ALIGNMENT (1H RIVER) ---
        # The 'Macro Trend' (1h) must support the 'Micro Entry' (15m).
        # We expect 'macro_trend' to be passed in 'structure_ctx' or derived.
        macro_trend = structure_ctx.get('macro_trend', 'NEUTRAL') if structure_ctx else 'NEUTRAL'
        
        # --- SECTOR PHYSICS OVERRIDE (Holistic 5b) ---
        # Allow energetic decoupling for memecoins
        sector_override = False
        if symbol in config.MEMECOIN_ASSETS:
             # Calculate local rvol for check
             vol_avg_chk = window_data['volume'].rolling(window=20).mean().iloc[-1]
             vol_curr_chk = window_data['volume'].iloc[-1]
             rvol_chk = vol_curr_chk / vol_avg_chk if vol_avg_chk > 0 else 1.0
             
             if rvol_chk > config.MEMECOIN_PUMP_RVOL:
                 self._safe_print(f"[{self.name}] 🚀 SECTOR PHYSICS: {symbol} Decoupling from Macro (RVOL {rvol_chk:.1f})")
                 sector_override = True
        
        if not sector_override:
            last_state = self.last_macro_state.get(symbol, 'UNKNOWN')
            prices = window_data['close']
            
            # Volatility & Momentum (Moved UP for Whale Logic)
            returns = prices.pct_change()
            volatility = returns.rolling(14).std().iloc[-1]
            
            # 1. Feature Engineering (Moved UP for Filter Logic)
            # RSI (14)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
            
            # Whale Volume
            vol_avg = window_data['volume'].rolling(window=20).mean().iloc[-1]
            vol_curr = window_data['volume'].iloc[-1]
            rvol = vol_curr / vol_avg if vol_avg > 0 else 1.0
            
            # --- PROJECT AHAB: WHALE DETECTION ---
            # 1. Base Whale (Rocket)
            is_rocket = rvol > config.WHALE_RVOL_THRESHOLD
            
            # 2. Stealth Accumulation
            atr = volatility # approximated by std dev of returns * price? 
            # Better re-calc precise ATR or use passed logic. 
            # Let's use the local volatility as a proxy or calculate ATR properly if needed.
            # We will use the 'volatility' calculated later (std dev of returns) for now.
            # Actually, let's calc simple range-based volatility here for 'detect_accumulation'
            tr = (window_data['high'] - window_data['low']).iloc[-1]
            avg_tr = (window_data['high'] - window_data['low']).rolling(20).mean().iloc[-1]
            
            is_accumulation = self.detect_accumulation(window_data, tr, avg_tr)
            
            # 3. Whale Defense
            is_defense = self.detect_whale_defense(structure_ctx, window_data)
            
            # 4. Front-Running (Book & Funding)
            book_ratio = self.calculate_book_pressure(book_data)
            is_book_skewed = book_ratio > config.WHALE_ORDER_IMBALANCE_RATIO
            
            is_squeeze = self.detect_short_squeeze(funding_rate, macro_trend)
            
            # Combine Signals
            whale_reason = []
            if is_rocket: whale_reason.append("ROCKET")
            if is_accumulation: whale_reason.append("ACCUMULATION")
            if is_defense: whale_reason.append("DEFENSE")
            if is_book_skewed: whale_reason.append(f"BID_WALL({book_ratio:.1f})")
            if is_squeeze: whale_reason.append("SQUEEZE")
            
            is_whale = bool(whale_reason) # True if any whale signal found


            if macro_trend == 'BULLISH':
                if last_state != 'BULLISH':
                     self._safe_print(f"[{self.name}] 🌊 MACRO FLOW (1H): {symbol} Turned BULLISH. (Restrictions Disabled)")
                     self.last_macro_state[symbol] = 'BULLISH'
                
                # RESTRICTION LOGIC DISABLED
                can_short = True 
                
            elif macro_trend == 'BEARISH':
                 if last_state != 'BEARISH':
                     self._safe_print(f"[{self.name}] 🌊 MACRO FLOW (1H): {symbol} Turned BEARISH. (Restrictions Disabled)")
                     self.last_macro_state[symbol] = 'BEARISH'
                 
                 # RESTRICTION LOGIC DISABLED
                 can_long = True
        else:
             can_long = True
             can_short = True
             
        # --- PATCH 6: POLYMARKET PATIENCE (The Wait) ---
        # If we are in the first 3 minutes of a 15m candle, we BLOCK standard signals.
        # This filters out the initial "Fakeout" noise.
        # We assume 'minutes_into_candle' is passed in structure_ctx
        minutes_in = structure_ctx.get('minutes_into_candle', 10) if structure_ctx else 10
        if minutes_in < config.POLYMARKET_PATIENCE_MINUTES and self.crisis_score < 0.8:
             # self._safe_print(f"[{self.name}] ⏳ PATIENCE: Waiting for candle settlement ({minutes_in}m < {config.POLYMARKET_PATIENCE_MINUTES}m).")
             # return None # STAND ASIDE - DISABLED PER USER REQUEST
             pass
             
        # --------------------------------------------------
        # --------------------------------------------------
        prices = window_data['close']
        current_price = float(prices.iloc[-1])
        
        # Volatility & Momentum (Already calculated above)
        # returns = prices.pct_change()
        # volatility = returns.rolling(14).std().iloc[-1]
        
        # 1. Feature Engineering (Remaining)
        # RSI (14) - Already calculated above
        # delta = prices.diff()
        # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        # Ensure rsi variable is available even if we took the sector_override path
        if sector_override:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        # BB %B
        bb_pct_b = (current_price - bb_vals['lower']) / (bb_vals['upper'] - bb_vals['lower']) if (bb_vals['upper'] - bb_vals['lower']) != 0 else 0.5
        
        # MACD (12, 26, 9)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal).iloc[-1]
        
        # Volatility & Momentum
        # returns = prices.pct_change()
        # volatility = returns.rolling(14).std().iloc[-1]
        
        # 2. Hierarchical Inference (Monolith-V4)
        # 2. Hierarchical Inference (Monolith-V4)
        # === WHALE TRACKING (Volume Physics) ===
        # Already calculated above or need calc
        if sector_override:
             vol_avg = window_data['volume'].rolling(window=20).mean().iloc[-1]
             vol_curr = window_data['volume'].iloc[-1]
             rvol = vol_curr / vol_avg if vol_avg > 0 else 1.0
             is_whale = rvol > config.WHALE_RVOL_THRESHOLD

        lstm_prob = self.predict_trend_lstm(prices, symbol)
        
        xgb_features = {
            'lstm_prob': lstm_prob,
            'rsi': rsi,
            'bb_pct_b': bb_pct_b,
            'macd_hist': macd_hist,
            'volatility': volatility
        }
        xgb_prob = self.predict_trend_xgboost(xgb_features)
        
        # --- ENTROPY INTEGRATION (Chaotic Dampening) ---
        # If the market is CHAOTIC (High Entropy), we reduce our confidence in the ML models.
        entropy_regime = structure_ctx.get('entropy_regime', 'TRANSITION') if structure_ctx else 'TRANSITION'
        
        if entropy_regime == 'CHAOTIC':
            # Dampen probabilities towards 0.5 (Uncertainty) by 50%
            original_xgb = xgb_prob
            xgb_prob = 0.5 + (xgb_prob - 0.5) * 0.5
            lstm_prob = 0.5 + (lstm_prob - 0.5) * 0.5
            self._safe_print(f"[{self.name}] 🌪️ CHAOS DETECTED ({symbol}): Dampening Confidence (XGB {original_xgb:.2f}->{xgb_prob:.2f})")
        # -----------------------------------------------

        # Store for GUI/Logging
        self.last_probes[symbol] = {
            'lstm': lstm_prob,
            'xgb': xgb_prob
        }
        
        # Final Decision from Master Brain (XGBoost)
        is_bullish = xgb_prob > config.STRATEGY_XGB_THRESHOLD
        high_conv_bullish = xgb_prob > 0.7
        high_conv_bearish = xgb_prob < 0.3
        
        kalman_price = self.get_kalman_estimate(symbol, window_data)
        market_bias = self.get_market_bias()
        is_market_bullish = market_bias >= config.GMB_THRESHOLD
        
        # Logging Consensus (Internal Diagnostic)
        if lstm_prob > 0.6 or xgb_prob > 0.6:
            self._safe_print(f"[{self.name}] Ensemble Check {symbol}: LSTM({lstm_prob:.2f}) XGB({xgb_prob:.2f})")

        # 8. LOG & EXECUTE
        # 8. LOG & EXECUTE (UNIFIED STRATEGY - "UNLEASHED")
        # Merging Scavenger and Predator Logic to allow High Value Trades regardless of mode.
        
        # --- BULLISH TRIGGERS ---
        is_below_middle = current_price <= bb_vals['middle']
        is_oversold = rsi < config.STRATEGY_RSI_ENTRY_MAX
        is_panic_buy = rsi < config.STRATEGY_RSI_PANIC_BUY # e.g. < 40
        
        # Trigger A: Trend Following (High Conviction)
        # Buy if Model is Bullish AND Market Bias is Supportive (or we have Physics Override)
        # Note: We relax GMB check if Whale is present
        trigger_trend_buy = is_bullish and (is_market_bullish or is_whale) and can_long
        
        # Trigger B: Mean Reversion (Dip Buying)
        # Buy if Price is low (Below Middle) AND RSI is safe
        trigger_dip_buy = is_below_middle and is_oversold and can_long
        
        # Trigger C: Panic Buy (Falling Knife Catch)
        # Buy if RSI is extreme (ignore other filters)
        trigger_panic_buy = is_panic_buy and can_long

        if (trigger_trend_buy or trigger_dip_buy or trigger_panic_buy):
             # VALIDATION: Kalman Check (Relaxed)
             # Allow if Price < Kalman (Value) OR if High Conviction Bullish (Momentum)
             # OR if it's a Panic Buy (Value is extreme)
             if current_price < kalman_price or high_conv_bullish or trigger_panic_buy:
                 reason = "TREND" if trigger_trend_buy else ("DIP" if trigger_dip_buy else "PANIC")
                 if is_whale:
                     reason = f"WHALE_{whale_reason[0]}" # Override reason with primary whale driver
                     self._safe_print(f"[{self.name}] 🐋 WHALE SIGHTING: {symbol} - {', '.join(whale_reason)}")

                 self._safe_print(f"[{self.name}] 🚀 {symbol} BUY SIGNAL ({reason}) | XGB:{xgb_prob:.2f} GMB:{market_bias:.2f}")
                 
                 
                 meta = {'is_whale': is_whale, 'whale_factors': whale_reason, 'structure': structure_ctx, 'reason': reason}
                 sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata=meta)
                 return self.apply_asset_personality(symbol, sig)

        # --- BEARISH TRIGGERS ---
        is_above_middle = current_price >= bb_vals['middle']
        is_overbought = rsi > 65
        is_panic_sell = rsi > 75
        
        # Trigger A: Trend Shorting
        trigger_trend_sell = (not is_bullish) and (not is_market_bullish or is_whale) and can_short
        
        # Trigger B: Mean Reversion Short (Top Selling)
        trigger_top_sell = is_above_middle and is_overbought and can_short
        
        # Trigger C: Panic Sell (Blow-off Top)
        trigger_panic_sell = is_panic_sell and can_short
        
        if (trigger_trend_sell or trigger_top_sell or trigger_panic_sell):
            # VALIDATION: Kalman Check (Relaxed)
            # Allow if Price > Kalman (Premium) OR if High Conviction Bearish (Momentum)
            if current_price > kalman_price or high_conv_bearish or trigger_panic_sell:
                 reason = "TREND" if trigger_trend_sell else ("TOP" if trigger_top_sell else "PANIC")
                 self._safe_print(f"[{self.name}] 🔻 {symbol} SELL SIGNAL ({reason}) | XGB:{xgb_prob:.2f} GMB:{market_bias:.2f}")
                 
                 meta = {'is_whale': False, 'structure': structure_ctx, 'reason': reason}
                 sig = GlobalTradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata=meta)
                 return self.apply_asset_personality(symbol, sig)

        return None

    def get_health(self) -> dict:
        last_lstm = 0.5
        last_xgb = 0.5
        if self.last_probes:
            # Get the last symbol analyzed
            last_sym = list(self.last_probes.keys())[-1]
            last_lstm = self.last_probes[last_sym]['lstm']
            last_xgb = self.last_probes[last_sym]['xgb']

        return {
            'status': 'OK' if (self.model or self.xgb_model) else 'HEURISTIC',
            'lstm_loaded': self.model is not None,
            'xgb_loaded': self.xgb_model is not None,
            'last_lstm': last_lstm,
            'last_xgb': last_xgb
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass

    def _calculate_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bill Williams 5-Bar Fractal Generation.
        Fractal Up: High[i] > High[i-2, i-1, i+1, i+2]
        Fractal Down: Low[i] < Low[i-2, i-1, i+1, i+2]
        """
        if len(df) < 5: return df
        
        # We need a copy to avoid SettingWithCopy warnings on the main DF
        # Actually, we usually want to modify the DF being analyzed.
        
        # Using verifying logic with shifting (Assuming future data is not available, we trigger these on lag)
        # But for historical analysis (df provided), we can look ahead.
        # Note: In live trading, the fractal at T is only confirmed at T+2.
        
        # Highs
        highs = df['high']
        df['fractal_high'] = (highs > highs.shift(1)) & \
                             (highs > highs.shift(2)) & \
                             (highs > highs.shift(-1)) & \
                             (highs > highs.shift(-2))

        # Lows
        lows = df['low']
        df['fractal_low'] = (lows < lows.shift(1)) & \
                            (lows < lows.shift(2)) & \
                            (lows < lows.shift(-1)) & \
                            (lows < lows.shift(-2))
                            
        return df

    def get_structural_context(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Finds the nearest Structural Support (Lower Fractal) and Resistance (Upper Fractal).
        Returns distances and levels.
        """
        if df.empty or 'fractal_high' not in df.columns:
            df = self._calculate_fractals(df)
            
        # Scan backwards for the last CONFIRMED fractals
        # Note: shift(-2) means we lose the last 2 bars of fractal data.
        # So valid fractals are from index 0 to -3.
        
        valid_df = df.iloc[:-2] # Exclude unconfirmed
        
        last_resistance = valid_df[valid_df['fractal_high']]['high'].iloc[-1] if valid_df['fractal_high'].any() else current_price * 2
        last_support = valid_df[valid_df['fractal_low']]['low'].iloc[-1] if valid_df['fractal_low'].any() else current_price * 0.5
        
        dist_to_res_pct = (last_resistance - current_price) / current_price
        dist_to_sup_pct = (current_price - last_support) / current_price
        
        structure = "RANGE"
        if current_price > last_resistance: structure = "BREAKOUT_UP"
        elif current_price < last_support: structure = "BREAKDOWN_DOWN"
        
        return {
            'resistance_price': last_resistance,
            'support_price': last_support,
            'dist_to_res_pct': dist_to_res_pct,
            'dist_to_sup_pct': dist_to_sup_pct,
            'structure_mode': structure
        }

    def profile_asset_class(self, symbol: str, ticker_data: dict) -> str:
        """
        The Scout's Eye: Classifies an asset based on lightweight Ticker Data.
        Returns: 'ANCHOR', 'ROCKET', or 'DEAD'
        """
        if not ticker_data: return 'DEAD'
        
        # Extract Metrics
        try:
            pct_change = float(ticker_data.get('percentage', 0.0))
            quote_vol = float(ticker_data.get('quoteVolume', 0.0)) # USDT Volume
            
            # 1. ROCKET CHECK (High Energy)
            # Volume > $10M and Move > 5% (Adjust thresholds as needed)
            if quote_vol > 5_000_000 and abs(pct_change) > 5.0:
                 return 'ROCKET'
                 
            # 2. ANCHOR CHECK (Steady Flow)
            # Volume > $50M (Liquid) and Move < 5% (Stable)
            if quote_vol > 50_000_000 and abs(pct_change) < 5.0:
                return 'ANCHOR'
                
            return 'DEAD' # Not interesting enough for the Active List
        except:
            return 'DEAD'
