import numpy as np
import pandas as pd
import logging
import sqlite3
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger("AssetProfiler")

class AssetProfiler:
    def __init__(self, db_path: str = 'holonic_trader.db'):
        self.db_path = db_path
        self.meme_list = ['PEPE', 'SHIB', 'BONK', 'WIF', 'FLOKI', 'DOGE']
        self.large_cap = ['BTC', 'ETH']
        self.mid_cap = ['SOL', 'AVAX', 'LINK', 'UNI', 'AAVE', 'BNB', 'NEAR', 'SUI']
        
    def classify_category(self, symbol: str) -> str:
        base = symbol.split('/')[0]
        if base in self.large_cap: return 'large_cap'
        if base in self.mid_cap: return 'mid_cap'
        if base in self.meme_list: return 'meme_coin'
        return 'alt_coin'

    def calculate_profile(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action to determine volatility regime and risk parameters."""
        if df is None or len(df) < 50:
            return {'symbol': symbol, 'category': self.classify_category(symbol), 'regime': 'UNKNOWN'}
            
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(365 * 24) # Annualized
        
        # Determine Regime
        if volatility > 1.5: regime = 'CHAOTIC'
        elif volatility > 0.8: regime = 'HIGH_VOL'
        elif volatility > 0.3: regime = 'NORMAL'
        else: regime = 'LOW_VOL'
        
        avg_range = (df['high'] - df['low']).mean() / df['close'].mean()
        
        profile = {
            'symbol': symbol,
            'category': self.classify_category(symbol),
            'volatility': float(volatility),
            'regime': regime,
            'avg_daily_range': float(avg_range),
            'max_drawdown_window': float(self._calculate_max_dd(df)),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Generate optimal seeds
        profile['recommended_params'] = self.generate_params(profile)
        
        self.save_profile(profile)
        return profile

    def _calculate_max_dd(self, df: pd.DataFrame) -> float:
        peak = df['close'].expanding(min_periods=1).max()
        dd = (df['close'] - peak) / peak
        return abs(dd.min())

    def generate_params(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        category = profile['category']
        regime = profile['regime']
        
        templates = {
            'large_cap': {
                'rsi_buy': 30.0, 'rsi_sell': 75.0, 'stop_loss': 0.03, 'take_profit': 0.10,
                'trailing_activation': 1.0, 'trailing_distance': 0.02, 'leverage_cap': 2.0
            },
            'mid_cap': {
                'rsi_buy': 25.0, 'rsi_sell': 80.0, 'stop_loss': 0.06, 'take_profit': 0.20,
                'trailing_activation': 1.5, 'trailing_distance': 0.04, 'leverage_cap': 1.5
            },
            'meme_coin': {
                'rsi_buy': 15.0, 'rsi_sell': 70.0, 'stop_loss': 0.20, 'take_profit': 2.0,
                'trailing_activation': 3.0, 'trailing_distance': 0.10, 'leverage_cap': 1.0
            },
            'alt_coin': {
                'rsi_buy': 20.0, 'rsi_sell': 80.0, 'stop_loss': 0.10, 'take_profit': 0.50,
                'trailing_activation': 2.0, 'trailing_distance': 0.05, 'leverage_cap': 1.2
            }
        }
        
        params = templates.get(category, templates['alt_coin']).copy()
        
        # Volatility Scaling
        if regime == 'CHAOTIC':
            params['stop_loss'] *= 1.5
            params['leverage_cap'] *= 0.5
        elif regime == 'LOW_VOL':
            params['stop_loss'] *= 0.7
            params['leverage_cap'] *= 1.5
            
        return params

    def save_profile(self, profile: Dict[str, Any]):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS asset_profiles 
                         (symbol TEXT PRIMARY KEY, category TEXT, regime TEXT, 
                          volatility REAL, data TEXT)''')
            
            c.execute("REPLACE INTO asset_profiles VALUES (?,?,?,?,?)",
                      (profile['symbol'], profile['category'], profile['regime'], 
                       profile['volatility'], json.dumps(profile)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save asset profile: {e}")

    def get_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT data FROM asset_profiles WHERE symbol=?", (symbol,))
            row = c.fetchone()
            conn.close()
            return json.loads(row[0]) if row else None
        except:
            return None
