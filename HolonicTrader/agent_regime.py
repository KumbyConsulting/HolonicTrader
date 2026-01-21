"""
RegimeController Holon (Phase 7)

Manages stateful capital regime graduation:
- MICRO ($0-$49): Maximum safety, minimal risk
- SMALL ($50-$249): Unlocks limited autonomy
- MEDIUM ($250-$999): Full trading capabilities

Promotion requires: Capital + Stability + Behavior Integrity
Demotion is fast and ruthless.
"""

import time
from typing import Dict, Optional, List
from collections import deque
import config


class RegimeController:
    """
    The Regime Controller is the authority on risk permissions.
    It tracks capital regime, handles promotion/demotion, and freezes trading during transitions.
    """
    
    def __init__(self):
        self.name = "RegimeController"
        
        # Current State
        # Current State
        # PATCH: Auto-detect regime on startup based on Config Capital (Paper Trading Fix)
        if config.INITIAL_CAPITAL >= 1000.0:
            self.current_regime = 'MEDIUM'
        elif config.INITIAL_CAPITAL >= 250.0:
            self.current_regime = 'SMALL'
        elif config.INITIAL_CAPITAL >= 50.0:
            self.current_regime = 'MICRO'
        else:
            self.current_regime = 'NANO'
            
        print(f"[{self.name}] Initialized. Auto-Detected Regime: {self.current_regime} (Cap: ${config.INITIAL_CAPITAL})")
        self.previous_regime = self.current_regime
        
        # Promotion Tracking
        self.equity_history: deque = deque(maxlen=1000)  # (timestamp, equity) tuples
        self.promotion_eligible_since: Optional[float] = None
        
        # Health Tracking
        self.health_events: deque = deque(maxlen=100)  # Track recent issues
        self.trade_count = 0
        self.solvency_rejections = 0
        self.gc_corrections = 0
        self.hwm_resets = 0
        self.avg_slippage = 0.0
        
        # Transition State
        self.transition_pending = False
        self.transition_target: Optional[str] = None
        
        # High Water Mark for Demotion
        self.peak_equity = 0.0
        
        print(f"[{self.name}] Initialized. Starting Regime: {self.current_regime}")
        
    def update_state(self, equity: float, health_metrics: Dict = None):
        """
        Called every cycle to update regime state.
        
        Args:
            equity: Current account equity
            health_metrics: Dict with keys like 'solvency_rejection', 'gc_correction', 'slippage'
        """
        now = time.time()
        
        # 1. Record Equity History
        self.equity_history.append((now, equity))
        
        # 2. Update Peak (for drawdown calculation)
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        # 3. Update Health Metrics
        if health_metrics:
            if health_metrics.get('solvency_rejection'):
                self.solvency_rejections += 1
                self.health_events.append(('solvency_rejection', now))
            if health_metrics.get('gc_correction'):
                self.gc_corrections += 1
                self.health_events.append(('gc_correction', now))
            if health_metrics.get('hwm_reset'):
                self.hwm_resets += 1
                self.health_events.append(('hwm_reset', now))
            if 'slippage' in health_metrics:
                # Rolling average
                self.avg_slippage = (self.avg_slippage * 0.9) + (health_metrics['slippage'] * 0.1)
            if health_metrics.get('trade_completed'):
                self.trade_count += 1
                
        # 4. Check for Demotion (FAST)
        demoted = self._check_demotion(equity)
        if demoted:
            return
            
        # 5. Check for Promotion (SLOW)
        self._check_promotion(equity)
        
    def _check_demotion(self, equity: float) -> bool:
        """
        Demotion is fast and ruthless.
        Returns True if demoted.
        """
        demoted = False
        target_regime = None
        reason = ""
        
        # A. Capital Below Lower Threshold
        if self.current_regime == 'SMALL' and equity < (50.0 - config.REGIME_DEMOTION_BUFFER):
            target_regime = 'MICRO'
            reason = f"Equity ${equity:.2f} < ${50.0 - config.REGIME_DEMOTION_BUFFER:.2f}"
            
        elif self.current_regime == 'MEDIUM' and equity < (250.0 - config.REGIME_DEMOTION_BUFFER):
            target_regime = 'SMALL'
            reason = f"Equity ${equity:.2f} < ${250.0 - config.REGIME_DEMOTION_BUFFER:.2f}"
            
        # B. Drawdown from Peak
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            
            # Dynamic Threshold
            limit = config.REGIME_DEMOTION_DRAWDOWN_PCT # Default 0.25
            if self.current_regime == 'VOL_WINDOW':
                limit = 0.12 # 12% Hard Stop for Vol Window
            
            if drawdown >= limit and self.current_regime != 'MICRO':
                # Demote to MICRO regardless of current regime
                # If in VOL_WINDOW, we definitely go back to MICRO/NORMAL
                target_regime = 'MICRO'
                reason = f"Drawdown {drawdown*100:.1f}% >= {limit*100:.1f}%"
                    
        if target_regime:
            self._execute_transition(target_regime, reason, is_demotion=True)
            demoted = True
            
        return demoted
        
    def _check_promotion(self, equity: float):
        """
        Promotion requires 3 conditions, all true:
        A. Capital Threshold
        B. Stability Window (72h continuous)
        C. Behavior Integrity (health score >= 0.95)
        """
        now = time.time()
        
        # Determine target regime based on equity
        if self.current_regime == 'MICRO' and equity >= 50.0:
            target_regime = 'SMALL'
            threshold = 50.0
        elif self.current_regime == 'SMALL' and equity >= 250.0:
            target_regime = 'MEDIUM'
            threshold = 250.0
        else:
            # No promotion possible
            self.promotion_eligible_since = None
            return
            
        # A. Capital Threshold (already checked above)
        
        # B. Stability Window
        if self.promotion_eligible_since is None:
            self.promotion_eligible_since = now
            print(f"[{self.name}] 📈 Promotion Eligibility Started: ${equity:.2f} >= ${threshold:.2f}")
            return
            
        # Check if equity stayed above threshold for the entire window
        stability_seconds = config.REGIME_PROMOTION_STABILITY_HOURS * 3600
        if (now - self.promotion_eligible_since) < stability_seconds:
            # Not enough time yet
            hours_remaining = (stability_seconds - (now - self.promotion_eligible_since)) / 3600
            return
            
        # Verify stability: Check all equity samples in the window
        window_start = now - stability_seconds
        for ts, eq in self.equity_history:
            if ts >= window_start and eq < threshold:
                # Equity dipped below threshold during window
                self.promotion_eligible_since = None
                print(f"[{self.name}] ⚠️ Promotion Reset: Equity dipped to ${eq:.2f} during stability window.")
                return
                
        # C. Behavior Integrity
        health_score = self._calculate_health_score()
        if health_score < config.REGIME_PROMOTION_HEALTH_THRESHOLD:
            print(f"[{self.name}] ⚠️ Promotion Blocked: Health Score {health_score:.2f} < {config.REGIME_PROMOTION_HEALTH_THRESHOLD:.2f}")
            return
            
        # All conditions met - PROMOTE
        self._execute_transition(target_regime, f"Equity ${equity:.2f} stable for {config.REGIME_PROMOTION_STABILITY_HOURS}h, Health {health_score:.2f}", is_demotion=False)

    def check_vol_window_conditions(self, observer_data: Dict[str, float]) -> bool:
        """
        Check if we should enter the VOL_WINDOW High-Entropy Regime.
        Requires:
        1. BTC 24h Realized Vol > 45%
        2. Avg Funding > 0.03% (Positive)
        3. Spread < 0.4%
        4. "Meme" Listing < 14 days (Optional check, assumed checked by caller or config)
        """
        btc_vol = observer_data.get('btc_vol', 0.0)
        avg_funding = observer_data.get('avg_funding', 0.0) # 8h rate
        avg_spread = observer_data.get('avg_spread', 0.0)
        
        # 1. Vol Check
        if btc_vol < config.VOL_WINDOW_BTC_VOL_THRESHOLD:
            return False
            
        # 2. Funding Check (Positive Bullish Sentiment)
        if avg_funding < config.VOL_WINDOW_FUNDING_THRESHOLD:
            return False
            
        # 3. Spread Check (Liquidity)
        if avg_spread > config.VOL_WINDOW_SPREAD_THRESHOLD:
            return False
            
        # All Clear
        return True

    def attempt_vol_window_entry(self, observer_data: Dict[str, float]):
        """
        Public method to trigger VOL_WINDOW entry if conditions met.
        Overrules standard regimes.
        """
        if self.current_regime == 'VOL_WINDOW':
            # Check exit conditions? (Reverse logic)
            # If Vol drops OR Funding turns negative -> Exit
            if not self.check_vol_window_conditions(observer_data):
                print(f"[{self.name}] 📉 VOL_WINDOW Conditions Lost. Reverting to NORMAL.")
                # Revert to appropriate capital regime
                # For safety, go to MICRO first or calculate based on current equity
                # We simply demote to MICRO to be safe, then let standard promotion take over.
                self._execute_transition('MICRO', "VOL_WINDOW Conditions Lost", is_demotion=True)
            return

        # Check Entry
        if self.check_vol_window_conditions(observer_data):
            self._execute_transition('VOL_WINDOW', "High-Entropy Conditions Detected (Vol+Funding+Spread)", is_demotion=False)
        
    def _calculate_health_score(self) -> float:
        """
        Calculate behavior integrity score (0.0 to 1.0).
        Penalizes: Solvency rejections, GC corrections, HWM resets, high slippage.
        """
        if self.trade_count < config.REGIME_PROMOTION_MIN_TRADES:
            return 0.5  # Neutral Start (Fixes 'Coma' bug)
            
        # Start at 1.0, deduct for each issue
        score = 1.0
        
        # Recent events (last 20 trades worth of time)
        recent_window = 20 * 3600  # Rough estimate: 1h per trade
        now = time.time()
        
        recent_solvency = sum(1 for e, t in self.health_events if e == 'solvency_rejection' and now - t < recent_window)
        recent_gc = sum(1 for e, t in self.health_events if e == 'gc_correction' and now - t < recent_window)
        recent_hwm = sum(1 for e, t in self.health_events if e == 'hwm_reset' and now - t < recent_window)
        
        # Deductions
        score -= recent_solvency * 0.05
        score -= recent_gc * 0.03
        score -= recent_hwm * 0.10
        score -= min(0.10, self.avg_slippage * 10)  # 1% slippage = 0.10 deduction
        
        return max(0.0, min(1.0, score))
        
    def _execute_transition(self, target_regime: str, reason: str, is_demotion: bool):
        """
        Execute a regime transition with proper handshake.
        """
        direction = "DEMOTION" if is_demotion else "PROMOTION"
        
        print(f"\n{'='*50}")
        print(f"[{self.name}] 🔄 REGIME {direction}: {self.current_regime} → {target_regime}")
        print(f"[{self.name}] Reason: {reason}")
        print(f"{'='*50}\n")
        
        # 1. Freeze new entries
        self.transition_pending = True
        self.transition_target = target_regime
        
        # 2. Update regime
        self.previous_regime = self.current_regime
        self.current_regime = target_regime
        
        # 3. Reset promotion eligibility
        self.promotion_eligible_since = None
        
        # 4. On demotion, reset peak to current (to avoid immediate re-demotion)
        if is_demotion:
            # Give some breathing room
            pass
            
        # 5. Clear transition flag (will be handled by Governor during next consolidation)
        # The transition_pending flag tells Governor to re-evaluate all positions
        
    def complete_transition(self):
        """
        Called by Governor after consolidation is complete.
        Unlocks trading.
        """
        if self.transition_pending:
            print(f"[{self.name}] ✅ Transition Complete. Regime: {self.current_regime}")
            self.transition_pending = False
            self.transition_target = None
            
    def get_current_regime(self) -> str:
        return self.current_regime
        
    def get_permissions(self) -> Dict:
        """
        Return the permissions dict for the current regime.
        """
        # Handle VOL_WINDOW specially if not in config dict yet (it should be)
         # If not in config yet, return a hardcoded high-entropy set
        if self.current_regime == 'VOL_WINDOW':
            return {
                'max_positions': config.VOL_WINDOW_MAX_POSITIONS,
                'max_stacks': 0,
                'max_exposure_ratio': config.VOL_WINDOW_LEVERAGE,
                'max_leverage': config.VOL_WINDOW_LEVERAGE,
                'allowed_pairs': config.ALLOWED_ASSETS, # All allowed
                'correlation_check': False # Speed over safety
            }
            
        return config.REGIME_PERMISSIONS.get(self.current_regime, config.REGIME_PERMISSIONS['MICRO'])
        
    def is_transition_pending(self) -> bool:
        return self.transition_pending
        
    def get_status_summary(self) -> Dict:
        """
        Return a summary for Dashboard display.
        """
        return {
            'regime': self.current_regime,
            'peak_equity': self.peak_equity,
            'health_score': self._calculate_health_score(),
            'promotion_eligible_since': self.promotion_eligible_since,
            'transition_pending': self.transition_pending,
            'trade_count': self.trade_count,
        }

