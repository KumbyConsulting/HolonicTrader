use std::collections::HashMap;

/// Configuration for the Governor
#[derive(Clone, Debug)]
pub struct GovernorConfig {
    pub initial_balance: f64,
    pub principal: f64,           // Principal to protect (e.g., $10)
    pub max_exposure_ratio: f64,  // Max total exposure as multiple of equity
    pub max_leverage: f64,        // Max leverage per position
    pub base_risk_pct: f64,       // Base risk per trade (e.g., 0.02 = 2%)
    pub drawdown_lock_pct: f64,   // Drawdown % to trigger lock
}

impl Default for GovernorConfig {
    fn default() -> Self {
        GovernorConfig {
            initial_balance: 10.0,
            principal: 10.0,
            max_exposure_ratio: 10.0,  // 10x
            max_leverage: 20.0,        // 20x
            base_risk_pct: 0.02,       // 2% per trade
            drawdown_lock_pct: 0.05,   // 5% drawdown lock
        }
    }
}

/// Tracks an open position
#[derive(Clone, Debug)]
pub struct Position {
    pub symbol: String,
    pub direction: i8,       // 1 = Long, -1 = Short
    pub entry_price: f64,
    pub quantity: f64,
    pub stop_loss: Option<f64>,
}

/// The Governor State Machine
#[derive(Clone, Debug)]
pub struct GovernorState {
    pub balance: f64,
    pub equity: f64,
    pub high_water_mark: f64,
    pub positions: HashMap<String, Position>,
    pub config: GovernorConfig,
    pub is_locked: bool,      // Drawdown Lock
    pub reference_atr: f64,
}

impl GovernorState {
    pub fn new(config: GovernorConfig) -> Self {
        GovernorState {
            balance: config.initial_balance,
            equity: config.initial_balance,
            high_water_mark: config.initial_balance,
            positions: HashMap::new(),
            config,
            is_locked: false,
            reference_atr: 0.0,
        }
    }

    /// Update equity and check for drawdown lock
    pub fn update_equity(&mut self, new_equity: f64) {
        self.equity = new_equity;
        
        // Update HWM
        if new_equity > self.high_water_mark {
            self.high_water_mark = new_equity;
        }
        
        // Drawdown Check
        let drawdown = (self.high_water_mark - new_equity) / self.high_water_mark;
        if drawdown >= self.config.drawdown_lock_pct {
            self.is_locked = true;
        }
        // Optional: Unlock logic if equity recovers
    }

    /// Minimax Constraint: Max risk per trade
    pub fn calculate_max_risk(&self) -> f64 {
        // Rule: Never risk the principal. Only risk 'house money' OR 1% of total.
        let house_money = (self.balance - self.config.principal).max(0.0);
        let one_pct_equity = self.equity * 0.01;
        
        // Take the smaller of house_money and 1% equity, but ensure at least some risk is allowed
        house_money.min(one_pct_equity).max(0.1) // Minimum $0.10 risk
    }

    /// Volatility Scalar based on ATR
    pub fn calculate_volatility_scalar(&self, current_atr: f64) -> f64 {
        if self.reference_atr <= 0.0 || current_atr <= 0.0 {
            return 1.0;
        }
        
        let scalar = self.reference_atr / current_atr;
        // Clamp to 0.5 - 2.0
        scalar.clamp(0.5, 2.0)
    }

    /// Check if current exposure allows new position
    pub fn check_exposure(&self, new_notional: f64) -> bool {
        let current_exposure = self.calculate_total_exposure();
        let max_allowed = self.equity * self.config.max_exposure_ratio;
        
        (current_exposure + new_notional) <= max_allowed
    }
    
    /// Calculate total notional exposure
    pub fn calculate_total_exposure(&self) -> f64 {
        self.positions.values()
            .map(|p| p.quantity * p.entry_price)
            .sum()
    }

    /// Calculate position size
    pub fn calculate_position_size(
        &self,
        asset_price: f64,
        current_atr: f64,
        conviction: f64,    // 0.0 - 1.0
    ) -> f64 {
        if self.is_locked || asset_price <= 0.0 {
            return 0.0;
        }

        // 1. Max Risk
        let max_risk_usd = self.calculate_max_risk();
        
        // 2. Volatility Scalar
        let vol_scalar = self.calculate_volatility_scalar(current_atr);
        
        // 3. Conviction Scalar (0.5 to 1.5 based on 0.0-1.0 input)
        let conviction_scalar = 0.5 + conviction; // e.g., conviction=0.5 -> 1.0x
        
        // 4. Risk-Adjusted Size
        let risk_adjusted = max_risk_usd * vol_scalar * conviction_scalar;
        
        // 5. Convert to Quantity
        // Assuming stop distance is ~2 ATR for sizing
        let stop_distance_pct = (2.0 * current_atr) / asset_price;
        if stop_distance_pct <= 0.0 {
             return 0.0;
        }
        
        let position_value = risk_adjusted / stop_distance_pct;
        let quantity = position_value / asset_price;
        
        // 6. Apply Micro Guard Rail (Max Notional Check)
        let notional = quantity * asset_price;
        let max_notional = self.equity * 1.5; // 150% of NAV
        
        if notional > max_notional {
            return max_notional / asset_price;
        }
        
        quantity
    }

    /// Check Cluster Risk (same family)
    pub fn check_cluster_risk(&self, symbol: &str) -> bool {
        // Simple family grouping
        let families: Vec<(&str, Vec<&str>)> = vec![
            ("BTC", vec!["BTC/USDT", "TBTC/USDT"]),
            ("ETH", vec!["ETH/USDT"]),
            ("ALTS", vec!["XRP/USDT", "ADA/USDT", "DOGE/USDT", "SOL/USDT"]),
        ];
        
        // Find symbol's family
        let mut symbol_family: Option<&str> = None;
        for (family, members) in &families {
            if members.contains(&symbol) {
                symbol_family = Some(family);
                break;
            }
        }
        
        if symbol_family.is_none() {
            return true; // Unknown symbol, allow
        }
        
        // Check if we hold any other symbol from same family
        for (held_symbol, _) in &self.positions {
            if held_symbol == symbol {
                continue; // Same symbol, not cluster
            }
            for (family, members) in &families {
                if *family == symbol_family.unwrap() && members.contains(&held_symbol.as_str()) {
                    return false; // Cluster risk detected!
                }
            }
        }
        
        true // Safe
    }
}
