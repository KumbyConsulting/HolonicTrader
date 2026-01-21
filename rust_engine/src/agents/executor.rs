use std::collections::HashMap;

/// Trade Direction
#[derive(Clone, Debug, PartialEq)]
pub enum Direction {
    Buy,
    Sell,
}

/// Trade Action (Result of decision)
#[derive(Clone, Debug, PartialEq)]
pub enum TradeAction {
    Execute,
    Halt,
    Reduce,
}

/// Market Regime
#[derive(Clone, Debug, PartialEq)]
pub enum Regime {
    Ordered,
    Chaotic,
    Transition,
}

/// A Trade Signal
#[derive(Clone, Debug)]
pub struct TradeSignal {
    pub symbol: String,
    pub direction: Direction,
    pub size: f64,
    pub price: f64,
    pub conviction: f64,
    pub stop_loss_price: Option<f64>,
}

/// A Trade Decision (Result of decide_trade)
#[derive(Clone, Debug)]
pub struct TradeDecision {
    pub action: TradeAction,
    pub adjusted_size: f64,
    pub entropy_score: f64,
    pub regime: Regime,
}

/// Executor State Machine
#[derive(Clone, Debug)]
pub struct ExecutorState {
    pub balance_usd: f64,
    pub held_assets: HashMap<String, f64>,  // Symbol -> Quantity
    pub entry_prices: HashMap<String, f64>, // Symbol -> Entry Price
    pub total_trades: u64,
    pub winning_trades: u64,
}

impl ExecutorState {
    pub fn new(initial_capital: f64) -> Self {
        ExecutorState {
            balance_usd: initial_capital,
            held_assets: HashMap::new(),
            entry_prices: HashMap::new(),
            total_trades: 0,
            winning_trades: 0,
        }
    }

    /// Decide whether to execute a trade based on entropy and regime
    /// Returns TradeDecision with adjusted size
    pub fn decide_trade(
        &self,
        signal: &TradeSignal,
        entropy_score: f64,
        regime: &Regime,
    ) -> TradeDecision {
        // Disposition Logic based on regime
        let (action, size_multiplier) = match regime {
            Regime::Chaotic => {
                // HALT in chaos - too risky
                (TradeAction::Halt, 0.0)
            }
            Regime::Transition => {
                // REDUCE in transition - partial execution
                (TradeAction::Reduce, 0.5)
            }
            Regime::Ordered => {
                // EXECUTE in order - full confidence
                (TradeAction::Execute, 1.0)
            }
        };

        // Entropy-based fine-tuning
        // High entropy (>2.0) = more uncertainty = reduce more
        let entropy_factor = if entropy_score > 2.0 {
            0.8
        } else if entropy_score < 1.0 {
            1.2 // Very low entropy = high confidence
        } else {
            1.0
        };

        let final_size = signal.size * size_multiplier * entropy_factor;

        TradeDecision {
            action,
            adjusted_size: final_size.max(0.0),
            entropy_score,
            regime: regime.clone(),
        }
    }

    /// Calculate current portfolio value
    pub fn calculate_equity(&self, prices: &HashMap<String, f64>) -> f64 {
        let mut equity = self.balance_usd;
        
        for (symbol, qty) in &self.held_assets {
            if let Some(&price) = prices.get(symbol) {
                equity += qty * price;
            }
        }
        
        equity
    }

    /// Record a completed trade
    pub fn record_trade(&mut self, pnl: f64) {
        self.total_trades += 1;
        if pnl > 0.0 {
            self.winning_trades += 1;
        }
    }

    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.5; // Default neutral
        }
        self.winning_trades as f64 / self.total_trades as f64
    }
}
