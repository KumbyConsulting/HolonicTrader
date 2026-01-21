use rayon::prelude::*;
use crate::trading_loop::TradingLoop;

/// Walk-Forward Window Configuration
#[derive(Clone, Debug)]
pub struct WFOWindow {
    pub start_idx: usize,
    pub train_end_idx: usize,
    pub test_end_idx: usize,
}

/// Walk-Forward Result for a single window
#[derive(Clone, Debug)]
pub struct WFOResult {
    pub window_id: usize,
    pub train_pnl: f64,
    pub test_pnl: f64,
    pub final_balance: f64,
}

/// Run a single WFO window (used by parallel executor)
fn run_single_window(
    window: &WFOWindow,
    window_id: usize,
    timestamps: &[i64],
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    rsi: &[f64],
    bb_lower: &[f64],
    bb_upper: &[f64],
    obv_slopes: &[f64],
    entropy_scores: &[f64],
    initial_capital: f64,
) -> WFOResult {
    let mut loop_state = TradingLoop::new(initial_capital);
    
    // Run backtest on test window only (train_end to test_end)
    let test_balance = loop_state.run_backtest(
        &timestamps[window.train_end_idx..window.test_end_idx],
        &opens[window.train_end_idx..window.test_end_idx],
        &highs[window.train_end_idx..window.test_end_idx],
        &lows[window.train_end_idx..window.test_end_idx],
        &closes[window.train_end_idx..window.test_end_idx],
        &rsi[window.train_end_idx..window.test_end_idx],
        &bb_lower[window.train_end_idx..window.test_end_idx],
        &bb_upper[window.train_end_idx..window.test_end_idx],
        &obv_slopes[window.train_end_idx..window.test_end_idx],
        &entropy_scores[window.train_end_idx..window.test_end_idx],
        "SCAVENGER",
    );

    WFOResult {
        window_id,
        train_pnl: 0.0,  // Training not implemented in this POC
        test_pnl: (test_balance / initial_capital - 1.0) * 100.0,
        final_balance: test_balance,
    }
}

/// Run Parallel Walk-Forward Optimization
/// Uses Rayon to process multiple windows concurrently
pub fn run_parallel_wfo(
    windows: Vec<WFOWindow>,
    timestamps: &[i64],
    opens: &[f64],
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    rsi: &[f64],
    bb_lower: &[f64],
    bb_upper: &[f64],
    obv_slopes: &[f64],
    entropy_scores: &[f64],
    initial_capital: f64,
) -> Vec<WFOResult> {
    windows
        .par_iter()  // Parallel Iterator
        .enumerate()
        .map(|(id, window)| {
            run_single_window(
                window,
                id,
                timestamps,
                opens,
                highs,
                lows,
                closes,
                rsi,
                bb_lower,
                bb_upper,
                obv_slopes,
                entropy_scores,
                initial_capital,
            )
        })
        .collect()
}
