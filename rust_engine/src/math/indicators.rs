use std::f64;

/// Simple Moving Average (SMA)
pub fn calculate_sma(data: &[f64], period: usize) -> Vec<f64> {
    let mut sma = vec![f64::NAN; data.len()];
    if data.len() < period { return sma; }

    let mut sum = 0.0;
    for i in 0..period {
        sum += data[i];
    }
    sma[period - 1] = sum / period as f64;

    for i in period..data.len() {
        sum += data[i] - data[i - period];
        sma[i] = sum / period as f64;
    }
    sma
}

/// Bollinger Bands
/// Returns (Upper, Middle, Lower)
pub fn calculate_bollinger_bands(data: &[f64], period: usize, std_dev: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sma = calculate_sma(data, period);
    let mut upper = vec![f64::NAN; data.len()];
    let mut lower = vec![f64::NAN; data.len()];
    
    // Middle Band IS the SMA
    // We need rolling std dev.
    // Optimization: Calculate variance in the same pass? 
    // Harder with floating window. Naive loop for Std.
    
    for i in (period-1)..data.len() {
        if sma[i].is_nan() { continue; }
        
        // Variance
        let window = &data[i+1-period ..= i];
        let mean = sma[i];
        let mut variance_sum = 0.0;
        for &x in window {
            variance_sum += (x - mean).powi(2);
        }
        let std = (variance_sum / period as f64).sqrt();
        
        upper[i] = mean + (std_dev * std);
        lower[i] = mean - (std_dev * std);
    }
    
    (upper, sma, lower)
}

/// Relative Strength Index (RSI)
pub fn calculate_rsi(data: &[f64], period: usize) -> Vec<f64> {
    let mut rsi = vec![f64::NAN; data.len()];
    if data.len() < period + 1 { return rsi; }

    let mut gains = 0.0;
    let mut losses = 0.0;

    // First period Average Gain/Loss
    for i in 1..=period {
        let change = data[i] - data[i-1];
        if change > 0.0 {
            gains += change;
        } else {
            losses -= change;
        }
    }

    let mut avg_gain = gains / period as f64;
    let mut avg_loss = losses / period as f64;

    if avg_loss == 0.0 {
        rsi[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    // Wilder's Smoothing
    for i in (period + 1)..data.len() {
        let change = data[i] - data[i-1];
        let mut current_gain = 0.0;
        let mut current_loss = 0.0;
        
        if change > 0.0 { current_gain = change; }
        else { current_loss = -change; }
        
        avg_gain = ((avg_gain * (period as f64 - 1.0)) + current_gain) / period as f64;
        avg_loss = ((avg_loss * (period as f64 - 1.0)) + current_loss) / period as f64;
        
        if avg_loss == 0.0 {
            rsi[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    rsi
}

/// Average True Range (ATR)
pub fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let len = high.len();
    let mut atr = vec![f64::NAN; len];
    if len < period + 1 { return atr; }

    let mut tr_sum = 0.0;
    
    // First TRs
    let mut trs = Vec::with_capacity(len);
    trs.push(high[0] - low[0]); // First candle TR is H-L
    
    for i in 1..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i-1]).abs();
        let lc = (low[i] - close[i-1]).abs();
        let tr = hl.max(hc).max(lc);
        trs.push(tr);
    }
    
    // First ATR is SimpleMA of TR
    for i in 0..period {
        tr_sum += trs[i];
    }
    
    let mut prev_atr = tr_sum / period as f64;
    atr[period-1] = prev_atr; // Index aligned? 
    // Standard python aligns at period-1 or period? Usually period-1 if we include current.
    // Let's stick to having first val at index `period-1`.
    
    // Wilder's Smoothing for subsequent
    for i in period..len {
        let current_tr = trs[i];
        let current_atr = ((prev_atr * (period as f64 - 1.0)) + current_tr) / period as f64;
        atr[i] = current_atr;
        prev_atr = current_atr;
    }
    
    atr
}
