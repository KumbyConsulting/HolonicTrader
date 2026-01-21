/// 1D Kalman Filter for Price Estimation
/// 
/// Model:
/// x_k = x_{k-1} + w_k  (Process noise w ~ N(0, Q))
/// z_k = x_k + v_k      (Measurement noise v ~ N(0, R))

#[derive(Clone, Debug)]
pub struct KalmanFilter1D {
    /// Process Noise Covariance (System volatility)
    pub q: f64,
    /// Measurement Noise Covariance (Market noise)
    pub r: f64,
    /// Estimation Error Covariance
    pub p: f64,
    /// State Estimate
    pub x: f64,
    /// Initialization flag
    initialized: bool,
}

impl KalmanFilter1D {
    /// Create a new Kalman Filter
    pub fn new(process_noise: f64, measurement_noise: f64, estimated_error: f64) -> Self {
        KalmanFilter1D {
            q: process_noise,
            r: measurement_noise,
            p: estimated_error,
            x: 0.0,
            initialized: false,
        }
    }

    /// Default configuration
    pub fn default() -> Self {
        Self::new(0.01, 0.1, 1.0)
    }

    /// Perform one update step (Predict + Update)
    /// Returns the filtered state estimate
    pub fn update(&mut self, measurement: f64) -> f64 {
        if !self.initialized {
            self.x = measurement;
            self.initialized = true;
            return self.x;
        }

        // 1. Prediction Step
        let x_pred = self.x;
        let p_pred = self.p + self.q;

        // 2. Update Step
        // Kalman Gain: K = P_pred / (P_pred + R)
        let k = p_pred / (p_pred + self.r);

        // Update State: x = x_pred + K * (measurement - x_pred)
        self.x = x_pred + k * (measurement - x_pred);

        // Update Covariance: P = (1 - K) * P_pred
        self.p = (1.0 - k) * p_pred;

        self.x
    }

    /// Process a batch of measurements
    pub fn update_batch(&mut self, measurements: &[f64]) -> Vec<f64> {
        measurements.iter().map(|&m| self.update(m)).collect()
    }

    /// Get current estimate without updating
    pub fn get_estimate(&self) -> f64 {
        self.x
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.p = 1.0;
        self.initialized = false;
    }
}

/// Holon Disposition (Autonomy/Integration balance)
#[derive(Clone, Debug)]
pub struct Disposition {
    pub autonomy: f64,
    pub integration: f64,
}

impl Disposition {
    pub fn new(autonomy: f64, integration: f64) -> Self {
        Disposition { autonomy, integration }
    }

    /// Default balanced disposition
    pub fn balanced() -> Self {
        Self::new(0.5, 0.5)
    }
}

/// Holon State
#[derive(Clone, Debug, PartialEq)]
pub enum HolonState {
    Active,
    Passive,
    Hibernate,
}

/// Base Holon struct (minimal implementation)
#[derive(Clone, Debug)]
pub struct HolonCore {
    pub name: String,
    pub disposition: Disposition,
    pub state: HolonState,
    pub reputation: f64,
}

impl HolonCore {
    pub fn new(name: &str, disposition: Disposition) -> Self {
        HolonCore {
            name: name.to_string(),
            disposition,
            state: HolonState::Active,
            reputation: 1.0,
        }
    }

    /// Update reputation and adjust autonomy accordingly
    pub fn update_reputation(&mut self, reward: f64) {
        self.reputation = (self.reputation + reward).max(0.1);

        // Performance-based autonomy scaling
        let new_autonomy = (0.8 * self.reputation).clamp(0.1, 0.95);
        self.disposition.autonomy = new_autonomy;
        self.disposition.integration = 1.0 - new_autonomy;
    }
}

// === CRYPTO UTILITIES ===

use sha2::{Sha256, Digest};

/// Compute SHA-256 hash of a string (for ledger blocks)
pub fn compute_sha256(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// Compute hash for a ledger block
pub fn compute_block_hash(
    timestamp: &str,
    entropy_score: f64,
    regime: &str,
    action: &str,
    prev_hash: &str,
) -> String {
    // Match Python's json.dumps(sort_keys=True) format
    let block_string = format!(
        r#"{{"action":"{}","entropy_score":{},"prev_hash":"{}","regime":"{}","timestamp":"{}"}}"#,
        action, entropy_score, prev_hash, regime, timestamp
    );
    compute_sha256(&block_string)
}
