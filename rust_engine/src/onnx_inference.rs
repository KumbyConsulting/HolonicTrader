use ort::{GraphOptimizationLevel, Session};
use std::path::Path;

/// ONNX Model Wrapper for LSTM Trend Prediction
pub struct OnnxPredictor {
    session: Session,
}

impl OnnxPredictor {
    /// Load an ONNX model from file path
    pub fn new(model_path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load model: {}", e))?;
        
        Ok(OnnxPredictor { session })
    }

    /// Predict trend probability from price sequence
    /// Input: Last N prices (e.g., 20 closing prices)
    /// Output: Probability of uptrend (0.0 to 1.0)
    pub fn predict(&self, prices: &[f32]) -> Result<f32, String> {
        // Normalize prices (simple min-max for now)
        let min_p = prices.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_p = prices.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_p - min_p;
        
        let normalized: Vec<f32> = if range > 0.0 {
            prices.iter().map(|p| (p - min_p) / range).collect()
        } else {
            vec![0.5; prices.len()]
        };

        // Create input tensor [1, sequence_length, 1] for LSTM
        let seq_len = normalized.len();
        let input_data: Vec<f32> = normalized;

        // Run inference
        let outputs = self.session
            .run(ort::inputs!["input" => ([1, seq_len, 1], input_data.as_slice())]
                .map_err(|e| format!("Failed to create inputs: {}", e))?)
            .map_err(|e| format!("Inference failed: {}", e))?;

        // Extract output probability
        let output = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract output: {}", e))?;
        
        let prob = output.view().iter().next().cloned().unwrap_or(0.5);
        
        Ok(prob)
    }
}

/// Batch prediction for multiple sequences
pub fn predict_batch(model: &OnnxPredictor, sequences: &[Vec<f32>]) -> Vec<f32> {
    sequences
        .iter()
        .map(|seq| model.predict(seq).unwrap_or(0.5))
        .collect()
}
