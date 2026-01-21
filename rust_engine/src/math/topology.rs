use std::f64;

/// Topological Holon Logic
/// Calculates 0-dimensional Persistent Homology (Connected Components)
/// on a Time-Delay Embedding (Takens' Theorem).
/// 
/// Returns a "Betti Score" representing topological complexity.
/// Low score = Collapse to lower dimension (Crisis/Crash).
/// High score = Healthy complex topology.

pub fn calculate_persistent_entropy(data: &[f64], window_size: usize, dimension: usize, delay: usize) -> f64 {
    let n = data.len();
    if n < window_size { return 0.0; }

    // 1. Convert to returns (percentage change) for scale-invariance
    // This ensures BTC at $90k and ADA at $0.35 produce comparable scores
    let start_idx = n - window_size;
    let raw_window = &data[start_idx..];
    
    // Calculate log returns for better normalization (no scaling)
    let returns: Vec<f64> = raw_window.windows(2)
        .map(|w| if w[0] > 0.0 { (w[1] / w[0]).ln() } else { 0.0 })
        .collect();
    
    if returns.len() < (dimension - 1) * delay + 2 { return 0.0; }
    
    // 2. Point Cloud Generation (Takens' Embedding) on returns
    let num_points = returns.len() - (dimension - 1) * delay;
    if num_points < 2 { return 0.0; }

    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let mut point = Vec::with_capacity(dimension);
        for d in 0..dimension {
            point.push(returns[i + d * delay]);
        }
        points.push(point);
    }

    // 3. Simplified Persistence: Euclidean MST (Minimum Spanning Tree)
    let mst_total_weight = prim_mst_weight(&points);
    
    // 4. Normalize by expected range for returns-based embedding
    // Healthy markets: varied returns -> spread-out embedding -> larger MST
    // Crash precursor: uniform returns -> clustered embedding -> smaller MST
    let complexity_score = mst_total_weight / (num_points as f64);
    
    // Clamp to reasonable range (0 to 2.0, typical healthy is 0.3-0.8)
    complexity_score.min(2.0).max(0.0)
}

fn prim_mst_weight(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    let mut min_dists = vec![f64::MAX; n];
    let mut in_mst = vec![false; n];
    let mut total_weight = 0.0;
    
    // Start with point 0
    min_dists[0] = 0.0;
    
    for _ in 0..n {
        // Find vertex u not in MST with min dist
        let mut u = usize::MAX;
        let mut min_val = f64::MAX;
        
        for i in 0..n {
            if !in_mst[i] && min_dists[i] < min_val {
                min_val = min_dists[i];
                u = i;
            }
        }
        
        if u == usize::MAX { break; } // Disconnected? Shouldn't happen in complete graph
        
        in_mst[u] = true;
        total_weight += min_val;
        
        // Update neighbors (all other nodes v)
        for v in 0..n {
            if !in_mst[v] {
                let dist = euclidean_dist(&points[u], &points[v]);
                if dist < min_dists[v] {
                    min_dists[v] = dist;
                }
            }
        }
    }
    
    total_weight
}

fn euclidean_dist(p1: &[f64], p2: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for (a, b) in p1.iter().zip(p2.iter()) {
        sum_sq += (a - b).powi(2);
    }
    sum_sq.sqrt()
}
