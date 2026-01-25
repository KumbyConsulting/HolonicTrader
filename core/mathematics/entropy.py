import numpy as np
import math
from typing import List, Union

# Try to import Numba for speed, fallback to pure Python if not available
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def calculate_sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate Sample Entropy (SampleEn) of a time series.
    
    Args:
        data: Time series data (numpy array)
        m: Template length (embedding dimension)
        r: Tolerance threshold (typically 0.2 * std_dev)
        
    Returns:
        float: Sample Entropy value
    """
    N = len(data)
    if N < m + 1:
        return 0.0

    # Normalize r by standard deviation if it's a relative value
    # Note: In standard SampleEn, caller usually passes r * std.
    # We accept r as the absolute threshold here for speed inside JIT.
    
    # Pre-compute correlation counts for m and m+1
    B = 0.0
    A = 0.0
    
    # Optimized loop for B (m)
    for i in range(N - m):
        template_m = data[i : i + m]
        count = 0
        for j in range(N - m):
            if i == j:
                continue
            # Chebyshev distance (max absolute difference)
            dist = np.max(np.abs(data[j : j + m] - template_m))
            if dist <= r:
                count += 1
        B += count
        
    # Optimized loop for A (m+1) - reusing logic but extended length
    for i in range(N - m - 1):
        template_m1 = data[i : i + m + 1]
        count = 0
        for j in range(N - m - 1):
            if i == j:
                continue
            dist = np.max(np.abs(data[j : j + m + 1] - template_m1))
            if dist <= r:
                count += 1
        A += count

    # Handle edge cases to avoid log(0)
    if A == 0 or B == 0:
        return -math.log((N - m - 1) / (N - m)) # Approx for zero matches
        
    return -math.log(A / B)

def calculate_permutation_entropy(data: Union[List[float], np.ndarray], m: int = 3, delay: int = 1) -> float:
    """
    Calculate Permutation Entropy (PermEn).
    Measures the complexity of the time series based on order patterns.
    
    Args:
        data: Time series
        m: Embedding dimension (3-7 usually)
        delay: Time delay
        
    Returns:
        float: Normalized Permutation Entropy (0 to 1)
    """
    n = len(data)
    if n < m:
        return 0.0
        
    if isinstance(data, list):
        data = np.array(data)
        
    permutations = {}
    
    # Extract embedded vectors
    for i in range(n - (m - 1) * delay):
        # Create vector based on delay
        idx = [i + j * delay for j in range(m)]
        vector = data[idx]
        
        # Get ordinal pattern (argsort gives the indices that would sort the array)
        # Convert to tuple to be hashable
        perm_tuple = tuple(np.argsort(vector))
        
        if perm_tuple in permutations:
            permutations[perm_tuple] += 1
        else:
            permutations[perm_tuple] = 1
            
    # Calculate probabilities
    total_patterns = sum(permutations.values())
    probs = [count / total_patterns for count in permutations.values()]
    
    # Calculate Shannon Entropy of patterns
    pe = 0.0
    for p in probs:
        if p > 0:
            pe -= p * math.log2(p)
            
    # Normalize: PE / log2(m!)
    # m! is the number of possible permutations
    max_entropy = math.log2(math.factorial(m))
    
    if max_entropy == 0:
        return 0.0
        
    return pe / max_entropy

# Helper for standard usage
def compute_entropy_profile(closes: List[float]) -> dict:
    """
    Compute a full entropy profile for the series.
    Returns dictionary with SampleEn and PermEn.
    """
    if not closes or len(closes) < 30:
        return {'sample_entropy': 0.0, 'perm_entropy': 0.0, 'regime': 'UNKNOWN'}
        
    arr = np.array(closes)
    std_dev = np.std(arr)
    
    # Standard settings: m=2, r=0.2*std
    samp_en = calculate_sample_entropy(arr, m=2, r=0.2 * std_dev)
    
    # Standard settings: m=4 for PermEn (captures turning points well)
    perm_en = calculate_permutation_entropy(arr, m=4, delay=1)
    
    return {
        'sample_entropy': samp_en,
        'perm_entropy': perm_en
    }
