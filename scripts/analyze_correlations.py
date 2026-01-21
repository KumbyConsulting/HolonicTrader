import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'HolonicTrader', 'market_data')
TIMEFRAME = '15m'
MIN_DATA_POINTS = 1000

def load_data():
    """Load and align Closes for all assets."""
    print(f"Loading data from {DATA_DIR}...")
    close_prices = {}
    
    pattern = os.path.join(DATA_DIR, f"*_{TIMEFRAME}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("❌ No data files found!")
        return None

    for f in files:
        try:
            # Extract symbol check
            base = os.path.basename(f).split('_')[0]
            # If base ends with USDT, strip it first? Or just map it.
            # Filename: BTCUSDT_15m.csv -> base: BTCUSDT
            # Desired: BTC/USDT
            if base.endswith('USDT'):
                clean_base = base[:-4]
                symbol = f"{clean_base}/USDT"
            else:
                symbol = f"{base}/USDT"
            
            df = pd.read_csv(f)
            if 'timestamp' not in df.columns or 'close' not in df.columns:
                continue
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            close_prices[symbol] = df['close']
        except Exception as e:
            print(f"  ⚠️ Error loading {f}: {e}")

    # Combine into single DataFrame
    print("Aligning timestamps...")
    aligned_df = pd.DataFrame(close_prices)
    
    # Forward fill small gaps, drop rows with too many NaNs
    aligned_df = aligned_df.fillna(method='ffill', limit=5)
    aligned_df = aligned_df.dropna(thresh=int(len(aligned_df.columns)*0.8)) # Keep rows with 80% data
    
    print(f"Aligned Data Shape: {aligned_df.shape}")
    return aligned_df

def write_report(content):
    with open("analysis_report.md", "a", encoding='utf-8') as f:
        f.write(content + "\n")

def analyze_correlations(df):
    """Calculate correlation matrix and find notable pairs."""
    returns = df.pct_change().dropna()
    corr_matrix = returns.corr()
    
    write_report("\n## Correlation Matrix (Top Pairs)")
    
    pairs = []
    seen = set()
    
    for sym1 in corr_matrix.columns:
        for sym2 in corr_matrix.columns:
            if sym1 == sym2: continue
            key = tuple(sorted([sym1, sym2]))
            if key in seen: continue
            seen.add(key)
            score = corr_matrix.loc[sym1, sym2]
            pairs.append((sym1, sym2, score))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    write_report("\n### High Positive (Sympathetic)")
    for p in pairs[:10]:
        write_report(f"- **{p[0]} <-> {p[1]}**: {p[2]:.3f}")
        
    write_report("\n### High Negative (Inverse/Hedge)")
    pairs.sort(key=lambda x: x[2])
    for p in pairs[:10]:
         if p[2] < 0:
            write_report(f"- **{p[0]} <-> {p[1]}**: {p[2]:.3f}")

def analyze_lead_lag(df, lag_max=4):
    write_report(f"\n## Lead-Lag Analysis (Max Lag {lag_max})")
    returns = df.pct_change().dropna()
    candidates = returns.columns
    leads = []
    
    for leader in candidates:
        for follower in candidates:
            if leader == follower: continue
            x = returns[leader].values
            y = returns[follower].values
            best_lag = 0
            best_corr = 0
            
            for lag in range(1, lag_max + 1):
                s1 = x[:-lag]
                s2 = y[lag:]
                if len(s1) < 100: continue
                c = np.corrcoef(s1, s2)[0, 1]
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            
            if abs(best_corr) > 0.15:
                 leads.append((leader, follower, best_lag, best_corr))

    leads.sort(key=lambda x: abs(x[3]), reverse=True)
    
    write_report("\n### Potential Predictive Pairs")
    for l in leads[:10]:
        write_report(f"- **{l[0]}** leads **{l[1]}** by {l[2]} candles (Corr: {l[3]:.3f})")

def classify_beta(df, benchmark='BTC/USDT'):
    write_report(f"\n## Beta Classification (vs {benchmark})")
    if benchmark not in df.columns:
        write_report("Benchmark not found.")
        return
        
    returns = df.pct_change().dropna()
    market = returns[benchmark]
    betas = []
    
    for sym in df.columns:
        if sym == benchmark: continue
        asset = returns[sym]
        covariance = np.cov(asset, market)[0, 1]
        variance = np.var(market)
        beta = covariance / variance
        betas.append((sym, beta))
        
    betas.sort(key=lambda x: x[1], reverse=True)
    
    write_report("\n### High Beta (Amplifiers)")
    for b in betas[:5]:
        write_report(f"- **{b[0]}**: {b[1]:.2f}")
        
    write_report("\n### Low Beta (Shields)")
    low_betas = [b for b in betas if 0 < b[1] < 0.6]
    for b in low_betas:
        write_report(f"- **{b[0]}**: {b[1]:.2f}")
        
    write_report("\n### Negative Beta (Hedges)")
    neg_betas = [b for b in betas if b[1] < 0]
    for b in neg_betas:
        write_report(f"- **{b[0]}**: {b[1]:.2f}")

def calculate_hurst(series):
    """
    Calculate the Hurst Exponent of a time series.
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def analyze_advanced_metrics(df):
    """
    Perform PCA, Clustering, and Hurst Analysis.
    """
    write_report("\n# Deep Dive Analytics (Asset DNA)")
    returns = df.pct_change().dropna()
    
    # --- 1. HURST EXPONENT ---
    write_report("\n## Hurst Exponent (Trendiness)")
    hursts = []
    for col in df.columns:
        # Use log prices for Hurst
        try:
            h = calculate_hurst(np.log(df[col].values))
            hursts.append((col, h))
        except:
            pass
            
    hursts.sort(key=lambda x: x[1], reverse=True)
    
    write_report("### 🦁 Predators (Trending H > 0.55)")
    for s, h in hursts:
        if h > 0.55: write_report(f"- **{s}**: {h:.3f}")
        
    write_report("\n### 🦅 Scavengers (Mean Reverting H < 0.45)")
    for s, h in hursts:
        if h < 0.45: write_report(f"- **{s}**: {h:.3f}")

    write_report("\n### 🎲 Random Walk (0.45 < H < 0.55)")
    for s, h in hursts:
        if 0.45 <= h <= 0.55: write_report(f"- **{s}**: {h:.3f}")

    # --- 2. PCA EXPOSURE ---
    write_report("\n## PCA Component Analysis")
    # Standardize returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    
    pca = PCA(n_components=2)
    pca.fit(scaled_returns)
    
    explained = pca.explained_variance_ratio_
    write_report(f"- **PC1 (Market Risk)**: {explained[0]*100:.1f}% Variance Explained")
    write_report(f"- **PC2 (Sector/Idiosyncratic)**: {explained[1]*100:.1f}% Variance Explained")
    
    # Loadings
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=returns.columns)
    write_report("\n### Asset Loadings (Alpha Potential)")
    # Assets with LOW PC1 loading are less driven by the general market
    sorted_pc1 = loadings.sort_values('PC1')
    
    write_report("#### Lowest Market Correlation (Potential Alpha/Hedge)")
    for s, row in sorted_pc1.head(5).iterrows():
        write_report(f"- **{s}**: PC1={row['PC1']:.2f}, PC2={row['PC2']:.2f}")

    # --- 3. CLUSTER ANALYSIS ---
    write_report("\n## Asset Families (K-Means Clustering)")
    # Transpose to cluster ASSETS, not time steps
    # We use correlation matrix as feature? Or just raw returns?
    # Using Transposed Returns is standard for time-series clustering
    X = returns.T
    
    # Estimate K: We have about 15 assets, maybe 4 families? 
    # (Safe, Alts, Memes, L1s)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    df_clusters = pd.DataFrame({'Cluster': kmeans.labels_}, index=returns.columns)
    
    for i in range(n_clusters):
        members = df_clusters[df_clusters['Cluster'] == i].index.tolist()
        write_report(f"\n### Family {i+1}")
        for m in members:
            write_report(f"- {m}")

if __name__ == "__main__":
    # Clear file
    with open("analysis_report.md", "w", encoding='utf-8') as f:
        f.write("# Market Taxonomy Report\n")
        f.write(f"Generated from {TIMEFRAME} candles.\n")
        
    df = load_data()
    if df is not None and not df.empty:
        analyze_correlations(df)
        classify_beta(df)
        analyze_lead_lag(df)
        analyze_advanced_metrics(df) # NEW
        print("Report generated: analysis_report.md")
    else:
        print("Data load failed or empty.")
