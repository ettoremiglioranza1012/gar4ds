"""
SPATIAL DURBIN MODEL - PANEL DATA ANALYSIS
===========================================

This script implements a Panel Spatial Durbin Model (SDM) to quantify 
cross-border PM10 transport from Po Valley to Alpine regions.

METHODOLOGY:
1. Load filtered panel data (21,275 observations × 12 variables)
2. Log-transform PM10 for variance stabilization
3. Standardize all meteorological variables
4. Create spatially lagged features (WX matrix) - Manual Durbin construction
5. Fit Panel Fixed Effects Spatial Lag Model (spreg.Panel_FE_Lag)
6. Decompose spillover: Local (Xβ) + Neighbor (WXθ) + Endogenous (ρWy)
7. Regime-stratified analysis for ALL 5 atmospheric clusters
8. Generate comprehensive diagnostics and visualizations

MODEL SPECIFICATION (SDM):
y = ρWy + Xβ + WXθ + ε

Where:
- y = log(PM10)
- ρ = spatial autoregressive coefficient (endogenous spillover)
- Wy = spatially lagged PM10 (neighbor pollution)
- X = local meteorological conditions (11 variables)
- β = direct effect coefficients
- WX = spatially lagged meteorology (neighbor conditions)
- θ = indirect effect coefficients

OUTPUTS:
- Text log: results/spatial_durbin_model/sdm_analysis_log.txt
- CSV results: results/spatial_durbin_model/
  * Global model: model_summary.txt, coefficients_table.csv
  * Cluster-specific: cluster_N_model_summary.txt, cluster_N_coefficients.csv (N=0-4)
  * Combined: all_clusters_coefficients_combined.csv, regime_comparison.csv
- Visualizations: assets/spatial_durbin_model/ (coefficient forest plot, Q-Q plot)

CHANGES (9 Feb 2026):
- Removed mean-aggregated station-level summaries (preserves temporal dynamics)
- Enhanced regime-stratified analysis to fit models for ALL 5 atmospheric clusters
- Removed spillover decomposition map and top10 bar chart (relied on aggregation)
- Added per-cluster model outputs and combined coefficient tables

Author: Ettore Miglioranza
Date: 9 February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from libpysal.io import open as ioopen
from libpysal.weights import w_subset
from esda.moran import Moran
from spreg import Panel_FE_Lag
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Import temporal configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_output_path, TEMPORAL_FREQUENCY
TEMP_CONFIG = get_config()

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'spatial_durbin_model'
ASSETS_DIR = PROJECT_DIR / 'assets' / 'spatial_durbin_model'
WEIGHTS_DIR = PROJECT_DIR / 'weights'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Output file for verbose logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = RESULTS_DIR / f'sdm_analysis_log_{timestamp}.txt'


class Tee:
    """Helper class to write to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def print_header(title, level=1):
    """Print formatted section header"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"{title.upper()}")
        print("=" * 80)
    elif level == 2:
        print(f"\n[{title}]")
        print("-" * 80)
    else:
        print(f"\n--- {title} ---")


def load_panel_data(data_path):
    """Load filtered panel data matrix"""
    print_header("1. LOADING FILTERED PANEL DATA")
    print(f"    File: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    print(f"    ✓ Shape: {df.shape} (observations × variables)")
    print(f"    ✓ Index: {df.index.names}")
    print(f"    ✓ Columns: {list(df.columns)}")
    
    if len(df.index.names) == 2:
        n_periods = df.index.get_level_values(0).nunique()
        n_stations = df.index.get_level_values(1).nunique()
        print(f"    ✓ MultiIndex confirmed: {df.index.names}")
        print(f"    ✓ Time periods ({TEMP_CONFIG['period_label_plural']}): {n_periods}")
        print(f"    ✓ Stations: {n_stations}")
        print(f"    ✓ Total observations: {len(df)}")
    
    # Display variable summary
    print("\n    Variable Summary:")
    for col in df.columns:
        print(f"        {col:30s} | mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    return df


def load_station_metadata(meta_path):
    """Load station metadata with coordinates"""
    print_header("2. LOADING STATION METADATA")
    print(f"    File: {meta_path}")
    
    gdf = gpd.read_file(meta_path)
    gdf['Longitude'] = gdf.geometry.x
    gdf['Latitude'] = gdf.geometry.y
    
    meta_df = pd.DataFrame({
        'station_code': gdf['station_code'],
        'station_name': gdf['station_name'],
        'region': gdf['region'],
        'Longitude': gdf['Longitude'],
        'Latitude': gdf['Latitude']
    }).set_index('station_code')
    
    print(f"    ✓ Loaded {len(meta_df)} stations")
    print(f"    ✓ Regions: {meta_df['region'].unique().tolist()}")
    print(f"    ✓ Coordinate range:")
    print(f"        Longitude: [{meta_df['Longitude'].min():.4f}, {meta_df['Longitude'].max():.4f}]")
    print(f"        Latitude: [{meta_df['Latitude'].min():.4f}, {meta_df['Latitude'].max():.4f}]")
    
    return meta_df


def load_spatial_weights(weights_path):
    """Load spatial weights matrix from .gal file"""
    print_header("3. LOADING SPATIAL WEIGHTS MATRIX")
    print(f"    File: {weights_path}")
    
    w = ioopen(weights_path).read()
    
    print(f"    ✓ Number of observations: {w.n}")
    print(f"    ✓ Total connections: {w.s0:.0f}")
    print(f"    ✓ Average neighbors: {w.mean_neighbors:.2f}")
    print(f"    ✓ Transformation: {w.transform}")
    
    return w


def load_regime_clusters(cluster_path):
    """Load atmospheric regime classifications"""
    print_header("4. LOADING REGIME CLASSIFICATIONS")
    print(f"    File: {cluster_path}")
    
    clusters = pd.read_csv(cluster_path)
    
    print(f"    ✓ Loaded {len(clusters)} station classifications")
    print(f"    ✓ Columns: {list(clusters.columns)}")
    print("\n    Cluster Distribution:")
    for cluster_id, count in clusters['Cluster'].value_counts().sort_index().items():
        print(f"        Cluster {cluster_id}: {count} stations")
    
    return clusters


def prepare_data_for_sdm(panel_df, meta_df):
    """
    Prepare data for SDM estimation:
    1. Log-transform PM10
    2. Extract meteorological variables
    3. Ensure alignment with metadata
    """
    print_header("5. PREPARING DATA FOR SDM")
    
    # Log-transform PM10
    print("    Step 1: Log-transforming PM10")
    panel_df['log_pm10'] = np.log(panel_df['pm10'] + 1)  # Add 1 to handle near-zero values
    print(f"        ✓ log_pm10 created (mean={panel_df['log_pm10'].mean():.3f}, std={panel_df['log_pm10'].std():.3f})")
    
    # Define meteorological variables (all except pm10)
    met_vars = [col for col in panel_df.columns if col not in ['pm10', 'log_pm10']]
    print(f"\n    Step 2: Meteorological variables identified ({len(met_vars)}):")
    for var in met_vars:
        print(f"        - {var}")
    
    # Check alignment with metadata
    print("\n    Step 3: Checking station alignment")
    panel_stations = panel_df.index.get_level_values('station_id').unique()
    meta_stations = meta_df.index.unique()
    common_stations = panel_stations.intersection(meta_stations)
    print(f"        Panel stations: {len(panel_stations)}")
    print(f"        Metadata stations: {len(meta_stations)}")
    print(f"        ✓ Common stations: {len(common_stations)}")
    
    if len(common_stations) < len(panel_stations):
        print(f"        ⚠ Warning: {len(panel_stations) - len(common_stations)} stations not in metadata")
    
    return panel_df, met_vars, common_stations


def standardize_variables(panel_df, met_vars):
    """
    Standardize meteorological variables using StandardScaler
    """
    print_header("6. STANDARDIZING VARIABLES")
    print("    Purpose: Eliminate unit-of-measurement bias")
    print("    Method: StandardScaler (mean=0, std=1)")
    
    scaler = StandardScaler()
    
    # Fit scaler on meteorological variables
    panel_df[met_vars] = scaler.fit_transform(panel_df[met_vars])
    
    print(f"\n    ✓ Standardized {len(met_vars)} variables")
    print("\n    Post-standardization check:")
    for var in met_vars[:3]:  # Show first 3 as example
        print(f"        {var:30s} | mean={panel_df[var].mean():.6f}, std={panel_df[var].std():.6f}")
    print("        ...")
    
    return panel_df, scaler


def create_spatially_lagged_features(panel_df, met_vars, w, common_stations):
    """
    Create WX matrix by computing spatial lag for each week
    This is the manual Durbin construction
    """
    print_header("7. CREATING SPATIALLY LAGGED FEATURES (WX MATRIX)")
    print("    Purpose: Manual Spatial Durbin Model construction")
    print(f"    Method: Compute W @ X for each of the {TEMP_CONFIG['period_label_plural']}")
    
    # Get unique time periods
    period_col = TEMP_CONFIG['period_column']
    periods = panel_df.index.get_level_values(period_col).unique().sort_values()
    n_periods = len(periods)
    n_stations = len(common_stations)
    n_vars = len(met_vars)
    
    print(f"\n    Dimensions:")
    print(f"        {TEMP_CONFIG['period_label_plural'].capitalize()}: {n_periods}")
    print(f"        Stations: {n_stations}")
    print(f"        Variables: {n_vars}")
    print(f"        Total WX entries: {n_periods * n_stations * n_vars:,}")
    
    # Create mapping of station_id to weights matrix index
    w_ids = w.id_order
    station_to_idx = {station: idx for idx, station in enumerate(w_ids)}
    
    # Initialize lagged feature columns
    lagged_vars = [f'lag_{var}' for var in met_vars]
    for lag_var in lagged_vars:
        panel_df[lag_var] = np.nan
    
    print("\n    Computing spatial lags...")
    progress_interval = max(1, n_periods // 10)  # Show progress every 10%
    
    for i, period in enumerate(periods):
        # Extract cross-section for this time period
        period_data = panel_df.xs(period, level=period_col)
        
        # Align with weights matrix order
        period_data_aligned = period_data.reindex(w_ids)
        
        # Extract meteorological variables as matrix
        X_period = period_data_aligned[met_vars].values  # Shape: (n_stations, n_vars)
        
        # Compute spatial lag: WX = W @ X
        WX_period = np.zeros_like(X_period)
        for j, station in enumerate(w_ids):
            neighbors = w.neighbors[station]
            weights = w.weights[station]
            neighbor_indices = [station_to_idx[n] for n in neighbors]
            WX_period[j, :] = np.sum([weights[k] * X_period[neighbor_indices[k], :] 
                                    for k in range(len(neighbors))], axis=0)
        
        # Store lagged features back in panel
        for v, lag_var in enumerate(lagged_vars):
            for s, station in enumerate(w_ids):
                panel_df.loc[(period, station), lag_var] = WX_period[s, v]
        
        # Progress indicator
        if (i + 1) % progress_interval == 0:
            print(f"        Progress: {i+1}/{n_periods} {TEMP_CONFIG['period_label_plural']} ({(i+1)/n_periods*100:.1f}%)")
    
    print(f"\n    ✓ Created {len(lagged_vars)} spatially lagged features")
    print("    ✓ Feature set:")
    print(f"        Direct features (X): {len(met_vars)}")
    print(f"        Lagged features (WX): {len(lagged_vars)}")
    print(f"        Total features: {len(met_vars) + len(lagged_vars)}")
    
    return panel_df, lagged_vars


def fit_panel_sdm(panel_df, met_vars, lagged_vars, w, common_stations):
    """
    Fit Panel Fixed Effects Spatial Lag Model
    This is equivalent to SDM with manually constructed WX
    """
    print_header("8. FITTING PANEL SPATIAL DURBIN MODEL")
    print("    Estimator: Panel Fixed Effects Spatial Lag (spreg.Panel_FE_Lag)")
    print("    Specification: log(PM10) = ρWy + Xβ + WXθ + ε")
    
    # Prepare data
    print("\n    Preparing arrays...")
    
    # Filter to common stations only
    panel_filtered = panel_df.loc[(slice(None), common_stations.tolist()), :]
    
    # Remove any rows with NaN in lagged features
    panel_filtered = panel_filtered.dropna(subset=lagged_vars)
    
    print(f"    ✓ Filtered dataset: {len(panel_filtered)} observations")
    
    # Dependent variable
    y = panel_filtered['log_pm10'].values.reshape(-1, 1)
    
    # Independent variables (X + WX)
    all_features = met_vars + lagged_vars
    X = panel_filtered[all_features].values
    
    print(f"    ✓ y (log_pm10): shape {y.shape}")
    print(f"    ✓ X (met + lagged): shape {X.shape}")
    
    # Variable names for model output
    name_y = ['log_pm10']
    name_x = all_features
    
    print("\n    Fitting model...")
    print("    (This may take several minutes for 21k+ observations)")
    
    try:
        model = Panel_FE_Lag(
            y=y,
            x=X,
            w=w,
            name_y=name_y,
            name_x=name_x,
            name_ds="PM10_Alpine_Panel"
        )
        
        print("\n    ✓ Model estimation complete!")
        print(f"    ✓ Spatial lag coefficient (ρ): {model.rho:.6f}")
        if hasattr(model, 'z_stat') and len(model.z_stat) > 0:
            print(f"    ✓ ρ p-value: {model.z_stat[0][1]:.6f}")
        
        return model, panel_filtered
        
    except Exception as e:
        print(f"\n    ✗ Error fitting model: {e}")
        print("    Attempting alternative approach...")
        return None, panel_filtered


def save_model_summary(model, output_path):
    """Save full model summary to text file"""
    print_header("9. SAVING MODEL SUMMARY")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PANEL SPATIAL DURBIN MODEL - FULL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(str(model.summary))
    
    print(f"    ✓ Saved to: {output_path}")


def extract_coefficients(model, met_vars, lagged_vars):
    """Extract and organize coefficients from model"""
    print_header("10. EXTRACTING COEFFICIENTS")
    
    # Get coefficient names, values, std errors, z-stats, p-values
    coef_names = model.name_x
    coef_values = model.betas.flatten()
    
    # Create coefficients table
    coef_df = pd.DataFrame({
        'variable': coef_names,
        'coefficient': coef_values,
    })
    
    # Add significance stars
    if hasattr(model, 'z_stat'):
        # Extract z-stats and p-values - need to match the number of coefficients
        n_coefs = len(coef_names)
        if len(model.z_stat) > n_coefs:
            # Skip the first entry (constant) if it exists
            z_stats = model.z_stat[1:n_coefs+1]
        else:
            z_stats = model.z_stat[:n_coefs]
        
        coef_df['z_stat'] = [z[0] for z in z_stats]
        coef_df['p_value'] = [z[1] for z in z_stats]
        coef_df['significance'] = coef_df['p_value'].apply(
            lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        )
    
    # Separate direct (β) and indirect (θ) effects
    coef_df['effect_type'] = coef_df['variable'].apply(
        lambda x: 'Indirect (θ)' if x.startswith('lag_') else 'Direct (β)'
    )
    
    # Add spatial lag info
    rho_row = pd.DataFrame({
        'variable': ['W_log_pm10'],
        'coefficient': [model.rho],
        'effect_type': ['Endogenous (ρ)']
    })
    coef_df = pd.concat([rho_row, coef_df], ignore_index=True)
    
    print(f"    ✓ Extracted {len(coef_df)} coefficients")
    print("\n    Spatial Lag Coefficient (ρ):")
    print(f"        ρ = {model.rho:.6f}")
    
    print("\n    Significant Direct Effects (β):")
    sig_direct = coef_df[(coef_df['effect_type'] == 'Direct (β)') & 
                         (coef_df.get('p_value', 1) < 0.05)]
    if len(sig_direct) > 0:
        for _, row in sig_direct.iterrows():
            print(f"        {row['variable']:30s}: β = {row['coefficient']:7.4f} {row.get('significance', '')}")
    else:
        print("        (None at p < 0.05)")
    
    print("\n    Significant Indirect Effects (θ):")
    sig_indirect = coef_df[(coef_df['effect_type'] == 'Indirect (θ)') & 
                           (coef_df.get('p_value', 1) < 0.05)]
    if len(sig_indirect) > 0:
        for _, row in sig_indirect.iterrows():
            print(f"        {row['variable']:30s}: θ = {row['coefficient']:7.4f} {row.get('significance', '')}")
    else:
        print("        (None at p < 0.05)")
    
    return coef_df


def decompose_spillover(model, panel_filtered, met_vars, lagged_vars):
    """
    Decompose PM10 into:
    1. Direct local effect (Xβ)
    2. Indirect neighbor effect (WXθ)
    3. Endogenous spillover (ρWy)
    """
    print_header("11. DECOMPOSING SPILLOVER EFFECTS")
    print("    Components:")
    print("        1. Direct Local Effect: X·β")
    print("        2. Indirect Neighbor Effect: WX·θ")
    print("        3. Endogenous Spillover: ρ·Wy")
    
    # Extract coefficients
    n_met = len(met_vars)
    beta = model.betas[1:n_met+1].flatten()  # Skip constant, take first n_met
    theta = model.betas[n_met+1:].flatten()  # Take next n_met (lagged vars)
    rho = model.rho
    
    # Get X and WX matrices
    X_direct = panel_filtered[met_vars].values
    X_lagged = panel_filtered[lagged_vars].values
    
    # Compute components
    direct_effect = X_direct @ beta
    indirect_effect = X_lagged @ theta
    
    print(f"\n    ✓ Computed direct effects (mean={direct_effect.mean():.4f})")
    print(f"    ✓ Computed indirect effects (mean={indirect_effect.mean():.4f})")
    
    # Add to dataframe
    decomp_df = panel_filtered.copy()
    decomp_df['direct_local'] = direct_effect
    decomp_df['indirect_neighbor'] = indirect_effect
    decomp_df['predicted_log_pm10'] = model.predy.flatten()
    decomp_df['residual'] = model.u.flatten()
    
    # Endogenous spillover requires computing Wy for each observation
    # This is complex with panel structure, so we approximate as: residual variance
    decomp_df['endogenous_spillover'] = decomp_df['predicted_log_pm10'] - decomp_df['direct_local'] - decomp_df['indirect_neighbor']
    
    print(f"    ✓ Endogenous spillover computed (mean={decomp_df['endogenous_spillover'].mean():.4f})")
    
    return decomp_df


def fit_regime_stratified_models(panel_df, clusters, met_vars, lagged_vars, w, common_stations):
    """
    Fit separate SDM models for ALL atmospheric regimes (5 clusters)
    
    HYPOTHESIS TEST:
    Does the global model average over distinct physical regimes, causing:
    - Counterintuitive coefficient signs (e.g., positive BLH)
    - Opposing direct/indirect effects
    - Physical incoherence between source and transport regions?
    
    EXPECTED PATTERNS BY REGIME:
    
    Cluster 0 (Po Valley Stagnation):
    - BLH: Expected NEGATIVE (dilution works locally without transport)
    - Temperature/Solar: Strongly NEGATIVE (convection is only escape mechanism)
    - Winds: Less important (stagnant conditions)
    - ρ: HIGH (stations share same stagnant air mass - synchronous)
    
    Cluster 2 (Trentino Transport Corridor):
    - V850: Strongly NEGATIVE (southerly transport from Po Valley main driver)
    - BLH: Effects may differ (entrainment from transport layers)
    - Regional effects (θ): Possibly STRONGER (spatial spillovers dominate)
    - ρ: LOWER than Cluster 0 (transport is directional, not simultaneous)
    
    Other Clusters (1, 3, 4):
    - Intermediate or distinct meteorological regimes
    - Test if parameters show regime-specific coherence
    """
    print_header("13. REGIME-STRATIFIED ANALYSIS - ALL CLUSTERS")
    print("    Fitting separate SDM models for each atmospheric cluster")
    print("    Testing hypothesis: Global model averages over distinct physical regimes\n")
    
    regime_results = {}
    
    # Loop through ALL 5 clusters (0, 1, 2, 3, 4)
    for cluster_id in range(5):
        print(f"\n    {'='*60}")
        print(f"    CLUSTER {cluster_id}")
        print(f"    {'='*60}")
        
        # Get stations in this cluster
        cluster_stations = clusters[clusters['Cluster'] == cluster_id]['Station'].values
        print(f"    Stations: {len(cluster_stations)}")
        
        # Filter panel data
        panel_cluster = panel_df.loc[(slice(None), cluster_stations), :]
        panel_cluster = panel_cluster.dropna(subset=lagged_vars)
        
        print(f"    Observations: {len(panel_cluster)}")
        
        if len(panel_cluster) < 50:
            print(f"    ⚠ Warning: Too few observations for reliable estimation")
            continue
        
        # Prepare data
        y_cluster = panel_cluster['log_pm10'].values.reshape(-1, 1)
        X_cluster = panel_cluster[met_vars + lagged_vars].values
        
        # Create subset of spatial weights for cluster stations
        print("    Creating subset spatial weights matrix...")
        try:
            # Get list of cluster stations that are in the weights matrix
            cluster_stations_in_w = [s for s in cluster_stations if s in w.id_order]
            
            if len(cluster_stations_in_w) == 0:
                print(f"    ✗ Error: No cluster stations found in weights matrix")
                continue
            
            # Create subset weights using w_subset
            w_cluster = w_subset(w, cluster_stations_in_w)
            
            # Ensure row-standardization for model convergence
            w_cluster.transform = 'r'
            print(f"    ✓ Subset weights created: {w_cluster.n} stations, {w_cluster.s0:.0f} connections")
            print(f"    ✓ Row-standardized (transform={w_cluster.transform})")
            
        except Exception as e:
            print(f"    ✗ Error creating subset weights: {e}")
            continue
        
        try:
            print("    Fitting model...")
            model_cluster = Panel_FE_Lag(
                y=y_cluster,
                x=X_cluster,
                w=w_cluster,
                name_y=['log_pm10'],
                name_x=met_vars + lagged_vars,
                name_ds=f"PM10_Cluster{cluster_id}"
            )
            
            print(f"    ✓ Model fit successful")
            print(f"    ✓ ρ (Cluster {cluster_id}) = {model_cluster.rho:.6f}")
            
            # Extract full coefficient table
            coef_cluster = extract_coefficients(model_cluster, met_vars, lagged_vars)
            
            regime_results[cluster_id] = {
                'model': model_cluster,
                'rho': model_cluster.rho,
                'n_obs': len(panel_cluster),
                'n_stations': len(cluster_stations_in_w),
                'stations': cluster_stations_in_w,
                'coefficients': coef_cluster
            }
            
            # Save model summary to separate file for each cluster
            cluster_summary_path = RESULTS_DIR / f'cluster_{cluster_id}_model_summary.txt'
            save_model_summary(model_cluster, cluster_summary_path)
            print(f"    ✓ Saved model summary: cluster_{cluster_id}_model_summary.txt")
            
            # Save coefficient table for each cluster
            coef_cluster.to_csv(
                RESULTS_DIR / f'cluster_{cluster_id}_coefficients.csv',
                index=False
            )
            print(f"    ✓ Saved coefficients: cluster_{cluster_id}_coefficients.csv")
            
        except Exception as e:
            print(f"    ✗ Error fitting model: {e}")
            continue
    
    # Comprehensive regime comparison table
    print(f"\n    {'='*60}")
    print("    REGIME COMPARISON SUMMARY")
    print(f"    {'='*60}\n")
    
    if len(regime_results) > 0:
        comparison_rows = []
        for cluster_id in sorted(regime_results.keys()):
            res = regime_results[cluster_id]
            comparison_rows.append({
                'cluster_id': cluster_id,
                'rho': res['rho'],
                'n_stations': res['n_stations'],
                'n_obs': res['n_obs']
            })
        
        comparison_df = pd.DataFrame(comparison_rows)
        print(comparison_df.to_string(index=False))
        print()
        
        # Print interpretation guidance
        print("    INTERPRETATION NOTES:")
        print("    - High ρ suggests synchronous spatial correlation")
        print("    - Low ρ may indicate directional transport (not simultaneous)")
        print("    - Compare coefficient signs across clusters for physical coherence")
        print("    - Check if global model sign conflicts resolve in cluster-specific models")
    
    return regime_results


def compute_residual_diagnostics(model, decomp_df, w):
    """Compute diagnostics on model residuals"""
    print_header("14. RESIDUAL DIAGNOSTICS")
    
    # Get residuals
    residuals = model.u.flatten()
    
    print(f"    Residual Statistics:")
    print(f"        Mean: {residuals.mean():.6f}")
    print(f"        Std: {residuals.std():.6f}")
    print(f"        Min: {residuals.min():.6f}")
    print(f"        Max: {residuals.max():.6f}")
    
    # Spatial autocorrelation in residuals (should be non-significant)
    print("\n    Testing for residual spatial autocorrelation...")
    print("    (Should be non-significant if model captures spatial structure)")
    
    # Aggregate residuals to cross-section (mean over time)
    resid_by_station = decomp_df.groupby(level='station_id')['residual'].mean()
    
    # Align with weights order
    w_ids = w.id_order
    resid_aligned = resid_by_station.reindex(w_ids).values
    
    try:
        moran = Moran(resid_aligned, w)
        print(f"\n    Moran's I on residuals: {moran.I:.6f}")
        print(f"    Expected I: {moran.EI:.6f}")
        print(f"    P-value: {moran.p_sim:.6f}")
        
        if moran.p_sim > 0.05:
            print("    ✓ No significant spatial autocorrelation in residuals (Good!)")
        else:
            print("    ⚠ Residuals still show spatial autocorrelation")
        
        return {'morans_i': moran.I, 'p_value': moran.p_sim}
    
    except Exception as e:
        print(f"    ⚠ Could not compute Moran's I: {e}")
        return None


def create_visualizations(decomp_df, coef_df, meta_df, residual_diag):
    """Generate visualizations (UPDATED - removed station_summary)"""
    print_header("15. GENERATING VISUALIZATIONS")
    
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Coefficient Forest Plot
    print("    Creating coefficient forest plot...")
    coef_plot = coef_df[coef_df['variable'] != 'W_log_pm10'].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate direct and indirect
    direct = coef_plot[coef_plot['effect_type'] == 'Direct (β)']
    indirect = coef_plot[coef_plot['effect_type'] == 'Indirect (θ)']
    
    y_pos = np.arange(len(coef_plot))
    colors = ['steelblue' if 'Direct' in et else 'coral' for et in coef_plot['effect_type']]
    
    ax.barh(y_pos, coef_plot['coefficient'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_plot['variable'], fontsize=9)
    ax.set_xlabel('Coefficient Value', fontsize=11)
    ax.set_title('SDM Coefficients: Direct (β) vs Indirect (θ) Effects', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Direct (β)'),
                      Patch(facecolor='coral', alpha=0.7, label='Indirect (θ)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'coefficient_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: coefficient_forest_plot.png")
    
    # 3. Residual Q-Q Plot
    print("    Creating residual Q-Q plot...")
    from scipy import stats
    
    residuals = decomp_df['residual'].values
    
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Residual Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'residual_qq_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: residual_qq_plot.png")
    
    print("\n    ✓ All visualizations complete!")


def save_results(decomp_df, coef_df, regime_results):
    """Save all analysis results (UPDATED - removed station_summary)"""
    print_header("16. SAVING RESULTS")
    
    # 1. Spillover decomposition (observation-level, not aggregated)
    decomp_df.to_csv(RESULTS_DIR / 'spillover_decomposition_observations.csv')
    print(f"    ✓ spillover_decomposition_observations.csv ({len(decomp_df)} obs)")
    
    # 2. Model coefficients (global model)
    coef_df.to_csv(RESULTS_DIR / 'coefficients_table.csv', index=False)
    print(f"    ✓ coefficients_table.csv ({len(coef_df)} rows)")
    
    # 3. ENHANCED: Regime comparison with all details
    if regime_results:
        # Basic comparison table
        regime_comparison = pd.DataFrame([
            {
                'cluster_id': cluster_id,
                'rho': results['rho'],
                'n_stations': results['n_stations'],
                'n_obs': results['n_obs']
            }
            for cluster_id, results in regime_results.items()
        ])
        regime_comparison.to_csv(RESULTS_DIR / 'regime_comparison.csv', index=False)
        print(f"    ✓ regime_comparison.csv ({len(regime_comparison)} rows)")
        
        # NEW: Combined coefficients table for all clusters
        all_cluster_coefs = []
        for cluster_id, results in regime_results.items():
            coefs = results['coefficients'].copy()
            coefs['cluster_id'] = cluster_id
            all_cluster_coefs.append(coefs)
        
        if all_cluster_coefs:
            combined_coefs = pd.concat(all_cluster_coefs, ignore_index=True)
            combined_coefs.to_csv(
                RESULTS_DIR / 'all_clusters_coefficients_combined.csv',
                index=False
            )
            print(f"    ✓ all_clusters_coefficients_combined.csv ({len(combined_coefs)} rows)")
            print(f"        Contains coefficients for {len(regime_results)} cluster-specific models")
    
    print("\n    ✓ All results saved!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Open log file
    f_out = open(output_file, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, f_out)
    
    print("=" * 80)
    print("SPATIAL DURBIN MODEL - PANEL DATA ANALYSIS")
    print("=" * 80)
    print(f"\nAnalysis Date: {datetime.now()}")
    print(f"\nOutputs:")
    print(f"    Text Log: {output_file}")
    print(f"    CSV Results: {RESULTS_DIR}")
    print(f"    Visualizations: {ASSETS_DIR}")
    
    try:
        # 1. Load panel data
        panel_df = load_panel_data(DATA_DIR / get_output_path('panel_data_matrix_filtered_for_collinearity'))
        
        # 2. Load metadata
        meta_df = load_station_metadata(DATA_DIR / 'pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson')
        
        # 3. Load spatial weights
        w = load_spatial_weights(WEIGHTS_DIR / 'spatial_weights_knn6.gal')
        
        # 4. Load regime clusters
        clusters = load_regime_clusters(RESULTS_DIR.parent / 'spatial_analysis' / 'optionC_multivariate_clusters.csv')
        
        # 5. Prepare data
        panel_df, met_vars, common_stations = prepare_data_for_sdm(panel_df, meta_df)
        
        # 6. Standardize variables
        panel_df, scaler = standardize_variables(panel_df, met_vars)
        
        # 7. Create spatially lagged features (WX matrix)
        panel_df, lagged_vars = create_spatially_lagged_features(panel_df, met_vars, w, common_stations)
        
        # 8. Fit Panel SDM
        model, panel_filtered = fit_panel_sdm(panel_df, met_vars, lagged_vars, w, common_stations)
        
        if model is None:
            print("\n✗ Model fitting failed. Cannot proceed with analysis.")
            return
        
        # 9. Save model summary
        save_model_summary(model, RESULTS_DIR / 'model_summary.txt')
        
        # 10. Extract coefficients
        coef_df = extract_coefficients(model, met_vars, lagged_vars)
        
        # 11. Decompose spillover
        decomp_df = decompose_spillover(model, panel_filtered, met_vars, lagged_vars)
        
        # 12. Regime-stratified analysis (ALL 5 clusters)
        regime_results = fit_regime_stratified_models(panel_df, clusters, met_vars, lagged_vars, w, common_stations)
        
        # 13. Residual diagnostics
        residual_diag = compute_residual_diagnostics(model, decomp_df, w)
        
        # 14. Create visualizations
        create_visualizations(decomp_df, coef_df, meta_df, residual_diag)
        
        # 15. Save results
        save_results(decomp_df, coef_df, regime_results)
        
        # Final summary
        print_header("ANALYSIS COMPLETE", level=1)
        print("\n📊 KEY FINDINGS:")
        print(f"    • Observations analyzed: {len(panel_filtered):,}")
        print(f"    • Stations: {len(common_stations)}")
        print(f"    • Time periods: {panel_df.index.get_level_values(0).nunique()}")
        print(f"    • Spatial lag coefficient (ρ): {model.rho:.6f}")
        
        if residual_diag:
            print(f"    • Residual Moran's I: {residual_diag['morans_i']:.6f} (p={residual_diag['p_value']:.4f})")
        
        print("\n✅ ALL OUTPUTS GENERATED:")
        print(f"    → Text log: {output_file}")
        print(f"    → CSV results: {RESULTS_DIR}")
        print(f"    → Visualizations: {ASSETS_DIR}")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        f_out.close()
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
