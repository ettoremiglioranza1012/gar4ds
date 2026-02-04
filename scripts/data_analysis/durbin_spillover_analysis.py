"""
Spatial Durbin Model for PM10 Pollution Spillover Analysis
===========================================================

This script implements a Spatial Durbin Model (SDM) to analyze spillover effects 
in PM10 pollution, with specific focus on decomposing pollution sources for 
target stations: Borgo Valsugana, Monte Gaza, Parco S. Chiara, and Piana Rotaliana.

The decomposition separates PM10 levels into:
1. Local Effect (Xβ): Pollution from station's own characteristics
2. Neighbor Context (WXθ): Effect of neighboring stations' characteristics
3. Spatial Spillover (ρWy): Pollution "imported" from neighboring stations

Mathematical Model:
y_i = ρ * Σ(w_ij * y_j) + x_i * β + Σ(w_ij * x_j) * θ + ε_i
"""

import polars as pl
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN
from spreg import ML_Lag
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Target stations for detailed analysis
TARGET_STATIONS = [
    'Borgo Valsugana',
    'Monte Gaza', 
    'Parco S. Chiara',
    'Piana Rotaliana'
]

# Spatial weights configuration
K_NEIGHBORS = 6  # Number of nearest neighbors for spatial weights

# Create output directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
ASSETS_DIR = PROJECT_DIR / 'assets' / 'spatial_durbin_model'
RESULTS_DIR = PROJECT_DIR / 'results' / 'spatial_durbin_model'
DATA_DIR = PROJECT_DIR / 'data'

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Output file for text results
OUTPUT_FILE = RESULTS_DIR / 'durbin_spillover_analysis.txt'


class TeeOutput:
    """Utility class to write output to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def load_data():
    """Load and prepare multi-variable spatial data and station metadata"""
    print("=" * 70)
    print("SPATIAL DURBIN MODEL: PM10 SPILLOVER ANALYSIS")
    print("(Enhanced with Meteorological Variables)")
    print("=" * 70)
    print("\n[1] Loading Data...")
    
    # Load multi-variable spatial matrix (PM10, TEMP, BLH, WS, PRECIP, PRESS, RAD)
    spatial_df = pl.read_parquet(DATA_DIR / 'spatial_full_matrix.parquet')
    
    # Load station metadata
    with open(DATA_DIR / 'data_stations_metadata.json', 'r') as f:
        stations_meta = json.load(f)
    
    meta_df = pd.DataFrame(stations_meta).set_index('Station_ID')
    
    # Get valid stations from PM10 columns
    pm10_cols = [c for c in spatial_df.columns if c.startswith('PM10_')]
    data_stations = [c.replace('PM10_', '') for c in pm10_cols]
    valid_stations = [s for s in data_stations if s in meta_df.index]
    
    print(f"    ✓ Loaded {len(valid_stations)} stations for analysis")
    print(f"    ✓ Variables available: PM10, TEMP, BLH, WS, PRECIP, PRESS, RAD")
    print(f"    ✓ Time series length: {len(spatial_df)} observations")
    
    # Check target stations availability
    available_targets = [s for s in TARGET_STATIONS if s in valid_stations]
    missing_targets = [s for s in TARGET_STATIONS if s not in valid_stations]
    
    if missing_targets:
        print(f"    ⚠ Missing target stations: {missing_targets}")
    print(f"    ✓ Available target stations: {available_targets}")
    
    return spatial_df, meta_df, valid_stations, available_targets


def create_spatial_weights(meta_df, valid_stations, k=K_NEIGHBORS):
    """Create row-standardized spatial weights matrix using KNN"""
    print(f"\n[2] Creating Spatial Weights Matrix (K={k} neighbors)...")
    
    coords = meta_df.loc[valid_stations, ['Longitude', 'Latitude']].values
    w = KNN.from_array(coords, k=k)
    w.transform = 'r'  # Row-standardize
    
    print(f"    ✓ Created {w.n} x {w.n} spatial weights matrix")
    print(f"    ✓ Average neighbors per station: {w.mean_neighbors:.2f}")
    
    return w, coords


def prepare_model_data(spatial_df, meta_df, valid_stations):
    """
    Prepare multi-variable model with meteorological features.
    
    This is the KEY UPGRADE from the original model:
    - Original: Used only Lat/Lon/Altitude (geographic proxies)
    - New: Uses actual meteorological variables that drive pollution dispersion
    
    Independent variables:
    - BLH (Boundary Layer Height): Low BLH traps pollutants
    - TEMP (Temperature): Affects atmospheric stability
    - WS (Wind Speed): Primary dispersion mechanism
    - PRECIP (Precipitation): Wet deposition removes particulates
    - PRESS (Pressure): Weather systems affect mixing
    - RAD (Radiation): Photochemical reactions
    - Altitude (optional): Terrain effects
    """
    print("\n[3] Preparing Multi-Variable Model...")
    
    # Dependent variable: Mean PM10 per station
    pm10_cols = [f'PM10_{s}' for s in valid_stations]
    pm10_means = spatial_df.select(pm10_cols).mean().to_pandas().iloc[0]
    pm10_means.index = valid_stations  # Rename index to station names
    y = pm10_means.values.reshape(-1, 1)
    
    # Independent variables: Meteorological characteristics
    feature_names = []
    feature_data = []
    
    # 1. Boundary Layer Height (critical for pollution dispersion)
    blh_cols = [f'BLH_{s}' for s in valid_stations]
    blh_means = spatial_df.select(blh_cols).mean().to_pandas().iloc[0]
    feature_data.append(blh_means.values)
    feature_names.append('BLH')
    
    # 2. Temperature (affects atmospheric stability)
    temp_cols = [f'TEMP_{s}' for s in valid_stations]
    temp_means = spatial_df.select(temp_cols).mean().to_pandas().iloc[0]
    feature_data.append(temp_means.values)
    feature_names.append('TEMP')
    
    # 3. Wind Speed (primary dispersion mechanism)
    ws_cols = [f'WS_{s}' for s in valid_stations]
    ws_means = spatial_df.select(ws_cols).mean().to_pandas().iloc[0]
    feature_data.append(ws_means.values)
    feature_names.append('WS')
    
    # 4. Precipitation (wet deposition)
    precip_cols = [f'PRECIP_{s}' for s in valid_stations]
    precip_means = spatial_df.select(precip_cols).mean().to_pandas().iloc[0]
    feature_data.append(precip_means.values)
    feature_names.append('PRECIP')
    
    # 5. Pressure (weather systems)
    press_cols = [f'PRESS_{s}' for s in valid_stations]
    press_means = spatial_df.select(press_cols).mean().to_pandas().iloc[0]
    feature_data.append(press_means.values)
    feature_names.append('PRESS')
    
    # 6. Radiation (photochemical reactions)
    rad_cols = [f'RAD_{s}' for s in valid_stations]
    rad_means = spatial_df.select(rad_cols).mean().to_pandas().iloc[0]
    feature_data.append(rad_means.values)
    feature_names.append('RAD')
    
    # Optional: Add Altitude from metadata (terrain effects)
    if 'Altitude' in meta_df.columns:
        feature_data.append(meta_df.loc[valid_stations, 'Altitude'].values)
        feature_names.append('Altitude')
    
    X = np.column_stack(feature_data)
    
    # Standardize features (CRITICAL: mixed units - meters, Kelvin, m/s, Pa, W/m², etc.)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"    ✓ Dependent variable: PM10 (n={len(y)})")
    print(f"    ✓ Independent variables: {feature_names}")
    print(f"    ✓ Features standardized for comparable coefficients")
    print(f"    ✓ Observations: {len(y)}, Features: {X_scaled.shape[1]}")
    
    # Store raw feature means for interpretation
    feature_means = {
        'BLH': blh_means,
        'TEMP': temp_means,
        'WS': ws_means,
        'PRECIP': precip_means,
        'PRESS': press_means,
        'RAD': rad_means
    }
    
    return y, X_scaled, feature_names, pm10_means, scaler, feature_means


def fit_spatial_durbin_model(y, X, w, feature_names):
    """
    Fit Spatial Lag Model with meteorological controls.
    
    The full Spatial Durbin Model includes both Wy and WX terms.
    We use ML_Lag which estimates ρ for the spatial lag (Wy).
    
    With meteorological variables, this model now captures:
    - ρ: "Pure" spatial spillover AFTER controlling for weather
    - β: Direct effects of local meteorology on local PM10
    """
    print("\n[4] Fitting Spatial Durbin Model with Meteorological Controls...")
    print("    Model: y = ρWy + Xβ + ε")
    print(f"    Controls: {feature_names}")
    
    # Fit Maximum Likelihood Spatial Lag model
    model = ML_Lag(y, X, w=w, name_y='PM10_Mean', name_x=feature_names)
    
    # Extract key parameters
    rho = model.rho  # Spatial autoregressive parameter
    betas = model.betas.flatten()
    
    print(f"\n    --- Model Results ---")
    print(f"    Spatial Lag (ρ): {rho:.4f}")
    print(f"    Interpretation: {rho*100:.1f}% of PM10 variance from spatial spillover")
    print(f"    → This is 'pure spillover' AFTER accounting for meteorology")
    
    if rho > 0.5:
        print(f"    ⚠ HIGH SPILLOVER: Spatial dependence remains dominant even after controls!")
    elif rho > 0.3:
        print(f"    → MODERATE SPILLOVER: Significant spatial effects beyond meteorology")
    else:
        print(f"    → LOW SPILLOVER: Meteorological factors explain most variation")
    
    print(f"\n    Model Fit:")
    print(f"      Log-Likelihood: {model.logll:.4f}")
    print(f"      AIC: {model.aic:.4f}")
    
    return model, rho, betas


def interpret_meteorological_effects(betas, feature_names, rho):
    """
    Interpret how meteorological variables affect PM10.
    
    Since features are standardized, coefficients represent:
    "Effect of 1 standard deviation change in X on PM10"
    """
    print("\n[5] Meteorological Influence Analysis...")
    
    print(f"\n    Direct Effects (own-station meteorology → own PM10):")
    print(f"    {'Variable':<12} {'Coef':<12} {'Interpretation'}")
    print(f"    {'-'*70}")
    
    # betas[0] is constant, betas[1:] are feature coefficients
    for i, name in enumerate(feature_names):
        beta = betas[i + 1]  # Skip constant
        
        if name == 'BLH':
            if beta < 0:
                interp = "↑ BLH → ↑ dispersion → ↓ PM10 ✓"
            else:
                interp = "⚠ Unexpected positive (check data)"
        elif name == 'WS':
            if beta < 0:
                interp = "↑ Wind → ↑ dispersion → ↓ PM10 ✓"
            else:
                interp = "⚠ May indicate pollution transport"
        elif name == 'PRECIP':
            if beta < 0:
                interp = "↑ Rain → ↑ washout → ↓ PM10 ✓"
            else:
                interp = "⚠ Unexpected (check precipitation units)"
        elif name == 'TEMP':
            interp = "Complex: affects stability + chemistry"
        elif name == 'PRESS':
            if beta > 0:
                interp = "↑ Pressure → stable air → ↑ PM10"
            else:
                interp = "↓ Pressure → mixing → ↓ PM10"
        elif name == 'RAD':
            interp = "Photochemistry: can increase/decrease"
        elif name == 'Altitude':
            if beta < 0:
                interp = "Higher elevation → cleaner air ✓"
            else:
                interp = "Valley effect or transport"
        else:
            interp = ""
        
        sign = "+" if beta > 0 else ""
        print(f"    {name:<12} {sign}{beta:<11.4f} {interp}")
    
    print(f"\n    Spatial Spillover Parameter (ρ): {rho:.4f}")
    print(f"    → This is the 'contagion' effect INDEPENDENT of local meteorology")
    print(f"    → With ρ={rho:.4f}, {rho*100:.1f}% of PM10 comes from neighbors")


def decompose_spillover(y, X, w, rho, betas, valid_stations, target_stations):
    """
    Decompose PM10 predictions into local and spillover components.
    
    For each station i:
    - Local Effect: X_i * β (own meteorological characteristics)
    - Spatial Spillover: ρ * Σ(w_ij * y_j) (from neighbors)
    
    With meteorological variables, the local effect now represents:
    - How local weather conditions (BLH, WS, TEMP, etc.) affect local PM10
    
    Total predicted: ŷ_i ≈ Local + Spillover
    """
    print("\n[6] Decomposing PM10 into Local (Meteo) and Spillover Components...")
    
    y_flat = y.flatten()
    n = len(y_flat)
    
    # Create dense weight matrix from libpysal weights
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = w.neighbors[i]
        weights = w.weights[i]
        for j, wt in zip(neighbors, weights):
            W[i, j] = wt
    
    # betas from spreg ML_Lag: [constant, x1_coef, x2_coef, ..., rho]
    # The last element is rho, first is constant, middle are X coefficients
    n_x_features = X.shape[1]
    constant = betas[0]
    x_coeffs = betas[1:1+n_x_features]  # Coefficients for X variables
    
    # Local Effect: constant + Xβ
    local_effect = constant + X @ x_coeffs
    
    # Spatial Spillover: ρ * Wy
    Wy = W @ y_flat
    spillover_effect = rho * Wy
    
    # Create decomposition DataFrame
    decomposition_df = pd.DataFrame({
        'Station': valid_stations,
        'Observed_PM10': y_flat,
        'Local_Effect': local_effect.flatten() if hasattr(local_effect, 'flatten') else local_effect,
        'Spatial_Spillover': spillover_effect,
        'Spillover_Pct': (spillover_effect / y_flat) * 100,
        'Wy_Raw': Wy  # Weighted average of neighbors' PM10
    })
    
    # Calculate spillover sources for target stations
    print("\n    --- Decomposition Summary ---")
    print(f"    {'Station':<30} {'Observed':<10} {'Local':<10} {'Spillover':<12} {'Spill%':<8}")
    print("    " + "-" * 70)
    
    for _, row in decomposition_df.iterrows():
        station = row['Station']
        marker = "⭐" if station in target_stations else "  "
        print(f"    {marker}{station:<28} {row['Observed_PM10']:<10.2f} {row['Local_Effect']:<10.2f} "
              f"{row['Spatial_Spillover']:<12.2f} {row['Spillover_Pct']:<8.1f}%")
    
    return decomposition_df, W


def identify_spillover_sources(decomposition_df, W, valid_stations, target_stations, y, feature_means):
    """
    Identify which neighboring stations contribute most to spillover
    for each target station, WITH METEOROLOGICAL CONTEXT.
    
    Enhanced analysis includes:
    - WHY each neighbor contributes (high emissions vs favorable transport)
    - Meteorological profile of top sources
    """
    print("\n[7] Identifying Spillover Sources with Meteorological Context...")
    
    y_flat = y.flatten()
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    
    spillover_sources = {}
    
    for target in target_stations:
        if target not in station_to_idx:
            print(f"    ⚠ Target station '{target}' not found in data")
            continue
            
        idx = station_to_idx[target]
        target_row = decomposition_df[decomposition_df['Station'] == target].iloc[0]
        
        print(f"\n    {'='*70}")
        print(f"    TARGET: {target}")
        print(f"    {'='*70}")
        print(f"    Observed PM10: {target_row['Observed_PM10']:.2f} μg/m³")
        print(f"    Spatial Spillover Received: {target_row['Spatial_Spillover']:.2f} μg/m³ "
              f"({target_row['Spillover_Pct']:.1f}%)")
        
        # Get neighbor contributions WITH meteorological context
        neighbor_contributions = []
        for j in range(len(valid_stations)):
            if W[idx, j] > 0:  # j is a neighbor of target
                neighbor_station = valid_stations[j]
                contribution = W[idx, j] * y_flat[j]  # Weight × neighbor's PM10
                
                # Get meteorological profile of neighbor
                neighbor_blh = feature_means['BLH'].iloc[j]
                neighbor_ws = feature_means['WS'].iloc[j]
                neighbor_temp = feature_means['TEMP'].iloc[j]
                
                # Classify pollution mechanism
                mechanism = classify_pollution_mechanism(
                    y_flat[j], neighbor_blh, neighbor_ws
                )
                
                neighbor_contributions.append({
                    'Neighbor': neighbor_station,
                    'Weight': W[idx, j],
                    'Neighbor_PM10': y_flat[j],
                    'Contribution': contribution,
                    'BLH': neighbor_blh,
                    'WindSpeed': neighbor_ws,
                    'Mechanism': mechanism
                })
        
        contrib_df = pd.DataFrame(neighbor_contributions)
        contrib_df = contrib_df.sort_values('Contribution', ascending=False)
        
        # Calculate percentage of total spillover from each neighbor
        total_spillover = target_row['Wy_Raw']
        contrib_df['Pct_of_Spillover'] = (contrib_df['Contribution'] / total_spillover) * 100
        
        print(f"\n    Top Contributing Neighbors (with Weather Context):")
        print(f"    {'Neighbor':<30} {'PM10':<8} {'BLH':<8} {'WS':<6} {'Contrib':<8} {'Mechanism'}")
        print("    " + "-" * 85)
        
        for _, row in contrib_df.head(8).iterrows():
            print(f"    {row['Neighbor']:<30} {row['Neighbor_PM10']:<8.1f} {row['BLH']:<8.1f} "
                  f"{row['WindSpeed']:<6.2f} {row['Contribution']:<8.2f} {row['Mechanism']}")
        
        spillover_sources[target] = contrib_df
        
        # Save to CSV (with meteorological context)
        csv_path = RESULTS_DIR / f"spillover_sources_{target.replace(' ', '_').replace(',', '').replace('/', '_')}.csv"
        contrib_df.to_csv(csv_path, index=False)
    
    return spillover_sources


def classify_pollution_mechanism(pm10, blh, ws):
    """
    Classify HOW a station becomes a pollution source.
    
    Categories:
    - High emitter + wind transport: High PM10, high wind
    - High emitter + low mixing: High PM10, low BLH
    - Transport corridor: Lower PM10 but high wind (passes pollution through)
    - Atmospheric trapping: Low BLH traps whatever is emitted
    """
    if pm10 > 30 and ws > 2.0:
        return "High emitter + wind transport"
    elif pm10 > 30 and blh < 350:
        return "High emitter + low mixing"
    elif pm10 < 25 and ws > 2.5:
        return "Transport corridor"
    elif blh < 300:
        return "Atmospheric trapping"
    else:
        return "Mixed factors"


def calculate_network_centrality(W, valid_stations, decomposition_df):
    """Calculate network centrality metrics to identify key pollution hubs"""
    print("\n[8] Calculating Network Centrality Metrics...")
    
    n = len(valid_stations)
    
    # Out-degree: How much pollution a station exports
    out_degree = np.sum(W, axis=1)
    
    # In-degree: How much pollution a station imports
    in_degree = np.sum(W, axis=0)
    
    # Eigenvector centrality (influence in the network)
    eigenvalues, eigenvectors = np.linalg.eig(W)
    max_idx = np.argmax(eigenvalues.real)
    eigenvector_centrality = np.abs(eigenvectors[:, max_idx].real)
    eigenvector_centrality = eigenvector_centrality / eigenvector_centrality.max()
    
    centrality_df = pd.DataFrame({
        'Station': valid_stations,
        'Out_Degree': out_degree,
        'In_Degree': in_degree,
        'Eigenvector_Centrality': eigenvector_centrality,
        'PM10_Mean': decomposition_df['Observed_PM10'].values,
        'Spillover_Received': decomposition_df['Spatial_Spillover'].values
    })
    
    # Pollution export potential = PM10 × Out_Degree
    centrality_df['Pollution_Export'] = centrality_df['PM10_Mean'] * centrality_df['Out_Degree']
    centrality_df = centrality_df.sort_values('Pollution_Export', ascending=False)
    
    print("\n    Top Pollution Exporters (Network Hubs):")
    print(f"    {'Station':<35} {'PM10':<10} {'Export Score':<12}")
    print("    " + "-" * 57)
    for _, row in centrality_df.head(10).iterrows():
        print(f"    {row['Station']:<35} {row['PM10_Mean']:<10.2f} {row['Pollution_Export']:<12.2f}")
    
    # Save centrality metrics
    centrality_df.to_csv(RESULTS_DIR / 'network_centrality.csv', index=False)
    
    return centrality_df


def create_visualizations(decomposition_df, spillover_sources, centrality_df, 
                          meta_df, valid_stations, target_stations, coords, W):
    """Create comprehensive visualizations of spillover analysis"""
    print("\n[9] Creating Visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Stacked Bar - Decomposition for Target Stations
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    target_data = decomposition_df[decomposition_df['Station'].isin(target_stations)].copy()
    target_data = target_data.set_index('Station').loc[target_stations]
    
    x = np.arange(len(target_stations))
    width = 0.6
    
    # Stack local and spillover
    bars1 = ax1.bar(x, target_data['Local_Effect'], width, label='Local Effect (Xβ)', 
                    color='#3498db', edgecolor='black')
    bars2 = ax1.bar(x, target_data['Spatial_Spillover'], width, 
                    bottom=target_data['Local_Effect'], label='Spatial Spillover (ρWy)', 
                    color='#e74c3c', edgecolor='black')
    
    # Add observed PM10 line
    ax1.scatter(x, target_data['Observed_PM10'], color='gold', s=150, zorder=5, 
                edgecolor='black', linewidth=2, label='Observed PM10')
    
    # Labels and formatting
    ax1.set_xlabel('Target Station', fontsize=12)
    ax1.set_ylabel('PM10 Concentration (μg/m³)', fontsize=12)
    ax1.set_title('PM10 Decomposition: Local vs Spillover Effects\n(Target Stations)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('_', '\n') for s in target_stations], fontsize=10)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add percentage annotations
    for i, (idx, row) in enumerate(target_data.iterrows()):
        pct = row['Spillover_Pct']
        ax1.annotate(f'{pct:.1f}%\nspillover', 
                     xy=(i, row['Local_Effect'] + row['Spatial_Spillover']/2),
                     ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    plt.tight_layout()
    fig1.savefig(ASSETS_DIR / 'spillover_decomposition_targets.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: spillover_decomposition_targets.png")
    plt.close(fig1)
    
    # -------------------------------------------------------------------------
    # FIGURE 2: All Stations Spillover Intensity
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    sorted_df = decomposition_df.sort_values('Spillover_Pct', ascending=True)
    colors = ['#e74c3c' if s in target_stations else '#3498db' for s in sorted_df['Station']]
    
    bars = ax2.barh(range(len(sorted_df)), sorted_df['Spillover_Pct'], color=colors, edgecolor='black')
    
    ax2.set_yticks(range(len(sorted_df)))
    ax2.set_yticklabels(sorted_df['Station'], fontsize=8)
    ax2.set_xlabel('Spillover Percentage (%)', fontsize=12)
    ax2.set_title('Spatial Spillover Intensity by Station\n(Red = Target Stations)', fontsize=14, fontweight='bold')
    ax2.axvline(x=sorted_df['Spillover_Pct'].mean(), color='gold', linestyle='--', 
                linewidth=2, label=f"Mean: {sorted_df['Spillover_Pct'].mean():.1f}%")
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    fig2.savefig(ASSETS_DIR / 'spillover_intensity_all_stations.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: spillover_intensity_all_stations.png")
    plt.close(fig2)
    
    # -------------------------------------------------------------------------
    # FIGURE 3: Spatial Map with Spillover Visualization
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    
    # Base scatter - all stations colored by spillover percentage
    scatter = ax3.scatter(coords[:, 0], coords[:, 1], 
                          c=decomposition_df['Spillover_Pct'], 
                          cmap='RdYlBu_r', s=150, edgecolor='black', 
                          linewidth=1, alpha=0.8)
    
    # Highlight target stations
    target_coords = []
    for target in target_stations:
        if target in valid_stations:
            idx = valid_stations.index(target)
            target_coords.append(coords[idx])
            ax3.scatter(coords[idx, 0], coords[idx, 1], 
                        s=400, facecolors='none', edgecolors='red', 
                        linewidth=3, label=target if len(target_coords) == 1 else None)
            ax3.annotate(target, (coords[idx, 0], coords[idx, 1]), 
                         fontsize=9, fontweight='bold', 
                         xytext=(5, 5), textcoords='offset points')
    
    # Draw weighted edges for target stations
    for target in target_stations:
        if target in valid_stations:
            idx = valid_stations.index(target)
            for j in range(len(valid_stations)):
                if W[idx, j] > 0:
                    ax3.plot([coords[idx, 0], coords[j, 0]], 
                             [coords[idx, 1], coords[j, 1]], 
                             'r-', alpha=W[idx, j]*2, linewidth=W[idx, j]*5)
    
    cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
    cbar.set_label('Spillover Percentage (%)', fontsize=11)
    
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.set_title('Spatial Distribution of PM10 Spillover Effects\n(Target Stations Highlighted)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig3.savefig(ASSETS_DIR / 'spatial_spillover_map.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: spatial_spillover_map.png")
    plt.close(fig3)
    
    # -------------------------------------------------------------------------
    # FIGURE 4: Local vs Spillover Scatter with Quadrants
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    
    colors = ['#e74c3c' if s in target_stations else '#3498db' for s in decomposition_df['Station']]
    sizes = [200 if s in target_stations else 80 for s in decomposition_df['Station']]
    
    ax4.scatter(decomposition_df['Local_Effect'], decomposition_df['Spatial_Spillover'], 
                c=colors, s=sizes, edgecolor='black', alpha=0.7)
    
    # Add station labels for targets
    for _, row in decomposition_df.iterrows():
        if row['Station'] in target_stations:
            ax4.annotate(row['Station'], 
                         (row['Local_Effect'], row['Spatial_Spillover']),
                         fontsize=9, fontweight='bold',
                         xytext=(5, 5), textcoords='offset points')
    
    # Add quadrant lines at medians
    med_local = decomposition_df['Local_Effect'].median()
    med_spill = decomposition_df['Spatial_Spillover'].median()
    ax4.axvline(x=med_local, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=med_spill, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax4.text(decomposition_df['Local_Effect'].max(), decomposition_df['Spatial_Spillover'].max(),
             'High Local\nHigh Spillover', ha='right', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax4.text(decomposition_df['Local_Effect'].min(), decomposition_df['Spatial_Spillover'].max(),
             'Low Local\nHigh Spillover', ha='left', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
    
    ax4.set_xlabel('Local Effect (Xβ) μg/m³', fontsize=12)
    ax4.set_ylabel('Spatial Spillover (ρWy) μg/m³', fontsize=12)
    ax4.set_title('Local vs Spillover Effects by Station\n(Red = Target Stations)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig4.savefig(ASSETS_DIR / 'local_vs_spillover_scatter.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: local_vs_spillover_scatter.png")
    plt.close(fig4)
    
    # -------------------------------------------------------------------------
    # FIGURE 5: Spillover Source Contributions for Each Target
    # -------------------------------------------------------------------------
    n_targets = len([t for t in target_stations if t in spillover_sources])
    if n_targets > 0:
        fig5, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for i, target in enumerate(target_stations):
            if target not in spillover_sources:
                continue
                
            ax = axes[i]
            contrib_df = spillover_sources[target].head(6)
            
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(contrib_df)))
            bars = ax.barh(range(len(contrib_df)), contrib_df['Contribution'], 
                          color=colors, edgecolor='black')
            
            ax.set_yticks(range(len(contrib_df)))
            ax.set_yticklabels(contrib_df['Neighbor'], fontsize=9)
            ax.set_xlabel('Contribution to Spillover (μg/m³)', fontsize=10)
            ax.set_title(f'Spillover Sources: {target}', fontsize=12, fontweight='bold')
            
            # Add percentage labels
            for j, (_, row) in enumerate(contrib_df.iterrows()):
                ax.text(row['Contribution'] + 0.1, j, f"{row['Pct_of_Spillover']:.1f}%", 
                        va='center', fontsize=9)
            
            ax.invert_yaxis()
        
        plt.tight_layout()
        fig5.savefig(ASSETS_DIR / 'spillover_source_contributions.png', dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: spillover_source_contributions.png")
        plt.close(fig5)


def generate_summary_statistics(decomposition_df, target_stations, rho):
    """Generate summary statistics for the analysis"""
    print("\n" + "=" * 70)
    print("SUMMARY: SPILLOVER ANALYSIS RESULTS")
    print("=" * 70)
    
    print(f"\n    Spatial Autoregressive Parameter (ρ): {rho:.4f}")
    print(f"    → {rho*100:.1f}% of pollution variance explained by spatial spillover")
    
    print(f"\n    Overall Statistics:")
    print(f"    - Mean Observed PM10: {decomposition_df['Observed_PM10'].mean():.2f} μg/m³")
    print(f"    - Mean Spillover Effect: {decomposition_df['Spatial_Spillover'].mean():.2f} μg/m³")
    print(f"    - Mean Spillover %: {decomposition_df['Spillover_Pct'].mean():.1f}%")
    print(f"    - Max Spillover %: {decomposition_df['Spillover_Pct'].max():.1f}%")
    print(f"    - Min Spillover %: {decomposition_df['Spillover_Pct'].min():.1f}%")
    
    print(f"\n    Target Stations Summary:")
    target_df = decomposition_df[decomposition_df['Station'].isin(target_stations)]
    for _, row in target_df.iterrows():
        print(f"\n    📍 {row['Station']}:")
        print(f"       - Observed PM10: {row['Observed_PM10']:.2f} μg/m³")
        print(f"       - From Local Sources: {row['Local_Effect']:.2f} μg/m³")
        print(f"       - From Spatial Spillover: {row['Spatial_Spillover']:.2f} μg/m³ ({row['Spillover_Pct']:.1f}%)")


def save_results(decomposition_df, centrality_df, target_stations):
    """Save all results to CSV files"""
    print("\n[10] Saving Results...")
    
    # Full decomposition
    decomposition_df.to_csv(RESULTS_DIR / 'station_spillover_decomposition.csv', index=False)
    print(f"    ✓ station_spillover_decomposition.csv")
    
    # Target stations summary
    target_summary = decomposition_df[decomposition_df['Station'].isin(target_stations)]
    target_summary.to_csv(RESULTS_DIR / 'target_stations_spillover.csv', index=False)
    print(f"    ✓ target_stations_spillover.csv")
    
    # Top external polluters (high PM10 + high network centrality)
    top_exporters = centrality_df.head(10)[['Station', 'PM10_Mean', 'Pollution_Export', 'Eigenvector_Centrality']]
    top_exporters.to_csv(RESULTS_DIR / 'top_external_polluters.csv', index=False)
    print(f"    ✓ top_external_polluters.csv")


def main():
    """Main execution function"""
    with open(OUTPUT_FILE, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(sys.stdout, f)
        
        try:
            # Load data
            spatial_df, meta_df, valid_stations, available_targets = load_data()
            
            # Create spatial weights
            w, coords = create_spatial_weights(meta_df, valid_stations)
            
            # Prepare model data with meteorological variables
            y, X, feature_names, pm10_means, scaler, feature_means = prepare_model_data(spatial_df, meta_df, valid_stations)
            
            # Fit Spatial Durbin Model with meteorological controls
            model, rho, betas = fit_spatial_durbin_model(y, X, w, feature_names)
            
            # Interpret meteorological effects
            interpret_meteorological_effects(betas, feature_names, rho)
            
            # Decompose spillover effects
            decomposition_df, W = decompose_spillover(y, X, w, rho, betas, valid_stations, available_targets)
            
            # Identify spillover sources with meteorological context
            spillover_sources = identify_spillover_sources(decomposition_df, W, valid_stations, available_targets, y, feature_means)
            
            # Calculate network centrality
            centrality_df = calculate_network_centrality(W, valid_stations, decomposition_df)
            
            # Create visualizations
            create_visualizations(decomposition_df, spillover_sources, centrality_df,
                                  meta_df, valid_stations, available_targets, coords, W)
            
            # Generate summary
            generate_summary_statistics(decomposition_df, available_targets, rho)
            
            # Save results
            save_results(decomposition_df, centrality_df, available_targets)
            
            print("\n" + "=" * 70)
            print("ANALYSIS COMPLETE")
            print("=" * 70)
            print(f"\n    📁 Text Output: {OUTPUT_FILE}")
            print(f"    📊 CSV Results: {RESULTS_DIR}")
            print(f"    🖼️  Visualizations: {ASSETS_DIR}")
            print("\n")
            
        finally:
            sys.stdout = original_stdout
    
    print(f"Script completed successfully!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Visualizations saved to: {ASSETS_DIR}")


if __name__ == '__main__':
    main()
