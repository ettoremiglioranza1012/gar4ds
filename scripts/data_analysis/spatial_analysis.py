"""
SPATIAL ANALYSIS - REFACTORED FOR PANEL DATA
=============================================

This script performs spatial autocorrelation and clustering analysis on the 
panel dataset (MultiIndex: week_start, station_id).

STRATEGY:
1. Load panel_data_matrix.parquet (MultiIndex format)
2. Time-aggregate to create cross-sectional "Station Profile" (mean over weeks)
3. Build KNN spatial weights matrix (k=6) from station coordinates
4. Calculate Moran's I (LISA) for PM10 and meteorological variables
5. Perform multivariate K-Means clustering (atmospheric regime detection)
6. Save spatial_weights_knn6.gal file for Spatial Durbin Model

OUTPUTS:
- Verbose text log: results/spatial_analysis/spatial_analysis_results.txt
- CSV results: results/spatial_analysis/
- Visualizations: assets/spatial_analysis/
- Weights file: weights/spatial_weights_knn6.gal
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
ASSETS_DIR = PROJECT_DIR / 'assets' / 'spatial_analysis'
RESULTS_DIR = PROJECT_DIR / 'results' / 'spatial_analysis'
WEIGHTS_DIR = PROJECT_DIR / 'weights'

# Create directories
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Output file for verbose logging
output_file = RESULTS_DIR / 'spatial_analysis_results.txt'


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

def load_panel_data(data_path):
    """
    Load panel data matrix with MultiIndex (week_start, station_id).
    
    Args:
        data_path: Path to panel_data_matrix.parquet
        
    Returns:
        pd.DataFrame with MultiIndex
    """
    print("\n[1] LOADING PANEL DATA")
    print("=" * 80)
    print(f"    Loading: {data_path}")
    
    df = pd.read_parquet(data_path)
    
    print(f"    ✓ Shape: {df.shape} (observations × variables)")
    print(f"    ✓ Index: {df.index.names}")
    print(f"    ✓ Columns: {list(df.columns)}")
    
    # Check index structure
    if len(df.index.names) == 2:
        print(f"    ✓ MultiIndex confirmed: {df.index.names}")
        n_weeks = df.index.get_level_values(0).nunique()
        n_stations = df.index.get_level_values(1).nunique()
        print(f"    ✓ Time periods: {n_weeks}")
        print(f"    ✓ Stations: {n_stations}")
    else:
        print(f"    ⚠ Warning: Expected MultiIndex, found {df.index.names}")
    
    return df


def load_station_metadata(meta_path):
    """
    Load station metadata with coordinates from GeoJSON.
    
    Args:
        meta_path: Path to pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson
        
    Returns:
        pd.DataFrame with station_code as index
    """
    print("\n[2] LOADING STATION METADATA")
    print("=" * 80)
    print(f"    Loading: {meta_path}")
    
    # Read GeoJSON with geopandas
    gdf = gpd.read_file(meta_path)
    
    # Extract coordinates from geometry
    gdf['Longitude'] = gdf.geometry.x
    gdf['Latitude'] = gdf.geometry.y
    
    # Build metadata DataFrame with all available fields
    meta_dict = {
        'station_code': gdf['station_code'],
        'station_name': gdf['station_name'],
        'region': gdf['region'],
        'Longitude': gdf['Longitude'],
        'Latitude': gdf['Latitude']
    }
    
    # Add elevation and classification fields if available
    if 'elevation' in gdf.columns:
        meta_dict['elevation'] = gdf['elevation']
    if 'terrain_type' in gdf.columns:
        meta_dict['terrain_type'] = gdf['terrain_type']
    if 'area_type' in gdf.columns:
        meta_dict['area_type'] = gdf['area_type']
    
    meta_df = pd.DataFrame(meta_dict).set_index('station_code')
    
    print(f"    ✓ Loaded {len(meta_df)} stations")
    print(f"    ✓ Columns: {list(meta_df.columns)}")
    print(f"    ✓ Regions: {meta_df['region'].unique().tolist()}")
    print(f"    ✓ Coordinate range:")
    print(f"        Longitude: [{meta_df['Longitude'].min():.4f}, {meta_df['Longitude'].max():.4f}]")
    print(f"        Latitude: [{meta_df['Latitude'].min():.4f}, {meta_df['Latitude'].max():.4f}]")
    
    # Display elevation statistics if available
    if 'elevation' in meta_df.columns:
        print(f"    ✓ Elevation range: [{meta_df['elevation'].min():.1f}, {meta_df['elevation'].max():.1f}] meters")
        print(f"    ✓ Average elevation: {meta_df['elevation'].mean():.1f} meters")
    
    # Display terrain classification if available
    if 'terrain_type' in meta_df.columns:
        terrain_counts = meta_df['terrain_type'].value_counts()
        print(f"    ✓ Terrain distribution:")
        for terrain, count in terrain_counts.items():
            print(f"        {terrain}: {count} stations")
    
    return meta_df


def create_cross_sectional_view(panel_df):
    """
    Time-aggregate panel data to create cross-sectional station profiles.
    
    Computes: X̄ᵢ = (1/T) Σₜ Xᵢₜ
    
    Args:
        panel_df: Panel DataFrame with MultiIndex (time, station_id)
        
    Returns:
        pd.DataFrame with one row per station (station_id as index)
    """
    print("\n[3] CREATING CROSS-SECTIONAL VIEW")
    print("=" * 80)
    print("    Strategy: Time-aggregate panel data (mean over all weeks)")
    print("    Formula: X̄ᵢ = (1/T) Σₜ Xᵢₜ")
    
    # Group by station (level 1 of MultiIndex) and compute mean
    station_profiles = panel_df.groupby(level=1).mean()
    
    print(f"\n    ✓ Aggregation complete")
    print(f"    ✓ Output shape: {station_profiles.shape} (stations × variables)")
    print(f"    ✓ Stations: {list(station_profiles.index)}")
    print(f"    ✓ Variables: {list(station_profiles.columns)}")
    
    # Display summary statistics
    print(f"\n    Summary statistics (cross-sectional means):")
    print(station_profiles.describe().T[['mean', 'std', 'min', 'max']].to_string())
    
    return station_profiles


def align_stations_with_metadata(station_profiles, meta_df):
    """
    Ensure stations in data match stations in metadata.
    
    Args:
        station_profiles: Cross-sectional DataFrame
        meta_df: Metadata DataFrame
        
    Returns:
        Tuple of (aligned_profiles, aligned_metadata, valid_stations)
    """
    print("\n[4] ALIGNING STATIONS WITH METADATA")
    print("=" * 80)
    
    data_stations = set(station_profiles.index)
    meta_stations = set(meta_df.index)
    
    valid_stations = sorted(data_stations & meta_stations)
    missing_in_meta = data_stations - meta_stations
    missing_in_data = meta_stations - data_stations
    
    print(f"    Data stations: {len(data_stations)}")
    print(f"    Metadata stations: {len(meta_stations)}")
    print(f"    Valid stations (intersection): {len(valid_stations)}")
    
    if missing_in_meta:
        print(f"    ⚠ Stations in data but not in metadata: {missing_in_meta}")
    if missing_in_data:
        print(f"    ℹ Stations in metadata but not in data: {missing_in_data}")
    
    # Filter to valid stations
    aligned_profiles = station_profiles.loc[valid_stations]
    aligned_metadata = meta_df.loc[valid_stations]
    
    print(f"\n    ✓ Aligned dataset shape: {aligned_profiles.shape}")
    print(f"    ✓ Valid stations: {valid_stations}")
    
    return aligned_profiles, aligned_metadata, valid_stations


def build_spatial_weights(coords, k=6):
    """
    Build KNN spatial weights matrix from station coordinates.
    
    Args:
        coords: Numpy array of shape (n_stations, 2) with (longitude, latitude)
        k: Number of nearest neighbors (default: 6)
        
    Returns:
        libpysal.weights.KNN object
    """
    print("\n[5] BUILDING SPATIAL WEIGHTS MATRIX")
    print("=" * 80)
    print(f"    Method: K-Nearest Neighbors (KNN)")
    print(f"    K: {k} neighbors")
    print(f"    Justification: KNN ensures logical connectivity in Alpine terrain")
    print(f"                   (avoids linking valleys across mountains)")
    
    w = KNN.from_array(coords, k=k)
    w.transform = 'r'  # Row-standardize weights
    
    print(f"\n    ✓ Weights matrix created")
    print(f"    ✓ Number of observations: {w.n}")
    print(f"    ✓ Average neighbors: {w.mean_neighbors:.2f}")
    print(f"    ✓ Transformation: Row-standardized")
    
    # Calculate distance statistics
    distances = []
    for i in range(w.n):
        origin = coords[i]
        for j in w.neighbors[i]:
            neighbor = coords[j]
            dist = np.sqrt(((origin - neighbor)**2).sum())
            distances.append(dist)
    
    distances = np.array(distances)
    print(f"\n    Connection distance statistics:")
    print(f"        Mean: {distances.mean():.4f}°")
    print(f"        Median: {np.median(distances):.4f}°")
    print(f"        Min: {distances.min():.4f}°")
    print(f"        Max: {distances.max():.4f}°")
    
    # Approximate conversion to km (1° ≈ 111 km at this latitude)
    print(f"\n    Approximate distance in km (1° ≈ 111 km):")
    print(f"        Mean: {distances.mean() * 111:.2f} km")
    print(f"        Median: {np.median(distances) * 111:.2f} km")
    print(f"        Max: {distances.max() * 111:.2f} km")
    
    return w


def save_spatial_weights_gal(w, station_ids, output_path):
    """
    Save spatial weights matrix in GAL format for Spatial Durbin Model.
    
    Args:
        w: libpysal.weights object
        station_ids: List of station identifiers
        output_path: Path to save .gal file
    """
    print("\n[6] SAVING SPATIAL WEIGHTS (.GAL FORMAT)")
    print("=" * 80)
    print(f"    Output: {output_path}")
    print(f"    Purpose: Input for Spatial Durbin Model regression")
    
    # Write GAL file
    with open(output_path, 'w') as f:
        # Header: number of observations
        f.write(f"0 {w.n} {output_path.name} station_id\n")
        
        # For each station, write neighbors
        for i, station_id in enumerate(station_ids):
            neighbors = w.neighbors[i]
            n_neighbors = len(neighbors)
            f.write(f"{station_id} {n_neighbors}\n")
            
            # Write neighbor IDs
            neighbor_ids = [station_ids[j] for j in neighbors]
            f.write(" ".join(neighbor_ids) + "\n")
    
    print(f"    ✓ GAL file saved successfully")
    print(f"    ✓ Format: Station_ID followed by neighbor Station_IDs")
    print(f"    ✓ Total connections: {sum(len(w.neighbors[i]) for i in range(w.n))}")


def calculate_global_morans_i(station_profiles, w):
    """
    Calculate Global Moran's I for all variables.
    
    Tests spatial autocorrelation: Are similar values clustered in space?
    
    Args:
        station_profiles: Cross-sectional DataFrame
        w: Spatial weights matrix
        
    Returns:
        pd.DataFrame with Moran's I statistics for each variable
    """
    print("\n[7] CALCULATING GLOBAL MORAN'S I")
    print("=" * 80)
    print("    Purpose: Test spatial autocorrelation (justify spatial lag term ρWy)")
    print("    Null hypothesis: Values are randomly distributed in space")
    print("    Interpretation: I > 0 → Clustering, I < 0 → Dispersion, I ≈ 0 → Random")
    
    results = []
    
    for var in station_profiles.columns:
        print(f"\n    --- {var} ---")
        
        values = station_profiles[var].values
        
        # Calculate Global Moran's I
        mi = Moran(values, w)
        
        pattern = 'Random'
        if mi.p_sim < 0.05:
            pattern = 'Clustered' if mi.I > 0 else 'Dispersed'
        
        print(f"        Moran's I: {mi.I:.4f}")
        print(f"        P-value: {mi.p_sim:.4f}")
        print(f"        Z-score: {mi.z_sim:.4f}")
        print(f"        Pattern: {pattern} {'***' if mi.p_sim < 0.001 else '**' if mi.p_sim < 0.01 else '*' if mi.p_sim < 0.05 else ''}")
        
        results.append({
            'Variable': var,
            'Morans_I': mi.I,
            'Expected_I': mi.EI,
            'Variance_I': mi.VI_sim,
            'Z_score': mi.z_sim,
            'P_value': mi.p_sim,
            'Significant': 'Yes' if mi.p_sim < 0.05 else 'No',
            'Pattern': pattern
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("GLOBAL MORAN'S I SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return results_df


def calculate_local_morans_i(station_profiles, w, station_ids):
    """
    Calculate Local Moran's I (LISA) for all variables.
    
    Identifies spatial clusters and outliers for each variable.
    
    Args:
        station_profiles: Cross-sectional DataFrame
        w: Spatial weights matrix
        station_ids: List of station identifiers
        
    Returns:
        pd.DataFrame with LISA statistics for each station-variable pair
    """
    print("\n[8] CALCULATING LOCAL MORAN'S I (LISA)")
    print("=" * 80)
    print("    Purpose: Identify spatial clusters and outliers")
    print("    Cluster types:")
    print("        High-High: High values surrounded by high values (hotspots)")
    print("        Low-Low: Low values surrounded by low values (coldspots)")
    print("        High-Low: High values surrounded by low values (spatial outliers)")
    print("        Low-High: Low values surrounded by high values (spatial outliers)")
    
    all_results = []
    
    for var in station_profiles.columns:
        print(f"\n    --- {var} ---")
        
        values = station_profiles[var].values
        
        # Calculate Local Moran's I
        lisa = Moran_Local(values, w)
        
        # Classify clusters
        sig = lisa.p_sim < 0.05
        hotspots = sig & (lisa.q == 1)  # High-High
        coldspots = sig & (lisa.q == 3)  # Low-Low
        high_low = sig & (lisa.q == 4)  # High-Low (outliers)
        low_high = sig & (lisa.q == 2)  # Low-High (outliers)
        
        print(f"        High-High (hotspots): {hotspots.sum()}")
        print(f"        Low-Low (coldspots): {coldspots.sum()}")
        print(f"        High-Low (outliers): {high_low.sum()}")
        print(f"        Low-High (outliers): {low_high.sum()}")
        print(f"        Not significant: {(~sig).sum()}")
        
        # Store results for each station
        for i, station_id in enumerate(station_ids):
            cluster_type = 'Not Significant'
            if hotspots[i]:
                cluster_type = 'High-High'
            elif coldspots[i]:
                cluster_type = 'Low-Low'
            elif high_low[i]:
                cluster_type = 'High-Low'
            elif low_high[i]:
                cluster_type = 'Low-High'
            
            all_results.append({
                'Station': station_id,
                'Variable': var,
                'Value': values[i],
                'Local_Morans_I': lisa.Is[i],
                'P_value': lisa.p_sim[i],
                'Z_score': lisa.z_sim[i],
                'Quadrant': lisa.q[i],
                'Cluster_Type': cluster_type,
                'Significant': sig[i]
            })
    
    results_df = pd.DataFrame(all_results)
    
    print("\n    ✓ LISA analysis complete for all variables")
    
    return results_df


def create_lisa_maps(station_profiles, w, station_ids, coords, assets_dir):
    """
    Create LISA cluster maps for PM10 (primary focus).
    
    Args:
        station_profiles: Cross-sectional DataFrame
        w: Spatial weights matrix
        station_ids: List of station identifiers
        coords: Station coordinates
        assets_dir: Directory to save plots
    """
    print("\n[9] CREATING LISA CLUSTER MAPS")
    print("=" * 80)
    print("    Focus: PM10 (primary pollutant)")
    
    if 'pm10' not in station_profiles.columns:
        print("    ⚠ PM10 not found in variables - skipping visualization")
        return
    
    var = 'pm10'
    values = station_profiles[var].values
    
    # Calculate LISA
    lisa = Moran_Local(values, w)
    
    # Classify clusters
    sig = lisa.p_sim < 0.05
    hotspots = sig & (lisa.q == 1)
    coldspots = sig & (lisa.q == 3)
    high_low = sig & (lisa.q == 4)
    low_high = sig & (lisa.q == 2)
    not_sig = ~sig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot not significant stations
    if not_sig.sum() > 0:
        ax.scatter(coords[not_sig, 0], coords[not_sig, 1],
                  c='lightgrey', edgecolor='k', s=80, alpha=0.6,
                  label='Not Significant', zorder=1)
    
    # Plot clusters
    if hotspots.sum() > 0:
        ax.scatter(coords[hotspots, 0], coords[hotspots, 1],
                  c='red', edgecolor='k', s=150, label='High-High (Hotspot)',
                  zorder=3, marker='s')
        for idx in np.where(hotspots)[0]:
            ax.annotate(station_ids[idx], (coords[idx, 0], coords[idx, 1]),
                       fontsize=8, fontweight='bold',
                       xytext=(3, 3), textcoords='offset points')
    
    if coldspots.sum() > 0:
        ax.scatter(coords[coldspots, 0], coords[coldspots, 1],
                  c='blue', edgecolor='k', s=150, label='Low-Low (Coldspot)',
                  zorder=3, marker='s')
    
    if high_low.sum() > 0:
        ax.scatter(coords[high_low, 0], coords[high_low, 1],
                  c='orange', edgecolor='k', s=120, label='High-Low (Outlier)',
                  zorder=2, marker='^')
    
    if low_high.sum() > 0:
        ax.scatter(coords[low_high, 0], coords[low_high, 1],
                  c='cyan', edgecolor='k', s=120, label='Low-High (Outlier)',
                  zorder=2, marker='v')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Spatial Clustering: PM10\n(LISA Analysis)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = assets_dir / 'lisa_cluster_map_pm10.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ LISA cluster map saved: {plot_path.name}")


def create_morans_i_comparison(global_morans_df, assets_dir):
    """
    Create bar chart comparing Moran's I across variables.
    
    Args:
        global_morans_df: DataFrame with Moran's I statistics
        assets_dir: Directory to save plot
    """
    print("\n[10] CREATING MORAN'S I COMPARISON PLOT")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars by significance
    colors = ['red' if p < 0.05 else 'gray' for p in global_morans_df['P_value']]
    
    bars = ax.barh(global_morans_df['Variable'], global_morans_df['Morans_I'],
                   color=colors, edgecolor='black')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Moran's I", fontsize=12)
    ax.set_title("Global Spatial Autocorrelation by Variable\n(Red = Significant p<0.05)",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (var, val, p) in enumerate(zip(global_morans_df['Variable'],
                                           global_morans_df['Morans_I'],
                                           global_morans_df['P_value'])):
        label = f"{val:.3f}*" if p < 0.05 else f"{val:.3f}"
        ax.text(val + 0.01, i, label, va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = assets_dir / 'global_morans_i_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Comparison plot saved: {plot_path.name}")


def perform_multivariate_clustering(station_profiles, coords, station_ids,
                                    assets_dir, results_dir):
    """
    Perform K-Means clustering on meteorological profiles.
    
    Identifies atmospheric regimes (e.g., mountain vs plain stations).
    
    Args:
        station_profiles: Cross-sectional DataFrame
        coords: Station coordinates
        station_ids: List of station identifiers
        assets_dir: Directory for visualizations
        results_dir: Directory for CSV results
        
    Returns:
        Tuple of (cluster_assignments_df, cluster_profiles_df)
    """
    print("\n[11] MULTIVARIATE CLUSTERING ANALYSIS")
    print("=" * 80)
    print("    Purpose: Detect atmospheric regimes (mountain vs plain)")
    print("    Method: K-Means clustering on standardized meteorological profiles")
    print("    Output: Cluster IDs → Dummy variables for regression")
    
    # Select clustering variables (exclude pm10 to avoid circularity)
    cluster_vars = [col for col in station_profiles.columns if col != 'pm10']
    
    if len(cluster_vars) < 2:
        print("    ⚠ Insufficient variables for clustering - skipping")
        return None, None
    
    print(f"\n    Clustering variables: {cluster_vars}")
    
    # Extract feature matrix
    X = station_profiles[cluster_vars].values
    
    print(f"    Feature matrix shape: {X.shape} (stations × variables)")
    
    # Standardize features
    print("\n    Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("    ✓ Features standardized (mean=0, std=1)")
    
    # Determine optimal number of clusters using elbow method
    print("\n    Determining optimal number of clusters...")
    optimal_k = determine_optimal_k(X_scaled, max_k=8)
    print(f"    ✓ Optimal K: {optimal_k}")
    
    # Perform K-Means clustering
    print(f"\n    Performing K-Means clustering (k={optimal_k})...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Count stations per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n    Cluster distribution:")
    for c, count in zip(unique, counts):
        print(f"        Cluster {c}: {count} stations")
    
    # PCA for visualization
    print("\n    Performing PCA for 2D visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_
    print(f"    ✓ PC1 explains {var_explained[0]:.1%} variance")
    print(f"    ✓ PC2 explains {var_explained[1]:.1%} variance")
    print(f"    ✓ Total explained: {var_explained.sum():.1%}")
    
    # Create cluster assignments DataFrame
    cluster_df = pd.DataFrame({
        'Station': station_ids,
        'Cluster': cluster_labels,
        'Longitude': coords[:, 0],
        'Latitude': coords[:, 1],
        'PCA_1': X_pca[:, 0],
        'PCA_2': X_pca[:, 1]
    })
    
    # Add original variable values
    for var in cluster_vars:
        cluster_df[var] = station_profiles[var].values
    
    if 'pm10' in station_profiles.columns:
        cluster_df['pm10'] = station_profiles['pm10'].values
    
    # Calculate cluster profiles
    print("\n    Calculating cluster profiles...")
    profiles = []
    for cluster_id in sorted(unique):
        mask = cluster_labels == cluster_id
        cluster_data = cluster_df[mask]
        
        profile = {
            'Cluster': cluster_id,
            'N_Stations': mask.sum()
        }
        
        # Mean and std for each variable
        for var in cluster_vars:
            profile[f'{var}_mean'] = cluster_data[var].mean()
            profile[f'{var}_std'] = cluster_data[var].std()
        
        if 'pm10' in cluster_df.columns:
            profile['pm10_mean'] = cluster_data['pm10'].mean()
            profile['pm10_std'] = cluster_data['pm10'].std()
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    print("\n" + "=" * 80)
    print("CLUSTER PROFILES - ATMOSPHERIC REGIMES")
    print("=" * 80)
    print(profiles_df.to_string(index=False))
    
    # Create visualizations
    create_clustering_visualizations(cluster_df, X_pca, cluster_labels, 
                                     optimal_k, var_explained, coords,
                                     cluster_vars, profiles_df, assets_dir)
    
    return cluster_df, profiles_df


def determine_optimal_k(X, max_k=10):
    """Determine optimal number of clusters using elbow method"""
    inertias = []
    K_range = range(2, min(max_k + 1, len(X) // 2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection
    if len(inertias) > 2:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        optimal_k = np.argmin(second_diffs) + 2
    else:
        optimal_k = 3
    
    return min(optimal_k, 6)  # Cap at 6 for interpretability


def create_clustering_visualizations(cluster_df, X_pca, cluster_labels,
                                     n_clusters, var_explained, coords,
                                     cluster_vars, profiles_df, assets_dir):
    """Create all clustering visualization plots"""
    
    print("\n[12] CREATING CLUSTERING VISUALIZATIONS")
    print("=" * 80)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # 1. PCA Scatter Plot
    print("    Creating PCA scatter plot...")
    fig, ax = plt.subplots(figsize=(12, 9))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[colors[cluster_id]], s=100, edgecolor='black',
                  label=f'Cluster {cluster_id}', alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
    ax.set_title('Multivariate Station Clustering\n(PCA Projection)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(assets_dir / 'cluster_pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ PCA scatter saved")
    
    # 2. Spatial Cluster Map
    print("    Creating spatial cluster map...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            cluster_coords = coords[mask]
            ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                      c=[colors[cluster_id]], s=150, edgecolor='black',
                      linewidth=1.5, label=f'Cluster {cluster_id}',
                      alpha=0.8, zorder=3)
            
            # Annotate some stations
            cluster_stations = cluster_df[mask]['Station'].values
            for i, (lon, lat) in enumerate(cluster_coords[:3]):
                ax.annotate(cluster_stations[i], (lon, lat),
                           fontsize=8, xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Spatial Distribution of Multivariate Clusters\n'
                '(Stations with similar environmental profiles)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(assets_dir / 'cluster_spatial_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Spatial cluster map saved")
    
    # 3. Cluster Profile Heatmap
    print("    Creating cluster profile heatmap...")
    mean_cols = [f'{var}_mean' for var in cluster_vars]
    
    if all(col in profiles_df.columns for col in mean_cols):
        heatmap_data = profiles_df[mean_cols].values
        
        # Normalize by column for visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        heatmap_normalized = scaler.fit_transform(heatmap_data.T).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(heatmap_normalized, cmap='RdYlBu_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(cluster_vars)))
        ax.set_yticks(np.arange(n_clusters))
        ax.set_xticklabels(cluster_vars, fontsize=11)
        ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=11)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value\n(0=Low, 1=High)', fontsize=10)
        
        # Add text annotations
        for i in range(n_clusters):
            for j in range(len(cluster_vars)):
                ax.text(j, i, f'{heatmap_normalized[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=9)
        
        ax.set_title('Cluster Environmental Profiles\n(Normalized Mean Values)',
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'cluster_profile_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    ✓ Cluster heatmap saved")
    
    print("    ✓ All clustering visualizations complete")


def create_connectivity_visualization(w, coords, station_ids, assets_dir):
    """
    Create spider map showing KNN connections.
    
    Args:
        w: Spatial weights matrix
        coords: Station coordinates
        station_ids: List of station identifiers
        assets_dir: Directory to save plot
    """
    print("\n[13] CREATING CONNECTIVITY VISUALIZATION")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot edges (connections)
    print("    Drawing connections...")
    for i in range(w.n):
        origin = coords[i]
        for j in w.neighbors[i]:
            dest = coords[j]
            ax.plot([origin[0], dest[0]], [origin[1], dest[1]],
                   color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Plot nodes (stations)
    print("    Drawing stations...")
    ax.scatter(coords[:, 0], coords[:, 1],
              c='red', s=100, edgecolor='black',
              linewidth=1.5, alpha=0.8, zorder=2)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Spatial Connectivity Network (KNN k={w.mean_neighbors:.0f})',
                fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = assets_dir / 'spatial_connectivity_network.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Connectivity network saved: {plot_path.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Open output file for logging
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        
        try:
            print("=" * 80)
            print("SPATIAL ANALYSIS - PANEL DATA REFACTORED VERSION")
            print("=" * 80)
            print(f"Analysis Date: {pd.Timestamp.now()}")
            print(f"\nOutputs:")
            print(f"    Text Log: {output_file}")
            print(f"    CSV Results: {RESULTS_DIR}")
            print(f"    Visualizations: {ASSETS_DIR}")
            print(f"    Weights File: {WEIGHTS_DIR}")
            
            # ================================================================
            # STEP 1-2: LOAD DATA
            # ================================================================
            
            panel_df = load_panel_data(PROJECT_DIR / 'data' / 'panel_data_matrix.parquet')
            meta_df = load_station_metadata(PROJECT_DIR / 'data' / 'pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson')
            
            # ================================================================
            # STEP 3: CREATE CROSS-SECTIONAL VIEW
            # ================================================================
            
            station_profiles = create_cross_sectional_view(panel_df)
            
            # ================================================================
            # STEP 4: ALIGN WITH METADATA
            # ================================================================
            
            station_profiles, meta_df, valid_stations = align_stations_with_metadata(
                station_profiles, meta_df
            )
            
            # Extract coordinates
            coords = meta_df[['Longitude', 'Latitude']].values
            
            # ================================================================
            # STEP 5-6: BUILD AND SAVE SPATIAL WEIGHTS
            # ================================================================
            
            w = build_spatial_weights(coords, k=6)
            
            gal_path = WEIGHTS_DIR / 'spatial_weights_knn6.gal'
            save_spatial_weights_gal(w, valid_stations, gal_path)
            
            # ================================================================
            # STEP 7-8: MORAN'S I ANALYSIS
            # ================================================================
            
            global_morans_df = calculate_global_morans_i(station_profiles, w)
            
            lisa_df = calculate_local_morans_i(station_profiles, w, valid_stations)
            
            # ================================================================
            # STEP 9-10: VISUALIZATIONS
            # ================================================================
            
            create_lisa_maps(station_profiles, w, valid_stations, coords, ASSETS_DIR)
            
            create_morans_i_comparison(global_morans_df, ASSETS_DIR)
            
            # ================================================================
            # STEP 11-12: MULTIVARIATE CLUSTERING
            # ================================================================
            
            cluster_df, profiles_df = perform_multivariate_clustering(
                station_profiles, coords, valid_stations, ASSETS_DIR, RESULTS_DIR
            )
            
            # ================================================================
            # STEP 13: CONNECTIVITY VISUALIZATION
            # ================================================================
            
            create_connectivity_visualization(w, coords, valid_stations, ASSETS_DIR)
            
            # ================================================================
            # SAVE ALL RESULTS
            # ================================================================
            
            print("\n" + "=" * 80)
            print("SAVING RESULTS")
            print("=" * 80)
            
            # Save cross-sectional data
            station_profiles_path = RESULTS_DIR / 'station_profiles_cross_sectional.csv'
            station_profiles.to_csv(station_profiles_path)
            print(f"    ✓ Station profiles: {station_profiles_path.name}")
            
            # Save Global Moran's I
            global_path = RESULTS_DIR / 'optionA_global_morans_I_by_variable.csv'
            global_morans_df.to_csv(global_path, index=False)
            print(f"    ✓ Global Moran's I: {global_path.name}")
            
            # Save LISA results
            lisa_path = RESULTS_DIR / 'optionA_lisa_results_all_variables.csv'
            lisa_df.to_csv(lisa_path, index=False)
            print(f"    ✓ LISA results: {lisa_path.name}")
            
            # Save clustering results
            if cluster_df is not None:
                cluster_path = RESULTS_DIR / 'optionC_multivariate_clusters.csv'
                cluster_df.to_csv(cluster_path, index=False)
                print(f"    ✓ Cluster assignments: {cluster_path.name}")
                
                profiles_path = RESULTS_DIR / 'optionC_cluster_profiles.csv'
                profiles_df.to_csv(profiles_path, index=False)
                print(f"    ✓ Cluster profiles: {profiles_path.name}")
            
            # ================================================================
            # FINAL SUMMARY
            # ================================================================
            
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)
            
            print("\n📊 KEY FINDINGS:")
            print(f"    • Stations analyzed: {len(valid_stations)}")
            print(f"    • Variables analyzed: {len(station_profiles.columns)}")
            print(f"    • Spatial connections: {sum(len(w.neighbors[i]) for i in range(w.n))}")
            
            # Summarize significant spatial autocorrelation
            sig_vars = global_morans_df[global_morans_df['Significant'] == 'Yes']
            print(f"\n    • Variables with significant spatial autocorrelation: {len(sig_vars)}")
            if len(sig_vars) > 0:
                print(f"        {list(sig_vars['Variable'].values)}")
            
            # Summarize PM10 clusters
            if 'pm10' in station_profiles.columns:
                pm10_lisa = lisa_df[lisa_df['Variable'] == 'pm10']
                hotspots = pm10_lisa[pm10_lisa['Cluster_Type'] == 'High-High']
                print(f"\n    • PM10 hotspots identified: {len(hotspots)}")
                if len(hotspots) > 0:
                    print(f"        {list(hotspots['Station'].values)}")
            
            if cluster_df is not None:
                print(f"\n    • Atmospheric regimes (clusters): {cluster_df['Cluster'].nunique()}")
            
            print(f"\n✅ CRITICAL OUTPUT FOR REGRESSION:")
            print(f"    → Spatial weights file: {gal_path}")
            print(f"    → Use this .gal file in your Spatial Durbin Model")
            
            print("\n" + "=" * 80)
            
        finally:
            sys.stdout = original_stdout
    
    print(f"\n✓ Script completed successfully")
    print(f"  Check {output_file} for complete log")


if __name__ == "__main__":
    main()
