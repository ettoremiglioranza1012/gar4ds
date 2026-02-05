import polars as pl
import pandas as pd
import numpy as np
import pyarrow
import json
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN, DistanceBand
from esda.moran import Moran, Moran_Local
import geopandas as gpd
from shapely.geometry import Point
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# Create output directories if they don't exist
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
ASSETS_DIR = PROJECT_DIR / 'assets' / 'spatial_analysis'
RESULTS_DIR = PROJECT_DIR / 'results' / 'spatial_analysis'
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set up output file for text results
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
# FUNCTION DEFINITIONS - Must be defined before main execution
# ============================================================================

def run_parallel_lisa_analysis(spatial_df, valid_stations, coords, w, 
                               available_vars, meta_df, assets_dir, results_dir):
    """
    Run separate LISA analysis for each environmental variable.
    This reveals which variables show spatial clustering patterns.
    
    REFINED STRATEGY:
    - Global Moran's I: Calculated for ALL variables (proves regressors are spatially structured)
    - LISA Cluster Maps: Generated ONLY for PM10 (focused visual narrative)
    - Meteorological Variables: Statistical validation only, no map generation
    """
    print("\n[A1] Running LISA for each variable...")
    
    # Focus on key environmental variables
    priority_vars = ['PM10', 'TEMP', 'BLH', 'WS', 'PRECIP', 'PRESS', 'RAD']
    vars_to_analyze = [v for v in priority_vars if v in available_vars]
    
    if not vars_to_analyze:
        print("    ⚠ No environmental variables found for analysis")
        return
    
    print(f"    ✓ Analyzing: {vars_to_analyze}")
    
    # Store results for each variable
    all_variable_results = []
    global_stats = []
    
    for var in vars_to_analyze:
        print(f"\n    --- Analyzing {var} ---")
        
        # Extract mean values for this variable
        var_cols = [f'{var}_{s}' for s in valid_stations]
        var_means = spatial_df.select(var_cols).mean().to_pandas().iloc[0].values
        
        # Global Moran's I
        mi = Moran(var_means, w)
        print(f"        Global Moran's I: {mi.I:.4f} (p={mi.p_sim:.4f})")
        
        global_stats.append({
            'Variable': var,
            'Morans_I': mi.I,
            'P_value': mi.p_sim,
            'Significant': 'Yes' if mi.p_sim < 0.05 else 'No',
            'Pattern': 'Clustered' if mi.I > 0 and mi.p_sim < 0.05 else 
                      'Dispersed' if mi.I < 0 and mi.p_sim < 0.05 else 'Random'
        })
        
        # Local Moran's I (LISA)
        lisa = Moran_Local(var_means, w)
        
        # Classify clusters
        sig = lisa.p_sim < 0.05
        hotspots = sig & (lisa.q == 1)  # High-High
        coldspots = sig & (lisa.q == 3)  # Low-Low
        high_low = sig & (lisa.q == 4)   # High-Low (outliers)
        low_high = sig & (lisa.q == 2)   # Low-High (outliers)
        
        print(f"        High-High: {hotspots.sum()}, Low-Low: {coldspots.sum()}")
        print(f"        High-Low: {high_low.sum()}, Low-High: {low_high.sum()}")
        
        # Store detailed results
        for i, station in enumerate(valid_stations):
            all_variable_results.append({
                'Station': station,
                'Variable': var,
                'Mean_Value': var_means[i],
                'Local_Morans_I': lisa.Is[i],
                'P_value': lisa.p_sim[i],
                'Cluster_Type': 'High-High' if hotspots[i] else
                               'Low-Low' if coldspots[i] else
                               'High-Low' if high_low[i] else
                               'Low-High' if low_high[i] else
                               'Not Significant',
                'Significant': sig[i]
            })
        
        # Visualization - only generate map for PM10
        if var == 'PM10':
            create_variable_lisa_map(var, var_means, coords, valid_stations, 
                                    hotspots, coldspots, high_low, low_high,
                                    assets_dir)
            print(f"        ✓ LISA cluster map generated for {var}")
        else:
            print(f"        → Meteorological variable - map generation skipped (statistical validation only)")
    
    # Save global statistics
    print(f"\n[A2] Saving results...")
    global_df = pd.DataFrame(global_stats)
    global_df.to_csv(results_dir / 'optionA_global_morans_I_by_variable.csv', index=False)
    print(f"    ✓ Global Moran's I results saved (all variables for statistical validation)")
    
    # Save detailed LISA results
    detailed_df = pd.DataFrame(all_variable_results)
    detailed_df.to_csv(results_dir / 'optionA_lisa_results_all_variables.csv', index=False)
    print(f"    ✓ Detailed LISA results saved (all variables)")
    
    # Create summary comparison plot
    create_variable_comparison_plot(global_df, assets_dir)
    
    # Count how many maps were generated
    pm10_count = sum(1 for v in vars_to_analyze if v == 'PM10')
    print(f"\n    📊 Visual outputs: {pm10_count} LISA cluster map(s) generated (PM10 only)")
    print(f"    📋 Meteorological variables: Statistical validation recorded, maps skipped")
    
    # Print summary
    print(f"\n[A3] Summary of spatial patterns (all variables):")
    print(global_df.to_string(index=False))


def create_variable_lisa_map(var_name, var_means, coords, valid_stations,
                             hotspots, coldspots, high_low, low_high, assets_dir):
    """Create LISA cluster map for a specific variable"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Base plot - not significant
    not_sig = ~(hotspots | coldspots | high_low | low_high)
    if not_sig.sum() > 0:
        ax.scatter(coords[not_sig, 0], coords[not_sig, 1],
                  c='lightgrey', edgecolor='k', s=80, alpha=0.6,
                  label='Not Significant', zorder=1)
    
    # Plot clusters
    if hotspots.sum() > 0:
        ax.scatter(coords[hotspots, 0], coords[hotspots, 1],
                  c='red', edgecolor='k', s=150, label='High-High',
                  zorder=3, marker='s')
        # Annotate hotspots
        for idx in np.where(hotspots)[0]:
            ax.annotate(valid_stations[idx], (coords[idx, 0], coords[idx, 1]),
                       fontsize=8, fontweight='bold',
                       xytext=(3, 3), textcoords='offset points')
    
    if coldspots.sum() > 0:
        ax.scatter(coords[coldspots, 0], coords[coldspots, 1],
                  c='blue', edgecolor='k', s=150, label='Low-Low',
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
    ax.set_title(f'Spatial Clustering: {var_name}\n(LISA Analysis)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_path = assets_dir / f'optionA_lisa_map_{var_name}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"        ✓ Saved map: {plot_path.name}")
    plt.close()


def create_variable_comparison_plot(global_df, assets_dir):
    """Create comparison plot of Moran's I across variables"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot: Moran's I values
    colors = ['red' if p < 0.05 else 'gray' for p in global_df['P_value']]
    bars = ax.barh(global_df['Variable'], global_df['Morans_I'], 
                    color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Moran's I", fontsize=12)
    ax.set_title("Global Spatial Autocorrelation by Variable\n(Red = Significant p<0.05)",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (var, val, p) in enumerate(zip(global_df['Variable'], 
                                           global_df['Morans_I'],
                                           global_df['P_value'])):
        label = f"{val:.3f}*" if p < 0.05 else f"{val:.3f}"
        ax.text(val + 0.01, i, label, va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = assets_dir / 'optionA_variable_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Comparison plot saved: {plot_path.name}")
    plt.close()


def run_multivariate_clustering(spatial_df, valid_stations, coords,
                                available_vars, meta_df, assets_dir, results_dir):
    """
    Multivariate clustering analysis combining all environmental variables.
    Identifies stations with similar meteorological profiles and overlays
    them spatially.
    
    CRITICAL FOR REGIME DEFINITION:
    - Defines "Source vs. Receptor" atmospheric regimes based on meteorological profiles
    - K-Means clustering identifies distinct environmental conditions
    - Cluster profiles are essential for understanding pollution dynamics
    """
    print("\n[C1] Preparing multivariate feature matrix...")
    print("    (This analysis defines atmospheric regimes for source/receptor classification)")
    
    
    # Select variables for clustering
    cluster_vars = ['PM10', 'TEMP', 'BLH', 'WS', 'PRECIP', 'PRESS']
    cluster_vars = [v for v in cluster_vars if v in available_vars]
    
    if len(cluster_vars) < 2:
        print("    ⚠ Insufficient variables for multivariate analysis")
        return
    
    print(f"    ✓ Using variables: {cluster_vars}")
    
    # Build feature matrix
    feature_matrix = []
    feature_names = []
    
    for var in cluster_vars:
        var_cols = [f'{var}_{s}' for s in valid_stations]
        var_means = spatial_df.select(var_cols).mean().to_pandas().iloc[0].values
        feature_matrix.append(var_means)
        feature_names.append(var)
    
    # Transpose: rows = stations, columns = variables
    X = np.column_stack(feature_matrix)
    
    print(f"    ✓ Feature matrix shape: {X.shape} (stations × variables)")
    
    # Standardize features
    print("\n[C2] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    ✓ Features standardized")
    
    # K-means clustering
    print("\n[C3] Performing K-means clustering...")
    optimal_k = determine_optimal_k(X_scaled, max_k=8)
    print(f"    ✓ Optimal number of clusters: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Count stations per cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"\n    Cluster sizes:")
    for c, count in zip(unique, counts):
        print(f"      Cluster {c}: {count} stations")
    
    # PCA for visualization
    print("\n[C4] Performing PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"    ✓ Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Station': valid_stations,
        'Cluster': cluster_labels,
        'Longitude': coords[:, 0],
        'Latitude': coords[:, 1],
        'PCA_1': X_pca[:, 0],
        'PCA_2': X_pca[:, 1]
    })
    
    # Add original variable values
    for i, var in enumerate(cluster_vars):
        results_df[var] = X[:, i]
    
    # Analyze cluster profiles
    print("\n[C5] Analyzing cluster profiles...")
    print("    (Defining atmospheric regimes for source/receptor classification)")
    cluster_profiles = analyze_cluster_profiles(results_df, cluster_vars, cluster_labels)
    
    # Save results
    print("\n[C6] Saving results...")
    results_df.to_csv(results_dir / 'optionC_multivariate_clusters.csv', index=False)
    cluster_profiles.to_csv(results_dir / 'optionC_cluster_profiles.csv', index=False)
    print(f"    ✓ Cluster assignments saved (station-to-regime mapping)")
    print(f"    ✓ Cluster profiles saved (regime environmental characteristics)")
    
    # Visualizations
    print("\n[C7] Creating visualizations...")
    create_pca_scatter(results_df, X_pca, cluster_labels, optimal_k, 
                      pca.explained_variance_ratio_, assets_dir)
    create_spatial_cluster_map(results_df, coords, cluster_labels, optimal_k, assets_dir)
    create_cluster_heatmap(cluster_profiles, cluster_vars, assets_dir)
    print(f"    ✓ Generated: PCA scatter, spatial cluster map, and regime heatmap")
    
    # Create spatial connectivity map
    print("\n[C8] Spatial connectivity analysis...")
    connectivity_stats = create_connectivity_map(results_df, k_neighbors=6, 
                                                 assets_dir=assets_dir, 
                                                 results_dir=results_dir)
    print(f"    ✓ Connectivity visualization complete")
    
    # Summary
    print("\n" + "="*80)
    print("[C9] CLUSTER PROFILE SUMMARY - ATMOSPHERIC REGIME DEFINITIONS")
    print("="*80)
    print("(Critical for identifying Source vs. Receptor station characteristics)\n")
    print(cluster_profiles.to_string(index=False))
    print("\n" + "="*80)


def determine_optimal_k(X, max_k=10):
    """Use elbow method to determine optimal number of clusters"""
    inertias = []
    K_range = range(2, min(max_k + 1, len(X) // 2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection: find maximum rate of change decrease
    if len(inertias) > 2:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        optimal_k = np.argmin(second_diffs) + 2  # +2 because we started at k=2
    else:
        optimal_k = 3  # Default
    
    return min(optimal_k, 6)  # Cap at 6 for interpretability


def analyze_cluster_profiles(results_df, cluster_vars, cluster_labels):
    """Calculate mean profile for each cluster"""
    profiles = []
    
    for cluster_id in sorted(np.unique(cluster_labels)):
        cluster_mask = results_df['Cluster'] == cluster_id
        cluster_data = results_df[cluster_mask]
        
        profile = {'Cluster': cluster_id, 'N_Stations': cluster_mask.sum()}
        
        for var in cluster_vars:
            profile[f'{var}_mean'] = cluster_data[var].mean()
            profile[f'{var}_std'] = cluster_data[var].std()
        
        # Characterize cluster
        if 'PM10' in cluster_vars:
            pm10_mean = cluster_data['PM10'].mean()
            if pm10_mean > results_df['PM10'].quantile(0.66):
                profile['Characterization'] = 'High Pollution'
            elif pm10_mean < results_df['PM10'].quantile(0.33):
                profile['Characterization'] = 'Low Pollution'
            else:
                profile['Characterization'] = 'Medium Pollution'
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


def create_pca_scatter(results_df, X_pca, cluster_labels, n_clusters, 
                       explained_var, assets_dir):
    """Create PCA scatter plot colored by cluster"""
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[colors[cluster_id]], s=100, edgecolor='black',
                  label=f'Cluster {cluster_id}', alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title('Multivariate Station Clustering\n(PCA Projection)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = assets_dir / 'optionC_pca_clusters.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ PCA scatter saved")
    plt.close()


def create_spatial_cluster_map(results_df, coords, cluster_labels, n_clusters, assets_dir):
    """Create spatial map colored by cluster membership"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            cluster_coords = coords[mask]
            ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                      c=[colors[cluster_id]], s=150, edgecolor='black',
                      linewidth=1.5, label=f'Cluster {cluster_id}',
                      alpha=0.8, zorder=3)
            
            # Annotate some stations in each cluster
            cluster_stations = results_df[mask]['Station'].values
            for i, (lon, lat) in enumerate(cluster_coords[:3]):  # Annotate first 3
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
    plot_path = assets_dir / 'optionC_spatial_clusters.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Spatial cluster map saved")
    plt.close()


def create_cluster_heatmap(cluster_profiles, cluster_vars, assets_dir):
    """Create heatmap of cluster profiles"""
    
    # Extract mean values for heatmap
    n_clusters = len(cluster_profiles)
    mean_cols = [f'{var}_mean' for var in cluster_vars]
    
    if not all(col in cluster_profiles.columns for col in mean_cols):
        print("    ⚠ Cannot create heatmap - missing columns")
        return
    
    heatmap_data = cluster_profiles[mean_cols].values
    
    # Normalize by column (variable) for better visualization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_normalized = scaler.fit_transform(heatmap_data.T).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(heatmap_normalized, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(cluster_vars)))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels(cluster_vars, fontsize=11)
    ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)], fontsize=11)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value\n(0=Low, 1=High)', fontsize=10)
    
    # Add text annotations
    for i in range(n_clusters):
        for j in range(len(cluster_vars)):
            text = ax.text(j, i, f'{heatmap_normalized[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title('Cluster Environmental Profiles\n(Normalized Mean Values)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plot_path = assets_dir / 'optionC_cluster_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Cluster heatmap saved")
    plt.close()


def create_connectivity_map(results_df, k_neighbors=6, assets_dir=None, results_dir=None):
    """
    Create spatial connectivity visualization (spider map) showing KNN connections.
    Calculates distance statistics and saves connectivity graph.
    
    Args:
        results_df: DataFrame with Station, Longitude, Latitude, and Cluster columns
        k_neighbors: Number of nearest neighbors for KNN weights (default: 6)
        assets_dir: Directory to save visualizations
        results_dir: Directory to save statistics
    """
    print(f"\n[D1] Creating spatial connectivity map (KNN k={k_neighbors})...")
    
    try:
        # Import contextily for basemap (optional)
        import contextily as ctx
        has_contextily = True
    except ImportError:
        has_contextily = False
        print("    ℹ contextily not available - basemap will be skipped")
    
    # Create GeoDataFrame from coordinates
    print("    → Converting to GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(results_df['Longitude'], results_df['Latitude'])]
    gdf = gpd.GeoDataFrame(results_df, geometry=geometry, crs="EPSG:4326")
    
    # Reproject to metric CRS for accurate distance calculation (UTM 32N for Alps region)
    print("    → Reprojecting to UTM 32N (metric CRS)...")
    gdf = gdf.to_crs(epsg=32632)
    
    # Build KNN weights matrix
    print(f"    → Building KNN weights matrix (k={k_neighbors})...")
    w = KNN.from_dataframe(gdf, k=k_neighbors)
    
    # Calculate distance statistics
    print("    → Calculating distance statistics...")
    min_dists = []
    max_dists = []
    all_dists = []
    
    for idx, neighbors in w.neighbors.items():
        origin = gdf.iloc[idx].geometry
        dists = [origin.distance(gdf.iloc[n].geometry) for n in neighbors]
        min_dists.append(min(dists))
        max_dists.append(max(dists))
        all_dists.extend(dists)
    
    # Convert to kilometers
    avg_nearest = sum(min_dists) / len(min_dists) / 1000
    avg_furthest = sum(max_dists) / len(max_dists) / 1000
    median_dist = np.median(all_dists) / 1000
    
    print(f"    ✓ Average distance to nearest neighbor: {avg_nearest:.2f} km")
    print(f"    ✓ Average distance to {k_neighbors}th neighbor: {avg_furthest:.2f} km")
    print(f"    ✓ Median connection distance: {median_dist:.2f} km")
    
    # Save statistics to file if results_dir provided
    if results_dir:
        stats_df = pd.DataFrame({
            'Metric': ['Average nearest neighbor distance (km)', 
                      f'Average {k_neighbors}th neighbor distance (km)', 
                      'Median connection distance (km)',
                      'Min connection distance (km)',
                      'Max connection distance (km)'],
            'Value': [avg_nearest, avg_furthest, median_dist, 
                     min(all_dists)/1000, max(all_dists)/1000]
        })
        stats_path = results_dir / 'spatial_connectivity_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        print(f"    ✓ Distance statistics saved to: {stats_path.name}")
    
    # Create connectivity visualization
    print("    → Creating connectivity visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Plot edges first (so they appear behind points)
    for i, neighbors in w.neighbors.items():
        origin = gdf.iloc[i].geometry
        for n in neighbors:
            dest = gdf.iloc[n].geometry
            ax.plot([origin.x, dest.x], [origin.y, dest.y], 
                   color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Plot points colored by cluster if available
    if 'Cluster' in gdf.columns:
        n_clusters = gdf['Cluster'].nunique()
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for cluster_id in sorted(gdf['Cluster'].unique()):
            cluster_data = gdf[gdf['Cluster'] == cluster_id]
            ax.scatter(cluster_data.geometry.x, cluster_data.geometry.y,
                      c=[colors[cluster_id]], s=100, edgecolor='black',
                      linewidth=1.5, label=f'Cluster {cluster_id}',
                      alpha=0.8, zorder=2)
    else:
        # Plot all points in red if no cluster information
        ax.scatter(gdf.geometry.x, gdf.geometry.y,
                  c='red', s=100, edgecolor='black',
                  linewidth=1.5, alpha=0.8, zorder=2)
    
    # Add basemap if contextily is available
    if has_contextily:
        try:
            print("    → Adding basemap...")
            ctx.add_basemap(ax, crs=gdf.crs.to_string(), 
                          source=ctx.providers.CartoDB.Positron,
                          attribution=False)
            print("    ✓ Basemap added")
        except Exception as e:
            print(f"    ⚠ Could not add basemap: {e}")
    
    # Styling
    ax.set_title(f'Spatial Connectivity Network (KNN k={k_neighbors})\n'
                f'Avg nearest neighbor: {avg_nearest:.1f} km | '
                f'Avg bandwidth: {avg_furthest:.1f} km',
                fontsize=14, fontweight='bold', pad=15)
    
    if 'Cluster' in gdf.columns:
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    ax.set_xlabel('Easting (UTM 32N, meters)', fontsize=11)
    ax.set_ylabel('Northing (UTM 32N, meters)', fontsize=11)
    ax.ticklabel_format(style='plain', axis='both')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    plt.tight_layout()
    
    # Save the figure
    if assets_dir:
        plot_path = assets_dir / 'spatial_connectivity_graph.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Connectivity map saved to: {plot_path.name}")
    
    plt.close()
    
    print("    ✓ Spatial connectivity analysis complete")
    
    return {
        'avg_nearest_km': avg_nearest,
        'avg_furthest_km': avg_furthest,
        'median_km': median_dist,
        'n_connections': len(all_dists)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Open output file and redirect stdout
with open(output_file, 'w') as f:
    # Create a Tee object that writes to both stdout and file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    
    try:
        print("=" * 80)
        print("MULTIVARIATE SPATIAL ANALYSIS")
        print("=" * 80)
        print("\n[1] Loading data...")
        
        # Load station metadata
        with open(PROJECT_DIR / 'data' / 'data_stations_metadata.json', 'r') as meta_file:
            stations_meta = json.load(meta_file)
        meta_df = pd.DataFrame(stations_meta).set_index('Station_ID')
        
        # Try new multi-variable dataset first, fallback to old if not available
        data_dir = PROJECT_DIR / 'data'
        if (data_dir / 'spatial_full_matrix.parquet').exists():
            spatial_df = pl.read_parquet(data_dir / 'spatial_full_matrix.parquet')
            print(f"    ✓ Loaded multi-variable dataset: {spatial_df.shape}")
        else:
            spatial_df = pl.read_parquet(data_dir / 'spatial_pollution_matrix.parquet')
            print(f"    ✓ Loaded PM10-only dataset: {spatial_df.shape}")
            
        print("\n[2] Aligning data and coordinates...")
        
        # Detect available variables in the dataset
        all_columns = spatial_df.columns
        available_vars = set()
        station_var_map = {}
        
        for col in all_columns:
            if col == 'Data':
                continue
            if '_' in col:
                var, station = col.split('_', 1)
                available_vars.add(var)
                if station not in station_var_map:
                    station_var_map[station] = []
                station_var_map[station].append(var)
        
        # Get valid stations (those in both data and metadata)
        data_stations = list(station_var_map.keys())
        valid_stations = [s for s in data_stations if s in meta_df.index]
        
        print(f"    ✓ Found {len(valid_stations)} valid stations")
        print(f"    ✓ Available variables: {sorted(available_vars)}")
        
        # Extract coordinates
        coords = meta_df.loc[valid_stations, ['Longitude', 'Latitude']].values
        
        # 3. Define Spatial Weights Matrix (W)
        print("\n[3] Creating spatial weights matrix...")
        w = KNN.from_array(coords, k=6)
        w.transform = 'r'  # Row-standardize weights
        print(f"    ✓ KNN weights created (k=6)")
        
        # =====================================================================
        # OPTION A: PARALLEL LISA ANALYSIS FOR EACH VARIABLE
        # =====================================================================
        print("\n" + "=" * 80)
        print("OPTION A: PARALLEL SPATIAL ANALYSIS FOR EACH VARIABLE")
        print("=" * 80)
        
        run_parallel_lisa_analysis(spatial_df, valid_stations, coords, w, 
                                   available_vars, meta_df, ASSETS_DIR, RESULTS_DIR)
        
        # =====================================================================
        # OPTION C: MULTIVARIATE CLUSTERING ANALYSIS
        # =====================================================================
        print("\n" + "=" * 80)
        print("OPTION C: MULTIVARIATE CLUSTERING ANALYSIS")
        print("=" * 80)
        
        run_multivariate_clustering(spatial_df, valid_stations, coords, 
                                    available_vars, meta_df, ASSETS_DIR, RESULTS_DIR)
        
        # Analysis complete
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n    📁 Text Output: {output_file}")
        print(f"    📊 CSV Results: {RESULTS_DIR}")
        print(f"    🖼️  Visualizations: {ASSETS_DIR}")
        print("\n")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

print(f"Script completed successfully. Check {RESULTS_DIR} for outputs and {ASSETS_DIR} for plots.")
