#!/usr/bin/env python3
"""
Spatial Analysis Module
=======================
Performs spatial autocorrelation and clustering analysis on the panel dataset.

Analysis steps:
1. Time-aggregate to create cross-sectional "Station Profile"
2. Build KNN spatial weights matrix (configurable k)
3. Calculate Moran's I (LISA) for PM10 and meteorological variables
4. Perform multivariate K-Means clustering (atmospheric regime detection)
5. Save spatial weights file for Spatial Durbin Model
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..config import PipelineConfig, load_config


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


def print_header(title: str, level: int = 1):
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


def load_panel_data(config: PipelineConfig) -> pd.DataFrame:
    """Load panel data matrix"""
    print_header("1. LOADING PANEL DATA")
    
    panel_path = config.get_panel_matrix_filtered_path()
    print(f"    Loading: {panel_path}")
    
    df = pd.read_parquet(panel_path)
    
    print(f"    ✓ Shape: {df.shape}")
    print(f"    ✓ Index: {df.index.names}")
    print(f"    ✓ Columns: {list(df.columns)}")
    
    return df


def load_station_metadata(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Load station metadata"""
    print_header("2. LOADING STATION METADATA")
    
    meta_path = config.get_stations_geojson_path(with_elevation=True)
    print(f"    Loading: {meta_path}")
    
    gdf = gpd.read_file(meta_path)
    print(f"    ✓ Stations: {len(gdf)}")
    
    return gdf


def create_station_profiles(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create cross-sectional station profiles by averaging over time"""
    print_header("3. CREATING STATION PROFILES")
    
    # Average all variables over time for each station
    df_profile = df.groupby(level='station_id').mean()
    
    print(f"    ✓ Station profiles created: {len(df_profile)}")
    print(f"    ✓ Variables: {list(df_profile.columns)}")
    
    return df_profile


def build_spatial_weights(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> KNN:
    """Build KNN spatial weights matrix"""
    print_header("4. BUILDING SPATIAL WEIGHTS")
    
    k = config.spatial.knn_neighbors
    print(f"    Building KNN weights with k={k}")
    
    # Create weights
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = 'r'  # Row-standardize
    
    print(f"    ✓ Weights created for {w.n} stations")
    print(f"    ✓ Mean neighbors: {w.mean_neighbors:.2f}")
    
    # Save weights
    weights_path = config.get_spatial_weights_path()
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    w.to_file(weights_path)
    print(f"    ✓ Saved to: {weights_path}")
    
    return w


def calculate_morans_i(df_profile: pd.DataFrame, w: KNN, config: PipelineConfig, results_dir: Path):
    """Calculate Moran's I for all variables"""
    print_header("5. GLOBAL MORAN'S I ANALYSIS")
    
    results = []
    
    for col in df_profile.columns:
        y = df_profile[col].values
        
        try:
            moran = Moran(y, w, permutations=config.spatial.lisa_permutations)
            results.append({
                'variable': col,
                'morans_I': moran.I,
                'expected_I': moran.EI,
                'p_value': moran.p_sim,
                'z_score': moran.z_sim
            })
            
            sig = "***" if moran.p_sim < 0.01 else "**" if moran.p_sim < 0.05 else "*" if moran.p_sim < 0.10 else ""
            print(f"    {col:30s}: I={moran.I:7.4f}, p={moran.p_sim:.4f} {sig}")
            
        except Exception as e:
            print(f"    {col:30s}: ERROR - {e}")
    
    results_df = pd.DataFrame(results)
    results_path = results_dir / 'global_morans_I_by_variable.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n    ✓ Saved results to: {results_path}")
    
    return results_df


def calculate_lisa(df_profile: pd.DataFrame, w: KNN, config: PipelineConfig, results_dir: Path):
    """Calculate LISA (Local Moran's I) for PM10"""
    print_header("6. LISA ANALYSIS (PM10)")
    
    if 'pm10' not in df_profile.columns:
        print("    ⚠ PM10 not found in data")
        return None
    
    y = df_profile['pm10'].values
    
    lisa = Moran_Local(y, w, permutations=config.spatial.lisa_permutations)
    
    # Create results dataframe
    lisa_df = pd.DataFrame({
        'station_id': df_profile.index,
        'pm10_mean': y,
        'local_I': lisa.Is,
        'p_value': lisa.p_sim,
        'quadrant': lisa.q
    })
    
    # Interpret quadrants
    quadrant_labels = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    lisa_df['cluster'] = lisa_df['quadrant'].map(quadrant_labels)
    lisa_df['significant'] = lisa_df['p_value'] < config.spatial.lisa_significance
    
    print(f"\n    Cluster distribution:")
    print(lisa_df.groupby(['cluster', 'significant']).size().unstack(fill_value=0))
    
    # Save
    lisa_path = results_dir / 'lisa_results_pm10.csv'
    lisa_df.to_csv(lisa_path, index=False)
    print(f"\n    ✓ Saved LISA results to: {lisa_path}")
    
    return lisa_df


def perform_clustering(df_profile: pd.DataFrame, gdf: gpd.GeoDataFrame, config: PipelineConfig, results_dir: Path):
    """Perform K-Means clustering for atmospheric regime detection"""
    print_header("7. MULTIVARIATE CLUSTERING")
    
    # Select variables for clustering
    cluster_vars = [col for col in df_profile.columns if col != 'pm10']
    
    X = df_profile[cluster_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    n_clusters = config.spatial.n_clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.spatial.random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add to dataframe
    df_profile['cluster'] = clusters
    
    print(f"    ✓ {n_clusters} clusters identified")
    print(f"\n    Cluster sizes:")
    print(df_profile['cluster'].value_counts().sort_index())
    
    # Cluster profiles
    print(f"\n    Cluster profiles (mean values):")
    cluster_profiles = df_profile.groupby('cluster').mean()
    print(cluster_profiles.round(2).to_string())
    
    # Save
    cluster_path = results_dir / 'multivariate_clusters.csv'
    df_profile.reset_index()[['station_id', 'cluster', 'pm10']].to_csv(cluster_path, index=False)
    print(f"\n    ✓ Saved cluster assignments to: {cluster_path}")
    
    cluster_profiles_path = results_dir / 'cluster_profiles.csv'
    cluster_profiles.to_csv(cluster_profiles_path)
    print(f"    ✓ Saved cluster profiles to: {cluster_profiles_path}")
    
    return df_profile


def create_visualizations(df_profile: pd.DataFrame, gdf: gpd.GeoDataFrame, 
                         lisa_df: pd.DataFrame, morans_df: pd.DataFrame,
                         config: PipelineConfig, assets_dir: Path):
    """Generate spatial analysis visualizations"""
    print_header("8. GENERATING VISUALIZATIONS")
    
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. LISA Cluster Map
    print("    Creating LISA cluster map...")
    try:
        # Merge with geometry
        lisa_geo = lisa_df.merge(
            gdf[['station_code', 'geometry']], 
            left_on='station_id', 
            right_on='station_code',
            how='left'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract coordinates
        coords = np.array([[g.x, g.y] for g in gdf.geometry])
        
        # Color by cluster
        colors = {'HH': 'red', 'LL': 'blue', 'HL': 'orange', 'LH': 'cyan', 'NS': 'lightgray'}
        
        for cluster_type in ['HH', 'LL', 'HL', 'LH']:
            mask = (lisa_df['cluster'] == cluster_type) & lisa_df['significant']
            if mask.sum() > 0:
                cluster_coords = coords[mask.values]
                ax.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                          c=colors[cluster_type], edgecolor='k', s=100, 
                          label=f'{cluster_type} (p<0.05)', zorder=3)
        
        # Non-significant
        ns_mask = ~lisa_df['significant']
        if ns_mask.sum() > 0:
            ns_coords = coords[ns_mask.values]
            ax.scatter(ns_coords[:, 0], ns_coords[:, 1],
                      c='lightgray', edgecolor='k', s=60, alpha=0.5,
                      label='Not Significant', zorder=2)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('LISA Cluster Map: PM10 Spatial Autocorrelation', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'lisa_cluster_map_pm10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'lisa_cluster_map_pm10.png'}")
    except Exception as e:
        print(f"    ⚠ LISA map failed: {e}")
    
    # 2. Global Moran's I Comparison
    print("    Creating Moran's I comparison plot...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by Moran's I
        morans_sorted = morans_df.sort_values('morans_I', ascending=True)
        
        # Color by significance
        colors = ['red' if p < 0.05 else 'gray' for p in morans_sorted['p_value']]
        
        ax.barh(morans_sorted['variable'], morans_sorted['morans_I'], color=colors, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel("Moran's I", fontsize=12)
        ax.set_title("Global Spatial Autocorrelation by Variable\n(Red = Significant p<0.05)",
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'global_morans_i_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'global_morans_i_comparison.png'}")
    except Exception as e:
        print(f"    ⚠ Moran's I plot failed: {e}")
    
    # 3. Cluster Spatial Map
    print("    Creating cluster spatial map...")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        coords = np.array([[g.x, g.y] for g in gdf.geometry])
        clusters = df_profile['cluster'].values
        
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters, 
                           cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Multivariate Clustering: Atmospheric Regimes', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'cluster_spatial_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'cluster_spatial_map.png'}")
    except Exception as e:
        print(f"    ⚠ Cluster map failed: {e}")
    
    # 4. Cluster Profile Heatmap
    print("    Creating cluster profile heatmap...")
    try:
        cluster_profiles = df_profile.groupby('cluster').mean()
        
        # Normalize for heatmap
        scaler = StandardScaler()
        profiles_norm = pd.DataFrame(
            scaler.fit_transform(cluster_profiles),
            index=cluster_profiles.index,
            columns=cluster_profiles.columns
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(profiles_norm.T, cmap='RdYlBu_r', center=0, 
                   annot=True, fmt='.2f', ax=ax, cbar_kws={'label': 'Z-score'})
        ax.set_title('Cluster Profiles (Standardized)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Variable', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'cluster_profile_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'cluster_profile_heatmap.png'}")
    except Exception as e:
        print(f"    ⚠ Profile heatmap failed: {e}")
    
    print(f"\n    ✓ All visualizations saved to: {assets_dir}")


def run_spatial_analysis(config: Optional[PipelineConfig] = None):
    """
    Main execution function for spatial analysis.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
    """
    if config is None:
        config = load_config()
    
    # Setup directories
    results_dir = config.get_results_subdir("spatial_analysis")
    assets_dir = config.get_assets_subdir("spatial_analysis")
    
    log_file = results_dir / 'spatial_analysis_results.txt'
    
    with open(log_file, 'w') as f:
        tee = Tee(sys.stdout, f)
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            print_header("SPATIAL ANALYSIS - PANEL DATA")
            print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Aggregation: {config.temporal.aggregation}")
            print(f"KNN neighbors: {config.spatial.knn_neighbors}")
            
            # Load data
            df = load_panel_data(config)
            gdf = load_station_metadata(config)
            
            # Create station profiles
            df_profile = create_station_profiles(df, config)
            
            # Build spatial weights
            w = build_spatial_weights(gdf, config)
            
            # Global Moran's I
            morans_df = calculate_morans_i(df_profile, w, config, results_dir)
            
            # LISA
            lisa_df = calculate_lisa(df_profile, w, config, results_dir)
            
            # Clustering
            df_profile = perform_clustering(df_profile, gdf, config, results_dir)
            
            # Generate visualizations
            create_visualizations(df_profile, gdf, lisa_df, morans_df, config, assets_dir)
            
            print_header("ANALYSIS COMPLETED")
            print(f"✓ Results saved to: {results_dir}")
            print(f"✓ Assets saved to: {assets_dir}")
            print(f"✓ Spatial weights saved to: {config.get_spatial_weights_path()}")
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            sys.stdout = old_stdout
            print(f"\n✓ Log saved to: {log_file}")


if __name__ == "__main__":
    run_spatial_analysis()
