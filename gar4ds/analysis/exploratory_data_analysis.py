#!/usr/bin/env python3
"""
Exploratory Data Analysis Module
================================
Performs comprehensive EDA on the panel matrix to verify physical
relationships before running regression models.

Analysis steps:
1. Correlation Analysis - Check physical relationships (PM10 vs BLH, wind)
2. Spatial Variance - Valley vs Plain stations analysis
3. Temporal Seasonality - Winter peaks and seasonal patterns
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import PipelineConfig, load_config


class OutputLogger:
    """Captures console output and saves to file"""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---\n")


def load_data(config: PipelineConfig):
    """Load panel matrix and metadata"""
    print_subsection("Loading Data")
    
    panel_path = config.get_panel_matrix_filtered_path()
    print(f"Loading panel matrix: {panel_path.name}")
    df_panel = pd.read_parquet(panel_path)
    print(f"  Shape: {df_panel.shape}")
    print(f"  Index: {df_panel.index.names}")
    print(f"  Variables: {len(df_panel.columns)}")
    
    meta_path = config.get_stations_geojson_path(with_elevation=True)
    print(f"\nLoading station metadata: {meta_path.name}")
    gdf_metadata = gpd.read_file(meta_path)
    print(f"  Stations: {len(gdf_metadata)}")
    
    return df_panel, gdf_metadata


def analyze_correlations(df_panel: pd.DataFrame, assets_dir: Path):
    """Analyze correlations between PM10 and other variables"""
    print_section("1. CORRELATION ANALYSIS")
    
    df_work = df_panel.reset_index()
    
    print_subsection("1.1 Overall Correlation Matrix")
    
    key_vars = ['pm10', 'blh', 'temperature_2m', 'total_precipitation', 
                'wind_u_10m', 'wind_v_10m', 'humidity_950']
    available_vars = [var for var in key_vars if var in df_work.columns]
    
    if len(available_vars) > 1:
        corr_matrix = df_work[available_vars].corr()
        
        print("Correlation Matrix (Key Physical Variables):")
        print(corr_matrix.round(3).to_string())
        
        # Save correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0, vmin=-1, vmax=1, square=True)
        plt.title('Correlation Matrix - Key Physical Variables')
        plt.tight_layout()
        heatmap_path = assets_dir / 'correlation_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {heatmap_path}")
    
    print_subsection("1.2 PM10 Correlations")
    
    if 'pm10' in df_work.columns:
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns
        pm10_corr = df_work[numeric_cols].corr()['pm10'].sort_values(ascending=False)
        
        print("PM10 Correlations with All Variables:")
        for var, corr_val in pm10_corr.items():
            if var != 'pm10':
                print(f"  {var:40s}: {corr_val:7.3f}")


def analyze_terrain_effects(df_panel: pd.DataFrame, gdf_metadata: gpd.GeoDataFrame, assets_dir: Path):
    """Analyze PM10 by terrain type"""
    print_section("2. TERRAIN ANALYSIS")
    
    df_work = df_panel.reset_index()
    
    # Merge with metadata
    if 'terrain_type' in gdf_metadata.columns:
        station_terrain = gdf_metadata.set_index('station_code')['terrain_type']
        df_work['terrain_type'] = df_work['station_id'].map(station_terrain)
        
        print_subsection("2.1 PM10 by Terrain Type")
        terrain_stats = df_work.groupby('terrain_type')['pm10'].agg(['mean', 'std', 'count'])
        print(terrain_stats.round(2).to_string())
        
        # Boxplot
        plt.figure(figsize=(10, 6))
        df_work.boxplot(column='pm10', by='terrain_type')
        plt.title('PM10 Distribution by Terrain Type')
        plt.suptitle('')
        plt.xlabel('Terrain Type')
        plt.ylabel('PM10 (μg/m³)')
        plt.tight_layout()
        boxplot_path = assets_dir / 'terrain_comparison_boxplots.png'
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {boxplot_path}")


def analyze_temporal_patterns(df_panel: pd.DataFrame, config: PipelineConfig, assets_dir: Path):
    """Analyze temporal/seasonal patterns"""
    print_section("3. TEMPORAL/SEASONAL ANALYSIS")
    
    df_work = df_panel.reset_index()
    time_col = config.temporal.time_label
    
    if time_col in df_work.columns:
        df_work['month'] = pd.to_datetime(df_work[time_col]).dt.month
        df_work['season'] = df_work['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        print_subsection("3.1 Seasonal PM10 Statistics")
        season_stats = df_work.groupby('season')['pm10'].agg(['mean', 'std', 'count'])
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        season_stats = season_stats.reindex(season_order)
        print(season_stats.round(2).to_string())
        
        # Seasonal plot
        plt.figure(figsize=(12, 6))
        df_work.groupby('month')['pm10'].mean().plot(kind='bar')
        plt.title(f'Mean PM10 by Month ({config.temporal.aggregation} data)')
        plt.xlabel('Month')
        plt.ylabel('Mean PM10 (μg/m³)')
        plt.xticks(rotation=0)
        plt.tight_layout()
        seasonal_path = assets_dir / 'temporal_seasonality_analysis.png'
        plt.savefig(seasonal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {seasonal_path}")


def run_eda(config: Optional[PipelineConfig] = None):
    """
    Main execution function for EDA.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
    """
    if config is None:
        config = load_config()
    
    # Setup directories
    results_dir = config.get_results_subdir("eda_analysis")
    assets_dir = config.get_assets_subdir("eda_analysis")
    
    log_file = results_dir / "eda_analysis.txt"
    
    logger = OutputLogger(log_file)
    sys.stdout = logger
    
    try:
        print_section("EXPLORATORY DATA ANALYSIS")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Aggregation: {config.temporal.aggregation}")
        
        # Load data
        df_panel, gdf_metadata = load_data(config)
        
        # Correlation analysis
        analyze_correlations(df_panel, assets_dir)
        
        # Terrain analysis
        analyze_terrain_effects(df_panel, gdf_metadata, assets_dir)
        
        # Temporal analysis
        analyze_temporal_patterns(df_panel, config, assets_dir)
        
        print_section("EDA COMPLETED")
        print(f"✓ Results saved to: {results_dir}")
        print(f"✓ Figures saved to: {assets_dir}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n✓ Console output saved to: {log_file}")


if __name__ == "__main__":
    run_eda()
