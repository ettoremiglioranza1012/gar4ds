#!/usr/bin/env python3
"""
Spatial Durbin Model Module
===========================
Implements a Panel Spatial Durbin Model (SDM) to quantify 
cross-border PM10 transport from Po Valley to Alpine regions.

Model Specification (SDM):
y = ρWy + Xβ + WXθ + ε

Where:
- y = log(PM10)
- ρ = spatial autoregressive coefficient
- X = local meteorological conditions
- WX = spatially lagged meteorology
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from libpysal.io import open as ioopen
from esda.moran import Moran
from spreg import Panel_FE_Lag
from sklearn.preprocessing import StandardScaler

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
    """Load filtered panel data"""
    print_header("1. LOADING FILTERED PANEL DATA")
    
    panel_path = config.get_panel_matrix_filtered_path()
    print(f"    File: {panel_path}")
    
    df = pd.read_parquet(panel_path)
    
    print(f"    ✓ Shape: {df.shape}")
    print(f"    ✓ Index: {df.index.names}")
    
    return df


def load_station_metadata(config: PipelineConfig) -> pd.DataFrame:
    """Load station metadata"""
    print_header("2. LOADING STATION METADATA")
    
    meta_path = config.get_stations_geojson_path(with_elevation=True)
    print(f"    File: {meta_path}")
    
    gdf = gpd.read_file(meta_path)
    gdf['Longitude'] = gdf.geometry.x
    gdf['Latitude'] = gdf.geometry.y
    
    meta_df = pd.DataFrame({
        'station_code': gdf['station_code'],
        'station_name': gdf['station_name'],
        'region': gdf['region'],
        'Longitude': gdf['Longitude'],
        'Latitude': gdf['Latitude'],
        'terrain_type': gdf.get('terrain_type', 'unknown')
    }).set_index('station_code')
    
    print(f"    ✓ Stations: {len(meta_df)}")
    
    return meta_df


def load_spatial_weights(config: PipelineConfig):
    """Load spatial weights"""
    print_header("3. LOADING SPATIAL WEIGHTS")
    
    weights_path = config.get_spatial_weights_path()
    print(f"    File: {weights_path}")
    
    w = ioopen(str(weights_path)).read()
    print(f"    ✓ Weights for {w.n} units")
    
    return w


def prepare_sdm_data(df: pd.DataFrame, w, config: PipelineConfig) -> tuple:
    """Prepare data for SDM estimation"""
    print_header("4. DATA PREPARATION")
    
    # Reset index
    df_work = df.reset_index()
    time_col = config.temporal.time_label
    
    # Log-transform PM10
    print("    Applying log transformation to PM10...")
    df_work['log_pm10'] = np.log(df_work['pm10'] + 1)
    
    # Get feature columns
    target = 'pm10'
    exclude_cols = [time_col, 'station_id', target, 'log_pm10']
    feature_cols = [col for col in df_work.columns if col not in exclude_cols]
    
    print(f"    Features: {feature_cols}")
    
    # Standardize features
    print("    Standardizing features...")
    scaler = StandardScaler()
    X_original = df_work[feature_cols].values
    X_scaled = scaler.fit_transform(X_original)
    
    # Create spatially lagged features (WX) for SDM
    print("    Creating spatially lagged features (WX)...")
    
    stations = df_work['station_id'].unique()
    n_stations = len(stations)
    n_periods = len(df_work) // n_stations
    
    # Reshape for spatial lag calculation
    X_panel = X_scaled.reshape(n_periods, n_stations, -1)
    
    # Calculate WX for each time period
    WX_list = []
    W_full = w.full()[0]  # Get full weight matrix
    
    for t in range(n_periods):
        Xt = X_panel[t]
        WXt = W_full @ Xt
        WX_list.append(WXt)
    
    WX_scaled = np.vstack(WX_list)
    
    # Combine X and WX for SDM
    X_sdm = np.hstack([X_scaled, WX_scaled])
    
    # Feature names for SDM
    wx_feature_cols = [f"W_{col}" for col in feature_cols]
    all_feature_cols = feature_cols + wx_feature_cols
    
    print(f"    ✓ Original features: {len(feature_cols)}")
    print(f"    ✓ Spatially lagged features: {len(wx_feature_cols)}")
    print(f"    ✓ Total SDM features: {len(all_feature_cols)}")
    
    # y must be 2D for spreg Panel models
    y = df_work['log_pm10'].values.reshape(-1, 1)
    
    return y, X_sdm, all_feature_cols, feature_cols, scaler


def estimate_sdm(y, X, w, feature_cols: List[str]) -> Dict[str, Any]:
    """Estimate Spatial Durbin Model using Panel FE Spatial Lag"""
    print_header("5. SDM ESTIMATION")
    
    print(f"    Estimating Panel FE Spatial Lag Model...")
    print(f"    Observations: {len(y)}")
    print(f"    Features: {X.shape[1]}")
    
    try:
        model = Panel_FE_Lag(
            y=y,
            x=X,
            w=w,
            name_y=['log_pm10'],
            name_x=feature_cols,
            name_ds="SDM_PM10_Panel"
        )
        
        print(f"\n    ✓ Model estimated successfully")
        print(f"    ✓ Rho (spatial lag): {model.rho:.4f}")
        print(f"    ✓ Log-likelihood: {model.logll:.2f}")
        
        # Extract coefficients - use model's name_x which matches betas
        betas = model.betas.flatten()
        
        # spreg model stores coefficient names in name_x
        if hasattr(model, 'name_x') and model.name_x:
            coef_names = list(model.name_x)
        else:
            coef_names = feature_cols
        
        # Ensure lengths match - betas may include constant
        if len(betas) != len(coef_names):
            print(f"    Note: betas={len(betas)}, names={len(coef_names)}, input features={len(feature_cols)}")
            # Use generic names if mismatch
            coef_names = [f"beta_{i}" for i in range(len(betas))]
        
        # Create coefficients table
        coef_df = pd.DataFrame({
            'variable': coef_names + ['rho'],
            'coefficient': list(betas) + [model.rho]
        })
        
        print_header("COEFFICIENTS", level=2)
        for _, row in coef_df.iterrows():
            print(f"    {row['variable']:35s}: {row['coefficient']:8.4f}")
        
        return {
            'model': model,
            'coefficients': coef_df,
            'rho': model.rho,
            'logll': model.logll,
            'betas': betas
        }
        
    except Exception as e:
        print(f"    ✗ Error in estimation: {e}")
        import traceback
        traceback.print_exc()
        return None


def decompose_effects(result: Dict[str, Any], feature_cols: List[str], 
                     original_feature_cols: List[str]) -> pd.DataFrame:
    """Decompose direct and indirect (spillover) effects"""
    print_header("6. SPILLOVER DECOMPOSITION")
    
    betas = result['betas']
    rho = result['rho']
    
    n_orig = len(original_feature_cols)
    
    # Direct effects (β)
    direct_effects = betas[:n_orig]
    
    # Indirect effects (θ from WX coefficients)
    indirect_effects = betas[n_orig:2*n_orig] if len(betas) > n_orig else np.zeros(n_orig)
    
    # Total effects
    total_effects = direct_effects + indirect_effects
    
    decomp_df = pd.DataFrame({
        'variable': original_feature_cols,
        'direct_effect': direct_effects,
        'indirect_effect': indirect_effects,
        'total_effect': total_effects,
        'spillover_ratio': np.abs(indirect_effects) / (np.abs(direct_effects) + 1e-10)
    })
    
    print("\n    Effect Decomposition:")
    print("-" * 80)
    print(f"    {'Variable':<30s} {'Direct':>10s} {'Indirect':>10s} {'Total':>10s} {'Spillover%':>12s}")
    print("-" * 80)
    
    for _, row in decomp_df.iterrows():
        spillover_pct = row['spillover_ratio'] * 100
        print(f"    {row['variable']:<30s} {row['direct_effect']:>10.4f} {row['indirect_effect']:>10.4f} "
              f"{row['total_effect']:>10.4f} {spillover_pct:>11.1f}%")
    
    return decomp_df


def save_results(result: Dict[str, Any], decomp_df: pd.DataFrame, 
                config: PipelineConfig, results_dir: Path):
    """Save model results"""
    print_header("7. SAVING RESULTS")
    
    # Coefficients
    coef_path = results_dir / 'coefficients_table.csv'
    result['coefficients'].to_csv(coef_path, index=False)
    print(f"    ✓ Coefficients: {coef_path}")
    
    # Spillover decomposition
    decomp_path = results_dir / 'spillover_decomposition.csv'
    decomp_df.to_csv(decomp_path, index=False)
    print(f"    ✓ Spillover decomposition: {decomp_path}")
    
    # Model summary
    summary_path = results_dir / 'model_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("SPATIAL DURBIN MODEL - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Aggregation: {config.temporal.aggregation}\n")
        f.write(f"Spatial lag (rho): {result['rho']:.4f}\n")
        f.write(f"Log-likelihood: {result['logll']:.2f}\n")
        f.write(f"\nModel: y = rho*Wy + X*beta + WX*theta + epsilon\n")
    print(f"    ✓ Model summary: {summary_path}")


def create_sdm_visualizations(result: Dict[str, Any], decomp_df: pd.DataFrame, 
                             config: PipelineConfig, assets_dir: Path):
    """Generate SDM visualizations"""
    print_header("8. GENERATING VISUALIZATIONS")
    
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 1. Coefficient Forest Plot
    print("    Creating coefficient forest plot...")
    try:
        coef_df = result['coefficients']
        # Exclude rho from plot
        coef_plot = coef_df[coef_df['variable'] != 'rho'].copy()
        
        # Identify direct vs indirect
        coef_plot['effect_type'] = coef_plot['variable'].apply(
            lambda x: 'Indirect (θ)' if x.startswith('W_') else 'Direct (β)'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['steelblue' if et == 'Direct (β)' else 'coral' 
                 for et in coef_plot['effect_type']]
        
        y_pos = np.arange(len(coef_plot))
        ax.barh(y_pos, coef_plot['coefficient'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(coef_plot['variable'], fontsize=9)
        ax.set_xlabel('Coefficient Value', fontsize=11)
        ax.set_title('SDM Coefficients: Direct (β) vs Indirect (θ) Effects', 
                    fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Direct (β)'),
                         Patch(facecolor='coral', alpha=0.7, label='Indirect (θ)')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'coefficient_forest_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'coefficient_forest_plot.png'}")
    except Exception as e:
        print(f"    ⚠ Coefficient plot failed: {e}")
    
    # 2. Spillover Decomposition Bar Chart
    print("    Creating spillover decomposition chart...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Stacked bar chart of effects
        ax1 = axes[0]
        x = np.arange(len(decomp_df))
        width = 0.35
        
        ax1.bar(x - width/2, decomp_df['direct_effect'], width, label='Direct Effect', color='steelblue')
        ax1.bar(x + width/2, decomp_df['indirect_effect'], width, label='Indirect Effect', color='coral')
        
        ax1.set_ylabel('Effect Magnitude')
        ax1.set_title('Direct vs Indirect Effects by Variable', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(decomp_df['variable'], rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # Right: Spillover ratio
        ax2 = axes[1]
        colors = ['red' if r > 0.5 else 'orange' if r > 0.25 else 'green' 
                 for r in decomp_df['spillover_ratio']]
        ax2.barh(decomp_df['variable'], decomp_df['spillover_ratio'] * 100, color=colors, edgecolor='black')
        ax2.set_xlabel('Spillover Ratio (%)')
        ax2.set_title('Spillover Contribution by Variable', fontweight='bold')
        ax2.axvline(x=25, color='orange', linestyle='--', alpha=0.7, label='25% threshold')
        ax2.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax2.legend(fontsize=9)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'spillover_decomposition_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'spillover_decomposition_chart.png'}")
    except Exception as e:
        print(f"    ⚠ Spillover chart failed: {e}")
    
    # 3. Total Effects Summary
    print("    Creating total effects summary...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_df = decomp_df.sort_values('total_effect')
        colors = ['green' if te < 0 else 'red' for te in sorted_df['total_effect']]
        
        ax.barh(sorted_df['variable'], sorted_df['total_effect'], color=colors, edgecolor='black')
        ax.set_xlabel('Total Effect on log(PM10)')
        ax.set_title('Total Effects (Direct + Indirect) on PM10', fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(assets_dir / 'total_effects_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {assets_dir / 'total_effects_summary.png'}")
    except Exception as e:
        print(f"    ⚠ Total effects plot failed: {e}")
    
    print(f"\n    ✓ All visualizations saved to: {assets_dir}")


def run_sdm(config: Optional[PipelineConfig] = None):
    """
    Main execution function for Spatial Durbin Model.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
    """
    if config is None:
        config = load_config()
    
    # Setup directories
    results_dir = config.get_results_subdir("spatial_durbin_model")
    assets_dir = config.get_assets_subdir("spatial_durbin_model")
    
    timestamp = datetime.now().strftime(config.logging.get("timestamp_format", "%Y%m%d_%H%M%S"))
    log_file = results_dir / f'sdm_analysis_log_{timestamp}.txt'
    
    with open(log_file, 'w') as f:
        tee = Tee(sys.stdout, f)
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            print_header("SPATIAL DURBIN MODEL - PANEL DATA ANALYSIS")
            print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Aggregation: {config.temporal.aggregation}")
            
            # Load data
            df = load_panel_data(config)
            meta_df = load_station_metadata(config)
            w = load_spatial_weights(config)
            
            # Prepare SDM data
            y, X, all_features, orig_features, scaler = \
                prepare_sdm_data(df, w, config)
            
            # Estimate SDM
            result = estimate_sdm(y, X, w, all_features)
            
            if result:
                # Decompose effects
                decomp_df = decompose_effects(result, all_features, orig_features)
                
                # Save results
                save_results(result, decomp_df, config, results_dir)
                
                # Generate visualizations
                create_sdm_visualizations(result, decomp_df, config, assets_dir)
            
            print_header("SDM ANALYSIS COMPLETED")
            print(f"✓ Results saved to: {results_dir}")
            print(f"✓ Assets saved to: {assets_dir}")
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            sys.stdout = old_stdout
            print(f"\n✓ Log saved to: {log_file}")


if __name__ == "__main__":
    run_sdm()
