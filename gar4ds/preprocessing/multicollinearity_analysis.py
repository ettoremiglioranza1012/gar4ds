#!/usr/bin/env python3
"""
Multicollinearity Analysis Module
=================================
Analyzes multicollinearity among ERA5 variables at different pressure levels
to determine which variables can be dropped without significant information loss.

Analyses performed:
1. Correlation matrices by variable type
2. Variance Inflation Factor (VIF) for all variables
3. Principal Component Analysis
4. Recommendations for variable selection
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

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
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---\n")


def group_variables_by_type(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group variables by type (temperature, humidity, wind, surface)"""
    groups = {
        'temperature': ['temperature_550', 'temperature_850', 'temperature_950', 'temperature_2m'],
        'humidity': ['humidity_550', 'humidity_850', 'humidity_950'],
        'u_wind': ['uwind_550', 'uwind_850', 'uwind_950', 'wind_u_10m'],
        'v_wind': ['Vwind_550', 'Vwind_850', 'Vwind_950', 'wind_v_10m'],
        'surface': ['surface_pressure', 'blh', 'total_precipitation', 'solar_radiation_downwards'],
        'target': ['pm10']
    }
    
    # Filter to only include variables present in data
    available_groups = {}
    for group_name, var_list in groups.items():
        available_vars = [v for v in var_list if v in df.columns]
        if available_vars:
            available_groups[group_name] = available_vars
    
    return available_groups


def analyze_correlations_by_group(df: pd.DataFrame, groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """Analyze correlations within each variable group"""
    print_section("CORRELATION ANALYSIS BY GROUP")
    
    correlation_results = {}
    
    for group_name, var_list in groups.items():
        if group_name == 'target' or len(var_list) < 2:
            continue
        
        print_subsection(f"{group_name.upper()} Variables")
        
        corr_matrix = df[var_list].corr()
        correlation_results[group_name] = corr_matrix
        
        print(f"Correlation Matrix:")
        print(corr_matrix.round(3).to_string())
        
        # Identify high correlations
        high_corr = []
        for i in range(len(var_list)):
            for j in range(i+1, len(var_list)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.85:
                    high_corr.append((var_list[i], var_list[j], corr_val))
        
        if high_corr:
            print(f"\n⚠ High correlations (|r| > 0.85):")
            for v1, v2, r in sorted(high_corr, key=lambda x: -x[2]):
                print(f"  {v1} <-> {v2}: r = {r:.3f}")
    
    return correlation_results


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for all variables"""
    print_section("VARIANCE INFLATION FACTOR (VIF)")
    
    # Exclude target variable
    X = df.drop(columns=['pm10'], errors='ignore').select_dtypes(include=[np.number])
    
    # Handle missing values
    X = X.dropna()
    
    print(f"Calculating VIF for {len(X.columns)} variables...")
    print(f"Sample size: {len(X):,} observations")
    
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'variable': col, 'VIF': vif})
        except Exception as e:
            print(f"  ✗ Error calculating VIF for {col}: {e}")
            vif_data.append({'variable': col, 'VIF': np.nan})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    print("\nVIF Results (sorted by VIF):")
    print("-" * 50)
    for _, row in vif_df.iterrows():
        if np.isnan(row['VIF']):
            print(f"  {row['variable']:35s}: ERROR")
        elif row['VIF'] > 10:
            print(f"  {row['variable']:35s}: {row['VIF']:,.1f} ⚠ HIGH")
        else:
            print(f"  {row['variable']:35s}: {row['VIF']:.2f}")
    
    high_vif = vif_df[vif_df['VIF'] > 10]
    print(f"\n⚠ Variables with VIF > 10: {len(high_vif)}")
    
    return vif_df


def perform_pca_analysis(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Perform PCA to identify redundant dimensions"""
    print_section("PRINCIPAL COMPONENT ANALYSIS")
    
    X = df.drop(columns=['pm10'], errors='ignore').select_dtypes(include=[np.number])
    X = X.dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print("Explained Variance by Component:")
    print("-" * 50)
    for i, (var, cum) in enumerate(zip(explained_var, cumulative_var)):
        print(f"  PC{i+1}: {var*100:5.2f}% (cumulative: {cum*100:5.2f}%)")
        if cum > 0.99:
            print(f"  ... (remaining components explain <1%)")
            break
    
    # Components needed for 95% variance
    n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_var >= 0.99) + 1
    
    print(f"\nComponents for 95% variance: {n_components_95} (of {len(X.columns)} variables)")
    print(f"Components for 99% variance: {n_components_99}")
    print(f"\n→ Potential variable reduction: {len(X.columns) - n_components_95} variables")
    
    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(X.columns))],
        index=X.columns
    )
    
    return explained_var, loadings


def generate_recommendations(vif_df: pd.DataFrame, groups: Dict[str, List[str]]) -> List[str]:
    """Generate variable selection recommendations"""
    print_section("RECOMMENDATIONS")
    
    # Variables to drop based on VIF and redundancy
    drop_candidates = []
    keep_candidates = []
    
    # Temperature - keep only surface
    if 'temperature' in groups:
        print("TEMPERATURE:")
        print("  → Keep temperature_2m (surface conditions, most relevant)")
        print("  → Drop upper-level temperatures (extreme VIF, highly correlated)")
        keep_candidates.append('temperature_2m')
        drop_candidates.extend(['temperature_550', 'temperature_850', 'temperature_950'])
    
    # Humidity - keep near-surface
    if 'humidity' in groups:
        print("\nHUMIDITY:")
        print("  → Keep humidity_950 (near-surface, affects hygroscopic growth)")
        print("  → Drop upper-level humidity (redundant)")
        keep_candidates.append('humidity_950')
        drop_candidates.extend(['humidity_550', 'humidity_850'])
    
    # Wind - keep boundary layer and surface
    if 'u_wind' in groups:
        print("\nU-WIND:")
        print("  → Keep wind_u_10m (surface), uwind_850, uwind_950 (boundary layer)")
        print("  → Drop uwind_550 (above boundary layer)")
        keep_candidates.extend(['wind_u_10m', 'uwind_850', 'uwind_950'])
        drop_candidates.append('uwind_550')
    
    if 'v_wind' in groups:
        print("\nV-WIND:")
        print("  → Keep wind_v_10m (surface), Vwind_850, Vwind_950 (boundary layer)")
        print("  → Drop Vwind_550 (above boundary layer)")
        keep_candidates.extend(['wind_v_10m', 'Vwind_850', 'Vwind_950'])
        drop_candidates.append('Vwind_550')
    
    # Surface variables
    print("\nSURFACE VARIABLES:")
    print("  → Keep all: blh, solar_radiation_downwards, total_precipitation")
    print("  → Drop surface_pressure (extreme VIF, indirect effects)")
    keep_candidates.extend(['blh', 'solar_radiation_downwards', 'total_precipitation'])
    drop_candidates.append('surface_pressure')
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nVariables to DROP ({len(drop_candidates)}):")
    for var in drop_candidates:
        print(f"  - {var}")
    
    print(f"\nVariables to KEEP ({len(keep_candidates)} + pm10):")
    for var in keep_candidates:
        print(f"  - {var}")
    print("  - pm10 (target)")
    
    return drop_candidates


def run_multicollinearity_analysis(config: Optional[PipelineConfig] = None) -> pd.DataFrame:
    """
    Main execution function for multicollinearity analysis.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
        
    Returns:
        DataFrame with VIF results
    """
    if config is None:
        config = load_config()
    
    # Setup logging
    results_dir = config.get_results_subdir("multicollinearity_analysis")
    timestamp = datetime.now().strftime(config.logging.get("timestamp_format", "%Y%m%d_%H%M%S"))
    log_file = results_dir / f"recommendations_{timestamp}.txt"
    
    logger = OutputLogger(log_file)
    sys.stdout = logger
    
    try:
        print_section("MULTICOLLINEARITY ANALYSIS")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Aggregation: {config.temporal.aggregation}")
        
        # Load data
        print_section("LOADING DATA")
        panel_path = config.get_panel_matrix_path()
        print(f"Loading: {panel_path.name}")
        
        df = pd.read_parquet(panel_path)
        print(f"  Shape: {df.shape}")
        print(f"  Variables: {list(df.columns)}")
        
        # Group variables
        groups = group_variables_by_type(df)
        
        print("\nVariable groups identified:")
        for group_name, var_list in groups.items():
            print(f"\n  {group_name.upper()}:")
            for var in var_list:
                print(f"    - {var}")
        
        # Correlation analysis
        analyze_correlations_by_group(df, groups)
        
        # VIF analysis
        vif_df = calculate_vif(df)
        
        # PCA analysis
        perform_pca_analysis(df)
        
        # Recommendations
        generate_recommendations(vif_df, groups)
        
        # Save VIF results
        vif_file = results_dir / f"vif_analysis_{timestamp}.csv"
        vif_df.to_csv(vif_file, index=False)
        print(f"\n✓ VIF results saved to: {vif_file}")
        
        print_section("ANALYSIS COMPLETED")
        print(f"Log saved to: {log_file}")
        
        return vif_df
        
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
    run_multicollinearity_analysis()
