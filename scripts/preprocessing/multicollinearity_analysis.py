"""
Multicollinearity Analysis for ERA5 Multi-Level Variables
==========================================================

This script analyzes multicollinearity among ERA5 variables at different
pressure levels (550, 850, 950 hPa) to determine if some levels or variables
can be dropped without significant information loss.

Analyses performed:
1. Correlation matrices by variable type (temperature, humidity, wind)
2. Variance Inflation Factor (VIF) for all variables
3. Principal Component Analysis to identify redundant dimensions
4. Recommendations for variable selection

Output: Saved to results/multicollinearity_analysis/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'multicollinearity_analysis'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for output files
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# VIF threshold (typically >10 indicates high multicollinearity)
VIF_THRESHOLD = 10.0

# Correlation threshold for redundancy (|r| > threshold)
CORRELATION_THRESHOLD = 0.85


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---\n")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_panel_data():
    """Load panel data matrix"""
    print_section("1. LOADING DATA")
    
    panel_path = DATA_DIR / 'panel_data_matrix.parquet'
    print(f"Loading: {panel_path.name}")
    
    df = pd.read_parquet(panel_path)
    
    print(f"  Shape: {df.shape}")
    print(f"  Index: {df.index.names}")
    print(f"  Variables: {list(df.columns)}")
    
    return df


# ============================================================================
# VARIABLE GROUPING
# ============================================================================

def group_variables_by_type(df):
    """Group variables by type (temperature, humidity, wind, surface)"""
    print_section("2. VARIABLE GROUPING")
    
    # Define variable groups
    groups = {
        'temperature': [
            'temperature_550', 'temperature_850', 'temperature_950', 'temperature_2m'
        ],
        'humidity': [
            'humidity_550', 'humidity_850', 'humidity_950'
        ],
        'u_wind': [
            'uwind_550', 'uwind_850', 'uwind_950', 'wind_u_10m'
        ],
        'v_wind': [
            'Vwind_550', 'Vwind_850', 'Vwind_950', 'wind_v_10m'
        ],
        'surface': [
            'surface_pressure', 'blh', 'total_precipitation', 'solar_radiation_downwards'
        ],
        'target': ['pm10']
    }
    
    # Filter to only include variables present in data
    available_groups = {}
    for group_name, var_list in groups.items():
        available_vars = [v for v in var_list if v in df.columns]
        if available_vars:
            available_groups[group_name] = available_vars
    
    print("Variable groups identified:")
    for group_name, var_list in available_groups.items():
        print(f"\n  {group_name.upper()}:")
        for var in var_list:
            print(f"    - {var}")
    
    return available_groups


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations_by_group(df, groups):
    """Analyze correlations within each variable group"""
    print_section("3. CORRELATION ANALYSIS BY GROUP")
    
    correlation_results = {}
    
    for group_name, var_list in groups.items():
        if group_name == 'target' or len(var_list) < 2:
            continue
        
        print_subsection(f"3.{list(groups.keys()).index(group_name) + 1} {group_name.upper()} Variables")
        
        # Calculate correlation matrix
        corr_matrix = df[var_list].corr()
        
        print(f"Correlation Matrix:")
        print(corr_matrix.round(3).to_string())
        
        # Identify highly correlated pairs
        high_corr_pairs = []
        for i in range(len(var_list)):
            for j in range(i + 1, len(var_list)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > CORRELATION_THRESHOLD:
                    high_corr_pairs.append((var_list[i], var_list[j], corr_value))
        
        if high_corr_pairs:
            print(f"\n⚠ Highly correlated pairs (|r| > {CORRELATION_THRESHOLD}):")
            for var1, var2, corr in high_corr_pairs:
                print(f"  {var1} <-> {var2}: {corr:.3f}")
        else:
            print(f"\n✓ No highly correlated pairs found (threshold: {CORRELATION_THRESHOLD})")
        
        correlation_results[group_name] = {
            'matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs
        }
    
    return correlation_results





# ============================================================================
# VARIANCE INFLATION FACTOR (VIF) ANALYSIS
# ============================================================================

def calculate_vif(df, exclude_vars=None):
    """Calculate VIF for all variables"""
    print_section("4. VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
    
    if exclude_vars is None:
        exclude_vars = ['pm10']
    
    # Select numeric columns, exclude target
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                    if col not in exclude_vars]
    
    # Drop any rows with NaN
    df_clean = df[numeric_cols].dropna()
    
    print(f"Calculating VIF for {len(numeric_cols)} variables...")
    print(f"Sample size: {len(df_clean)} observations\n")
    
    # Calculate VIF
    vif_data = []
    for i, col in enumerate(numeric_cols):
        try:
            vif = variance_inflation_factor(df_clean.values, i)
            vif_data.append({
                'Variable': col,
                'VIF': vif
            })
        except Exception as e:
            print(f"  Warning: Could not calculate VIF for {col}: {e}")
            vif_data.append({
                'Variable': col,
                'VIF': np.nan
            })
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    print("VIF Results (sorted by VIF value):")
    print(vif_df.to_string(index=False))
    
    # Identify problematic variables
    high_vif = vif_df[vif_df['VIF'] > VIF_THRESHOLD]
    
    if len(high_vif) > 0:
        print(f"\n⚠ Variables with high multicollinearity (VIF > {VIF_THRESHOLD}):")
        for _, row in high_vif.iterrows():
            print(f"  {row['Variable']}: VIF = {row['VIF']:.2f}")
    else:
        print(f"\n✓ No variables with excessive multicollinearity (all VIF < {VIF_THRESHOLD})")
    
    # Save VIF results
    output_path = RESULTS_DIR / f'vif_analysis_{TIMESTAMP}.csv'
    vif_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved VIF results to: {output_path.name}")
    
    return vif_df


# ============================================================================
# PRINCIPAL COMPONENT ANALYSIS
# ============================================================================

def perform_pca_by_group(df, groups):
    """Perform PCA on each variable group to assess dimensionality"""
    print_section("5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    
    pca_results = {}
    
    for group_name, var_list in groups.items():
        if group_name == 'target' or len(var_list) < 2:
            continue
        
        print_subsection(f"5.{list(groups.keys()).index(group_name) + 1} PCA: {group_name.upper()}")
        
        # Prepare data
        df_group = df[var_list].dropna()
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_group)
        
        # Perform PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"Number of variables: {len(var_list)}")
        print(f"\nExplained variance by component:")
        for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
            print(f"  PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")
        
        # Determine number of components needed for 95% variance
        n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
        print(f"\n✓ Components needed for 95% variance: {n_components_95} (out of {len(var_list)})")
        
        if n_components_95 < len(var_list):
            print(f"  → Potential to reduce from {len(var_list)} to {n_components_95} dimensions")
        
        pca_results[group_name] = {
            'pca': pca,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'n_components_95': n_components_95,
            'n_original': len(var_list)
        }

    
    return pca_results





# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def generate_recommendations(correlation_results, vif_df, pca_results, groups):
    """Generate recommendations for variable selection"""
    print_section("6. RECOMMENDATIONS FOR VARIABLE SELECTION")
    
    recommendations = []
    
    # Analyze each group
    for group_name, var_list in groups.items():
        if group_name == 'target' or len(var_list) < 2:
            continue
        
        print_subsection(f"{group_name.upper()} Variables")
        
        # Check correlation results
        high_corr = correlation_results.get(group_name, {}).get('high_corr_pairs', [])
        
        # Check PCA results
        pca_info = pca_results.get(group_name, {})
        n_orig = pca_info.get('n_original', len(var_list))
        n_needed = pca_info.get('n_components_95', n_orig)
        
        if high_corr:
            print(f"⚠ High correlation detected ({len(high_corr)} pairs)")
            print(f"  Recommendation: Consider removing redundant variables")
            
            # Suggest which variables to keep/remove
            vars_to_remove = set()
            for var1, var2, corr in high_corr:
                # Prefer keeping lower altitude (higher pressure) measurements
                if '950' in var1 or '2m' in var1 or '10m' in var1:
                    vars_to_remove.add(var2)
                else:
                    vars_to_remove.add(var1)
            
            if vars_to_remove:
                print(f"  Suggested removals: {', '.join(vars_to_remove)}")
                recommendations.append({
                    'group': group_name,
                    'issue': 'high_correlation',
                    'variables_to_remove': list(vars_to_remove),
                    'reason': f'High correlation (|r| > {CORRELATION_THRESHOLD})'
                })
        
        if n_needed < n_orig:
            reduction = n_orig - n_needed
            print(f"✓ PCA suggests dimension reduction: {n_orig} → {n_needed} ({reduction} fewer)")
            print(f"  Recommendation: Consider using PCA or dropping {reduction} variable(s)")
            recommendations.append({
                'group': group_name,
                'issue': 'high_dimensionality',
                'original_dims': n_orig,
                'recommended_dims': n_needed,
                'reason': 'PCA shows redundancy in dimensions'
            })
        else:
            print(f"✓ All {n_orig} variables contribute unique information")
        
        # Check VIF for this group
        group_vif = vif_df[vif_df['Variable'].isin(var_list)]
        high_vif_vars = group_vif[group_vif['VIF'] > VIF_THRESHOLD]
        
        if len(high_vif_vars) > 0:
            print(f"⚠ High VIF detected for {len(high_vif_vars)} variable(s)")
            for _, row in high_vif_vars.iterrows():
                print(f"  {row['Variable']}: VIF = {row['VIF']:.2f}")
    
    print_subsection("SUMMARY OF RECOMMENDATIONS")
    
    if not recommendations:
        print("✓ No significant multicollinearity issues detected.")
        print("✓ All variable groups appear to contribute unique information.")
    else:
        print(f"Found {len(recommendations)} potential improvements:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['group'].upper()}: {rec['reason']}")
            if 'variables_to_remove' in rec:
                print(f"   → Remove: {', '.join(rec['variables_to_remove'])}")
            elif 'original_dims' in rec:
                print(f"   → Reduce dimensions: {rec['original_dims']} → {rec['recommended_dims']}")
    
    # Save recommendations
    output_path = RESULTS_DIR / f'recommendations_{TIMESTAMP}.txt'
    with open(output_path, 'w') as f:
        f.write("MULTICOLLINEARITY ANALYSIS RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not recommendations:
            f.write("No significant multicollinearity issues detected.\n")
        else:
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec['group'].upper()}\n")
                f.write(f"   Issue: {rec['issue']}\n")
                f.write(f"   Reason: {rec['reason']}\n")
                if 'variables_to_remove' in rec:
                    f.write(f"   Variables to remove: {', '.join(rec['variables_to_remove'])}\n")
                elif 'original_dims' in rec:
                    f.write(f"   Dimension reduction: {rec['original_dims']} → {rec['recommended_dims']}\n")
                f.write("\n")
    
    print(f"\n✓ Saved recommendations to: {output_path.name}")
    
    return recommendations


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("MULTICOLLINEARITY ANALYSIS FOR ERA5 MULTI-LEVEL VARIABLES")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {RESULTS_DIR}")
    print(f"Timestamp: {TIMESTAMP}\n")
    
    # Load data
    df = load_panel_data()
    
    # Group variables by type
    groups = group_variables_by_type(df)
    
    # Correlation analysis
    correlation_results = analyze_correlations_by_group(df, groups)
    
    # VIF analysis
    vif_df = calculate_vif(df)
    
    # PCA analysis
    pca_results = perform_pca_by_group(df, groups)
    
    # Generate recommendations
    recommendations = generate_recommendations(correlation_results, vif_df, pca_results, groups)
    
    print_section("ANALYSIS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"\nFiles created:")
    print(f"  - vif_analysis_{TIMESTAMP}.csv")
    print(f"  - recommendations_{TIMESTAMP}.txt")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
