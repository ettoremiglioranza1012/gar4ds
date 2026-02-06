"""
MODEL SPECIFICATION TESTS - SDM vs SAR vs SEM
==============================================

This script performs Likelihood Ratio Tests (LRT) to assess the superiority
of the Spatial Durbin Model (SDM) over alternative specifications.

TESTS PERFORMED:
1. LRT: SDM vs SAR (Spatial Autoregressive)
   - H₀: θ = 0 (spatially lagged covariates have no effect)
   - If rejected → SDM is preferred over SAR

2. LRT: SDM vs SEM (Spatial Error Model)
   - H₀: θ + ρβ = 0 (spatial dependence only in errors)
   - If rejected → SDM is preferred over SEM

MODELS:
- SDM: y = ρWy + Xβ + WXθ + ε
- SAR: y = ρWy + Xβ + ε (restricted SDM with θ=0)
- SEM: y = Xβ + u, where u = λWu + ε (spatial error structure)

METHODOLOGY:
1. Load pre-processed panel data with spatially lagged features
2. Fit three competing models: SDM, SAR, SEM
3. Compute LRT statistics and p-values
4. Compare AIC/BIC for robustness
5. Generate diagnostic visualizations

OUTPUTS:
- Text log: results/model_specification_tests/lrt_tests_log.txt
- CSV results: results/model_specification_tests/
- Visualizations: assets/model_specification_tests/

Author: Ettore Miglioranza
Date: 6 February 2026
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal.io import open as ioopen
from spreg import Panel_FE_Lag, Panel_FE_Error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'model_specification_tests'
WEIGHTS_DIR = PROJECT_DIR / 'weights'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Output file for verbose logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = RESULTS_DIR / f'lrt_tests_log_{timestamp}.txt'


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
        n_weeks = df.index.get_level_values(0).nunique()
        n_stations = df.index.get_level_values(1).nunique()
        print(f"    ✓ MultiIndex confirmed: {df.index.names}")
        print(f"    ✓ Time periods: {n_weeks}")
        print(f"    ✓ Stations: {n_stations}")
        print(f"    ✓ Total observations: {len(df)}")
    
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


def prepare_data(panel_df, meta_df, w):
    """
    Prepare data for model estimation:
    1. Log-transform PM10
    2. Extract meteorological variables
    3. Standardize variables
    4. Create spatially lagged features (WX)
    5. Filter to common stations
    """
    print_header("4. PREPARING DATA FOR MODEL ESTIMATION")
    
    # Log-transform PM10
    print("    Step 1: Log-transforming PM10")
    panel_df['log_pm10'] = np.log(panel_df['pm10'] + 1)
    print(f"        ✓ log_pm10 created (mean={panel_df['log_pm10'].mean():.3f}, std={panel_df['log_pm10'].std():.3f})")
    
    # Define meteorological variables
    met_vars = [col for col in panel_df.columns if col not in ['pm10', 'log_pm10']]
    print(f"\n    Step 2: Meteorological variables identified ({len(met_vars)}):")
    for var in met_vars:
        print(f"        - {var}")
    
    # Check station alignment
    print("\n    Step 3: Checking station alignment")
    panel_stations = panel_df.index.get_level_values('station_id').unique()
    meta_stations = meta_df.index.unique()
    common_stations = panel_stations.intersection(meta_stations)
    print(f"        Panel stations: {len(panel_stations)}")
    print(f"        Metadata stations: {len(meta_stations)}")
    print(f"        ✓ Common stations: {len(common_stations)}")
    
    # Standardize variables
    print("\n    Step 4: Standardizing meteorological variables")
    scaler = StandardScaler()
    panel_df[met_vars] = scaler.fit_transform(panel_df[met_vars])
    print(f"        ✓ Standardized {len(met_vars)} variables (mean≈0, std≈1)")
    
    # Create spatially lagged features (WX)
    print("\n    Step 5: Creating spatially lagged features (WX matrix)")
    weeks = panel_df.index.get_level_values('week_start').unique().sort_values()
    n_weeks = len(weeks)
    n_stations = len(common_stations)
    
    print(f"        Dimensions: {n_weeks} weeks × {n_stations} stations × {len(met_vars)} vars")
    
    # Initialize lagged feature columns
    lagged_vars = [f'lag_{var}' for var in met_vars]
    for lag_var in lagged_vars:
        panel_df[lag_var] = np.nan
    
    # Create mapping
    w_ids = w.id_order
    station_to_idx = {station: idx for idx, station in enumerate(w_ids)}
    
    # Compute spatial lags
    progress_interval = max(1, n_weeks // 10)
    for i, week in enumerate(weeks):
        week_data = panel_df.xs(week, level='week_start')
        week_data_aligned = week_data.reindex(w_ids)
        X_week = week_data_aligned[met_vars].values
        
        WX_week = np.zeros_like(X_week)
        for j, station in enumerate(w_ids):
            neighbors = w.neighbors[station]
            weights = w.weights[station]
            neighbor_indices = [station_to_idx[n] for n in neighbors]
            WX_week[j, :] = np.sum([weights[k] * X_week[neighbor_indices[k], :] 
                                    for k in range(len(neighbors))], axis=0)
        
        for v, lag_var in enumerate(lagged_vars):
            for s, station in enumerate(w_ids):
                panel_df.loc[(week, station), lag_var] = WX_week[s, v]
        
        if (i + 1) % progress_interval == 0:
            print(f"        Progress: {i+1}/{n_weeks} weeks ({(i+1)/n_weeks*100:.1f}%)")
    
    print(f"        ✓ Created {len(lagged_vars)} spatially lagged features")
    
    # Filter to common stations and remove NaN
    panel_filtered = panel_df.loc[(slice(None), common_stations.tolist()), :]
    panel_filtered = panel_filtered.dropna(subset=lagged_vars)
    
    print(f"\n    Step 6: Final dataset")
    print(f"        ✓ Observations: {len(panel_filtered):,}")
    print(f"        ✓ Stations: {len(common_stations)}")
    print(f"        ✓ Direct features (X): {len(met_vars)}")
    print(f"        ✓ Lagged features (WX): {len(lagged_vars)}")
    
    return panel_filtered, met_vars, lagged_vars, common_stations


def fit_sdm_model(panel_filtered, met_vars, lagged_vars, w):
    """
    Fit Spatial Durbin Model (SDM)
    Specification: y = ρWy + Xβ + WXθ + ε
    """
    print_header("5. FITTING SPATIAL DURBIN MODEL (SDM)")
    print("    Specification: y = ρWy + Xβ + WXθ + ε")
    print("    Estimator: Panel Fixed Effects Spatial Lag")
    
    # Prepare arrays
    y = panel_filtered['log_pm10'].values.reshape(-1, 1)
    X = panel_filtered[met_vars + lagged_vars].values
    
    print(f"\n    Data dimensions:")
    print(f"        y: {y.shape}")
    print(f"        X (met + lagged): {X.shape}")
    print(f"        Features: {len(met_vars)} direct + {len(lagged_vars)} lagged = {len(met_vars) + len(lagged_vars)}")
    
    print("\n    Fitting SDM...")
    try:
        model_sdm = Panel_FE_Lag(
            y=y,
            x=X,
            w=w,
            name_y=['log_pm10'],
            name_x=met_vars + lagged_vars,
            name_ds="SDM_PM10_Panel"
        )
        
        print(f"    ✓ SDM estimation complete!")
        print(f"    ✓ Log-likelihood: {model_sdm.logll:.2f}")
        print(f"    ✓ ρ (spatial lag): {model_sdm.rho:.6f}")
        print(f"    ✓ Parameters: {len(model_sdm.betas) + 1}")  # +1 for rho
        
        return model_sdm
        
    except Exception as e:
        print(f"    ✗ Error fitting SDM: {e}")
        return None


def fit_sar_model(panel_filtered, met_vars, w):
    """
    Fit Spatial Autoregressive Model (SAR)
    Specification: y = ρWy + Xβ + ε
    This is SDM with θ=0 (no spatially lagged covariates)
    """
    print_header("6. FITTING SPATIAL AUTOREGRESSIVE MODEL (SAR)")
    print("    Specification: y = ρWy + Xβ + ε")
    print("    Note: Restricted SDM with θ=0 (no WX terms)")
    
    # Prepare arrays (only direct effects, no lagged)
    y = panel_filtered['log_pm10'].values.reshape(-1, 1)
    X = panel_filtered[met_vars].values
    
    print(f"\n    Data dimensions:")
    print(f"        y: {y.shape}")
    print(f"        X (met only): {X.shape}")
    print(f"        Features: {len(met_vars)} direct effects")
    
    print("\n    Fitting SAR...")
    try:
        model_sar = Panel_FE_Lag(
            y=y,
            x=X,
            w=w,
            name_y=['log_pm10'],
            name_x=met_vars,
            name_ds="SAR_PM10_Panel"
        )
        
        print(f"    ✓ SAR estimation complete!")
        print(f"    ✓ Log-likelihood: {model_sar.logll:.2f}")
        print(f"    ✓ ρ (spatial lag): {model_sar.rho:.6f}")
        print(f"    ✓ Parameters: {len(model_sar.betas) + 1}")
        
        return model_sar
        
    except Exception as e:
        print(f"    ✗ Error fitting SAR: {e}")
        return None


def fit_sem_model(panel_filtered, met_vars, w):
    """
    Fit Spatial Error Model (SEM)
    Specification: y = Xβ + u, where u = λWu + ε
    Spatial dependence only in error term
    """
    print_header("7. FITTING SPATIAL ERROR MODEL (SEM)")
    print("    Specification: y = Xβ + u, where u = λWu + ε")
    print("    Note: Spatial dependence only in error structure")
    
    # Prepare arrays
    y = panel_filtered['log_pm10'].values.reshape(-1, 1)
    X = panel_filtered[met_vars].values
    
    print(f"\n    Data dimensions:")
    print(f"        y: {y.shape}")
    print(f"        X (met only): {X.shape}")
    print(f"        Features: {len(met_vars)} direct effects")
    
    print("\n    Fitting SEM...")
    try:
        model_sem = Panel_FE_Error(
            y=y,
            x=X,
            w=w,
            name_y=['log_pm10'],
            name_x=met_vars,
            name_ds="SEM_PM10_Panel"
        )
        
        print(f"    ✓ SEM estimation complete!")
        print(f"    ✓ Log-likelihood: {model_sem.logll:.2f}")
        print(f"    ✓ λ (spatial error): {model_sem.lam:.6f}")
        print(f"    ✓ Parameters: {len(model_sem.betas) + 1}")
        
        return model_sem
        
    except Exception as e:
        print(f"    ✗ Error fitting SEM: {e}")
        return None


def compute_lrt(model_unrestricted, model_restricted, restriction_name, df_diff):
    """
    Compute Likelihood Ratio Test
    
    LRT statistic = 2(LL_unrestricted - LL_restricted)
    Under H₀, LRT ~ χ²(df_diff)
    
    Parameters:
    -----------
    model_unrestricted : Model object
        Unrestricted model (typically SDM)
    model_restricted : Model object
        Restricted model (SAR or SEM)
    restriction_name : str
        Name of the restriction being tested
    df_diff : int
        Degrees of freedom difference (number of restrictions)
    """
    print_header(f"LRT: {restriction_name}", level=3)
    
    ll_unrestricted = model_unrestricted.logll
    ll_restricted = model_restricted.logll
    
    lrt_stat = 2 * (ll_unrestricted - ll_restricted)
    p_value = 1 - stats.chi2.cdf(lrt_stat, df_diff)
    
    print(f"    Log-likelihood (unrestricted): {ll_unrestricted:.2f}")
    print(f"    Log-likelihood (restricted):   {ll_restricted:.2f}")
    print(f"    LRT statistic: {lrt_stat:.2f}")
    print(f"    Degrees of freedom: {df_diff}")
    print(f"    P-value: {p_value:.6f}")
    
    if p_value < 0.001:
        significance = "***"
        conclusion = "STRONGLY REJECT H₀"
    elif p_value < 0.01:
        significance = "**"
        conclusion = "REJECT H₀"
    elif p_value < 0.05:
        significance = "*"
        conclusion = "REJECT H₀"
    elif p_value < 0.10:
        significance = "†"
        conclusion = "MARGINALLY REJECT H₀"
    else:
        significance = ""
        conclusion = "FAIL TO REJECT H₀"
    
    print(f"    Significance: {significance}")
    print(f"    Conclusion: {conclusion}")
    
    return {
        'll_unrestricted': ll_unrestricted,
        'll_restricted': ll_restricted,
        'lrt_statistic': lrt_stat,
        'df': df_diff,
        'p_value': p_value,
        'significance': significance,
        'conclusion': conclusion
    }


def perform_specification_tests(model_sdm, model_sar, model_sem, n_met_vars):
    """
    Perform all specification tests
    """
    print_header("8. LIKELIHOOD RATIO TESTS")
    
    results = {}
    
    # Test 1: SDM vs SAR (H₀: θ=0)
    print("\n" + "-" * 80)
    print("TEST 1: SDM vs SAR")
    print("-" * 80)
    print("H₀: θ = 0 (spatially lagged covariates have no effect)")
    print("H₁: θ ≠ 0 (spatially lagged covariates matter)")
    print("\nIf H₀ is rejected → SDM is preferred over SAR")
    
    # Degrees of freedom = number of WX terms
    df_sdm_sar = n_met_vars
    
    results['sdm_vs_sar'] = compute_lrt(
        model_sdm, model_sar,
        "SDM vs SAR (H₀: θ=0)",
        df_sdm_sar
    )
    
    # Test 2: SDM vs SEM (H₀: θ+ρβ=0)
    print("\n" + "-" * 80)
    print("TEST 2: SDM vs SEM")
    print("-" * 80)
    print("H₀: θ + ρβ = 0 (spatial dependence only in errors)")
    print("H₁: θ + ρβ ≠ 0 (spatial spillovers in covariates)")
    print("\nIf H₀ is rejected → SDM is preferred over SEM")
    print("\nNote: This is a more complex restriction.")
    print("      Direct test requires Wald test on θ + ρβ = 0.")
    print("      LRT provides approximate comparison.")
    
    # For LRT approximation, df = number of restrictions
    # θ + ρβ = 0 implies n_met_vars restrictions
    df_sdm_sem = n_met_vars
    
    results['sdm_vs_sem'] = compute_lrt(
        model_sdm, model_sem,
        "SDM vs SEM (H₀: θ+ρβ=0)",
        df_sdm_sem
    )
    
    return results


def compute_information_criteria(model_sdm, model_sar, model_sem):
    """
    Compute AIC and BIC for model comparison
    Lower values indicate better fit
    """
    print_header("9. INFORMATION CRITERIA COMPARISON")
    print("    Lower values indicate better model fit")
    
    models = {
        'SDM': model_sdm,
        'SAR': model_sar,
        'SEM': model_sem
    }
    
    ic_results = []
    
    for name, model in models.items():
        # Get log-likelihood and number of parameters
        ll = model.logll
        
        # Count parameters: betas + spatial parameter (rho/lambda)
        k = len(model.betas) + 1
        n = model.n
        
        # AIC = -2*LL + 2*k
        aic = -2 * ll + 2 * k
        
        # BIC = -2*LL + k*ln(n)
        bic = -2 * ll + k * np.log(n)
        
        ic_results.append({
            'model': name,
            'log_likelihood': ll,
            'n_parameters': k,
            'n_observations': n,
            'AIC': aic,
            'BIC': bic
        })
        
        print(f"\n    {name}:")
        print(f"        Log-likelihood: {ll:.2f}")
        print(f"        Parameters: {k}")
        print(f"        AIC: {aic:.2f}")
        print(f"        BIC: {bic:.2f}")
    
    ic_df = pd.DataFrame(ic_results)
    
    # Identify best models
    best_aic = ic_df.loc[ic_df['AIC'].idxmin(), 'model']
    best_bic = ic_df.loc[ic_df['BIC'].idxmin(), 'model']
    
    print("\n    Best Model:")
    print(f"        By AIC: {best_aic}")
    print(f"        By BIC: {best_bic}")
    
    return ic_df


def summarize_results(lrt_results, ic_df):
    """
    Provide comprehensive summary of all tests
    """
    print_header("10. COMPREHENSIVE SUMMARY")
    
    print("\n    LIKELIHOOD RATIO TESTS:")
    print("    " + "-" * 76)
    
    # SDM vs SAR
    sar_result = lrt_results['sdm_vs_sar']
    print(f"\n    1. SDM vs SAR (H₀: θ=0)")
    print(f"        LRT statistic: {sar_result['lrt_statistic']:.2f}")
    print(f"        P-value: {sar_result['p_value']:.6f} {sar_result['significance']}")
    print(f"        → {sar_result['conclusion']}")
    if 'REJECT' in sar_result['conclusion']:
        print(f"        ✓ SDM is PREFERRED over SAR")
    else:
        print(f"        ⚠ SAR may be sufficient")
    
    # SDM vs SEM
    sem_result = lrt_results['sdm_vs_sem']
    print(f"\n    2. SDM vs SEM (H₀: θ+ρβ=0)")
    print(f"        LRT statistic: {sem_result['lrt_statistic']:.2f}")
    print(f"        P-value: {sem_result['p_value']:.6f} {sem_result['significance']}")
    print(f"        → {sem_result['conclusion']}")
    if 'REJECT' in sem_result['conclusion']:
        print(f"        ✓ SDM is PREFERRED over SEM")
    else:
        print(f"        ⚠ SEM may be sufficient")
    
    print("\n    INFORMATION CRITERIA:")
    print("    " + "-" * 76)
    
    best_aic = ic_df.loc[ic_df['AIC'].idxmin(), 'model']
    best_bic = ic_df.loc[ic_df['BIC'].idxmin(), 'model']
    
    print(f"\n    Best model by AIC: {best_aic}")
    print(f"    Best model by BIC: {best_bic}")
    
    print("\n" + "    " + "=" * 76)
    print("\n    FINAL RECOMMENDATION:")
    print("    " + "=" * 76)
    
    # Count how many tests favor SDM
    sdm_support = 0
    if 'REJECT' in sar_result['conclusion']:
        sdm_support += 1
    if 'REJECT' in sem_result['conclusion']:
        sdm_support += 1
    if best_aic == 'SDM':
        sdm_support += 1
    if best_bic == 'SDM':
        sdm_support += 1
    
    print(f"\n    Evidence supporting SDM: {sdm_support}/4 criteria")
    
    if sdm_support >= 3:
        print("\n    ✓✓✓ STRONG EVIDENCE FOR SDM")
        print("        The Spatial Durbin Model is STRONGLY PREFERRED.")
        print("        Spatially lagged covariates (WX) are essential for")
        print("        capturing cross-border PM10 transport dynamics.")
    elif sdm_support == 2:
        print("\n    ✓✓ MODERATE EVIDENCE FOR SDM")
        print("        The Spatial Durbin Model is PREFERRED.")
        print("        Results suggest importance of neighbor characteristics.")
    else:
        print("\n    ⚠ LIMITED EVIDENCE FOR SDM")
        print("        Simpler specifications (SAR/SEM) may be adequate.")
        print("        Review theoretical justification for SDM.")



def save_results(lrt_results, ic_df):
    """
    Save all results to CSV files
    """
    print_header("12. SAVING RESULTS")
    
    # 1. LRT Results
    print("    Saving LRT test results...")
    lrt_df = pd.DataFrame([
        {
            'test': 'SDM vs SAR',
            'null_hypothesis': 'θ = 0',
            'df': lrt_results['sdm_vs_sar']['df'],
            'lrt_statistic': lrt_results['sdm_vs_sar']['lrt_statistic'],
            'p_value': lrt_results['sdm_vs_sar']['p_value'],
            'significance': lrt_results['sdm_vs_sar']['significance'],
            'conclusion': lrt_results['sdm_vs_sar']['conclusion']
        },
        {
            'test': 'SDM vs SEM',
            'null_hypothesis': 'θ + ρβ = 0',
            'df': lrt_results['sdm_vs_sem']['df'],
            'lrt_statistic': lrt_results['sdm_vs_sem']['lrt_statistic'],
            'p_value': lrt_results['sdm_vs_sem']['p_value'],
            'significance': lrt_results['sdm_vs_sem']['significance'],
            'conclusion': lrt_results['sdm_vs_sem']['conclusion']
        }
    ])
    lrt_df.to_csv(RESULTS_DIR / 'lrt_test_results.csv', index=False)
    print(f"    ✓ lrt_test_results.csv")
    
    # 2. Information Criteria
    print("    Saving information criteria...")
    ic_df.to_csv(RESULTS_DIR / 'information_criteria.csv', index=False)
    print(f"    ✓ information_criteria.csv")
    
    # 3. Summary
    print("    Creating summary report...")
    
    sdm_support = 0
    if 'REJECT' in lrt_results['sdm_vs_sar']['conclusion']:
        sdm_support += 1
    if 'REJECT' in lrt_results['sdm_vs_sem']['conclusion']:
        sdm_support += 1
    if ic_df.loc[ic_df['AIC'].idxmin(), 'model'] == 'SDM':
        sdm_support += 1
    if ic_df.loc[ic_df['BIC'].idxmin(), 'model'] == 'SDM':
        sdm_support += 1
    
    summary_df = pd.DataFrame([{
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sdm_vs_sar_p_value': lrt_results['sdm_vs_sar']['p_value'],
        'sdm_vs_sar_rejected': 'REJECT' in lrt_results['sdm_vs_sar']['conclusion'],
        'sdm_vs_sem_p_value': lrt_results['sdm_vs_sem']['p_value'],
        'sdm_vs_sem_rejected': 'REJECT' in lrt_results['sdm_vs_sem']['conclusion'],
        'best_model_aic': ic_df.loc[ic_df['AIC'].idxmin(), 'model'],
        'best_model_bic': ic_df.loc[ic_df['BIC'].idxmin(), 'model'],
        'sdm_support_score': f"{sdm_support}/4",
        'recommendation': 'SDM STRONGLY PREFERRED' if sdm_support >= 3 else 
                         'SDM PREFERRED' if sdm_support == 2 else 
                         'MIXED EVIDENCE'
    }])
    summary_df.to_csv(RESULTS_DIR / 'specification_test_summary.csv', index=False)
    print(f"    ✓ specification_test_summary.csv")
    
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
    print("MODEL SPECIFICATION TESTS - SDM vs SAR vs SEM")
    print("=" * 80)
    print(f"\nAnalysis Date: {datetime.now()}")
    print(f"\nOutputs:")
    print(f"    Text Log: {output_file}")
    print(f"    CSV Results: {RESULTS_DIR}")
    
    try:
        # 1-3. Load data
        panel_df = load_panel_data(DATA_DIR / 'panel_data_matrix_filtered_for_collinearity.parquet')
        meta_df = load_station_metadata(DATA_DIR / 'pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson')
        w = load_spatial_weights(WEIGHTS_DIR / 'spatial_weights_knn6.gal')
        
        # 4. Prepare data
        panel_filtered, met_vars, lagged_vars, common_stations = prepare_data(panel_df, meta_df, w)
        
        # 5-7. Fit three models
        model_sdm = fit_sdm_model(panel_filtered, met_vars, lagged_vars, w)
        model_sar = fit_sar_model(panel_filtered, met_vars, w)
        model_sem = fit_sem_model(panel_filtered, met_vars, w)
        
        if model_sdm is None or model_sar is None or model_sem is None:
            print("\n✗ Model fitting failed. Cannot proceed with tests.")
            return
        
        # 8. Perform specification tests
        lrt_results = perform_specification_tests(model_sdm, model_sar, model_sem, len(met_vars))
        
        # 9. Compute information criteria
        ic_df = compute_information_criteria(model_sdm, model_sar, model_sem)
        
        # 10. Summarize results
        summarize_results(lrt_results, ic_df)
        
        # 11. Save results
        save_results(lrt_results, ic_df)
        
        # Final summary
        print_header("ANALYSIS COMPLETE", level=1)
        print("\n✅ ALL TESTS COMPLETED:")
        print(f"    → LRT: SDM vs SAR (p={lrt_results['sdm_vs_sar']['p_value']:.6f})")
        print(f"    → LRT: SDM vs SEM (p={lrt_results['sdm_vs_sem']['p_value']:.6f})")
        print(f"    → Best model (AIC): {ic_df.loc[ic_df['AIC'].idxmin(), 'model']}")
        print(f"    → Best model (BIC): {ic_df.loc[ic_df['BIC'].idxmin(), 'model']}")
        
        print("\n✅ ALL OUTPUTS GENERATED:")
        print(f"    → Text log: {output_file}")
        print(f"    → CSV results: {RESULTS_DIR}")
        
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
