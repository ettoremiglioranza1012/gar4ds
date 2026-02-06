#!/usr/bin/env python3
"""
Model Specification Tests Module
================================
Performs Likelihood Ratio Tests (LRT) to assess model specifications.

Tests performed:
1. LRT: SDM vs SAR (Spatial Autoregressive)
2. LRT: SDM vs SEM (Spatial Error Model)
3. Information Criteria comparison (AIC, BIC)
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from libpysal.io import open as ioopen
from spreg import Panel_FE_Lag, Panel_FE_Error
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


def load_data(config: PipelineConfig) -> tuple:
    """Load panel data and spatial weights"""
    print_header("1. LOADING DATA")
    
    # Load panel data
    panel_path = config.get_panel_matrix_filtered_path()
    print(f"    Panel data: {panel_path}")
    df = pd.read_parquet(panel_path)
    print(f"    ✓ Shape: {df.shape}")
    
    # Load weights
    weights_path = config.get_spatial_weights_path()
    print(f"    Spatial weights: {weights_path}")
    w = ioopen(str(weights_path)).read()
    print(f"    ✓ Weights for {w.n} units")
    
    return df, w


def prepare_model_data(df: pd.DataFrame, config: PipelineConfig) -> tuple:
    """Prepare data for spatial models"""
    print_header("2. DATA PREPARATION")
    
    # Reset index for manipulation
    df_work = df.reset_index()
    
    # Get time and entity columns
    time_col = config.temporal.time_label
    
    # Log-transform PM10
    print("    Log-transforming PM10...")
    df_work['log_pm10'] = np.log(df_work['pm10'] + 1)
    
    # Identify feature columns
    target = 'pm10'
    exclude_cols = [time_col, 'station_id', target, 'log_pm10']
    feature_cols = [col for col in df_work.columns if col not in exclude_cols]
    
    print(f"    Features ({len(feature_cols)}): {feature_cols}")
    
    # Standardize features
    print("    Standardizing features...")
    scaler = StandardScaler()
    X = df_work[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    
    # Prepare model matrices - y needs to be 2D for spreg
    y = df_work['log_pm10'].values.reshape(-1, 1)
    
    # Get unique stations for panel structure
    stations = df_work['station_id'].unique()
    n_stations = len(stations)
    n_periods = len(df_work) // n_stations
    
    print(f"    ✓ Observations: {len(y)}")
    print(f"    ✓ Stations: {n_stations}")
    print(f"    ✓ Time periods: {n_periods}")
    
    return y, X_scaled, feature_cols


def fit_sar_model(y, X, w, feature_names: list) -> Dict[str, Any]:
    """Fit Spatial Autoregressive (SAR) model using Panel FE Lag"""
    print_header("3. FITTING SAR MODEL", level=2)
    print("    Specification: y = ρWy + Xβ + αᵢ + ε")
    
    try:
        model = Panel_FE_Lag(
            y=y,
            x=X,
            w=w,
            name_y=['log_pm10'],
            name_x=feature_names,
            name_ds="SAR_PM10_Panel"
        )
        
        print(f"    ✓ Rho (spatial lag): {model.rho:.4f}")
        print(f"    ✓ Log-likelihood: {model.logll:.2f}")
        print(f"    ✓ Parameters: {len(model.betas) + 1}")
        
        return {
            'model': model,
            'logll': model.logll,
            'rho': model.rho,
            'n_params': len(model.betas) + 1  # coefficients + rho
        }
    except Exception as e:
        print(f"    ✗ Error fitting SAR: {e}")
        import traceback
        traceback.print_exc()
        return None


def fit_sem_model(y, X, w, feature_names: list) -> Dict[str, Any]:
    """Fit Spatial Error Model (SEM) using Panel FE Error"""
    print_header("4. FITTING SEM MODEL", level=2)
    print("    Specification: y = Xβ + αᵢ + u, where u = λWu + ε")
    
    try:
        model = Panel_FE_Error(
            y=y,
            x=X,
            w=w,
            name_y=['log_pm10'],
            name_x=feature_names,
            name_ds="SEM_PM10_Panel"
        )
        
        print(f"    ✓ Lambda (error lag): {model.lam:.4f}")
        print(f"    ✓ Log-likelihood: {model.logll:.2f}")
        print(f"    ✓ Parameters: {len(model.betas) + 1}")
        
        return {
            'model': model,
            'logll': model.logll,
            'lam': model.lam,
            'n_params': len(model.betas) + 1
        }
    except Exception as e:
        print(f"    ✗ Error fitting SEM: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_information_criteria(results: Dict, n_obs: int) -> pd.DataFrame:
    """Compute AIC and BIC for model comparison"""
    print_header("5. INFORMATION CRITERIA")
    
    criteria = []
    
    for name, result in results.items():
        if result is None:
            continue
            
        k = result['n_params']
        logll = result['logll']
        
        aic = 2 * k - 2 * logll
        bic = k * np.log(n_obs) - 2 * logll
        
        criteria.append({
            'model': name,
            'log_likelihood': logll,
            'n_params': k,
            'AIC': aic,
            'BIC': bic
        })
        
        print(f"    {name}: AIC={aic:.2f}, BIC={bic:.2f}")
    
    return pd.DataFrame(criteria)


def perform_lrt(model1_logll: float, model1_params: int, 
                model2_logll: float, model2_params: int,
                test_name: str) -> Dict[str, Any]:
    """Perform Likelihood Ratio Test"""
    # LRT statistic: -2 * (log L_restricted - log L_unrestricted)
    lrt_stat = -2 * (model1_logll - model2_logll)
    df = abs(model2_params - model1_params)
    p_value = 1 - stats.chi2.cdf(abs(lrt_stat), df)
    
    print(f"\n    {test_name}:")
    print(f"    LRT statistic: {lrt_stat:.4f}")
    print(f"    Degrees of freedom: {df}")
    print(f"    P-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"    → Reject H₀ at 5% level")
    else:
        print(f"    → Fail to reject H₀ at 5% level")
    
    return {
        'test': test_name,
        'lrt_statistic': lrt_stat,
        'df': df,
        'p_value': p_value,
        'significant_5pct': p_value < 0.05
    }


def run_specification_tests(config: Optional[PipelineConfig] = None):
    """
    Main execution function for model specification tests.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
    """
    if config is None:
        config = load_config()
    
    # Setup directories
    results_dir = config.get_results_subdir("model_specification_tests")
    
    timestamp = datetime.now().strftime(config.logging.get("timestamp_format", "%Y%m%d_%H%M%S"))
    log_file = results_dir / f'lrt_tests_log_{timestamp}.txt'
    
    with open(log_file, 'w') as f:
        tee = Tee(sys.stdout, f)
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            print_header("MODEL SPECIFICATION TESTS")
            print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Aggregation: {config.temporal.aggregation}")
            
            # Load data
            df, w = load_data(config)
            
            # Prepare model data
            y, X, feature_cols = prepare_model_data(df, config)
            n_obs = len(y)
            
            # Fit models
            results = {}
            results['SAR'] = fit_sar_model(y, X, w, feature_cols)
            results['SEM'] = fit_sem_model(y, X, w, feature_cols)
            
            # Check if any models were successfully fit
            successful_models = {k: v for k, v in results.items() if v is not None}
            
            if not successful_models:
                print("\n    ⚠ WARNING: No models could be fit successfully.")
                print("    This may be due to data issues or insufficient variation.")
                print_header("TESTS INCOMPLETE")
                return
            
            # Information criteria
            ic_df = compute_information_criteria(results, n_obs)
            ic_path = results_dir / 'information_criteria.csv'
            ic_df.to_csv(ic_path, index=False)
            print(f"\n    ✓ Saved to: {ic_path}")
            
            # LRT tests
            print_header("6. LIKELIHOOD RATIO TESTS")
            
            lrt_results = []
            
            if results['SAR'] and results['SEM']:
                lrt = perform_lrt(
                    results['SAR']['logll'], results['SAR']['n_params'],
                    results['SEM']['logll'], results['SEM']['n_params'],
                    "SAR vs SEM"
                )
                lrt_results.append(lrt)
            
            # Save LRT results
            if lrt_results:
                lrt_df = pd.DataFrame(lrt_results)
                lrt_path = results_dir / 'lrt_test_results.csv'
                lrt_df.to_csv(lrt_path, index=False)
                print(f"\n    ✓ Saved LRT results to: {lrt_path}")
            
            # Summary
            print_header("SUMMARY")
            
            if len(ic_df) > 0 and 'AIC' in ic_df.columns:
                print("\nBest model by AIC:", ic_df.loc[ic_df['AIC'].idxmin(), 'model'])
                print("Best model by BIC:", ic_df.loc[ic_df['BIC'].idxmin(), 'model'])
            else:
                print("\n⚠ Insufficient models for comparison.")
            
            print_header("TESTS COMPLETED")
            
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            sys.stdout = old_stdout
            print(f"\n✓ Log saved to: {log_file}")


if __name__ == "__main__":
    run_specification_tests()
