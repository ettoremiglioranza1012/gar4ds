#!/usr/bin/env python3
"""
Filter Multicollinearity Module
===============================
Removes highly collinear variables from the panel data matrix based on
multicollinearity analysis results.

Decision criteria:
- Variables with VIF > 10 and high correlation with others: DROP
- Variables with unique physical interpretation: KEEP
- Surface/near-surface measurements preferred over upper-level
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import pandas as pd
import numpy as np

from ..config import PipelineConfig, load_config


# Default variable selection (can be overridden by config)
DEFAULT_VARIABLES_TO_DROP = [
    'temperature_550', 'temperature_850', 'temperature_950',
    'surface_pressure',
    'humidity_550', 'humidity_850',
    'uwind_550', 'Vwind_550',
]


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


def load_and_filter_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Load panel data and filter variables"""
    print_section("LOADING PANEL DATA")
    
    input_path = config.get_panel_matrix_path()
    print(f"Input file: {input_path}")
    
    df = pd.read_parquet(input_path)
    
    print(f"  Shape: {df.shape}")
    print(f"  Index: {df.index.names}")
    print(f"  Variables ({len(df.columns)}):")
    for col in sorted(df.columns):
        print(f"    - {col}")
    
    return df


def filter_variables(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Filter to keep only specified variables"""
    print_section("FILTERING VARIABLES")
    
    # Get variables to keep from config
    vars_to_keep = config.variables.keep_after_filtering
    
    if not vars_to_keep:
        # Use default: drop known collinear variables
        vars_to_drop = DEFAULT_VARIABLES_TO_DROP
        vars_to_keep = [col for col in df.columns if col not in vars_to_drop]
        
        print("Using default filtering (no config specified)")
        print(f"\nVariables to DROP ({len(vars_to_drop)}):")
        for var in vars_to_drop:
            if var in df.columns:
                print(f"  ✗ {var}")
    else:
        print("Using config-specified variables to keep")
    
    # Filter to only available variables
    available_vars = [v for v in vars_to_keep if v in df.columns]
    missing_vars = [v for v in vars_to_keep if v not in df.columns]
    
    if missing_vars:
        print(f"\n⚠ Variables in config but not in data:")
        for var in missing_vars:
            print(f"    - {var}")
    
    print(f"\nVariables to KEEP ({len(available_vars)}):")
    for var in sorted(available_vars):
        print(f"  ✓ {var}")
    
    # Filter dataframe
    df_filtered = df[available_vars].copy()
    
    print_subsection("Filtering Result")
    print(f"Original variables: {len(df.columns)}")
    print(f"Filtered variables: {len(df_filtered.columns)}")
    print(f"Variables removed: {len(df.columns) - len(df_filtered.columns)}")
    
    return df_filtered


def validate_filtered_data(df_filtered: pd.DataFrame):
    """Validate filtered dataset"""
    print_section("VALIDATION")
    
    # Check for missing values
    missing = df_filtered.isna().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        print("Missing values by variable:")
        for col in df_filtered.columns:
            if missing[col] > 0:
                pct = (missing[col] / len(df_filtered)) * 100
                print(f"  {col}: {missing[col]:,} ({pct:.2f}%)")
    else:
        print("✓ No missing values in filtered dataset")
    
    # Summary statistics
    print_subsection("Summary Statistics")
    print(df_filtered.describe().round(2).to_string())
    
    # Correlation check
    print_subsection("Post-Filter Correlations (Top 10)")
    corr = df_filtered.corr()
    
    # Get upper triangle correlation pairs
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append((
                corr.columns[i],
                corr.columns[j],
                abs(corr.iloc[i, j])
            ))
    
    corr_pairs.sort(key=lambda x: -x[2])
    
    for v1, v2, r in corr_pairs[:10]:
        flag = "⚠" if r > 0.85 else ""
        print(f"  {v1:30s} <-> {v2:30s}: {r:.3f} {flag}")


def save_filtered_data(df_filtered: pd.DataFrame, config: PipelineConfig):
    """Save filtered dataset"""
    print_section("SAVING FILTERED DATASET")
    
    output_path = config.get_panel_matrix_filtered_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output file: {output_path}")
    
    df_filtered.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    file_size = output_path.stat().st_size / (1024**2)
    print(f"✓ Filtered dataset saved successfully")
    print(f"  File size: {file_size:.2f} MB")


def run_filter_multicollinearity(config: Optional[PipelineConfig] = None) -> pd.DataFrame:
    """
    Main execution function for filtering multicollinearity.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
        
    Returns:
        Filtered DataFrame
    """
    if config is None:
        config = load_config()
    
    # Setup logging
    results_dir = config.get_results_subdir("dataset_documentation")
    timestamp = datetime.now().strftime(config.logging.get("timestamp_format", "%Y%m%d_%H%M%S"))
    log_file = results_dir / f"multicollinearity_filter_{timestamp}.txt"
    
    logger = OutputLogger(log_file)
    sys.stdout = logger
    
    try:
        print_section("FILTER MULTICOLLINEARITY")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Aggregation: {config.temporal.aggregation}")
        
        # Load data
        df = load_and_filter_dataset(config)
        
        # Filter variables
        df_filtered = filter_variables(df, config)
        
        # Validate
        validate_filtered_data(df_filtered)
        
        # Save
        save_filtered_data(df_filtered, config)
        
        print_section("FILTERING COMPLETED")
        print(f"✓ Original: {df.shape[0]:,} obs × {df.shape[1]} vars")
        print(f"✓ Filtered: {df_filtered.shape[0]:,} obs × {df_filtered.shape[1]} vars")
        print(f"✓ Log saved to: {log_file}")
        
        return df_filtered
        
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
    run_filter_multicollinearity()
