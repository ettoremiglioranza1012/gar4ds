"""
Filter Panel Dataset for Multicollinearity
===========================================

This script removes highly collinear variables from the panel data matrix
based on the multicollinearity analysis results (VIF and correlation analysis).

Decision criteria:
- Variables with VIF > 10 and high correlation with others: DROP
- Variables with unique physical interpretation despite high VIF: KEEP
- Surface/near-surface measurements preferred over upper-level

Variables to DROP (8):
- temperature_850, temperature_950, temperature_550 (extreme VIF, redundant)
- surface_pressure (VIF = 2,335)
- humidity_850, humidity_550 (high VIF, redundant)
- uwind_550, Vwind_550 (above boundary layer, minimal impact)

Variables to KEEP (12):
- temperature_2m, humidity_950 (surface conditions)
- blh, solar_radiation_downwards (atmospheric state)
- wind_u_10m, wind_v_10m (surface winds)
- uwind_850, uwind_950, Vwind_850, Vwind_950 (boundary layer winds)
- total_precipitation (wet deposition)
- pm10 (target)

Output: panel_data_matrix_filtered_for_collinearity.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'dataset_documentation'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input/Output files
INPUT_FILE = DATA_DIR / 'panel_data_matrix.parquet'
OUTPUT_FILE = DATA_DIR / 'panel_data_matrix_filtered_for_collinearity.parquet'

# Output log
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_LOG = RESULTS_DIR / f'multicollinearity_filter_{TIMESTAMP}.txt'


# ============================================================================
# VARIABLE SELECTION DECISIONS
# ============================================================================

VARIABLES_TO_DROP = [
    # Temperature - upper levels (extreme VIF, highly correlated with temp_2m)
    'temperature_550',   # VIF = 58,255; r = 0.85 with temp_2m
    'temperature_850',   # VIF = 331,115; r = 0.96 with temp_2m
    'temperature_950',   # VIF = 307,653; r = 0.93 with temp_2m
    
    # Pressure (extreme VIF, indirect effects)
    'surface_pressure',  # VIF = 2,335
    
    # Humidity - upper levels (high VIF, redundant)
    'humidity_550',      # VIF = 35; upper-level moisture less relevant
    'humidity_850',      # VIF = 251; r = 0.80 with humidity_950
    
    # Wind - upper levels (above boundary layer)
    'uwind_550',         # VIF = 3.8; above BL, minimal surface impact
    'Vwind_550',         # VIF = 2.7; above BL, minimal surface impact
]

VARIABLES_TO_KEEP = [
    # Target
    'pm10',
    
    # Surface temperature (most policy-relevant)
    'temperature_2m',    # VIF drops to ~4 after removals
    
    # Near-surface humidity (hygroscopic growth)
    'humidity_950',      # VIF drops after removals
    
    # Atmospheric state
    'blh',                          # Critical PM10 dispersion variable
    'solar_radiation_downwards',    # Photochemistry driver
    
    # Surface winds (direct dispersion)
    'wind_u_10m',        # VIF = 4.1
    'wind_v_10m',        # VIF = 3.8
    
    # Boundary layer winds (transport within mixing layer)
    'uwind_850',         # VIF = 4.3; mid-level transport
    'uwind_950',         # VIF = 8.0; near-surface transport
    'Vwind_850',         # VIF = 4.0; mid-level meridional
    'Vwind_950',         # VIF = 3.3; near-surface meridional
    
    # Wet deposition
    'total_precipitation',  # VIF = 3.4
]


# ============================================================================
# OUTPUT LOGGER
# ============================================================================

class OutputLogger:
    """Captures console output and saves to file"""
    def __init__(self, filepath):
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
# MAIN PROCESSING
# ============================================================================

def load_and_filter_dataset():
    """Load panel data and filter based on multicollinearity decisions"""
    
    print_section("MULTICOLLINEARITY-BASED DATASET FILTERING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {INPUT_FILE.name}")
    print(f"Output: {OUTPUT_FILE.name}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print_subsection("1. LOADING ORIGINAL DATASET")
    
    df = pd.read_parquet(INPUT_FILE)
    
    print(f"Loaded: {INPUT_FILE.name}")
    print(f"  Shape: {df.shape}")
    print(f"  Index: {df.index.names}")
    print(f"  Variables ({len(df.columns)}): {sorted(df.columns.tolist())}")
    
    # ========================================================================
    # VERIFY VARIABLES
    # ========================================================================
    
    print_subsection("2. VERIFYING VARIABLE LISTS")
    
    # Check which variables to drop actually exist
    present_to_drop = [v for v in VARIABLES_TO_DROP if v in df.columns]
    missing_to_drop = [v for v in VARIABLES_TO_DROP if v not in df.columns]
    
    print(f"Variables marked for REMOVAL ({len(VARIABLES_TO_DROP)}):")
    for var in VARIABLES_TO_DROP:
        status = "✓ FOUND" if var in present_to_drop else "✗ NOT FOUND"
        print(f"  - {var:30s} {status}")
    
    if missing_to_drop:
        print(f"\n⚠ Warning: {len(missing_to_drop)} variables marked for removal not found in dataset:")
        for var in missing_to_drop:
            print(f"  - {var}")
    
    # Check which variables to keep actually exist
    present_to_keep = [v for v in VARIABLES_TO_KEEP if v in df.columns]
    missing_to_keep = [v for v in VARIABLES_TO_KEEP if v not in df.columns]
    
    print(f"\nVariables marked to KEEP ({len(VARIABLES_TO_KEEP)}):")
    for var in VARIABLES_TO_KEEP:
        status = "✓ FOUND" if var in present_to_keep else "✗ NOT FOUND"
        print(f"  - {var:30s} {status}")
    
    if missing_to_keep:
        print(f"\n⚠ Warning: {len(missing_to_keep)} variables marked to keep not found in dataset:")
        for var in missing_to_keep:
            print(f"  - {var}")
    
    # ========================================================================
    # APPLY FILTER
    # ========================================================================
    
    print_subsection("3. APPLYING MULTICOLLINEARITY FILTER")
    
    # Filter to keep only specified variables
    df_filtered = df[present_to_keep].copy()
    
    print(f"✓ Filtered dataset created")
    print(f"  Original variables: {len(df.columns)}")
    print(f"  Removed variables: {len(present_to_drop)}")
    print(f"  Retained variables: {len(df_filtered.columns)}")
    print(f"  Reduction: {len(present_to_drop) / len(df.columns) * 100:.1f}%")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print_subsection("4. FILTERED DATASET SUMMARY")
    
    print(f"Shape: {df_filtered.shape} (observations × variables)")
    print(f"Index: {df_filtered.index.names}")
    print(f"\nVariable list ({len(df_filtered.columns)}):")
    for i, col in enumerate(sorted(df_filtered.columns), 1):
        print(f"  {i:2d}. {col}")
    
    # Check for missing values
    missing_counts = df_filtered.isna().sum()
    if missing_counts.sum() > 0:
        print(f"\n⚠ Missing values detected:")
        for var, count in missing_counts[missing_counts > 0].items():
            pct = count / len(df_filtered) * 100
            print(f"  {var}: {count} ({pct:.2f}%)")
    else:
        print(f"\n✓ No missing values detected")
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(df_filtered.describe().T[['count', 'mean', 'std', 'min', 'max']].to_string())
    
    # ========================================================================
    # VARIABLE CATEGORIZATION
    # ========================================================================
    
    print_subsection("5. VARIABLE CATEGORIZATION")
    
    # Categorize retained variables
    categories = {
        'Target': ['pm10'],
        'Temperature': ['temperature_2m'],
        'Humidity': ['humidity_950'],
        'Boundary Layer': ['blh'],
        'Radiation': ['solar_radiation_downwards'],
        'Precipitation': ['total_precipitation'],
        'Surface Winds': ['wind_u_10m', 'wind_v_10m'],
        'Upper-Level Winds': ['uwind_850', 'uwind_950', 'Vwind_850', 'Vwind_950']
    }
    
    for category, vars_list in categories.items():
        present_vars = [v for v in vars_list if v in df_filtered.columns]
        if present_vars:
            print(f"\n{category}:")
            for var in present_vars:
                print(f"  ✓ {var}")
    
    # ========================================================================
    # SAVE FILTERED DATASET
    # ========================================================================
    
    print_subsection("6. SAVING FILTERED DATASET")
    
    df_filtered.to_parquet(OUTPUT_FILE)
    
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print(f"✓ Saved filtered dataset to: {OUTPUT_FILE.name}")
    print(f"  Location: {OUTPUT_FILE}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Format: Parquet (compressed)")
    
    # ========================================================================
    # DROPPED VARIABLES SUMMARY
    # ========================================================================
    
    print_subsection("7. DROPPED VARIABLES SUMMARY")
    
    dropped_vars_info = {
        'temperature_550': {'vif': 58255, 'reason': 'Extreme VIF; r=0.85 with temp_2m'},
        'temperature_850': {'vif': 331115, 'reason': 'Extreme VIF; r=0.96 with temp_2m'},
        'temperature_950': {'vif': 307653, 'reason': 'Extreme VIF; r=0.93 with temp_2m'},
        'surface_pressure': {'vif': 2335, 'reason': 'High VIF; indirect effects captured'},
        'humidity_550': {'vif': 35, 'reason': 'Moderate VIF; upper-level less relevant'},
        'humidity_850': {'vif': 251, 'reason': 'High VIF; r=0.80 with humidity_950'},
        'uwind_550': {'vif': 3.8, 'reason': 'Above boundary layer; minimal surface impact'},
        'Vwind_550': {'vif': 2.7, 'reason': 'Above boundary layer; minimal surface impact'}
    }
    
    print("Removed variables with justifications:")
    for var in present_to_drop:
        info = dropped_vars_info.get(var, {})
        vif = info.get('vif', 'N/A')
        reason = info.get('reason', 'See multicollinearity analysis')
        print(f"\n  {var}:")
        print(f"    VIF: {vif}")
        print(f"    Reason: {reason}")
    
    # ========================================================================
    # VERIFICATION
    # ========================================================================
    
    print_subsection("8. VERIFICATION")
    
    # Verify index integrity
    assert df_filtered.index.equals(df.index), "Index mismatch after filtering!"
    print("✓ Index integrity verified")
    
    # Verify no dropped variables remain
    remaining_dropped = set(present_to_drop) & set(df_filtered.columns)
    if remaining_dropped:
        print(f"✗ ERROR: Dropped variables still present: {remaining_dropped}")
    else:
        print("✓ All intended variables removed")
    
    # Verify all kept variables present
    missing_kept = set(present_to_keep) - set(df_filtered.columns)
    if missing_kept:
        print(f"⚠ Warning: Some kept variables missing: {missing_kept}")
    else:
        print("✓ All intended variables retained")
    
    print(f"\n✓ Filtering complete and verified")
    
    return df_filtered


def main():
    """Main execution function"""
    
    # Redirect output to file
    logger = OutputLogger(OUTPUT_LOG)
    sys.stdout = logger
    
    try:
        # Process dataset
        df_filtered = load_and_filter_dataset()
        
        # Final summary
        print_section("FILTERING COMPLETE")
        print(f"Original dataset: {INPUT_FILE.name}")
        print(f"  Variables: {len(pd.read_parquet(INPUT_FILE).columns)}")
        print(f"\nFiltered dataset: {OUTPUT_FILE.name}")
        print(f"  Variables: {len(df_filtered.columns)}")
        print(f"\nVariables removed: {len(VARIABLES_TO_DROP)}")
        print(f"Reduction: {len(VARIABLES_TO_DROP) / (len(df_filtered.columns) + len(VARIABLES_TO_DROP)) * 100:.1f}%")
        print(f"\nLog saved to: {OUTPUT_LOG}")
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()
        
        print(f"\n✓ Filtering complete!")
        print(f"✓ Filtered dataset saved to: {OUTPUT_FILE}")
        print(f"✓ Log saved to: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
