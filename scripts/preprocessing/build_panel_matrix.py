#!/usr/bin/env python3
"""
Panel Matrix Builder
===================
This script transforms the wide-format hourly dataset into a weekly aggregated
panel matrix with proper Multi-Index structure.

Transformation steps:
1. Parse column headers to extract variable names and station IDs
2. Reshape from wide to long format
3. Pivot variables into separate columns
4. Aggregate to weekly frequency (mean)
5. Create Multi-Index (timestamp, station_id)

All disruptive operations are documented in the console output.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "dataset_documentation"

# File paths
INPUT_PARQUET = DATA_DIR / "pm10_era5_land_era5_reanalysis_blh.parquet"
OUTPUT_PARQUET = DATA_DIR / "panel_data_matrix.parquet"

# Output file
OUTPUT_LOG = RESULTS_DIR / f"panel_matrix_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


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


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---\n")


def parse_column_name(col_name):
    """
    Parse column name to extract variable and station_id.
    
    Pattern: Variable_Region_StationID
    Where Region may contain hyphens and StationID may contain underscores.
    
    Known regions: Alto-Adige, Lombardia, Trentino, Veneto
    
    Example: 'pm10_Alto-Adige_AB2' -> variable='pm10', station_id='AB2'
    Example: 'temperature_2m_Lombardia_ARPAL_001' -> variable='temperature_2m', station_id='ARPAL_001'
    Example: 'blh_Veneto_502604' -> variable='blh', station_id='502604'
    """
    if col_name == 'datetime':
        return None, None
    
    # Known regions in the dataset
    regions = ['Alto-Adige', 'Lombardia', 'Trentino', 'Veneto']
    
    # Find which region is in the column name
    found_region = None
    for region in regions:
        if region in col_name:
            found_region = region
            break
    
    if not found_region:
        # Fallback to old logic if no region found
        parts = col_name.split('_')
        station_id = parts[-1]
        variable = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
        return variable, station_id
    
    # Split by the region to get variable and station parts
    parts = col_name.split(f'_{found_region}_')
    
    if len(parts) != 2:
        # Handle edge case
        return None, None
    
    variable = parts[0]  # Everything before the region
    station_id = parts[1]  # Everything after the region
    
    return variable, station_id


def load_and_parse_data():
    """Load the parquet file and parse column structure"""
    print_subsection("Loading Source Dataset")
    
    print(f"Reading: {INPUT_PARQUET.name}")
    df = pd.read_parquet(INPUT_PARQUET)
    
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"  Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df


def transform_to_long_format(df):
    """Transform from wide to long format"""
    print_subsection("Step 1: Parsing Column Headers")
    
    # Parse all columns
    column_mapping = {}
    station_columns = [col for col in df.columns if col != 'datetime']
    
    print(f"Total columns to parse: {len(station_columns)}")
    
    # Create mapping
    parsed_data = []
    variables_found = set()
    stations_found = set()
    
    for col in station_columns:
        variable, station_id = parse_column_name(col)
        if variable and station_id:
            column_mapping[col] = {'variable': variable, 'station_id': station_id}
            variables_found.add(variable)
            stations_found.add(station_id)
            parsed_data.append({'original_col': col, 'variable': variable, 'station_id': station_id})
    
    print(f"  Variables identified: {len(variables_found)}")
    print(f"  Stations identified: {len(stations_found)}")
    
    print("\nVariable types found:")
    variable_list = sorted(variables_found)
    for i in range(0, len(variable_list), 5):
        print(f"  {', '.join(variable_list[i:i+5])}")
    
    print(f"\nStation IDs found ({len(stations_found)}):")
    station_list = sorted(stations_found)
    print(f"  {', '.join(station_list)}")
    
    print_subsection("Step 2: Reshaping to Long Format")
    print("⚠ DISRUPTIVE OPERATION: Transforming Wide → Long format")
    print(f"  Original shape: {df.shape}")
    
    # Melt the dataframe
    print("\nMelting dataframe...")
    df_long = df.melt(
        id_vars=['datetime'],
        value_vars=station_columns,
        var_name='original_column',
        value_name='value'
    )
    
    print(f"  Long format shape: {df_long.shape}")
    print(f"  Memory usage: {df_long.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Add variable and station_id columns
    print("\nExtracting variable and station_id...")
    df_long['variable'] = df_long['original_column'].map(
        lambda x: column_mapping.get(x, {}).get('variable')
    )
    df_long['station_id'] = df_long['original_column'].map(
        lambda x: column_mapping.get(x, {}).get('station_id')
    )
    
    # Drop the original column name
    df_long = df_long.drop('original_column', axis=1)
    
    print(f"✓ Long format created: {df_long.shape}")
    print(f"  Columns: {list(df_long.columns)}")
    
    return df_long, len(variables_found), len(stations_found)


def pivot_variables(df_long):
    """Pivot so variables become columns"""
    print_subsection("Step 3: Pivoting Variables to Columns")
    print("⚠ DISRUPTIVE OPERATION: Pivoting variables into separate columns")
    
    print("Pivoting on variable column...")
    print("  Using efficient pivot operation...")
    
    # Use pivot instead of pivot_table for better performance
    # First ensure we don't have duplicate (datetime, station_id, variable) combinations
    df_pivot = df_long.pivot(
        index=['datetime', 'station_id'],
        columns='variable',
        values='value'
    )
    
    # Reset to regular columns
    df_pivot = df_pivot.reset_index()
    
    # Flatten column names if they're MultiIndex
    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_pivot.columns.values]
    
    print(f"✓ Pivoted shape: {df_pivot.shape}")
    print(f"  Variables as columns: {df_pivot.shape[1] - 2}")  # minus datetime and station_id
    print(f"  Sample columns: {list(df_pivot.columns[:10])}")
    
    return df_pivot


def aggregate_to_weekly(df_pivot):
    """Aggregate to weekly frequency"""
    print_subsection("Step 4: Aggregating to Weekly Frequency")
    print("⚠ DISRUPTIVE OPERATION: Temporal aggregation (Hourly → Weekly)")
    
    print(f"Original temporal resolution: Hourly")
    print(f"Original observations: {len(df_pivot):,}")
    print(f"Original date range: {df_pivot['datetime'].min()} to {df_pivot['datetime'].max()}")
    
    # Set datetime as index temporarily for resampling
    df_pivot['datetime'] = pd.to_datetime(df_pivot['datetime'])
    
    print("\nResampling to weekly frequency (Week start = Monday)...")
    print("  Using MEAN for most variables (temperature, wind, etc.)")
    print("  Using SUM for precipitation (accumulation over week)")
    
    # Identify precipitation columns
    precip_cols = [col for col in df_pivot.columns if 'precipitation' in col.lower()]
    
    # Set up aggregation dictionary
    agg_dict = {}
    for col in df_pivot.columns:
        if col not in ['datetime', 'station_id']:
            if col in precip_cols:
                agg_dict[col] = 'sum'  # Sum for precipitation
            else:
                agg_dict[col] = 'mean'  # Mean for other variables
    
    print(f"  Precipitation columns using SUM: {len(precip_cols)}")
    print(f"  Other columns using MEAN: {len(agg_dict) - len(precip_cols)}")
    
    # Group by station_id and resample with different aggregations
    df_weekly = df_pivot.set_index('datetime').groupby('station_id').resample('W-MON').agg(agg_dict)
    
    # Reset index to get datetime and station_id as columns
    df_weekly = df_weekly.reset_index()
    
    # Rename datetime to be clearer
    df_weekly = df_weekly.rename(columns={'datetime': 'week_start'})
    
    print(f"✓ Weekly aggregation complete")
    print(f"  New shape: {df_weekly.shape}")
    print(f"  New observations: {len(df_weekly):,}")
    print(f"  Weeks covered: {df_weekly['week_start'].nunique()}")
    print(f"  Date range: {df_weekly['week_start'].min()} to {df_weekly['week_start'].max()}")
    
    # Calculate aggregation statistics
    original_hours = len(df_pivot)
    weekly_records = len(df_weekly)
    reduction = (1 - weekly_records / original_hours) * 100
    
    print(f"\nAggregation statistics:")
    print(f"  Original hourly records: {original_hours:,}")
    print(f"  Weekly aggregated records: {weekly_records:,}")
    print(f"  Data reduction: {reduction:.1f}%")
    print(f"  Average hours per weekly record: {original_hours / weekly_records:.1f}")
        # Convert precipitation from meters to millimeters for interpretability
    print(f'\n⚠ UNIT CONVERSION: Converting precipitation from meters to mm')
    for col in precip_cols:
        if col in df_weekly.columns:
            df_weekly[col] = df_weekly[col] * 1000  # meters to millimeters
            print(f'  {col}: m → mm')
    
    if precip_cols:
        print(f'\nPrecipitation statistics (after conversion to mm):')
        for col in precip_cols[:3]:  # Show first 3 as sample
            if col in df_weekly.columns:
                non_zero = df_weekly[col][df_weekly[col] > 0]
                print(f'  {col}:')
                print(f'    Mean (all weeks): {df_weekly[col].mean():.3f} mm')
                print(f'    Mean (rainy weeks): {non_zero.mean():.3f} mm' if len(non_zero) > 0 else '    No rainy weeks')
                print(f'    Max: {df_weekly[col].max():.3f} mm')
        return df_weekly


def create_panel_index(df_weekly):
    """Create Multi-Index panel structure"""
    print_subsection("Step 5: Creating Panel Multi-Index Structure")
    print("⚠ DISRUPTIVE OPERATION: Creating Multi-Index [week_start, station_id]")
    
    # Sort by time and station
    print("Sorting by timestamp and station_id...")
    df_panel = df_weekly.sort_values(['week_start', 'station_id'])
    
    # Set Multi-Index
    print("Setting Multi-Index...")
    df_panel = df_panel.set_index(['week_start', 'station_id'])
    
    print(f"✓ Panel structure created")
    print(f"  Index: {df_panel.index.names}")
    print(f"  Shape: {df_panel.shape}")
    print(f"  Weeks: {df_panel.index.get_level_values('week_start').nunique()}")
    print(f"  Stations: {df_panel.index.get_level_values('station_id').nunique()}")
    
    return df_panel


def analyze_panel_matrix(df_panel):
    """Provide comprehensive analysis of the panel matrix"""
    print_section("PANEL MATRIX ANALYSIS")
    
    print_subsection("1. Panel Structure")
    print(f"Dimensions: {df_panel.shape[0]:,} observations × {df_panel.shape[1]} variables")
    print(f"Index Type: {type(df_panel.index).__name__}")
    print(f"Index Levels: {df_panel.index.names}")
    
    # Temporal coverage
    weeks = df_panel.index.get_level_values('week_start')
    stations = df_panel.index.get_level_values('station_id')
    
    print(f"\nTemporal Coverage:")
    print(f"  Start Date: {weeks.min()}")
    print(f"  End Date: {weeks.max()}")
    print(f"  Total Weeks: {weeks.nunique()}")
    print(f"  Duration: {(weeks.max() - weeks.min()).days} days ({(weeks.max() - weeks.min()).days/7:.1f} weeks)")
    
    print(f"\nEntity Coverage:")
    print(f"  Total Stations: {stations.nunique()}")
    print(f"  Station IDs: {', '.join(sorted(stations.unique()))}")
    
    print_subsection("2. Variable Summary")
    print(f"Total Variables: {len(df_panel.columns)}")
    print(f"\nVariables in panel:")
    for col in sorted(df_panel.columns):
        non_null = df_panel[col].notna().sum()
        pct = (non_null / len(df_panel)) * 100
        print(f"  {col:40s}: {non_null:6,} non-null ({pct:5.1f}%)")
    
    print_subsection("3. Panel Balance")
    # Check if panel is balanced
    weeks_per_station = df_panel.groupby(level='station_id').size()
    is_balanced = weeks_per_station.nunique() == 1
    
    print(f"Panel Type: {'Balanced' if is_balanced else 'Unbalanced'}")
    print(f"  Min observations per station: {weeks_per_station.min()}")
    print(f"  Max observations per station: {weeks_per_station.max()}")
    print(f"  Mean observations per station: {weeks_per_station.mean():.1f}")
    
    if not is_balanced:
        print("\n⚠ Panel is unbalanced. Observations per station:")
        for station, count in weeks_per_station.items():
            print(f"    {station}: {count} weeks")
    
    print_subsection("4. Data Quality")
    # Missing values
    missing = df_panel.isna().sum()
    if missing.sum() > 0:
        print("Missing values by variable:")
        for col in df_panel.columns:
            if missing[col] > 0:
                pct = (missing[col] / len(df_panel)) * 100
                print(f"  {col:40s}: {missing[col]:6,} ({pct:5.1f}%)")
    else:
        print("✓ No missing values in panel matrix")
    
    print_subsection("5. Sample Preview")
    print("First 10 rows of panel matrix:")
    print(df_panel.head(10).to_string())
    
    print("\n\nLast 10 rows of panel matrix:")
    print(df_panel.tail(10).to_string())
    
    print_subsection("6. Descriptive Statistics")
    print("Summary statistics for all variables:")
    print(df_panel.describe().round(2).to_string())
    
    print_subsection("7. Panel Matrix Memory Usage")
    memory_mb = df_panel.memory_usage(deep=True).sum() / (1024**2)
    print(f"Total Memory Usage: {memory_mb:.2f} MB")


def save_panel_matrix(df_panel):
    """Save the panel matrix to parquet"""
    print_subsection("Saving Panel Matrix")
    
    print(f"Output file: {OUTPUT_PARQUET}")
    print(f"Format: Parquet (compressed)")
    
    df_panel.to_parquet(OUTPUT_PARQUET, engine='pyarrow', compression='snappy')
    
    file_size = OUTPUT_PARQUET.stat().st_size / (1024**2)
    print(f"✓ Panel matrix saved successfully")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Location: {OUTPUT_PARQUET}")


def main():
    """Main execution function"""
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = OutputLogger(OUTPUT_LOG)
    sys.stdout = logger
    
    try:
        print_section("PANEL MATRIX BUILDER")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"Data Directory: {DATA_DIR}")
        print(f"Results Directory: {RESULTS_DIR}")
        
        print_section("TRANSFORMATION PIPELINE")
        
        # Step 0: Load data
        df = load_and_parse_data()
        
        # Step 1-2: Parse columns and reshape to long
        df_long, n_variables, n_stations = transform_to_long_format(df)
        
        # Step 3: Pivot variables to columns
        df_pivot = pivot_variables(df_long)
        
        # Step 4: Aggregate to weekly
        df_weekly = aggregate_to_weekly(df_pivot)
        
        # Step 5: Create panel index
        df_panel = create_panel_index(df_weekly)
        
        # Analyze the panel matrix
        analyze_panel_matrix(df_panel)
        
        # Save the panel matrix
        save_panel_matrix(df_panel)
        
        # Final summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Output Log Saved: {OUTPUT_LOG}")
        print(f"\nPanel Matrix Created:")
        print(f"  - File: {OUTPUT_PARQUET}")
        print(f"  - Structure: Multi-Index (week_start, station_id)")
        print(f"  - Dimensions: {df_panel.shape[0]:,} × {df_panel.shape[1]}")
        print(f"  - Variables: {n_variables}")
        print(f"  - Stations: {n_stations}")
        print(f"  - Weeks: {df_panel.index.get_level_values('week_start').nunique()}")
        print(f"\n✓ All operations completed without errors")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close logger
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n✓ Console output saved to: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
