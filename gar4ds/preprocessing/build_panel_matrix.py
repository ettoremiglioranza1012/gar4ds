#!/usr/bin/env python3
"""
Panel Matrix Builder
===================
This module transforms the wide-format hourly dataset into an aggregated
panel matrix with proper Multi-Index structure.

Transformation steps:
1. Parse column headers to extract variable names and station IDs
2. Reshape from wide to long format
3. Pivot variables into separate columns
4. Aggregate to configured frequency (hourly/daily/weekly/monthly)
5. Create Multi-Index (timestamp, station_id)

The aggregation level is controlled by config/pipeline.yaml
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import numpy as np

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


def parse_column_name(col_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse column name to extract variable and station_id.
    
    Pattern: Variable_Region_StationID
    Where Region may contain hyphens and StationID may contain underscores.
    
    Known regions: Alto-Adige, Lombardia, Trentino, Veneto
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
        return None, None
    
    variable = parts[0]
    station_id = parts[1]
    
    return variable, station_id


def load_and_parse_data(config: PipelineConfig) -> pd.DataFrame:
    """Load the parquet file and parse column structure"""
    print_subsection("Loading Source Dataset")
    
    input_path = config.get_parquet_path()
    print(f"Reading: {input_path.name}")
    df = pd.read_parquet(input_path)
    
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"  Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df


def transform_to_long_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Transform from wide to long format"""
    print_subsection("Step 1: Parsing Column Headers")
    
    column_mapping = {}
    station_columns = [col for col in df.columns if col != 'datetime']
    
    print(f"Total columns to parse: {len(station_columns)}")
    
    variables_found = set()
    stations_found = set()
    
    for col in station_columns:
        variable, station_id = parse_column_name(col)
        if variable and station_id:
            column_mapping[col] = {'variable': variable, 'station_id': station_id}
            variables_found.add(variable)
            stations_found.add(station_id)
    
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
    
    print("\nMelting dataframe...")
    df_long = df.melt(
        id_vars=['datetime'],
        value_vars=station_columns,
        var_name='original_column',
        value_name='value'
    )
    
    print(f"  Long format shape: {df_long.shape}")
    print(f"  Memory usage: {df_long.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    print("\nExtracting variable and station_id...")
    df_long['variable'] = df_long['original_column'].map(
        lambda x: column_mapping.get(x, {}).get('variable')
    )
    df_long['station_id'] = df_long['original_column'].map(
        lambda x: column_mapping.get(x, {}).get('station_id')
    )
    
    df_long = df_long.drop('original_column', axis=1)
    
    print(f"✓ Long format created: {df_long.shape}")
    print(f"  Columns: {list(df_long.columns)}")
    
    return df_long, len(variables_found), len(stations_found)


def pivot_variables(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot so variables become columns"""
    print_subsection("Step 3: Pivoting Variables to Columns")
    print("⚠ DISRUPTIVE OPERATION: Pivoting variables into separate columns")
    
    print("Pivoting on variable column...")
    
    df_pivot = df_long.pivot(
        index=['datetime', 'station_id'],
        columns='variable',
        values='value'
    )
    
    df_pivot = df_pivot.reset_index()
    
    if isinstance(df_pivot.columns, pd.MultiIndex):
        df_pivot.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_pivot.columns.values]
    
    print(f"✓ Pivoted shape: {df_pivot.shape}")
    print(f"  Variables as columns: {df_pivot.shape[1] - 2}")
    
    return df_pivot


def aggregate_temporal(df_pivot: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Aggregate to configured frequency"""
    aggregation = config.temporal.aggregation
    resample_rule = config.temporal.resample_rule
    time_label = config.temporal.time_label
    
    print_subsection(f"Step 4: Temporal Aggregation ({aggregation.upper()})")
    
    # If hourly, no aggregation needed
    if aggregation == "hourly":
        print("✓ Keeping original hourly resolution (no aggregation)")
        df_pivot = df_pivot.rename(columns={'datetime': time_label})
        return df_pivot
    
    print(f"⚠ DISRUPTIVE OPERATION: Temporal aggregation (Hourly → {aggregation.capitalize()})")
    print(f"Original temporal resolution: Hourly")
    print(f"Original observations: {len(df_pivot):,}")
    print(f"Original date range: {df_pivot['datetime'].min()} to {df_pivot['datetime'].max()}")
    
    df_pivot['datetime'] = pd.to_datetime(df_pivot['datetime'])
    
    print(f"\nResampling to {aggregation} frequency (rule: {resample_rule})...")
    
    # Identify precipitation columns
    precip_cols = config.variables.precipitation_vars
    precip_in_data = [col for col in df_pivot.columns if any(p in col.lower() for p in precip_cols)]
    
    # Set up aggregation dictionary
    agg_dict = {}
    for col in df_pivot.columns:
        if col not in ['datetime', 'station_id']:
            method = config.get_aggregation_method(col)
            agg_dict[col] = method
    
    print(f"  Precipitation columns using SUM: {len(precip_in_data)}")
    print(f"  Other columns using MEAN: {len(agg_dict) - len(precip_in_data)}")
    
    # Group by station_id and resample
    df_agg = df_pivot.set_index('datetime').groupby('station_id').resample(resample_rule).agg(agg_dict)
    df_agg = df_agg.reset_index()
    df_agg = df_agg.rename(columns={'datetime': time_label})
    
    print(f"✓ {aggregation.capitalize()} aggregation complete")
    print(f"  New shape: {df_agg.shape}")
    print(f"  New observations: {len(df_agg):,}")
    print(f"  Periods covered: {df_agg[time_label].nunique()}")
    print(f"  Date range: {df_agg[time_label].min()} to {df_agg[time_label].max()}")
    
    # Calculate aggregation statistics
    original_hours = len(df_pivot)
    agg_records = len(df_agg)
    reduction = (1 - agg_records / original_hours) * 100
    
    print(f"\nAggregation statistics:")
    print(f"  Original hourly records: {original_hours:,}")
    print(f"  {aggregation.capitalize()} records: {agg_records:,}")
    print(f"  Data reduction: {reduction:.1f}%")
    
    # Convert precipitation from meters to millimeters
    precip_factor = config.variables.precipitation_vars
    if precip_in_data:
        print(f'\n⚠ UNIT CONVERSION: Converting precipitation from meters to mm')
        for col in precip_in_data:
            if col in df_agg.columns:
                df_agg[col] = df_agg[col] * 1000
                print(f'  {col}: m → mm')
    
    return df_agg


def create_panel_index(df_agg: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create Multi-Index panel structure"""
    time_label = config.temporal.time_label
    
    print_subsection("Step 5: Creating Panel Multi-Index Structure")
    print(f"⚠ DISRUPTIVE OPERATION: Creating Multi-Index [{time_label}, station_id]")
    
    print("Sorting by timestamp and station_id...")
    df_panel = df_agg.sort_values([time_label, 'station_id'])
    
    print("Setting Multi-Index...")
    df_panel = df_panel.set_index([time_label, 'station_id'])
    
    print(f"✓ Panel structure created")
    print(f"  Index: {df_panel.index.names}")
    print(f"  Shape: {df_panel.shape}")
    print(f"  Time periods: {df_panel.index.get_level_values(0).nunique()}")
    print(f"  Stations: {df_panel.index.get_level_values(1).nunique()}")
    
    return df_panel


def analyze_panel_matrix(df_panel: pd.DataFrame, config: PipelineConfig):
    """Provide comprehensive analysis of the panel matrix"""
    time_label = config.temporal.time_label
    
    print_section("PANEL MATRIX ANALYSIS")
    
    print_subsection("1. Panel Structure")
    print(f"Dimensions: {df_panel.shape[0]:,} observations × {df_panel.shape[1]} variables")
    print(f"Index Type: {type(df_panel.index).__name__}")
    print(f"Index Levels: {df_panel.index.names}")
    print(f"Temporal Aggregation: {config.temporal.aggregation}")
    
    times = df_panel.index.get_level_values(0)
    stations = df_panel.index.get_level_values(1)
    
    print(f"\nTemporal Coverage:")
    print(f"  Start: {times.min()}")
    print(f"  End: {times.max()}")
    print(f"  Total periods: {times.nunique()}")
    
    print(f"\nEntity Coverage:")
    print(f"  Total Stations: {stations.nunique()}")
    
    print_subsection("2. Variable Summary")
    print(f"Total Variables: {len(df_panel.columns)}")
    
    print_subsection("3. Panel Balance")
    periods_per_station = df_panel.groupby(level='station_id').size()
    is_balanced = periods_per_station.nunique() == 1
    
    print(f"Panel Type: {'Balanced' if is_balanced else 'Unbalanced'}")
    print(f"  Min observations per station: {periods_per_station.min()}")
    print(f"  Max observations per station: {periods_per_station.max()}")
    
    print_subsection("4. Data Quality")
    missing = df_panel.isna().sum()
    if missing.sum() > 0:
        print("Missing values by variable:")
        for col in df_panel.columns:
            if missing[col] > 0:
                pct = (missing[col] / len(df_panel)) * 100
                print(f"  {col:40s}: {missing[col]:6,} ({pct:5.1f}%)")
    else:
        print("✓ No missing values in panel matrix")
    
    print_subsection("5. Descriptive Statistics")
    print("Summary statistics for all variables:")
    print(df_panel.describe().round(2).to_string())


def save_panel_matrix(df_panel: pd.DataFrame, config: PipelineConfig):
    """Save the panel matrix to parquet"""
    print_subsection("Saving Panel Matrix")
    
    output_path = config.get_panel_matrix_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output file: {output_path}")
    print(f"Format: Parquet (compressed)")
    
    df_panel.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    file_size = output_path.stat().st_size / (1024**2)
    print(f"✓ Panel matrix saved successfully")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Location: {output_path}")


def run_panel_builder(config: Optional[PipelineConfig] = None) -> pd.DataFrame:
    """
    Main execution function for panel matrix builder.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
        
    Returns:
        DataFrame with panel matrix
    """
    if config is None:
        config = load_config()
    
    # Setup logging
    results_dir = config.get_results_subdir("dataset_documentation")
    timestamp = datetime.now().strftime(config.logging.get("timestamp_format", "%Y%m%d_%H%M%S"))
    log_file = results_dir / f"panel_matrix_info_{timestamp}.txt"
    
    logger = OutputLogger(log_file)
    sys.stdout = logger
    
    try:
        print_section("PANEL MATRIX BUILDER")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {config.temporal.aggregation} aggregation")
        print(config.summary())
        
        print_section("TRANSFORMATION PIPELINE")
        
        # Step 0: Load data
        df = load_and_parse_data(config)
        
        # Step 1-2: Parse columns and reshape to long
        df_long, n_variables, n_stations = transform_to_long_format(df)
        
        # Step 3: Pivot variables to columns
        df_pivot = pivot_variables(df_long)
        
        # Step 4: Aggregate to configured frequency
        df_agg = aggregate_temporal(df_pivot, config)
        
        # Step 5: Create panel index
        df_panel = create_panel_index(df_agg, config)
        
        # Analyze the panel matrix
        analyze_panel_matrix(df_panel, config)
        
        # Save the panel matrix
        save_panel_matrix(df_panel, config)
        
        # Final summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Output Log Saved: {log_file}")
        print(f"\nPanel Matrix Created:")
        print(f"  - File: {config.get_panel_matrix_path()}")
        print(f"  - Aggregation: {config.temporal.aggregation}")
        print(f"  - Structure: Multi-Index ({config.temporal.time_label}, station_id)")
        print(f"  - Dimensions: {df_panel.shape[0]:,} × {df_panel.shape[1]}")
        print(f"\n✓ All operations completed without errors")
        
        return df_panel
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n✓ Console output saved to: {log_file}")


# Allow running as script
if __name__ == "__main__":
    run_panel_builder()
