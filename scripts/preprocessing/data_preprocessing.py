#!/usr/bin/env python3
"""
Data Preprocessing Script
=========================
This script performs the following operations on the raw datasets:
1. Checks for existing processed files (GeoJSON and Parquet)
2. Converts metadata CSV to GeoJSON format
3. Converts main dataset CSV to Parquet format for efficient storage
4. Provides comprehensive schema and statistical information

All disruptive operations are documented in the console output.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "dataset_documentation"

# File paths
METADATA_CSV = DATA_DIR / "pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv"
METADATA_GEOJSON = DATA_DIR / "pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson"
DATASET_CSV = DATA_DIR / "pm10_era5_land_era5_reanalysis_blh.csv"
DATASET_PARQUET = DATA_DIR / "pm10_era5_land_era5_reanalysis_blh.parquet"

# Output file
OUTPUT_LOG = RESULTS_DIR / f"dataset_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


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


def convert_metadata_to_geojson():
    """Convert metadata CSV to GeoJSON format"""
    print_subsection("Converting Metadata CSV to GeoJSON")
    
    if METADATA_GEOJSON.exists():
        print(f"✓ GeoJSON already exists: {METADATA_GEOJSON.name}")
        print("  Skipping conversion.")
        return False
    
    print(f"Reading metadata from: {METADATA_CSV.name}")
    df = pd.read_csv(METADATA_CSV)
    
    print(f"  Records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Create geometry from coordinates
    print("\nCreating Point geometries from latitude/longitude...")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Save to GeoJSON
    print(f"Saving GeoJSON to: {METADATA_GEOJSON.name}")
    gdf.to_file(METADATA_GEOJSON, driver='GeoJSON')
    
    print("✓ Metadata successfully converted to GeoJSON")
    return True


def convert_dataset_to_parquet():
    """Convert dataset CSV to Parquet format"""
    print_subsection("Converting Dataset CSV to Parquet")
    
    if DATASET_PARQUET.exists():
        print(f"✓ Parquet already exists: {DATASET_PARQUET.name}")
        print("  Skipping conversion.")
        return False
    
    print(f"Reading dataset from: {DATASET_CSV.name}")
    print(f"  File size: {DATASET_CSV.stat().st_size / (1024**3):.2f} GB")
    print("  This may take a few minutes...")
    
    # Read CSV with datetime parsing
    print("\nParsing CSV (detecting data types)...")
    df = pd.read_csv(DATASET_CSV, parse_dates=['datetime'])
    
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Save to Parquet
    print(f"\nSaving to Parquet format: {DATASET_PARQUET.name}")
    df.to_parquet(DATASET_PARQUET, engine='pyarrow', compression='snappy', index=False)
    
    parquet_size = DATASET_PARQUET.stat().st_size / (1024**3)
    compression_ratio = (1 - parquet_size / (DATASET_CSV.stat().st_size / (1024**3))) * 100
    
    print(f"✓ Dataset successfully converted to Parquet")
    print(f"  Parquet size: {parquet_size:.2f} GB")
    print(f"  Compression: {compression_ratio:.1f}% reduction")
    
    return True


def analyze_dataset():
    """Read Parquet and provide comprehensive dataset information"""
    print_section("COMPREHENSIVE DATASET ANALYSIS")
    
    print(f"Reading Parquet file: {DATASET_PARQUET.name}\n")
    df = pd.read_parquet(DATASET_PARQUET)
    
    # Basic Information
    print_subsection("1. Basic Dataset Information")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"File Size: {DATASET_PARQUET.stat().st_size / (1024**2):.2f} MB")
    
    # Datetime Range
    print_subsection("2. Temporal Coverage")
    print(f"Start Date: {df['datetime'].min()}")
    print(f"End Date: {df['datetime'].max()}")
    print(f"Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    print(f"Total Observations: {len(df):,}")
    
    # Column Structure
    print_subsection("3. Column Structure")
    print(f"Total Columns: {len(df.columns)}")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    
    # Station columns (assuming columns starting with certain patterns)
    station_cols = [col for col in df.columns if col != 'datetime']
    print(f"\nStation/Variable Columns: {len(station_cols)}")
    
    # Parse column names to understand structure
    print("\nColumn Name Patterns (first 20 examples):")
    for col in station_cols[:20]:
        print(f"  - {col}")
    
    # Data Quality Analysis
    print_subsection("4. Data Quality Assessment")
    
    print("Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    # Summary statistics
    total_missing = missing_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"  Total Missing Values: {total_missing:,} ({total_missing/total_cells*100:.2f}% of all cells)")
    
    # Columns with missing data
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        print(f"  Columns with Missing Data: {len(cols_with_missing)}")
        print(f"\nTop 10 columns by missing value percentage:")
        top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
        for col, pct in top_missing.items():
            print(f"    {col}: {pct:.2f}% ({missing_counts[col]:,} values)")
    else:
        print("  ✓ No missing values detected")
    
    # Statistical Summary for numeric columns
    print_subsection("5. Statistical Summary (Sample Columns)")
    
    # Get numeric columns (excluding datetime)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_cols) > 0:
        print(f"Numeric Columns: {len(numeric_cols)}")
        print(f"\nSample Statistics (first 10 columns):")
        print(df[numeric_cols[:10]].describe().round(2))
    
    # Value distributions
    print_subsection("6. Value Distribution Analysis")
    
    # Check for zero/negative values in PM10 columns
    pm10_cols = [col for col in station_cols if col.startswith('pm10_')]
    if pm10_cols:
        print(f"PM10 Columns: {len(pm10_cols)}")
        
        # Sample a few PM10 columns for analysis
        sample_pm10 = pm10_cols[:5]
        for col in sample_pm10:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n  {col}:")
                print(f"    Count: {len(values):,}")
                print(f"    Mean: {values.mean():.2f}")
                print(f"    Std: {values.std():.2f}")
                print(f"    Min: {values.min():.2f}")
                print(f"    Max: {values.max():.2f}")
                print(f"    Negative values: {(values < 0).sum()}")
                print(f"    Zero values: {(values == 0).sum()}")
    
    # ERA5 variable detection
    print_subsection("7. Variable Categories")
    
    variable_types = {}
    for col in station_cols:
        if col.startswith('pm10_'):
            var_type = 'PM10 Measurements'
        elif 'blh' in col.lower():
            var_type = 'Boundary Layer Height'
        elif any(x in col.lower() for x in ['temp', 't2m', 'temperature']):
            var_type = 'Temperature'
        elif any(x in col.lower() for x in ['precip', 'tp', 'rain']):
            var_type = 'Precipitation'
        elif any(x in col.lower() for x in ['wind', 'u10', 'v10']):
            var_type = 'Wind'
        elif any(x in col.lower() for x in ['pressure', 'sp']):
            var_type = 'Pressure'
        else:
            var_type = 'Other'
        
        variable_types[var_type] = variable_types.get(var_type, 0) + 1
    
    print("Variable Type Distribution:")
    for var_type, count in sorted(variable_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {var_type}: {count} columns")
    
    # Schema Documentation
    print_subsection("8. Full Schema")
    print("\nColumn Name | Data Type | Non-Null Count | Null %")
    print("-" * 80)
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_pct = (len(df) - non_null) / len(df) * 100
        print(f"{col:50s} | {str(dtype):15s} | {non_null:14,d} | {null_pct:6.2f}%")
    
    # Additional Insights
    print_subsection("9. Additional Insights")
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    print(f"Duplicate Rows: {dup_count}")
    
    # Check datetime continuity
    if 'datetime' in df.columns:
        df_sorted = df.sort_values('datetime')
        time_diffs = df_sorted['datetime'].diff()
        mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
        print(f"Most Common Time Interval: {mode_diff}")
        
        gaps = time_diffs[time_diffs > mode_diff]
        if len(gaps) > 0:
            print(f"⚠ Time Gaps Detected: {len(gaps)} instances")
            print(f"  Largest Gap: {gaps.max()}")
        else:
            print("✓ No time gaps detected")
    
    print_subsection("10. Data Quality Summary")
    print("✓ Dataset successfully loaded and analyzed")
    print(f"✓ Schema documentation complete")
    print(f"✓ {len(df):,} observations across {len(station_cols)} variables")
    print(f"✓ Temporal span: {(df['datetime'].max() - df['datetime'].min()).days} days")


def main():
    """Main execution function"""
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = OutputLogger(OUTPUT_LOG)
    sys.stdout = logger
    
    try:
        print_section("DATA PREPROCESSING PIPELINE")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base Directory: {BASE_DIR}")
        print(f"Data Directory: {DATA_DIR}")
        print(f"Results Directory: {RESULTS_DIR}")
        
        # Step 0: Check existing files
        print_section("STEP 0: Checking Existing Processed Files")
        
        metadata_exists = METADATA_GEOJSON.exists()
        parquet_exists = DATASET_PARQUET.exists()
        
        print(f"Metadata GeoJSON exists: {'✓ Yes' if metadata_exists else '✗ No'}")
        print(f"Dataset Parquet exists: {'✓ Yes' if parquet_exists else '✗ No'}")
        
        if metadata_exists and parquet_exists:
            print("\n⚠ All processed files already exist. Skipping conversion steps.")
            print("  To regenerate, delete the processed files and run again.")
        
        # Step 1: Convert metadata
        print_section("STEP 1: Metadata Conversion")
        converted_metadata = convert_metadata_to_geojson()
        
        if converted_metadata:
            print("\n⚠ DISRUPTIVE OPERATION: New GeoJSON file created")
            print(f"  File: {METADATA_GEOJSON}")
            print(f"  Action: Converted CSV to GeoJSON with spatial geometry")
        
        # Step 2: Convert dataset
        print_section("STEP 2: Dataset Conversion")
        converted_dataset = convert_dataset_to_parquet()
        
        if converted_dataset:
            print("\n⚠ DISRUPTIVE OPERATION: New Parquet file created")
            print(f"  File: {DATASET_PARQUET}")
            print(f"  Action: Converted CSV to compressed Parquet format")
            print(f"  Note: Original CSV preserved, no data loss")
        
        # Step 3: Analyze dataset
        analyze_dataset()
        
        # Final summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Output Log Saved: {OUTPUT_LOG}")
        print(f"\nProcessed Files:")
        print(f"  - Metadata GeoJSON: {METADATA_GEOJSON}")
        print(f"  - Dataset Parquet: {DATASET_PARQUET}")
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
