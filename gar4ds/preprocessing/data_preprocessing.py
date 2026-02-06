#!/usr/bin/env python3
"""
Data Preprocessing Module
=========================
Converts raw CSV files to efficient formats (GeoJSON and Parquet).

This module:
1. Converts metadata CSV to GeoJSON format with spatial geometry
2. Converts main dataset CSV to Parquet for efficient storage
3. Provides comprehensive schema and statistical analysis
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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


def convert_metadata_to_geojson(config: PipelineConfig) -> bool:
    """Convert metadata CSV to GeoJSON format"""
    print_subsection("Converting Metadata CSV to GeoJSON")
    
    metadata_csv = config.paths.source_metadata
    metadata_geojson = config.get_stations_geojson_path()
    
    if metadata_geojson.exists():
        print(f"✓ GeoJSON already exists: {metadata_geojson.name}")
        print("  Skipping conversion.")
        return False
    
    print(f"Reading metadata from: {metadata_csv.name}")
    df = pd.read_csv(metadata_csv)
    
    print(f"  Records: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    print("\nCreating Point geometries from latitude/longitude...")
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print(f"Saving GeoJSON to: {metadata_geojson.name}")
    gdf.to_file(metadata_geojson, driver='GeoJSON')
    
    print("✓ Metadata successfully converted to GeoJSON")
    return True


def convert_dataset_to_parquet(config: PipelineConfig) -> bool:
    """Convert dataset CSV to Parquet format"""
    print_subsection("Converting Dataset CSV to Parquet")
    
    dataset_csv = config.paths.source_main
    dataset_parquet = config.get_parquet_path()
    
    if dataset_parquet.exists():
        print(f"✓ Parquet already exists: {dataset_parquet.name}")
        print("  Skipping conversion.")
        return False
    
    print(f"Reading dataset from: {dataset_csv.name}")
    print(f"  File size: {dataset_csv.stat().st_size / (1024**3):.2f} GB")
    print("  This may take a few minutes...")
    
    print("\nParsing CSV (detecting data types)...")
    df = pd.read_csv(dataset_csv, parse_dates=['datetime'])
    
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    print(f"\nSaving to Parquet format: {dataset_parquet.name}")
    df.to_parquet(dataset_parquet, engine='pyarrow', compression='snappy', index=False)
    
    parquet_size = dataset_parquet.stat().st_size / (1024**3)
    compression_ratio = (1 - parquet_size / (dataset_csv.stat().st_size / (1024**3))) * 100
    
    print(f"✓ Dataset successfully converted to Parquet")
    print(f"  Parquet size: {parquet_size:.2f} GB")
    print(f"  Compression: {compression_ratio:.1f}% reduction")
    
    return True


def analyze_dataset(config: PipelineConfig):
    """Read Parquet and provide comprehensive dataset information"""
    print_section("COMPREHENSIVE DATASET ANALYSIS")
    
    dataset_parquet = config.get_parquet_path()
    print(f"Reading Parquet file: {dataset_parquet.name}\n")
    df = pd.read_parquet(dataset_parquet)
    
    # Basic Information
    print_subsection("1. Basic Dataset Information")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"File Size: {dataset_parquet.stat().st_size / (1024**2):.2f} MB")
    
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
    
    # Data Quality
    print_subsection("4. Data Quality Assessment")
    
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"  Total Missing Values: {total_missing:,} ({total_missing/total_cells*100:.2f}% of all cells)")
    
    cols_with_missing = missing_counts[missing_counts > 0]
    if len(cols_with_missing) > 0:
        print(f"  Columns with Missing Data: {len(cols_with_missing)}")
    else:
        print("  ✓ No missing values detected")
    
    print_subsection("5. Data Quality Summary")
    print("✓ Dataset successfully loaded and analyzed")
    print(f"✓ {len(df):,} observations")
    print(f"✓ Temporal span: {(df['datetime'].max() - df['datetime'].min()).days} days")


def run_preprocessing(config: Optional[PipelineConfig] = None):
    """
    Main execution function for data preprocessing.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
    """
    if config is None:
        config = load_config()
    
    # Setup logging
    results_dir = config.get_results_subdir("dataset_documentation")
    timestamp = datetime.now().strftime(config.logging.get("timestamp_format", "%Y%m%d_%H%M%S"))
    log_file = results_dir / f"dataset_info_{timestamp}.txt"
    
    logger = OutputLogger(log_file)
    sys.stdout = logger
    
    try:
        print_section("DATA PREPROCESSING PIPELINE")
        print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(config.summary())
        
        # Check existing files
        print_section("STEP 0: Checking Existing Processed Files")
        
        metadata_exists = config.get_stations_geojson_path().exists()
        parquet_exists = config.get_parquet_path().exists()
        
        print(f"Metadata GeoJSON exists: {'✓ Yes' if metadata_exists else '✗ No'}")
        print(f"Dataset Parquet exists: {'✓ Yes' if parquet_exists else '✗ No'}")
        
        # Step 1: Convert metadata
        print_section("STEP 1: Metadata Conversion")
        convert_metadata_to_geojson(config)
        
        # Step 2: Convert dataset
        print_section("STEP 2: Dataset Conversion")
        convert_dataset_to_parquet(config)
        
        # Step 3: Analyze dataset
        analyze_dataset(config)
        
        # Final summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Output Log Saved: {log_file}")
        print(f"\n✓ All operations completed without errors")
        
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
    run_preprocessing()
