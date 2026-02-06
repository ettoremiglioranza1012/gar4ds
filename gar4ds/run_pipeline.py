#!/usr/bin/env python3
"""
GAR4DS Pipeline Runner
======================
Main entry point for running the complete data processing and analysis pipeline.

Usage:
    # Run full pipeline with default config (weekly aggregation)
    python -m gar4ds.run_pipeline
    
    # Run full pipeline with custom config
    python -m gar4ds.run_pipeline --config path/to/config.yaml
    
    # Run specific stages
    python -m gar4ds.run_pipeline --stages preprocessing analysis
    
    # Show current config
    python -m gar4ds.run_pipeline --show-config

Available stages:
    1. preprocessing: Data conversion and panel matrix creation
    2. filtering: Multicollinearity analysis and filtering
    3. analysis: EDA, spatial analysis, model tests
    4. modeling: Spatial Durbin Model estimation
    5. visualization: Generate interactive maps
"""

import argparse
import sys
from datetime import datetime
from typing import List, Optional

from .config import PipelineConfig, load_config


def run_preprocessing(config: PipelineConfig):
    """Run preprocessing stage"""
    print("\n" + "=" * 60)
    print("STAGE 1: PREPROCESSING")
    print("=" * 60)
    
    from .preprocessing import run_preprocessing, run_panel_builder, run_add_elevation
    
    print("\n[1.1] Data Preprocessing (CSV → Parquet/GeoJSON)...")
    run_preprocessing(config)
    
    print("\n[1.2] Building Panel Matrix...")
    run_panel_builder(config)
    
    print("\n[1.3] Adding Elevation Data...")
    run_add_elevation(config)


def run_filtering(config: PipelineConfig):
    """Run filtering stage"""
    print("\n" + "=" * 60)
    print("STAGE 2: MULTICOLLINEARITY FILTERING")
    print("=" * 60)
    
    from .preprocessing import run_multicollinearity_analysis, run_filter_multicollinearity
    
    print("\n[2.1] Multicollinearity Analysis...")
    run_multicollinearity_analysis(config)
    
    print("\n[2.2] Filtering Collinear Variables...")
    run_filter_multicollinearity(config)


def run_analysis(config: PipelineConfig):
    """Run analysis stage"""
    print("\n" + "=" * 60)
    print("STAGE 3: DATA ANALYSIS")
    print("=" * 60)
    
    from .analysis import run_eda, run_spatial_analysis, run_specification_tests
    
    print("\n[3.1] Exploratory Data Analysis...")
    run_eda(config)
    
    print("\n[3.2] Spatial Analysis...")
    run_spatial_analysis(config)
    
    print("\n[3.3] Model Specification Tests...")
    run_specification_tests(config)


def run_modeling(config: PipelineConfig):
    """Run modeling stage"""
    print("\n" + "=" * 60)
    print("STAGE 4: SPATIAL DURBIN MODEL")
    print("=" * 60)
    
    from .analysis import run_sdm
    
    print("\n[4.1] Estimating Spatial Durbin Model...")
    run_sdm(config)


def run_visualization(config: PipelineConfig):
    """Run visualization stage"""
    print("\n" + "=" * 60)
    print("STAGE 5: VISUALIZATION")
    print("=" * 60)
    
    from .visualization import generate_all_maps
    
    print("\n[5.1] Generating Interactive Maps...")
    generate_all_maps(config)


STAGES = {
    'preprocessing': run_preprocessing,
    'filtering': run_filtering,
    'analysis': run_analysis,
    'modeling': run_modeling,
    'visualization': run_visualization,
}

STAGE_ORDER = ['preprocessing', 'filtering', 'analysis', 'modeling', 'visualization']


def run_pipeline(
    config_path: Optional[str] = None,
    stages: Optional[List[str]] = None
):
    """
    Run the GAR4DS pipeline.
    
    Args:
        config_path: Path to config file. If None, uses default.
        stages: List of stages to run. If None, runs all.
    """
    # Load config
    config = load_config(config_path)
    
    # Print header
    print("=" * 70)
    print("GAR4DS - PM10 SPATIAL ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temporal aggregation: {config.temporal.aggregation}")
    print("-" * 70)
    
    # Determine stages to run
    if stages is None:
        stages = STAGE_ORDER
    else:
        # Validate stages
        for stage in stages:
            if stage not in STAGES:
                print(f"Error: Unknown stage '{stage}'")
                print(f"Available stages: {', '.join(STAGE_ORDER)}")
                sys.exit(1)
    
    print(f"Stages to run: {', '.join(stages)}")
    
    # Run stages
    for stage in stages:
        if stage in STAGES:
            try:
                STAGES[stage](config)
            except Exception as e:
                print(f"\n✗ ERROR in stage '{stage}': {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Execution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput locations:")
    print(f"  Data: {config.paths.data}")
    print(f"  Results: {config.paths.results}")
    print(f"  Assets: {config.paths.assets}")
    print(f"  Weights: {config.paths.weights}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GAR4DS Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m gar4ds.run_pipeline
  
  # Run with custom config
  python -m gar4ds.run_pipeline --config config/daily.yaml
  
  # Run specific stages
  python -m gar4ds.run_pipeline --stages preprocessing filtering
  
  # Show current config
  python -m gar4ds.run_pipeline --show-config
"""
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--stages', '-s',
        nargs='+',
        choices=STAGE_ORDER,
        help='Stages to run (default: all)'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List available stages and exit'
    )
    
    args = parser.parse_args()
    
    # Handle info flags
    if args.list_stages:
        print("Available pipeline stages:")
        for i, stage in enumerate(STAGE_ORDER, 1):
            print(f"  {i}. {stage}")
        sys.exit(0)
    
    if args.show_config:
        config = load_config(args.config)
        print(config.summary())
        sys.exit(0)
    
    # Run pipeline
    run_pipeline(
        config_path=args.config,
        stages=args.stages
    )


if __name__ == "__main__":
    main()
