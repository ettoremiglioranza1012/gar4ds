"""
INTERACTIVE MAPS GENERATOR
===========================

Master script to generate all interactive HTML maps for the
Po Valley - Alpine Region PM10 pollution analysis project.

MAPS GENERATED:
1. LISA Spatial Clusters Explorer - Spatial autocorrelation visualization
2. PM10 & Meteorological Explorer - Pollution levels with wind patterns
3. Seasonal PM10 Patterns - Temporal/seasonal analysis

USAGE:
    uv run scripts/interactive_maps/generate_all_maps.py

OUTPUT:
    assets/maps/lisa_clusters_explorer.html
    assets/maps/pm10_meteorological_explorer.html
    assets/maps/seasonal_pm10_patterns.html
"""

from pathlib import Path
import sys

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.interactive_maps.lisa_clusters_map import generate_lisa_map
from scripts.interactive_maps.seasonal_patterns_map import generate_seasonal_map


def main():
    """Generate all interactive maps"""
    
    print("\n" + "=" * 80)
    print("  INTERACTIVE MAPS GENERATION - PO VALLEY PM10 ANALYSIS")
    print("=" * 80)
    
    output_paths = []
    
    # 1. LISA Clusters Map
    print("\n" + "-" * 80)
    print("  [1/3] LISA Spatial Clusters Map")
    print("-" * 80)
    try:
        path = generate_lisa_map()
        output_paths.append(('LISA Clusters Explorer', path))
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # 3. Seasonal Patterns Map
    print("\n" + "-" * 80)
    print("  [3/3] Seasonal PM10 Patterns Map")
    print("-" * 80)
    try:
        path = generate_seasonal_map()
        output_paths.append(('Seasonal PM10 Patterns', path))
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("  GENERATION SUMMARY")
    print("=" * 80)
    print(f"\n  Successfully generated {len(output_paths)} maps:\n")
    
    for name, path in output_paths:
        print(f"    🗺️  {name}")
        print(f"       {path}\n")
    
    print("\n  Open any HTML file in a web browser to explore the interactive maps.")
    print("=" * 80 + "\n")
    
    return output_paths


if __name__ == '__main__':
    main()
