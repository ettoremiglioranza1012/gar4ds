"""
Interactive Maps Module
========================

Scripts for generating interactive HTML maps for the
Po Valley - Alpine Region PM10 pollution analysis.

Available Maps:
- lisa_clusters_map: LISA Spatial Clusters Explorer
- seasonal_patterns_map: Seasonal PM10 Patterns

Usage:
    from scripts.interactive_maps import generate_all_maps
    generate_all_maps.main()
"""

from .lisa_clusters_map import generate_lisa_map
from .seasonal_patterns_map import generate_seasonal_map

__all__ = [
    'generate_lisa_map',
    'generate_seasonal_map'
]
