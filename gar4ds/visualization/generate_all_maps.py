#!/usr/bin/env python3
"""
Generate All Maps
=================
Utility to generate all interactive maps at once.
"""

from typing import Optional, List

from ..config import PipelineConfig, load_config
from .lisa_clusters_map import generate_lisa_map
from .pm10_meteorological_map import generate_pm10_map
from .seasonal_patterns_map import generate_seasonal_map


def generate_all_maps(config: Optional[PipelineConfig] = None) -> List[str]:
    """
    Generate all interactive maps.
    
    Returns:
        List of paths to generated HTML files
    """
    if config is None:
        config = load_config()
    
    print("=" * 60)
    print("GENERATING ALL INTERACTIVE MAPS")
    print("=" * 60)
    
    generated = []
    
    # PM10 map (always available)
    try:
        path = generate_pm10_map(config)
        generated.append(path)
    except Exception as e:
        print(f"⚠ Failed to generate PM10 map: {e}")
    
    # Seasonal map
    try:
        path = generate_seasonal_map(config)
        generated.append(path)
    except Exception as e:
        print(f"⚠ Failed to generate seasonal map: {e}")
    
    # LISA map (requires spatial analysis results)
    try:
        path = generate_lisa_map(config)
        generated.append(path)
    except FileNotFoundError:
        print("⚠ LISA map skipped (run spatial analysis first)")
    except Exception as e:
        print(f"⚠ Failed to generate LISA map: {e}")
    
    print("\n" + "=" * 60)
    print(f"Generated {len(generated)} maps")
    print("=" * 60)
    
    return generated


if __name__ == "__main__":
    generate_all_maps()
