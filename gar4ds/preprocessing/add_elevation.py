#!/usr/bin/env python3
"""
Add Elevation Data Module
=========================
Adds elevation data to station metadata using Open-Elevation API.

This module:
1. Loads the station metadata GeoJSON file
2. Batch requests elevation data from Open-Elevation API
3. Classifies terrain as mountain/hills/plain based on elevation
4. Classifies area as urban/suburban/rural
5. Saves the enriched GeoJSON file
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from ..config import PipelineConfig, load_config


def load_geojson(filepath: Path) -> Dict:
    """Load GeoJSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_geojson(data: Dict, filepath: Path) -> None:
    """Save GeoJSON file with proper formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved enriched data to: {filepath}")


def extract_coordinates(geojson_data: Dict) -> List[Dict[str, float]]:
    """Extract coordinates from GeoJSON for batch API request."""
    locations = []
    for feature in geojson_data['features']:
        coords = feature['geometry']['coordinates']
        locations.append({
            "latitude": coords[1],
            "longitude": coords[0]
        })
    return locations


def fetch_elevations(locations: List[Dict[str, float]], api_url: str, batch_size: int = 100) -> List[Dict]:
    """
    Fetch elevation data from Open-Elevation API.
    
    Args:
        locations: List of {latitude, longitude} dictionaries
        api_url: API endpoint URL
        batch_size: Number of locations per request
    
    Returns:
        List of elevation results
    """
    all_elevations = []
    headers = {'Content-Type': 'application/json'}
    
    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        print(f"Fetching elevations for locations {i+1} to {i+len(batch)}...")
        
        try:
            response = requests.post(
                api_url,
                json={'locations': batch},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            results = response.json()['results']
            all_elevations.extend(results)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error retrieving elevation data for batch {i//batch_size + 1}: {e}")
            all_elevations.extend([None] * len(batch))
    
    return all_elevations


def classify_terrain(elevation: float, thresholds: Dict) -> str:
    """Classify terrain type based on elevation."""
    plain_max = thresholds.get('plain_max', 300)
    hills_max = thresholds.get('hills_max', 600)
    
    if elevation < plain_max:
        return 'plain'
    elif elevation < hills_max:
        return 'hills'
    else:
        return 'mountain'


def classify_area_type(station_name: str, elevation: float) -> str:
    """
    Classify area as urban, suburban, or rural.
    Based on station name patterns and elevation.
    """
    urban_keywords = ['via', 'p.zza', 'piazza', 'centro', 'city', 'borgo', 'urban', 'città']
    name_lower = station_name.lower()
    
    is_likely_urban = any(keyword in name_lower for keyword in urban_keywords)
    
    if is_likely_urban and elevation < 400:
        return 'urban'
    elif is_likely_urban or elevation < 400:
        return 'suburban'
    else:
        return 'rural'


def enrich_geojson_with_elevation(geojson_data: Dict, elevations: List[Dict], thresholds: Dict) -> Dict:
    """Add elevation data and classifications to GeoJSON features."""
    enriched_data = geojson_data.copy()
    success_count = 0
    failure_count = 0
    
    for i, feature in enumerate(enriched_data['features']):
        elevation_result = elevations[i]
        
        if elevation_result and 'elevation' in elevation_result:
            elevation = elevation_result['elevation']
            station_name = feature['properties'].get('station_name', '')
            
            # Add Z coordinate to geometry (making it 3D)
            feature['geometry']['coordinates'].append(elevation)
            
            # Add elevation and classifications to properties
            feature['properties']['elevation'] = elevation
            feature['properties']['terrain_type'] = classify_terrain(elevation, thresholds)
            feature['properties']['area_type'] = classify_area_type(station_name, elevation)
            
            success_count += 1
        else:
            feature['properties']['elevation'] = None
            feature['properties']['terrain_type'] = 'unknown'
            feature['properties']['area_type'] = 'unknown'
            failure_count += 1
            print(f"  ✗ Failed to get elevation for station: {feature['properties'].get('station_name', 'unknown')}")
    
    print(f"\n✓ Successfully added elevation data to {success_count} stations")
    if failure_count > 0:
        print(f"✗ Failed to retrieve elevation for {failure_count} stations")
    
    return enriched_data


def print_elevation_summary(geojson_data: Dict) -> None:
    """Print summary statistics of elevation data."""
    elevations = [
        f['properties']['elevation'] 
        for f in geojson_data['features'] 
        if f['properties']['elevation'] is not None
    ]
    
    if not elevations:
        print("No elevation data available")
        return
    
    print("\n" + "=" * 60)
    print("ELEVATION SUMMARY")
    print("=" * 60)
    print(f"  Stations with elevation data: {len(elevations)}")
    print(f"  Min elevation: {min(elevations):.1f} m")
    print(f"  Max elevation: {max(elevations):.1f} m")
    print(f"  Mean elevation: {sum(elevations)/len(elevations):.1f} m")
    
    # Terrain type distribution
    terrain_counts = {}
    area_counts = {}
    for f in geojson_data['features']:
        terrain = f['properties'].get('terrain_type', 'unknown')
        area = f['properties'].get('area_type', 'unknown')
        terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
        area_counts[area] = area_counts.get(area, 0) + 1
    
    print("\nTerrain type distribution:")
    for terrain, count in sorted(terrain_counts.items()):
        print(f"  {terrain}: {count}")
    
    print("\nArea type distribution:")
    for area, count in sorted(area_counts.items()):
        print(f"  {area}: {count}")


def run_add_elevation(config: Optional[PipelineConfig] = None):
    """
    Main execution function to add elevation data to stations.
    
    Args:
        config: PipelineConfig instance. If None, loads from default.
    """
    if config is None:
        config = load_config()
    
    print("=" * 60)
    print("ADD ELEVATION DATA TO STATION METADATA")
    print("=" * 60)
    print(f"\nExecution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    input_file = config.get_stations_geojson_path(with_elevation=False)
    output_file = config.get_stations_geojson_path(with_elevation=True)
    
    print(f"\nInput file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Check if output already exists
    if output_file.exists():
        print(f"\n⚠ Output file already exists: {output_file}")
        print("  Skipping elevation fetch to avoid overwriting.")
        geojson_data = load_geojson(output_file)
        print_elevation_summary(geojson_data)
        return
    
    # Check input file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load GeoJSON
    print("\nLoading station metadata...")
    geojson_data = load_geojson(input_file)
    print(f"  Loaded {len(geojson_data['features'])} stations")
    
    # Extract coordinates
    locations = extract_coordinates(geojson_data)
    
    # Fetch elevation data
    api_url = config.elevation.get('api_url', 'https://api.open-elevation.com/api/v1/lookup')
    print(f"\nFetching elevation data from: {api_url}")
    elevations = fetch_elevations(locations, api_url)
    
    # Enrich GeoJSON
    thresholds = config.elevation.get('terrain_thresholds', {'plain_max': 300, 'hills_max': 600})
    enriched_data = enrich_geojson_with_elevation(geojson_data, elevations, thresholds)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_geojson(enriched_data, output_file)
    
    # Print summary
    print_elevation_summary(enriched_data)
    
    print("\n✓ Elevation data added successfully")


if __name__ == "__main__":
    run_add_elevation()
