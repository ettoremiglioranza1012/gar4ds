"""
Add elevation data to station metadata GeoJSON using Open-Elevation API.

This script:
1. Loads the station metadata GeoJSON file
2. Extracts coordinates and batch requests elevation data from Open-Elevation API
3. Adds elevation data to both geometry (Z coordinate) and properties
4. Classifies areas as mountain/urban/plain based on elevation thresholds
5. Saves the enriched GeoJSON file

Author: Generated for gar4ds project
Date: 2026-02-06
"""

import json
import requests
from pathlib import Path
from typing import Dict, List
from datetime import datetime


# Configuration
INPUT_FILE = "data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson"
OUTPUT_FILE = "data/pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson"
ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"

# Elevation thresholds for classification (in meters)
# Based on general topographic classifications for Northern Italy
# Includes below-sea-level areas (e.g., Venice lagoon) in plains
ELEVATION_THRESHOLDS = {
    'plain': (-10, 300),         # Plains, lowlands, and coastal/lagoon areas
    'hills': (300, 600),         # Hilly areas
    'mountain': (600, float('inf'))  # Mountain areas
}

# Urban classification could be enhanced with additional data sources
# For now, it's based on station name patterns
URBAN_KEYWORDS = [
    'via', 'p.zza', 'piazza', 'centro', 'city',
    'borgo', 'urban', 'città'
]


def load_geojson(filepath: str) -> Dict:
    """Load GeoJSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_geojson(data: Dict, filepath: str) -> None:
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


def fetch_elevations(locations: List[Dict[str, float]], batch_size: int = 100) -> List[Dict]:
    """
    Fetch elevation data from Open-Elevation API.
    
    Args:
        locations: List of {latitude, longitude} dictionaries
        batch_size: Number of locations per request (API limit consideration)
    
    Returns:
        List of elevation results
    """
    all_elevations = []
    headers = {'Content-Type': 'application/json'}
    
    # Process in batches to avoid potential API limits
    for i in range(0, len(locations), batch_size):
        batch = locations[i:i + batch_size]
        print(f"Fetching elevations for locations {i+1} to {i+len(batch)}...")
        
        try:
            response = requests.post(
                ELEVATION_API_URL,
                json={'locations': batch},
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            results = response.json()['results']
            all_elevations.extend(results)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error retrieving elevation data for batch {i//batch_size + 1}: {e}")
            # Add None values for failed batch
            all_elevations.extend([None] * len(batch))
    
    return all_elevations


def classify_terrain(elevation: float) -> str:
    """
    Classify terrain type based on elevation.
    
    Args:
        elevation: Elevation in meters
    
    Returns:
        Terrain classification: 'plain', 'hills', or 'mountain'
    """
    for terrain_type, (min_elev, max_elev) in ELEVATION_THRESHOLDS.items():
        if min_elev <= elevation < max_elev:
            return terrain_type
    return 'unknown'


def classify_area_type(station_name: str, elevation: float) -> str:
    """
    Classify area as urban, suburban, or rural.
    
    This is a simplified classification based on station name patterns.
    A more sophisticated approach would use additional geographic data.
    
    Args:
        station_name: Name of the monitoring station
        elevation: Elevation in meters
    
    Returns:
        Area type: 'urban', 'suburban', or 'rural'
    """
    name_lower = station_name.lower()
    
    # Check if station name contains urban indicators
    is_likely_urban = any(keyword in name_lower for keyword in URBAN_KEYWORDS)
    
    if is_likely_urban and elevation < 400:
        return 'urban'
    elif is_likely_urban or elevation < 400:
        return 'suburban'
    else:
        return 'rural'


def enrich_geojson_with_elevation(geojson_data: Dict, elevations: List[Dict]) -> Dict:
    """
    Add elevation data and classifications to GeoJSON features.
    
    Args:
        geojson_data: Original GeoJSON data
        elevations: List of elevation results from API
    
    Returns:
        Enriched GeoJSON data
    """
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
            feature['properties']['terrain_type'] = classify_terrain(elevation)
            feature['properties']['area_type'] = classify_area_type(station_name, elevation)
            
            success_count += 1
        else:
            # Mark as missing data
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
        if f['properties'].get('elevation') is not None
    ]
    
    if not elevations:
        print("No elevation data to summarize.")
        return
    
    print("\n" + "="*60)
    print("ELEVATION DATA SUMMARY")
    print("="*60)
    print(f"Total stations: {len(geojson_data['features'])}")
    print(f"Stations with elevation data: {len(elevations)}")
    print(f"\nElevation statistics (meters):")
    print(f"  Minimum: {min(elevations):.2f}")
    print(f"  Maximum: {max(elevations):.2f}")
    print(f"  Average: {sum(elevations)/len(elevations):.2f}")
    
    # Terrain type distribution
    terrain_counts = {}
    area_counts = {}
    for feature in geojson_data['features']:
        terrain = feature['properties'].get('terrain_type', 'unknown')
        area = feature['properties'].get('area_type', 'unknown')
        terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
        area_counts[area] = area_counts.get(area, 0) + 1
    
    print(f"\nTerrain type distribution:")
    for terrain, count in sorted(terrain_counts.items()):
        print(f"  {terrain.capitalize()}: {count}")
    
    print(f"\nArea type distribution:")
    for area, count in sorted(area_counts.items()):
        print(f"  {area.capitalize()}: {count}")
    print("="*60 + "\n")


def main():
    """Main execution function."""
    print("="*60)
    print("ELEVATION DATA ENRICHMENT SCRIPT")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_path = project_root / INPUT_FILE
    output_path = project_root / OUTPUT_FILE
    
    # Load GeoJSON
    print(f"Loading GeoJSON from: {input_path}")
    try:
        geojson_data = load_geojson(input_path)
        print(f"✓ Loaded {len(geojson_data['features'])} stations\n")
    except Exception as e:
        print(f"✗ Error loading GeoJSON: {e}")
        return
    
    # Extract coordinates
    print("Extracting coordinates...")
    locations = extract_coordinates(geojson_data)
    print(f"✓ Extracted {len(locations)} coordinate pairs\n")
    
    # Fetch elevation data
    print("Fetching elevation data from Open-Elevation API...")
    print("(This may take a moment...)\n")
    elevations = fetch_elevations(locations)
    
    # Enrich GeoJSON with elevation data
    print("\nEnriching GeoJSON with elevation and classification data...")
    enriched_data = enrich_geojson_with_elevation(geojson_data, elevations)
    
    # Save enriched data
    save_geojson(enriched_data, output_path)
    
    # Print summary
    print_elevation_summary(enriched_data)
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
