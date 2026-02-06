#!/usr/bin/env python3
"""
Visualization Module - LISA Clusters Map
========================================
Creates interactive Folium map showing LISA cluster results.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

from ..config import PipelineConfig, load_config


LISA_COLORS = {
    'HH': '#d7191c',  # High-High (red)
    'LL': '#2c7bb6',  # Low-Low (blue)
    'HL': '#fdae61',  # High-Low (orange)
    'LH': '#abd9e9',  # Low-High (light blue)
    'NS': '#ffffbf',  # Not significant (yellow)
}


def generate_lisa_map(config: Optional[PipelineConfig] = None) -> str:
    """
    Generate LISA clusters interactive map.
    
    Returns:
        Path to saved HTML file
    """
    if config is None:
        config = load_config()
    
    print("Generating LISA clusters map...")
    
    # Load data
    results_dir = config.get_results_subdir("spatial_analysis")
    lisa_path = results_dir / 'lisa_results_pm10.csv'
    
    if not lisa_path.exists():
        raise FileNotFoundError(f"LISA results not found: {lisa_path}")
    
    lisa_df = pd.read_csv(lisa_path)
    
    # Load station metadata
    gdf = gpd.read_file(config.get_stations_geojson_path(with_elevation=True))
    
    # Merge
    gdf = gdf.merge(lisa_df, left_on='station_code', right_on='station_id', how='left')
    
    # Create map
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add markers
    for _, row in gdf.iterrows():
        cluster = row.get('cluster', 'NS')
        significant = row.get('significant', False)
        
        color = LISA_COLORS.get(cluster if significant else 'NS', '#ffffbf')
        
        popup_html = f"""
        <b>{row.get('station_name', 'Unknown')}</b><br>
        Station: {row.get('station_code', 'N/A')}<br>
        PM10: {row.get('pm10_mean', 0):.1f} μg/m³<br>
        LISA Cluster: {cluster}<br>
        Significant: {significant}
        """
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=8,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey;">
        <h4>LISA Clusters</h4>
        <p><span style="color:#d7191c;">●</span> High-High</p>
        <p><span style="color:#2c7bb6;">●</span> Low-Low</p>
        <p><span style="color:#fdae61;">●</span> High-Low</p>
        <p><span style="color:#abd9e9;">●</span> Low-High</p>
        <p><span style="color:#ffffbf;">●</span> Not Significant</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save
    assets_dir = config.get_assets_subdir("maps")
    output_path = assets_dir / 'lisa_clusters_explorer.html'
    m.save(str(output_path))
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_lisa_map()
