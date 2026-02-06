#!/usr/bin/env python3
"""
Visualization Module - Seasonal Patterns Map
============================================
Creates interactive map showing seasonal PM10 patterns.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

from ..config import PipelineConfig, load_config


SEASON_COLORS = {
    'Winter': '#1f78b4',
    'Spring': '#33a02c',
    'Summer': '#ff7f00',
    'Autumn': '#6a3d9a'
}


def generate_seasonal_map(config: Optional[PipelineConfig] = None) -> str:
    """
    Generate seasonal PM10 patterns interactive map.
    
    Returns:
        Path to saved HTML file
    """
    if config is None:
        config = load_config()
    
    print("Generating seasonal patterns map...")
    
    # Load data
    df = pd.read_parquet(config.get_panel_matrix_filtered_path())
    df = df.reset_index()
    
    time_col = config.temporal.time_label
    df['month'] = pd.to_datetime(df[time_col]).dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    # Calculate seasonal means per station
    seasonal_means = df.groupby(['station_id', 'season'])['pm10'].mean().unstack()
    
    # Find dominant season (highest PM10)
    dominant_season = seasonal_means.idxmax(axis=1)
    winter_pm10 = seasonal_means.get('Winter', pd.Series())
    summer_pm10 = seasonal_means.get('Summer', pd.Series())
    
    # Load station metadata
    gdf = gpd.read_file(config.get_stations_geojson_path(with_elevation=True))
    gdf = gdf.set_index('station_code')
    gdf['dominant_season'] = dominant_season
    gdf['winter_pm10'] = winter_pm10
    gdf['summer_pm10'] = summer_pm10
    gdf = gdf.reset_index()
    
    # Create map
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add markers
    for _, row in gdf.iterrows():
        season = row.get('dominant_season', 'Winter')
        color = SEASON_COLORS.get(season, '#808080')
        
        popup_html = f"""
        <b>{row.get('station_name', 'Unknown')}</b><br>
        Region: {row.get('region', 'N/A')}<br>
        <br>
        <b>Seasonal PM10 (μg/m³):</b><br>
        Winter: {row.get('winter_pm10', 0):.1f}<br>
        Summer: {row.get('summer_pm10', 0):.1f}<br>
        <br>
        Dominant Season: <b>{season}</b>
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
        <h4>Dominant PM10 Season</h4>
        <p><span style="color:#1f78b4;">●</span> Winter</p>
        <p><span style="color:#33a02c;">●</span> Spring</p>
        <p><span style="color:#ff7f00;">●</span> Summer</p>
        <p><span style="color:#6a3d9a;">●</span> Autumn</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save
    assets_dir = config.get_assets_subdir("maps")
    output_path = assets_dir / 'seasonal_pm10_patterns.html'
    m.save(str(output_path))
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_seasonal_map()
