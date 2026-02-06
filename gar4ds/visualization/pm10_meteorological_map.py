#!/usr/bin/env python3
"""
Visualization Module - PM10 Meteorological Map
==============================================
Creates interactive map showing PM10 and meteorological data.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap

from ..config import PipelineConfig, load_config


def generate_pm10_map(config: Optional[PipelineConfig] = None) -> str:
    """
    Generate PM10 and meteorological data interactive map.
    
    Returns:
        Path to saved HTML file
    """
    if config is None:
        config = load_config()
    
    print("Generating PM10 meteorological map...")
    
    # Load data
    df = pd.read_parquet(config.get_panel_matrix_filtered_path())
    df_mean = df.groupby(level='station_id').mean()
    
    gdf = gpd.read_file(config.get_stations_geojson_path(with_elevation=True))
    gdf = gdf.merge(df_mean, left_on='station_code', right_index=True, how='left')
    
    # Create map
    center_lat = gdf.geometry.y.mean()
    center_lon = gdf.geometry.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Color scale for PM10
    pm10_min = gdf['pm10'].min()
    pm10_max = gdf['pm10'].max()
    
    def get_color(pm10_val):
        if pd.isna(pm10_val):
            return '#808080'
        normalized = (pm10_val - pm10_min) / (pm10_max - pm10_min + 1e-10)
        if normalized < 0.25:
            return '#2c7bb6'  # Blue (low)
        elif normalized < 0.5:
            return '#abd9e9'  # Light blue
        elif normalized < 0.75:
            return '#fdae61'  # Orange
        else:
            return '#d7191c'  # Red (high)
    
    # Add markers
    for _, row in gdf.iterrows():
        pm10 = row.get('pm10', 0)
        color = get_color(pm10)
        
        popup_html = f"""
        <b>{row.get('station_name', 'Unknown')}</b><br>
        Region: {row.get('region', 'N/A')}<br>
        PM10: {pm10:.1f} μg/m³<br>
        BLH: {row.get('blh', 0):.0f} m<br>
        Temp: {row.get('temperature_2m', 0):.1f} K<br>
        Wind U: {row.get('wind_u_10m', 0):.2f} m/s<br>
        Wind V: {row.get('wind_v_10m', 0):.2f} m/s
        """
        
        # Size proportional to PM10
        radius = 5 + (pm10 / pm10_max) * 10
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    
    # Add legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey;">
        <h4>Mean PM10 (μg/m³)</h4>
        <p><span style="color:#d7191c;">●</span> High ({pm10_max:.0f}+)</p>
        <p><span style="color:#fdae61;">●</span> Medium-High</p>
        <p><span style="color:#abd9e9;">●</span> Medium-Low</p>
        <p><span style="color:#2c7bb6;">●</span> Low ({pm10_min:.0f})</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save
    assets_dir = config.get_assets_subdir("maps")
    output_path = assets_dir / 'pm10_meteorological_explorer.html'
    m.save(str(output_path))
    
    print(f"✓ Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_pm10_map()
