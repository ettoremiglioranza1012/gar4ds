"""
INTERACTIVE PM10 POLLUTION & METEOROLOGICAL EXPLORER MAP
==========================================================

This script generates an interactive HTML map for exploring PM10 pollution levels
and meteorological conditions across the Po Valley - Alpine region monitoring network.

FEATURES:
- Stations colored by PM10 levels with customizable threshold
- Wind vectors showing direction and magnitude
- Filter by terrain type (plain, hills, mountain)
- Filter by region (Veneto, Lombardia, Trentino, Alto-Adige)
- Seasonal pattern visualization
- Spatial connections from KNN weights matrix
- Detailed popup with all meteorological variables

OUTPUT:
- assets/maps/pm10_meteorological_explorer.html
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap, AntPath
from pathlib import Path
import json
import math
from branca.element import Element
from branca.colormap import LinearColormap

# ============================================================================
# DIRECTORY SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RESULTS_DIR = PROJECT_DIR / 'results'
ASSETS_DIR = PROJECT_DIR / 'assets' / 'maps'
WEIGHTS_DIR = PROJECT_DIR / 'weights'

# Create output directory
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_station_metadata():
    """Load station metadata with coordinates"""
    meta_path = DATA_DIR / 'pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson'
    gdf = gpd.read_file(meta_path)
    
    # Extract coordinates
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    
    return gdf


def load_cross_sectional_data():
    """Load cross-sectional station profiles (mean values)"""
    profile_path = RESULTS_DIR / 'spatial_analysis' / 'station_profiles_cross_sectional.csv'
    df = pd.read_csv(profile_path)
    return df


def load_panel_data():
    """Load panel data for temporal analysis"""
    panel_path = DATA_DIR / 'panel_data_matrix_filtered_for_collinearity.parquet'
    if panel_path.exists():
        return pd.read_parquet(panel_path)
    return None


def load_spatial_weights():
    """Load spatial weights matrix to draw connections"""
    weights_path = WEIGHTS_DIR / 'spatial_weights_knn6.gal'
    
    if not weights_path.exists():
        return None
    
    # Parse GAL file manually
    connections = {}
    with open(weights_path, 'r') as f:
        lines = f.readlines()
        
    # Skip header line
    i = 1
    while i < len(lines):
        # Station line: station_id n_neighbors
        parts = lines[i].strip().split()
        if len(parts) >= 2:
            station_id = parts[0]
            n_neighbors = int(parts[1])
            i += 1
            
            # Neighbors line
            if i < len(lines) and n_neighbors > 0:
                neighbors = lines[i].strip().split()
                connections[station_id] = neighbors
                i += 1
        else:
            i += 1
    
    return connections


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pm10_color(value, min_val=10, max_val=35):
    """Get color based on PM10 value using color scale"""
    # Normalize to 0-1 range
    norm_val = (value - min_val) / (max_val - min_val)
    norm_val = max(0, min(1, norm_val))
    
    # Green -> Yellow -> Orange -> Red gradient
    if norm_val < 0.33:
        # Green to Yellow
        r = int(255 * (norm_val / 0.33))
        g = 200
        b = 0
    elif norm_val < 0.66:
        # Yellow to Orange
        r = 255
        g = int(200 - 100 * ((norm_val - 0.33) / 0.33))
        b = 0
    else:
        # Orange to Red
        r = 255
        g = int(100 - 100 * ((norm_val - 0.66) / 0.34))
        b = 0
    
    return f'#{r:02x}{g:02x}{b:02x}'


def get_terrain_icon(terrain_type):
    """Get emoji icon for terrain type"""
    icons = {
        'plain': '🏭',
        'mountain': '⛰️',
        'hills': '🌄'
    }
    return icons.get(terrain_type, '📍')


def calculate_wind_arrow(u, v, lat, lon, scale=0.02):
    """Calculate arrow endpoint from wind components"""
    # Wind speed
    speed = math.sqrt(u**2 + v**2)
    
    if speed < 0.01:
        return None
    
    # Normalize and scale
    arrow_len = min(speed * scale, 0.1)  # Max arrow length
    
    # Calculate end point
    end_lat = lat + v * scale / speed * arrow_len * 10 if speed > 0 else lat
    end_lon = lon + u * scale / speed * arrow_len * 10 if speed > 0 else lon
    
    return (end_lat, end_lon, speed)


def create_detailed_popup(station, profile):
    """Create detailed popup HTML for a station"""
    
    terrain_icon = get_terrain_icon(station.get('terrain_type', 'plain'))
    pm10_val = profile['pm10']
    pm10_color = get_pm10_color(pm10_val)
    
    # Wind speed and direction
    u = profile['wind_u_10m']
    v = profile['wind_v_10m']
    wind_speed = math.sqrt(u**2 + v**2)
    wind_dir = (math.degrees(math.atan2(-u, -v)) + 360) % 360  # Meteorological convention
    
    html = f"""
    <div style="font-family: Arial, sans-serif; width: 380px;">
        <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 3px solid {pm10_color};">
            {terrain_icon} {station['station_name']}
        </h4>
        
        <div style="display: flex; gap: 10px; margin-bottom: 12px;">
            <div style="flex: 1; background: #f0f7ff; padding: 8px; border-radius: 6px;">
                <strong style="color: #555;">Station ID</strong><br>
                <span style="font-size: 14px;">{station['station_code']}</span>
            </div>
            <div style="flex: 1; background: #f0f7ff; padding: 8px; border-radius: 6px;">
                <strong style="color: #555;">Region</strong><br>
                <span style="font-size: 14px;">{station['region']}</span>
            </div>
        </div>
        
        <div style="display: flex; gap: 10px; margin-bottom: 12px;">
            <div style="flex: 1; background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <strong style="color: #555;">Terrain</strong><br>
                <span style="font-size: 14px;">{station.get('terrain_type', 'N/A').title()}</span>
            </div>
            <div style="flex: 1; background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <strong style="color: #555;">Elevation</strong><br>
                <span style="font-size: 14px;">{station.get('elevation', 'N/A'):.0f} m</span>
            </div>
            <div style="flex: 1; background: #f5f5f5; padding: 8px; border-radius: 6px;">
                <strong style="color: #555;">Area</strong><br>
                <span style="font-size: 14px;">{station.get('area_type', 'N/A').title()}</span>
            </div>
        </div>
        
        <div style="background: {pm10_color}30; padding: 12px; border-radius: 6px; border-left: 4px solid {pm10_color}; margin-bottom: 12px;">
            <strong style="font-size: 16px; color: #333;">PM₁₀ Mean: {pm10_val:.1f} μg/m³</strong>
        </div>
        
        <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <tr style="background: #f9f9f9;">
                <th style="padding: 6px; text-align: left; border-bottom: 1px solid #ddd;">Variable</th>
                <th style="padding: 6px; text-align: right; border-bottom: 1px solid #ddd;">Value</th>
            </tr>
            <tr>
                <td style="padding: 5px;">🌡️ Temperature (2m)</td>
                <td style="padding: 5px; text-align: right;">{profile['temperature_2m'] - 273.15:.1f} °C</td>
            </tr>
            <tr style="background: #f9f9f9;">
                <td style="padding: 5px;">💧 Humidity (950 hPa)</td>
                <td style="padding: 5px; text-align: right;">{profile['humidity_950']:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 5px;">📊 Boundary Layer Height</td>
                <td style="padding: 5px; text-align: right;">{profile['blh']:.0f} m</td>
            </tr>
            <tr style="background: #f9f9f9;">
                <td style="padding: 5px;">☀️ Solar Radiation</td>
                <td style="padding: 5px; text-align: right;">{profile['solar_radiation_downwards']/1000:.0f} kJ/m²</td>
            </tr>
            <tr>
                <td style="padding: 5px;">💨 Wind Speed (10m)</td>
                <td style="padding: 5px; text-align: right;">{wind_speed:.2f} m/s</td>
            </tr>
            <tr style="background: #f9f9f9;">
                <td style="padding: 5px;">🧭 Wind Direction</td>
                <td style="padding: 5px; text-align: right;">{wind_dir:.0f}°</td>
            </tr>
            <tr>
                <td style="padding: 5px;">🌧️ Precipitation</td>
                <td style="padding: 5px; text-align: right;">{profile['total_precipitation']:.1f} mm</td>
            </tr>
        </table>
        
        <div style="margin-top: 10px; padding: 8px; background: #fff9e6; border-radius: 4px; font-size: 11px;">
            <strong>Upper Level Winds (850 hPa):</strong><br>
            U: {profile['uwind_850']:.2f} m/s | V: {profile['Vwind_850']:.2f} m/s
        </div>
    </div>
    """
    return html


# ============================================================================
# MAP GENERATION
# ============================================================================

def generate_pm10_map():
    """Generate the interactive PM10 and meteorological explorer map"""
    
    print("=" * 70)
    print("  GENERATING PM10 & METEOROLOGICAL EXPLORER MAP")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    stations_gdf = load_station_metadata()
    profile_df = load_cross_sectional_data()
    panel_df = load_panel_data()
    weights = load_spatial_weights()
    
    print(f"    • Stations: {len(stations_gdf)}")
    print(f"    • Profile variables: {len(profile_df.columns)}")
    if weights:
        print(f"    • Spatial connections loaded: {len(weights)} stations")
    if panel_df is not None:
        print(f"    • Panel data: {len(panel_df)} observations")
    
    # Merge station metadata with profiles
    stations_gdf['station_code'] = stations_gdf['station_code'].astype(str)
    profile_df['station_id'] = profile_df['station_id'].astype(str)
    
    # Calculate map center
    center_lat = stations_gdf['lat'].mean()
    center_lon = stations_gdf['lon'].mean()
    
    # PM10 statistics
    pm10_min = profile_df['pm10'].min()
    pm10_max = profile_df['pm10'].max()
    pm10_mean = profile_df['pm10'].mean()
    
    print(f"\n    PM10 Statistics:")
    print(f"      Min: {pm10_min:.1f} μg/m³")
    print(f"      Max: {pm10_max:.1f} μg/m³")
    print(f"      Mean: {pm10_mean:.1f} μg/m³")
    
    # Create base map
    print("\n[2] Creating base map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None
    )
    
    # Add tile layers
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Light Base Map',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        control=True
    ).add_to(m)
    
    # Create color scale for PM10
    colormap = LinearColormap(
        colors=['#00c853', '#ffeb3b', '#ff9800', '#f44336'],
        vmin=10,
        vmax=35,
        caption='PM₁₀ Mean Concentration (μg/m³)'
    )
    colormap.add_to(m)
    
    # Create feature groups
    print("\n[3] Creating feature groups...")
    
    # All stations layer
    all_stations_fg = folium.FeatureGroup(name='🏭 All Stations (PM₁₀)', show=True)
    
    # Terrain-based layers
    plain_fg = folium.FeatureGroup(name='🏭 Plain Stations', show=False)
    hills_fg = folium.FeatureGroup(name='🌄 Hill Stations', show=False)
    mountain_fg = folium.FeatureGroup(name='⛰️ Mountain Stations', show=False)
    
    # Region-based layers
    region_fgs = {}
    for region in stations_gdf['region'].unique():
        region_fgs[region] = folium.FeatureGroup(name=f'📍 {region}', show=False)
    
    # Wind vectors layer
    wind_fg = folium.FeatureGroup(name='💨 Wind Vectors (10m)', show=True)
    
    # Spatial connections layer
    connections_fg = folium.FeatureGroup(name='🔗 Spatial Connections (KNN6)', show=False)
    
    # Add markers and wind vectors
    print("\n[4] Adding station markers and wind vectors...")
    
    for _, station in stations_gdf.iterrows():
        station_code = station['station_code']
        
        # Get profile data
        profile_row = profile_df[profile_df['station_id'] == station_code]
        if profile_row.empty:
            continue
        profile = profile_row.iloc[0]
        
        lat, lon = station['lat'], station['lon']
        pm10_val = profile['pm10']
        terrain = station.get('terrain_type', 'plain')
        region = station['region']
        
        # Create marker
        marker_color = get_pm10_color(pm10_val)
        radius = 8 + (pm10_val / pm10_max) * 8  # Scale radius with PM10
        
        popup = folium.Popup(create_detailed_popup(station, profile), max_width=420)
        
        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color='#333',
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            popup=popup,
            tooltip=f"{station['station_name']}: {pm10_val:.1f} μg/m³"
        )
        
        # Add to all stations
        marker.add_to(all_stations_fg)
        
        # Add to terrain layer
        marker_copy = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color='#333',
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            popup=popup,
            tooltip=f"{station['station_name']}: {pm10_val:.1f} μg/m³"
        )
        
        if terrain == 'plain':
            marker_copy.add_to(plain_fg)
        elif terrain == 'hills':
            marker_copy.add_to(hills_fg)
        elif terrain == 'mountain':
            marker_copy.add_to(mountain_fg)
        
        # Add to region layer
        marker_region = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color='#333',
            weight=2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            popup=popup,
            tooltip=f"{station['station_name']}: {pm10_val:.1f} μg/m³"
        )
        marker_region.add_to(region_fgs[region])
        
        # Add wind vector
        u = profile['wind_u_10m']
        v = profile['wind_v_10m']
        wind_result = calculate_wind_arrow(u, v, lat, lon, scale=0.08)
        
        if wind_result:
            end_lat, end_lon, speed = wind_result
            
            # Scale arrow opacity with wind speed
            arrow_opacity = min(0.3 + speed * 0.5, 1.0)
            
            folium.PolyLine(
                locations=[[lat, lon], [end_lat, end_lon]],
                weight=2,
                color='#1565c0',
                opacity=arrow_opacity,
                tooltip=f"Wind: {speed:.2f} m/s"
            ).add_to(wind_fg)
            
            # Add arrow head
            folium.CircleMarker(
                location=[end_lat, end_lon],
                radius=3,
                color='#1565c0',
                fill=True,
                fill_color='#1565c0',
                fill_opacity=arrow_opacity
            ).add_to(wind_fg)
    
    # Add spatial connections
    print("\n[5] Adding spatial connections...")
    if weights:
        added_connections = set()
        
        for station_id, neighbors in weights.items():
            # Get station coordinates
            station_row = stations_gdf[stations_gdf['station_code'] == station_id]
            if station_row.empty:
                continue
            
            lat1, lon1 = station_row.iloc[0]['lat'], station_row.iloc[0]['lon']
            
            for neighbor_id in neighbors:
                # Avoid duplicate connections
                conn_key = tuple(sorted([station_id, neighbor_id]))
                if conn_key in added_connections:
                    continue
                added_connections.add(conn_key)
                
                neighbor_row = stations_gdf[stations_gdf['station_code'] == neighbor_id]
                if neighbor_row.empty:
                    continue
                
                lat2, lon2 = neighbor_row.iloc[0]['lat'], neighbor_row.iloc[0]['lon']
                
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    weight=1.5,
                    color='#9e9e9e',
                    opacity=0.5,
                    dash_array='5, 5',
                    tooltip=f"Connection: {station_id} ↔ {neighbor_id}"
                ).add_to(connections_fg)
        
        print(f"    • Added {len(added_connections)} spatial connections")
    
    # Add feature groups to map
    all_stations_fg.add_to(m)
    wind_fg.add_to(m)
    connections_fg.add_to(m)
    plain_fg.add_to(m)
    hills_fg.add_to(m)
    mountain_fg.add_to(m)
    
    for fg in region_fgs.values():
        fg.add_to(m)
    
    # Add legend
    print("\n[6] Adding UI elements...")
    legend_html = f'''
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        max-width: 260px;
    ">
        <h4 style="margin: 0 0 10px 0; color: #333;">Station Properties</h4>
        
        <div style="margin-bottom: 10px;">
            <strong style="font-size: 12px; color: #555;">Terrain Types:</strong>
            <div style="margin-top: 5px; font-size: 12px;">
                🏭 Plain ({len(stations_gdf[stations_gdf['terrain_type']=='plain'])} stations)<br>
                🌄 Hills ({len(stations_gdf[stations_gdf['terrain_type']=='hills'])} stations)<br>
                ⛰️ Mountain ({len(stations_gdf[stations_gdf['terrain_type']=='mountain'])} stations)
            </div>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        
        <div style="margin-bottom: 10px;">
            <strong style="font-size: 12px; color: #555;">Marker Size:</strong>
            <div style="font-size: 11px; color: #666;">Proportional to PM₁₀ level</div>
        </div>
        
        <div>
            <strong style="font-size: 12px; color: #555;">Wind Arrows:</strong>
            <div style="font-size: 11px; color: #666;">
                Direction: meteorological convention<br>
                Opacity: proportional to speed
            </div>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        
        <div style="font-size: 11px; color: #888;">
            <strong>Data:</strong> Weekly averages (2014-2024)
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        background-color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
    ">
        <h3 style="margin: 0; color: #333; text-align: center;">
            🗺️ PM₁₀ & Meteorological Explorer
        </h3>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666; text-align: center;">
            Po Valley - Alpine Region Air Quality Monitoring Network
        </p>
    </div>
    '''
    m.get_root().html.add_child(Element(title_html))
    
    # Add instructions
    instructions_html = '''
    <div style="
        position: fixed;
        bottom: 50px;
        right: 50px;
        z-index: 1000;
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        max-width: 260px;
    ">
        <h4 style="margin: 0 0 10px 0; color: #333;">📋 How to Explore</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #555;">
            <li>Toggle <strong>terrain filters</strong> in layer control</li>
            <li>Enable <strong>wind vectors</strong> to see patterns</li>
            <li>Show <strong>spatial connections</strong> (KNN-6)</li>
            <li><strong>Click stations</strong> for full details</li>
            <li>Use <strong>color scale</strong> at bottom-right</li>
        </ul>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        <div style="font-size: 11px; color: #888;">
            <strong>Key Pattern:</strong><br>
            Higher PM₁₀ in Po Valley (red/orange);<br>
            Lower in Alpine regions (green).
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(instructions_html))
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    print("\n[7] Saving map...")
    output_path = ASSETS_DIR / 'pm10_meteorological_explorer.html'
    m.save(str(output_path))
    print(f"    ✓ Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("  MAP GENERATION COMPLETE")
    print("=" * 70)
    
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    output_path = generate_pm10_map()
    print(f"\n🗺️  Open the map in a browser: {output_path}")
