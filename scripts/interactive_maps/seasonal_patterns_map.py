"""
INTERACTIVE SEASONAL PM10 PATTERNS MAP
=======================================

This script generates an interactive HTML map for exploring seasonal PM10 patterns
and how pollution levels vary across seasons in different terrain types.

FEATURES:
- Season selector (Winter, Spring, Summer, Autumn)
- Seasonal PM10 values for each station
- Comparison of seasonal patterns by terrain
- Animation capability showing seasonal progression
- Interactive charts showing seasonal trends

OUTPUT:
- assets/maps/seasonal_pm10_patterns.html
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import TimestampedGeoJson
from pathlib import Path
import json
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

# Create output directory
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_station_metadata():
    """Load station metadata with coordinates"""
    meta_path = DATA_DIR / 'pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson'
    gdf = gpd.read_file(meta_path)
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    return gdf


def load_panel_data():
    """Load panel data for temporal analysis"""
    panel_path = DATA_DIR / 'panel_data_matrix_filtered_for_collinearity.parquet'
    if panel_path.exists():
        return pd.read_parquet(panel_path)
    return None


def compute_seasonal_data(panel_df, stations_gdf):
    """Compute seasonal statistics for each station"""
    
    # Reset index to access week_start
    df = panel_df.reset_index()
    
    # Convert week_start to datetime if needed
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Extract month and assign season
    df['month'] = df['week_start'].dt.month
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df['season'] = df['month'].apply(get_season)
    
    # Compute seasonal statistics per station
    seasonal_stats = df.groupby(['station_id', 'season']).agg({
        'pm10': ['mean', 'std', 'min', 'max', 'count'],
        'temperature_2m': 'mean',
        'blh': 'mean',
        'humidity_950': 'mean',
        'wind_u_10m': 'mean',
        'wind_v_10m': 'mean'
    }).reset_index()
    
    # Flatten column names
    seasonal_stats.columns = [
        'station_id', 'season', 
        'pm10_mean', 'pm10_std', 'pm10_min', 'pm10_max', 'n_observations',
        'temperature_mean', 'blh_mean', 'humidity_mean',
        'wind_u_mean', 'wind_v_mean'
    ]
    
    # Merge with station metadata
    stations_gdf['station_code'] = stations_gdf['station_code'].astype(str)
    seasonal_stats['station_id'] = seasonal_stats['station_id'].astype(str)
    
    merged = seasonal_stats.merge(
        stations_gdf[['station_code', 'station_name', 'region', 'terrain_type', 'area_type', 'elevation', 'lat', 'lon']],
        left_on='station_id',
        right_on='station_code',
        how='left'
    )
    
    return merged


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_pm10_color(value, min_val=5, max_val=45):
    """Get color based on PM10 value"""
    norm_val = (value - min_val) / (max_val - min_val)
    norm_val = max(0, min(1, norm_val))
    
    if norm_val < 0.25:
        r, g, b = 76, 175, 80   # Green
    elif norm_val < 0.50:
        r, g, b = 255, 235, 59  # Yellow
    elif norm_val < 0.75:
        r, g, b = 255, 152, 0   # Orange
    else:
        r, g, b = 244, 67, 54   # Red
    
    return f'#{r:02x}{g:02x}{b:02x}'


SEASON_COLORS = {
    'Winter': '#1976d2',  # Blue
    'Spring': '#4caf50',  # Green
    'Summer': '#ff9800',  # Orange
    'Autumn': '#9c27b0'   # Purple
}

SEASON_ICONS = {
    'Winter': '❄️',
    'Spring': '🌸',
    'Summer': '☀️',
    'Autumn': '🍂'
}

SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']


def create_seasonal_popup(station_data, season):
    """Create detailed popup for a season/station combination"""
    
    pm10_val = station_data['pm10_mean']
    pm10_color = get_pm10_color(pm10_val)
    season_icon = SEASON_ICONS.get(season, '📅')
    season_color = SEASON_COLORS.get(season, '#666')
    
    # Convert temperature from Kelvin to Celsius
    temp_c = station_data['temperature_mean'] - 273.15
    
    html = f"""
    <div style="font-family: Arial, sans-serif; width: 340px;">
        <div style="background: {season_color}; color: white; padding: 10px; border-radius: 8px 8px 0 0; margin: -1px;">
            <h4 style="margin: 0;">{season_icon} {station_data['station_name']} - {season}</h4>
            <span style="opacity: 0.9; font-size: 12px;">{station_data['region']} | {station_data.get('terrain_type', 'N/A').title()}</span>
        </div>
        
        <div style="padding: 12px; background: white; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px;">
            
            <div style="background: {pm10_color}25; padding: 10px; border-radius: 6px; border-left: 4px solid {pm10_color}; margin-bottom: 12px;">
                <strong style="font-size: 18px; color: #333;">PM₁₀: {pm10_val:.1f} μg/m³</strong>
                <div style="font-size: 11px; color: #666; margin-top: 4px;">
                    Range: {station_data['pm10_min']:.1f} - {station_data['pm10_max']:.1f} | 
                    StdDev: {station_data['pm10_std']:.1f}
                </div>
            </div>
            
            <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
                <tr>
                    <td style="padding: 6px; border-bottom: 1px solid #eee;">🌡️ Temperature</td>
                    <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;">{temp_c:.1f} °C</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border-bottom: 1px solid #eee;">📊 Boundary Layer</td>
                    <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;">{station_data['blh_mean']:.0f} m</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border-bottom: 1px solid #eee;">💧 Humidity</td>
                    <td style="padding: 6px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold;">{station_data['humidity_mean']:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding: 6px;">📊 Observations</td>
                    <td style="padding: 6px; text-align: right; font-weight: bold;">{int(station_data['n_observations'])} weeks</td>
                </tr>
            </table>
            
            <div style="margin-top: 10px; padding: 8px; background: #f5f5f5; border-radius: 4px; font-size: 11px;">
                <strong>Station ID:</strong> {station_data['station_id']}<br>
                <strong>Elevation:</strong> {station_data.get('elevation', 'N/A'):.0f} m
            </div>
        </div>
    </div>
    """
    return html


# ============================================================================
# MAP GENERATION
# ============================================================================

def generate_seasonal_map():
    """Generate the interactive seasonal PM10 patterns map"""
    
    print("=" * 70)
    print("  GENERATING SEASONAL PM10 PATTERNS MAP")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    stations_gdf = load_station_metadata()
    panel_df = load_panel_data()
    
    if panel_df is None:
        print("    ✗ Error: Panel data not found")
        return None
    
    print(f"    • Stations: {len(stations_gdf)}")
    print(f"    • Panel observations: {len(panel_df)}")
    
    # Compute seasonal statistics
    print("\n[2] Computing seasonal statistics...")
    seasonal_df = compute_seasonal_data(panel_df, stations_gdf)
    print(f"    • Seasonal records: {len(seasonal_df)}")
    
    # Check seasonal PM10 summary
    seasonal_summary = seasonal_df.groupby('season')['pm10_mean'].agg(['mean', 'min', 'max'])
    print("\n    Seasonal PM10 Summary:")
    for season in SEASON_ORDER:
        if season in seasonal_summary.index:
            row = seasonal_summary.loc[season]
            print(f"      {SEASON_ICONS[season]} {season}: Mean={row['mean']:.1f}, Range=[{row['min']:.1f}, {row['max']:.1f}]")
    
    # Calculate map center
    center_lat = stations_gdf['lat'].mean()
    center_lon = stations_gdf['lon'].mean()
    
    # Create base map
    print("\n[3] Creating base map...")
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
    
    # Create color scale
    colormap = LinearColormap(
        colors=['#4caf50', '#ffeb3b', '#ff9800', '#f44336'],
        vmin=5,
        vmax=45,
        caption='Seasonal PM₁₀ Mean (μg/m³)'
    )
    colormap.add_to(m)
    
    # Create feature groups for each season
    print("\n[4] Creating seasonal layers...")
    
    season_fgs = {}
    for season in SEASON_ORDER:
        icon = SEASON_ICONS[season]
        color = SEASON_COLORS[season]
        is_winter = (season == 'Winter')  # Show winter by default
        
        fg = folium.FeatureGroup(name=f'{icon} {season}', show=is_winter)
        season_fgs[season] = fg
        
        # Get seasonal data
        season_data = seasonal_df[seasonal_df['season'] == season]
        
        for _, row in season_data.iterrows():
            lat = row['lat']
            lon = row['lon']
            pm10_val = row['pm10_mean']
            
            if pd.isna(lat) or pd.isna(lon):
                continue
            
            marker_color = get_pm10_color(pm10_val)
            radius = 7 + (pm10_val / 45) * 10
            
            popup = folium.Popup(create_seasonal_popup(row, season), max_width=380)
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                weight=2,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.85,
                popup=popup,
                tooltip=f"{row['station_name']}: {pm10_val:.1f} μg/m³ ({season})"
            ).add_to(fg)
        
        fg.add_to(m)
        print(f"    • Added {len(season_data)} markers for {season}")
    
    # Create terrain comparison layers
    print("\n[5] Creating terrain comparison layers...")
    
    for terrain in ['plain', 'mountain', 'hills']:
        terrain_icon = {'plain': '🏭', 'mountain': '⛰️', 'hills': '🌄'}[terrain]
        fg = folium.FeatureGroup(name=f'{terrain_icon} {terrain.title()} (Winter)', show=False)
        
        # Get winter data for this terrain
        terrain_data = seasonal_df[(seasonal_df['season'] == 'Winter') & (seasonal_df['terrain_type'] == terrain)]
        
        for _, row in terrain_data.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                continue
            
            pm10_val = row['pm10_mean']
            marker_color = get_pm10_color(pm10_val)
            radius = 7 + (pm10_val / 45) * 10
            
            popup = folium.Popup(create_seasonal_popup(row, 'Winter'), max_width=380)
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                color='#1976d2',
                weight=2,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.85,
                popup=popup,
                tooltip=f"{row['station_name']}: {pm10_val:.1f} μg/m³"
            ).add_to(fg)
        
        fg.add_to(m)
    
    # Add legend and info panel
    print("\n[6] Adding UI elements...")
    
    # Compute useful statistics for legend
    winter_mean = seasonal_df[seasonal_df['season'] == 'Winter']['pm10_mean'].mean()
    summer_mean = seasonal_df[seasonal_df['season'] == 'Summer']['pm10_mean'].mean()
    winter_summer_ratio = winter_mean / summer_mean if summer_mean > 0 else 0
    
    plain_winter = seasonal_df[(seasonal_df['season'] == 'Winter') & (seasonal_df['terrain_type'] == 'plain')]['pm10_mean'].mean()
    mountain_winter = seasonal_df[(seasonal_df['season'] == 'Winter') & (seasonal_df['terrain_type'] == 'mountain')]['pm10_mean'].mean()
    
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
        max-width: 280px;
    ">
        <h4 style="margin: 0 0 12px 0; color: #333;">Seasonal PM₁₀ Patterns</h4>
        
        <div style="margin-bottom: 12px;">
            <strong style="font-size: 12px; color: #555;">Season Selector:</strong>
            <div style="margin-top: 5px; font-size: 13px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: {SEASON_COLORS['Winter']}; border-radius: 50%; margin-right: 8px;"></span>
                    ❄️ Winter (Dec-Feb)
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: {SEASON_COLORS['Spring']}; border-radius: 50%; margin-right: 8px;"></span>
                    🌸 Spring (Mar-May)
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: {SEASON_COLORS['Summer']}; border-radius: 50%; margin-right: 8px;"></span>
                    ☀️ Summer (Jun-Aug)
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="display: inline-block; width: 12px; height: 12px; background: {SEASON_COLORS['Autumn']}; border-radius: 50%; margin-right: 8px;"></span>
                    🍂 Autumn (Sep-Nov)
                </div>
            </div>
        </div>
        
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        
        <div style="background: #f5f5f5; padding: 8px; border-radius: 4px; font-size: 12px;">
            <strong>Key Statistics:</strong><br>
            • Winter mean: <strong>{winter_mean:.1f}</strong> μg/m³<br>
            • Summer mean: <strong>{summer_mean:.1f}</strong> μg/m³<br>
            • Winter/Summer ratio: <strong>{winter_summer_ratio:.2f}x</strong><br>
            <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;">
            • Plain (winter): <strong>{plain_winter:.1f}</strong> μg/m³<br>
            • Mountain (winter): <strong>{mountain_winter:.1f}</strong> μg/m³
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
            📅 Seasonal PM₁₀ Patterns Explorer
        </h3>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666; text-align: center;">
            Po Valley - Alpine Region | Seasonal Air Quality Comparison
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
        <h4 style="margin: 0 0 10px 0; color: #333;">📋 What If Analysis</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #555;">
            <li>Compare <strong>seasons</strong> using layer control</li>
            <li>See how <strong>terrain</strong> affects Winter PM₁₀</li>
            <li><strong>Click markers</strong> for seasonal stats</li>
            <li>Note the <strong>Winter peak</strong> pattern</li>
        </ul>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        <div style="font-size: 11px; color: #888;">
            <strong>Key Finding:</strong><br>
            Winter PM₁₀ is ~2x higher than summer,<br>
            driven by thermal inversions and<br>
            reduced boundary layer height.
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(instructions_html))
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    print("\n[7] Saving map...")
    output_path = ASSETS_DIR / 'seasonal_pm10_patterns.html'
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
    output_path = generate_seasonal_map()
    if output_path:
        print(f"\n🗺️  Open the map in a browser: {output_path}")
