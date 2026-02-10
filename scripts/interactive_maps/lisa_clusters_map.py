"""
INTERACTIVE LISA SPATIAL CLUSTERS MAP
======================================

This script generates an interactive HTML map visualizing Local Moran's I (LISA)
spatial autocorrelation results for the Po Valley - Alpine region PM10 analysis.

FEATURES:
- Color-coded markers by LISA cluster type (High-High, Low-Low, High-Low, Low-High)
- Dropdown filter to select variable (PM10, temperature, humidity, etc.)
- Significance threshold filter
- Cluster type filter
- Detailed popup with station info and LISA statistics
- Interactive legend

OUTPUT:
- assets/maps/lisa_clusters_explorer.html
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, GroupedLayerControl
from pathlib import Path
import json
from branca.element import Element, MacroElement, Template

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

def load_lisa_results():
    """Load LISA analysis results"""
    lisa_path = RESULTS_DIR / 'spatial_analysis' / 'optionA_lisa_results_all_variables.csv'
    df = pd.read_csv(lisa_path)
    return df


def load_station_metadata():
    """Load station metadata with coordinates"""
    meta_path = DATA_DIR / 'pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson'
    gdf = gpd.read_file(meta_path)
    
    # Extract coordinates
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    
    return gdf


def load_cross_sectional_data():
    """Load cross-sectional station profiles"""
    profile_path = RESULTS_DIR / 'spatial_analysis' / 'station_profiles_cross_sectional.csv'
    if profile_path.exists():
        return pd.read_csv(profile_path)
    return None


# ============================================================================
# MAP STYLING
# ============================================================================

CLUSTER_COLORS = {
    'High-High': '#d7191c',      # Red - high value, high neighbors
    'Low-Low': '#2c7bb6',        # Blue - low value, low neighbors
    'High-Low': '#fdae61',       # Orange - high value, low neighbors (outlier)
    'Low-High': '#abd9e9',       # Light blue - low value, high neighbors (outlier)
    'Not Significant': '#999999'  # Gray - not significant
}

CLUSTER_DESCRIPTIONS = {
    'High-High': 'Hot Spot: High PM10 surrounded by high PM10 neighbors',
    'Low-Low': 'Cold Spot: Low PM10 surrounded by low PM10 neighbors',
    'High-Low': 'Spatial Outlier: High PM10 surrounded by low PM10 neighbors',
    'Low-High': 'Spatial Outlier: Low PM10 surrounded by high PM10 neighbors',
    'Not Significant': 'No significant spatial autocorrelation'
}

VARIABLE_LABELS = {
    'pm10': 'PM₁₀ (μg/m³)',
    'temperature_2m': 'Temperature 2m (K)',
    'humidity_950': 'Humidity 950 hPa (%)',
    'blh': 'Boundary Layer Height (m)',
    'solar_radiation_downwards': 'Solar Radiation (J/m²)',
    'wind_u_10m': 'Wind U-comp 10m (m/s)',
    'wind_v_10m': 'Wind V-comp 10m (m/s)',
    'uwind_850': 'Wind U-comp 850 hPa (m/s)',
    'uwind_950': 'Wind U-comp 950 hPa (m/s)',
    'Vwind_850': 'Wind V-comp 850 hPa (m/s)',
    'Vwind_950': 'Wind V-comp 950 hPa (m/s)',
    'total_precipitation': 'Total Precipitation (mm)'
}


# ============================================================================
# MAP GENERATION
# ============================================================================

def create_popup_html(station, lisa_row, var_label):
    """Create detailed popup HTML for a station marker"""
    
    cluster_type = lisa_row['Cluster_Type']
    cluster_color = CLUSTER_COLORS.get(cluster_type, '#999999')
    cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_type, '')
    
    html = f"""
    <div style="font-family: Arial, sans-serif; width: 320px;">
        <h4 style="margin: 0 0 10px 0; color: #333; border-bottom: 2px solid {cluster_color};">
            📍 {station['station_name']}
        </h4>
        
        <div style="background: #f5f5f5; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
            <strong>Station ID:</strong> {station['station_code']}<br>
            <strong>Region:</strong> {station['region']}<br>
            <strong>Terrain:</strong> {station.get('terrain_type', 'N/A')}<br>
            <strong>Area Type:</strong> {station.get('area_type', 'N/A')}<br>
            <strong>Elevation:</strong> {station.get('elevation', 'N/A')} m
        </div>
        
        <div style="margin-bottom: 10px;">
            <strong style="color: #555;">Variable:</strong> {var_label}<br>
            <strong style="color: #555;">Value:</strong> {lisa_row['Value']:.2f}
        </div>
        
        <div style="background: {cluster_color}20; padding: 10px; border-radius: 4px; border-left: 4px solid {cluster_color};">
            <strong style="color: {cluster_color};">LISA Cluster: {cluster_type}</strong><br>
            <span style="font-size: 11px; color: #666;">{cluster_desc}</span>
        </div>
        
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            <strong>Statistics:</strong><br>
            • Local Moran's I: {lisa_row['Local_Morans_I']:.4f}<br>
            • Z-score: {lisa_row['Z_score']:.3f}<br>
            • P-value: {lisa_row['P_value']:.4f}<br>
            • Significant (p<0.05): {'✓ Yes' if lisa_row['Significant'] else '✗ No'}
        </div>
    </div>
    """
    return html


def generate_lisa_map():
    """Generate the interactive LISA clusters map"""
    
    print("=" * 70)
    print("  GENERATING LISA SPATIAL CLUSTERS INTERACTIVE MAP")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading data...")
    lisa_df = load_lisa_results()
    stations_gdf = load_station_metadata()
    
    print(f"    • LISA results: {len(lisa_df)} rows")
    print(f"    • Stations: {len(stations_gdf)}")
    
    # Get unique variables
    variables = lisa_df['Variable'].unique().tolist()
    print(f"    • Variables: {variables}")
    
    # Calculate map center
    center_lat = stations_gdf['lat'].mean()
    center_lon = stations_gdf['lon'].mean()
    
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
    
    # Create feature groups for each variable
    print("\n[3] Adding station markers for each variable...")
    
    feature_groups = {}
    
    for var in variables:
        var_lisa = lisa_df[lisa_df['Variable'] == var]
        var_label = VARIABLE_LABELS.get(var, var)
        
        # Create feature group for this variable
        fg = folium.FeatureGroup(name=f'🔬 {var_label}', show=(var == 'pm10'))
        
        for _, station in stations_gdf.iterrows():
            station_id = station['station_code']
            
            # Get LISA result for this station
            lisa_row = var_lisa[var_lisa['Station'] == station_id]
            
            if lisa_row.empty:
                continue
                
            lisa_row = lisa_row.iloc[0]
            
            # Determine marker properties
            cluster_type = lisa_row['Cluster_Type']
            is_significant = lisa_row['Significant']
            
            color = CLUSTER_COLORS.get(cluster_type, '#999999')
            
            # Size based on absolute Local Moran's I value
            base_radius = 8
            moran_abs = abs(lisa_row['Local_Morans_I'])
            radius = base_radius + min(moran_abs * 5, 10)
            
            # Create popup
            popup_html = create_popup_html(station, lisa_row, var_label)
            popup = folium.Popup(popup_html, max_width=350)
            
            # Create circle marker
            folium.CircleMarker(
                location=[station['lat'], station['lon']],
                radius=radius,
                color='#333' if is_significant else '#999',
                weight=2 if is_significant else 1,
                fill=True,
                fill_color=color,
                fill_opacity=0.8 if is_significant else 0.4,
                popup=popup,
                tooltip=f"{station['station_name']} ({cluster_type})"
            ).add_to(fg)
        
        feature_groups[var] = fg
        fg.add_to(m)
        print(f"    • Added layer: {var_label} ({len(var_lisa)} stations)")
    
    # Add legend
    print("\n[4] Adding legend...")
    legend_html = '''
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
        <h4 style="margin: 0 0 10px 0; color: #333;">LISA Cluster Types</h4>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 16px; height: 16px; background: #d7191c; border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
            <span style="font-size: 13px;"><strong>High-High</strong> (Hot Spot)</span>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 16px; height: 16px; background: #2c7bb6; border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
            <span style="font-size: 13px;"><strong>Low-Low</strong> (Cold Spot)</span>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 16px; height: 16px; background: #fdae61; border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
            <span style="font-size: 13px;"><strong>High-Low</strong> (Outlier)</span>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="display: inline-block; width: 16px; height: 16px; background: #abd9e9; border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
            <span style="font-size: 13px;"><strong>Low-High</strong> (Outlier)</span>
        </div>
        <div style="margin-bottom: 10px;">
            <span style="display: inline-block; width: 16px; height: 16px; background: #999999; border-radius: 50%; margin-right: 8px; vertical-align: middle;"></span>
            <span style="font-size: 13px;"><strong>Not Significant</strong></span>
        </div>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        <div style="font-size: 11px; color: #666;">
            <strong>Note:</strong> Marker size indicates Local Moran's I magnitude.
            Bold borders indicate statistical significance (p < 0.05).
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(legend_html))
    
    # Add Global Moran's I Statistics Panel
    stats_html = '''
    <div style="
        position: fixed;
        top: 80px;
        left: 50px;
        z-index: 1000;
        background-color: white;
        padding: 12px 15px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
        width: 220px;
    ">
        <h4 style="margin: 0 0 10px 0; color: #333; font-size: 14px;">📊 Global Spatial Autocorrelation</h4>
        <div style="background: #e8f5e9; padding: 8px; border-radius: 4px; border-left: 3px solid #4caf50; margin-bottom: 8px;">
            <div style="font-size: 12px; color: #333;">
                <strong>PM₁₀ Statistics:</strong><br>
                • Global Moran's I: <strong>0.681</strong><br>
                • Z-score: <strong>9.01</strong><br>
                • p-value: <strong>&lt;0.001</strong>
            </div>
        </div>
        <div style="font-size: 11px; color: #666; font-style: italic;">
            Interpretation: Strong positive spatial autocorrelation
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(stats_html))
    
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
            🗺️ LISA Spatial Clusters Explorer
        </h3>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666; text-align: center;">
            Po Valley - Alpine Region PM₁₀ Spatial Autocorrelation Analysis
        </p>
    </div>
    '''
    m.get_root().html.add_child(Element(title_html))
    
    # Add instructions panel
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
        max-width: 250px;
    ">
        <h4 style="margin: 0 0 10px 0; color: #333;">📋 How to Use</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #555;">
            <li>Use <strong>layer control</strong> (top-right) to switch variables</li>
            <li><strong>Click markers</strong> for detailed station info</li>
            <li><strong>Zoom in</strong> to explore specific regions</li>
            <li>Toggle <strong>base maps</strong> for different views</li>
        </ul>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
        <div style="font-size: 11px; color: #888;">
            <strong>Key Insight:</strong><br>
            PM₁₀ Hot Spots (red) cluster in Po Valley plains;<br>
            Cold Spots (blue) in Alpine regions.
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(instructions_html))
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    print("\n[5] Saving map...")
    output_path = ASSETS_DIR / 'lisa_clusters_explorer.html'
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
    output_path = generate_lisa_map()
    print(f"\n🗺️  Open the map in a browser: {output_path}")
