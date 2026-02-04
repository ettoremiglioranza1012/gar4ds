import polars as pl
import pandas as pd
import numpy as np
import pyarrow
import json
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import KNN, DistanceBand
from esda.moran import Moran, Moran_Local
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster
from pathlib import Path
import sys

# Create output directories if they don't exist
ASSETS_DIR = Path(__file__).parent.parent.parent / 'assets'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
ASSETS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Set up output file for text results
output_file = RESULTS_DIR / 'spatial_analysis_results.txt'

class Tee:
    """Helper class to write to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Open output file and redirect stdout
with open(output_file, 'w') as f:
    # Create a Tee object that writes to both stdout and file
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    
    try:
        # 1. Load Data
        print("Loading data...")
        spatial_df = pl.read_parquet('../../data/spatial_pollution_matrix.parquet')
        with open('../../data/data_stations_metadata.json', 'r') as meta_file:
            stations_meta = json.load(meta_file)
        
        # Convert metadata to a DataFrame for easy alignment
        meta_df = pd.DataFrame(stations_meta).set_index('Station_ID')
        
        # 2. Align Data and Coordinates
        # Extract the list of stations from the pollution matrix (excluding 'Data' column)
        data_stations = [c for c in spatial_df.columns if c != 'Data']
        
        # Intersection of stations in both Data and Metadata
        valid_stations = [s for s in data_stations if s in meta_df.index]
        print(f"Aligned {len(valid_stations)} stations for spatial analysis.")
        
        # Extract the pollution vector (y) - Average PM10 over the entire period
        # (You can also filter by specific dates, e.g., Winter 2024)
        pm10_means = spatial_df.select(valid_stations).mean().to_pandas().iloc[0]
        coords = meta_df.loc[valid_stations, ['Longitude', 'Latitude']].values
        
        # 3. Define Spatial Weights Matrix (W)
        # We use K-Nearest Neighbors (k=6) to ensure every station has connections
        # Alternatively, use DistanceBand for a fixed radius (e.g., 50km)
        w = KNN.from_array(coords, k=6)
        w.transform = 'r'  # Row-standardize weights
        
        # 4. Global Moran's I
        # Tests the null hypothesis of spatial randomness
        mi = Moran(pm10_means, w)
        print(f"\n--- Global Moran's I ---")
        print(f"Index: {mi.I:.4f}")
        print(f"P-value: {mi.p_sim:.4f}")
        if mi.p_sim < 0.05:
            print("Result: Significant spatial clustering detected.")
        else:
            print("Result: No significant spatial pattern (random distribution).")
        
        # 5. Local Moran's I (LISA)
        # Identifies specific clusters (High-High, Low-Low) and outliers
        lisa = Moran_Local(pm10_means, w)
        
        # Classify clusters (at p < 0.05 significance)
        # 1=HH, 2=LH, 3=LL, 4=HL
        sig = lisa.p_sim < 0.05
        hotspots = sig & (lisa.q == 1)  # High-High
        coldspots = sig & (lisa.q == 3) # Low-Low
        
        print(f"\n--- Local Spatial Clusters ---")
        print(f"High-High Clusters (Hotspots): {hotspots.sum()} stations")
        print(f"Low-Low Clusters (Coldspots): {coldspots.sum()} stations")
        
        # 6. Visualization
        plt.figure(figsize=(14, 10))
        
        # Base plot of all stations
        plt.scatter(meta_df.loc[valid_stations, 'Longitude'], 
                    meta_df.loc[valid_stations, 'Latitude'], 
                    c='lightgrey', edgecolor='k', s=50, label='Not Significant')
        
        # Plot Hotspots (Red)
        if hotspots.sum() > 0:
            plt.scatter(coords[hotspots, 0], coords[hotspots, 1], 
                        c='red', edgecolor='k', s=100, label='High-High (Pollution Cluster)')
            # Annotate HH stations
            for idx in np.where(hotspots)[0]:
                plt.annotate(valid_stations[idx], (coords[idx, 0], coords[idx, 1]), fontsize=9)
        
        # Plot Coldspots (Blue)
        if coldspots.sum() > 0:
            plt.scatter(coords[coldspots, 0], coords[coldspots, 1], 
                        c='blue', edgecolor='k', s=100, label='Low-Low (Clean Cluster)')
        
        plt.title('Spatial Clustering of PM10 Concentrations (LISA Analysis)', fontsize=15)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save the plot
        plot_path = ASSETS_DIR / 'spatial_clustering_lisa.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n[INFO] Plot saved to: {plot_path}")
        
        # Also show the plot (comment out if running headless)
        # plt.show()
        plt.close()
        
        # 7. Create Results DataFrame
        results_df = pd.DataFrame({
            'Station': valid_stations,
            'PM10_Mean': pm10_means.values,
            'Is_Hotspot': hotspots,
            'Is_Coldspot': coldspots,
            'Local_Moran_I': lisa.Is
        }).sort_values('Local_Moran_I', ascending=False)
        
        print("\nTop 5 Stations contributing to Clustering:")
        print(results_df.head(5))
        
        # Save results DataFrame to CSV
        results_csv_path = RESULTS_DIR / 'spatial_clustering_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n[INFO] Results DataFrame saved to: {results_csv_path}")
        
        print(f"\n[INFO] Text output saved to: {output_file}")
        print("\n=== Analysis Complete ===")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout

print(f"Script completed successfully. Check {RESULTS_DIR} for outputs and {ASSETS_DIR} for plots.")
