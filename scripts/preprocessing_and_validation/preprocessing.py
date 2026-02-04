import polars as pl
import numpy as np
import os

# Configuration
INPUT_FILE = "../../data/dataset.parquet"
OUTPUT_DIR = "../../data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "spatial_full_matrix.parquet")

def preprocess_data():
    print(f"Loading data from {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)

    # =========================================================================
    # 1. PROCESS APPA STATIONS (Long Format -> Wide)
    # =========================================================================
    # Select only relevant columns for APPA stations
    # Note: 'Temperatura_(°C)' values look like Kelvin (>250), so we treat them as such or standardize later.
    appa_cols = {
        "PM10_(ug.m-3)": "PM10",
        "Temperatura_(°C)": "TEMP",
        "blh_mean_daily": "BLH",
        "Vel_Vento_media_(m/s)": "WS",
        "Precipitazione_(mm)": "PRECIP",
        "Pressione_Atm_(hPa)": "PRESS",
        "Radiaz_Solare_tot_(kJ/m2)": "RAD"
    }
    
    print("Processing APPA Stations...")
    
    # Filter for rows where we have APPA station names
    appa_long = df.filter(pl.col("Stazione_APPA").is_not_null()).select(
        ["Data", "Stazione_APPA"] + list(appa_cols.keys())
    ).rename(appa_cols)

    # Pivot each variable to create columns like "PM10_Borgo Valsugana", "TEMP_Borgo Valsugana"
    appa_wide = None
    
    for var in appa_cols.values():
        pivot = appa_long.pivot(
            on="Stazione_APPA",
            index="Data",
            values=var,
            aggregate_function="mean"
        )
        # Rename columns to {VAR}_{Station}
        rename_map = {c: f"{var}_{c}" for c in pivot.columns if c != "Data"}
        pivot = pivot.rename(rename_map)
        
        if appa_wide is None:
            appa_wide = pivot
        else:
            appa_wide = appa_wide.join(pivot, on="Data", how="full", coalesce=True)

    # =========================================================================
    # 2. PROCESS EXTERNAL STATIONS (Wide Columns -> Standardized Wide)
    # =========================================================================
    print("Processing External Stations...")
    
    # Identify external stations based on 'pm10_' prefix
    ext_pm10_cols = [c for c in df.columns if c.lower().startswith("pm10_") and "PM10" not in c[:4]] 
    # Logic: "pm10_" prefix but ignore the main "PM10_(ug.m-3)" if checks were looser
    
    # Extract station names from "pm10_StationName"
    ext_stations = [c.replace("pm10_", "") for c in ext_pm10_cols]
    
    # Since external data repeats for every APPA row, we group by Data and take the first/mean
    ext_data = df.group_by("Data").first()
    
    ext_exprs = []
    
    for station in ext_stations:
        # PM10
        ext_exprs.append(pl.col(f"pm10_{station}").alias(f"PM10_{station}"))
        
        # Temperature (Map temperature_2m -> TEMP)
        if f"temperature_2m_{station}" in df.columns:
            ext_exprs.append(pl.col(f"temperature_2m_{station}").alias(f"TEMP_{station}"))
            
        # BLH
        if f"blh_{station}" in df.columns:
            ext_exprs.append(pl.col(f"blh_{station}").alias(f"BLH_{station}"))
            
        # Precipitation
        if f"total_precipitation_{station}" in df.columns:
            ext_exprs.append(pl.col(f"total_precipitation_{station}").alias(f"PRECIP_{station}"))

        # --- NEW: Pressure & Radiation ---
        if f"surface_pressure_{station}" in df.columns:
            ext_exprs.append(pl.col(f"surface_pressure_{station}").alias(f"PRESS_{station}"))
            
        if f"solar_radiation_downwards_{station}" in df.columns:
            ext_exprs.append(pl.col(f"solar_radiation_downwards_{station}").alias(f"RAD_{station}"))
            
        # Wind Speed (Calculate from U and V if WS not explicit)
        # Check for different casing/naming conventions in schema
        u_col = f"wind_u_10m_{station}"
        v_col = f"wind_v_10m_{station}"
        
        if u_col in df.columns and v_col in df.columns:
            # WS = sqrt(u^2 + v^2)
            ext_exprs.append(
                ((pl.col(u_col)**2 + pl.col(v_col)**2).sqrt()).alias(f"WS_{station}")
            )
            
    ext_wide = ext_data.select(["Data"] + ext_exprs)

    # =========================================================================
    # 3. MERGE AND SAVE
    # =========================================================================
    print("Merging datasets...")
    full_matrix = appa_wide.join(ext_wide, on="Data", how="full").sort("Data")
    
    print(f"Saving to {OUTPUT_FILE}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    full_matrix.write_parquet(OUTPUT_FILE)
    
    print("--- Processing Complete ---")
    print(f"Shape: {full_matrix.shape}")
    print("Columns Example:", full_matrix.columns[:10])

if __name__ == "__main__":
    preprocess_data()