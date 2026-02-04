import polars as pl
import os
import sys

# Configuration
INPUT_FILE = "../../data/spatial_full_matrix.parquet"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    print(f"Loading {INPUT_FILE}...\n")
    try:
        df = pl.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"[CRITICAL] Failed to load parquet: {e}")
        return

    # =========================================================================
    # 1. AI-READABLE SCHEMA OVERVIEW
    # =========================================================================
    print("--- DATASET METADATA (AI CONTEXT) ---")
    print(f"Row_Count: {df.height}")
    print(f"Col_Count: {df.width}")
    
    # Time Range extraction
    if "Data" in df.columns:
        try:
            # Check strictly if it's string or datetime
            if df["Data"].dtype == pl.Utf8:
                dates = df.select(pl.col("Data").str.to_datetime(strict=False)).sort("Data")
            else:
                dates = df.select("Data").sort("Data")
                
            start = dates.row(0)[0]
            end = dates.row(df.height-1)[0]
            print(f"Time_Start: {start}")
            print(f"Time_End:   {end}")
        except Exception as e:
            print(f"Time_Range: Error parsing 'Data' column ({e})")

    # =========================================================================
    # 2. STATION & VARIABLE MATRIX (COMPACT)
    # =========================================================================
    print("\n--- DETECTED STATIONS & VARIABLES ---")
    
    # Logic to group columns: {Station: [List of Vars]}
    # Assumes format: VAR_StationName (e.g., PM10_Borgo Valsugana)
    prefixes = ["PM10", "TEMP", "BLH", "WS", "PRECIP", "PRESS", "RAD"]
    station_map = {}

    for col in df.columns:
        if col in ["Data", "Stazione_APPA"]: continue
        
        # Heuristic to split Prefix_Station
        parts = col.split('_', 1)
        if len(parts) == 2 and parts[0] in prefixes:
            var, station = parts[0], parts[1]
            if station not in station_map:
                station_map[station] = []
            station_map[station].append(var)

    # Print in a format an AI can easily read to understand coverage
    # Format: StationName | Var1, Var2, Var3...
    print(f"{'STATION_KEY':<45} | {'AVAILABLE_VARIABLES'}")
    print("-" * 80)
    
    sorted_stations = sorted(station_map.keys())
    for s in sorted_stations:
        vars_found = sorted(station_map[s])
        print(f"{s:<45} | {', '.join(vars_found)}")

    # =========================================================================
    # 3. RAW COLUMN LIST (SAMPLED)
    # =========================================================================
    # Printing 600+ columns is bad for tokens. We print a structured sample.
    print("\n--- RAW COLUMN STRUCTURE (SAMPLE) ---")
    print("Format: [Index] ColumnName (DataType)")
    
    all_cols = df.columns
    dtypes = df.dtypes
    
    # Print first 10 (Metadata/Key columns)
    for i in range(min(10, len(all_cols))):
        print(f"[{i:03}] {all_cols[i]:<40} ({dtypes[i]})")
        
    if len(all_cols) > 20:
        print(f"... (Skipping {len(all_cols) - 20} intermediate columns) ...")
        
    # Print last 10 (To verify end of wide format)
    for i in range(max(10, len(all_cols)-10), len(all_cols)):
        print(f"[{i:03}] {all_cols[i]:<40} ({dtypes[i]})")

    # =========================================================================
    # 4. DATA SAMPLE (HEAD)
    # =========================================================================
    print("\n--- DATA PREVIEW (FIRST 3 ROWS) ---")
    # Set Polars to print wide so the AI sees the values
    pl.Config.set_tbl_rows(3)
    pl.Config.set_tbl_cols(10) # Limit width to avoid line wrapping mess
    print(df.head(3))

if __name__ == "__main__":
    main()