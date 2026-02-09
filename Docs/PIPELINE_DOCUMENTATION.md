# GAR4DS Data Processing Pipeline
**Complete Chain of Dependencies**

Generated: 2026-02-06

---

## Pipeline Overview

```
data_preprocessing.py
    ↓ produces: pm10_era5_land_era5_reanalysis_blh.parquet
    ↓           pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson
    ↓
build_panel_matrix.py
    ↓ produces: panel_data_matrix.parquet (20 variables)
    ↓
add_elevation_data.py
    ↓ produces: pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson
    ↓
multicollinearity_analysis.py
    ↓ produces: VIF analysis, correlation matrices, PCA results
    ↓
filter_multicollinearity.py
    ↓ produces: panel_data_matrix_filtered_for_collinearity.parquet (12 variables)
    ↓
    ├──→ exploratory_data_analysis.py
    │    └── produces: EDA results, correlation heatmaps, temporal analysis
    │
    └──→ spatial_analysis_new.py
         └── produces: spatial weights (KNN6.gal), Moran's I, LISA clusters
```

---

## Detailed Pipeline Steps

### 1. data_preprocessing.py
**Location:** `scripts/preprocessing/data_preprocessing.py`  
**Purpose:** Convert raw CSV to efficient formats

**Inputs:**
- `data/pm10_era5_land_era5_reanalysis_blh.csv`
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv`

**Outputs:**
- `data/pm10_era5_land_era5_reanalysis_blh.parquet`
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`
- `results/dataset_documentation/dataset_info_*.txt`

**Run command:**
```bash
uv run scripts/preprocessing/data_preprocessing.py
```

---

### 2. build_panel_matrix.py
**Location:** `scripts/preprocessing/build_panel_matrix.py`  
**Purpose:** Create panel data matrix with MultiIndex (week_start, station_id)

**Inputs:**
- `data/pm10_era5_land_era5_reanalysis_blh.parquet`

**Outputs:**
- `data/panel_data_matrix.parquet` (21,275 obs × 20 vars)
- `results/dataset_documentation/panel_matrix_info_*.txt`

**Variables included (20):**
- Target: pm10
- Temperature: temperature_550, temperature_850, temperature_950, temperature_2m
- Humidity: humidity_550, humidity_850, humidity_950
- Wind U: uwind_550, uwind_850, uwind_950, wind_u_10m
- Wind V: Vwind_550, Vwind_850, Vwind_950, wind_v_10m
- Surface: surface_pressure, blh, total_precipitation, solar_radiation_downwards

**Run command:**
```bash
uv run scripts/preprocessing/build_panel_matrix.py
```

---

### 3. add_elevation_data.py
**Location:** `scripts/preprocessing/add_elevation_data.py`  
**Purpose:** Add elevation data and terrain classification using Open-Elevation API

**Inputs:**
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`

**Outputs:**
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson`
- Enhanced with: elevation, terrain_type (plain/hills/mountain), area_type (urban/suburban/rural)

**API:** Open-Elevation (no key required)

**Terrain thresholds:**
- Plain: -10 to 300m
- Hills: 300 to 600m
- Mountain: >600m

**Run command:**
```bash
uv run scripts/preprocessing/add_elevation_data.py
```

---

### 4. multicollinearity_analysis.py
**Location:** `scripts/preprocessing/multicollinearity_analysis.py`  
**Purpose:** Analyze VIF, correlations, and PCA to identify redundant variables

**Inputs:**
- `data/panel_data_matrix.parquet`

**Outputs:**
- `results/multicollinearity_analysis/vif_analysis_*.csv`
- `results/multicollinearity_analysis/recommendations_*.txt`
- `assets/multicollinearity_analysis/correlation_heatmap_*.png`
- `assets/multicollinearity_analysis/pca_scree_*.png`

**Key findings:**
- Temperature levels: VIF > 50,000 (extreme collinearity)
- Humidity levels: VIF 35-340
- Recommendation: Drop 8 variables (40% reduction)

**Run command:**
```bash
uv run scripts/preprocessing/multicollinearity_analysis.py
```

---

### 5. filter_multicollinearity.py
**Location:** `scripts/preprocessing/filter_multicollinearity.py`  
**Purpose:** Create filtered dataset based on multicollinearity analysis

**Inputs:**
- `data/panel_data_matrix.parquet` (20 variables)

**Outputs:**
- `data/panel_data_matrix_filtered_for_collinearity.parquet` (12 variables)
- `results/dataset_documentation/multicollinearity_filter_*.txt`

**Variables KEPT (12):**
1. pm10 (target)
2. temperature_2m (surface temperature)
3. humidity_950 (near-surface humidity)
4. blh (boundary layer height)
5. solar_radiation_downwards
6. wind_u_10m, wind_v_10m (surface winds)
7. uwind_850, uwind_950 (upper-level U winds)
8. Vwind_850, Vwind_950 (upper-level V winds)
9. total_precipitation

**Variables DROPPED (8):**
- temperature_550, temperature_850, temperature_950 (VIF > 50k)
- surface_pressure (VIF = 2,335)
- humidity_550, humidity_850 (VIF > 35)
- uwind_550, Vwind_550 (above boundary layer)

**Run command:**
```bash
uv run scripts/preprocessing/filter_multicollinearity.py
```

---

### 6. exploratory_data_analysis.py
**Location:** `scripts/data_analysis/exploratory_data_analysis.py`  
**Purpose:** Validate physical relationships and data quality

**Inputs:**
- `data/panel_data_matrix_filtered_for_collinearity.parquet` (12 vars)
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson`

**Outputs:**
- `results/eda_analysis/eda_analysis.txt`
- `assets/eda_analysis/correlation_heatmap.png`
- `assets/eda_analysis/terrain_comparison_boxplots.png`
- `assets/eda_analysis/temporal_seasonality_analysis.png`

**Key validations:**
- PM10 vs BLH: r = -0.47 ✓ (expected negative)
- Terrain effects confirmed
- Winter peaks verified

**Run command:**
```bash
uv run scripts/data_analysis/exploratory_data_analysis.py
```

---

### 7. spatial_analysis.py
**Location:** `scripts/data_analysis/spatial_analysis.py`  
**Purpose:** Spatial autocorrelation analysis and weights matrix generation

**Inputs:**
- `data/panel_data_matrix_filtered_for_collinearity.parquet` (12 vars)
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`

**Outputs:**
- `weights/spatial_weights_knn6.gal` ← **CRITICAL FOR SDM**
- `results/spatial_analysis/optionA_global_morans_I_by_variable.csv`
- `results/spatial_analysis/optionA_lisa_results_all_variables.csv`
- `results/spatial_analysis/optionC_multivariate_clusters.csv`
- `results/spatial_analysis/optionC_cluster_profiles.csv`
- `assets/maps/lisa_clusters_explorer.html` (interactive map)
- `assets/maps/seasonal_pm10_patterns.html` (interactive map)

**Key findings:**
- PM10 Moran's I: 0.68 (strong spatial clustering)
- 11 PM10 hotspots identified
- 5 atmospheric regimes detected (multivariate clustering)

**Run command:**
```bash
uv run scripts/data_analysis/spatial_analysis.py
```

---

### 8. model_specification_tests.py
**Location:** `scripts/data_analysis/model_specification_tests.py`  
**Purpose:** Test which spatial model specification fits best (SLX, SAR, SEM, SDM, etc.)

**Inputs:**
- `data/panel_data_matrix_filtered_for_collinearity.parquet`
- `weights/spatial_weights_knn6.gal`

**Outputs:**
- `results/model_specification_tests/lrt_test_results.csv`
- `results/model_specification_tests/information_criteria.csv`
- `results/model_specification_tests/specification_test_summary.csv`
- `results/model_specification_tests/lrt_tests_log_*.txt`

**Key findings:**
- SDM (Spatial Durbin Model) strongly preferred
- Captures both direct and indirect (spillover) effects
- Properly handles spatial autocorrelation

**Run command:**
```bash
uv run scripts/data_analysis/model_specification_tests.py
```

---

### 9. spatial_durbin_model.py ⭐
**Location:** `scripts/data_analysis/spatial_durbin_model.py`  
**Purpose:** Fit Panel Spatial Durbin Model with regime-stratified analysis

**Inputs:**
- `data/panel_data_matrix_filtered_for_collinearity.parquet` (12 vars)
- `weights/spatial_weights_knn6.gal`
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`
- `results/spatial_analysis/optionC_multivariate_clusters.csv` (5 atmospheric regimes)

**Model Specification:**
```
log(PM10) = ρWy + Xβ + WXθ + αᵢ + γₜ + εᵢₜ
```
Where:
- ρ = spatial autoregressive parameter (endogenous spillover)
- β = direct effect coefficients (11 meteorological variables)
- θ = indirect effect coefficients (neighbor spillover)
- αᵢ = station fixed effects, γₜ = time fixed effects

**Outputs:**
- **Global Model:**
  - `results/spatial_durbin_model/model_summary.txt`
  - `results/spatial_durbin_model/coefficients_table.csv`
  - `results/spatial_durbin_model/spillover_decomposition_observations.csv`

- **Cluster-Specific Models (5 atmospheric regimes):**
  - `results/spatial_durbin_model/cluster_0_model_summary.txt`
  - `results/spatial_durbin_model/cluster_0_coefficients.csv`
  - ... (repeated for clusters 1-4)

- **Combined Analysis:**
  - `results/spatial_durbin_model/regime_comparison.csv`
  - `results/spatial_durbin_model/all_clusters_coefficients_combined.csv`

- **Visualizations:**
  - `assets/spatial_durbin_model/coefficient_forest_plot.png`
  - `assets/spatial_durbin_model/residual_qq_plot.png`

**Key Features (Updated 9 Feb 2026):**
- Fits separate SDM models for ALL 5 atmospheric clusters
- Tests hypothesis: global model averages over distinct physical regimes
- Preserves temporal dynamics (no mean aggregation)
- Comprehensive spillover decomposition: Direct (Xβ) + Indirect (WXθ) + Endogenous (ρWy)

**Run command:**
```bash
uv run scripts/data_analysis/spatial_durbin_model.py
```

---

## Pipeline Verification

### Check all dependencies exist:
```bash
ls -lh data/panel_data_matrix.parquet
ls -lh data/panel_data_matrix_filtered_for_collinearity.parquet
ls -lh data/*metadata*elevation.geojson
ls -lh weights/spatial_weights_knn6.gal
```

### Expected file sizes:
- panel_data_matrix.parquet: ~2.0 MB (20 variables)
- panel_data_matrix_filtered_for_collinearity.parquet: ~1.9 MB (12 variables)
- metadata_with_elevation.geojson: ~10 KB
- spatial_weights_knn6.gal: ~3 KB

### Variable reduction summary:
- Original (raw CSV): ~30 columns
- After panel matrix: 20 variables (cleaned)
- After collinearity filter: 12 variables (40% reduction)
- All VIF < 10 (except BLH = 22, theoretically justified)

---

## Run Complete Pipeline

To execute the entire pipeline from scratch:

```bash
# Navigate to project
cd /Users/ettoremiglioranza/Projects/gar4ds

# ═══════════════════════════════════════════════════════════
# PHASE 1: PREPROCESSING
# ═══════════════════════════════════════════════════════════

# 1. Convert raw CSV to efficient formats
uv run scripts/preprocessing/data_preprocessing.py

# 2. Build panel matrix (20 variables)
uv run scripts/preprocessing/build_panel_matrix.py

# 3. Add elevation data (optional, for terrain analysis)
uv run scripts/preprocessing/add_elevation_data.py

# 4. Analyze multicollinearity
uv run scripts/preprocessing/multicollinearity_analysis.py

# 5. Filter dataset (12 variables)
uv run scripts/preprocessing/filter_multicollinearity.py

# ═══════════════════════════════════════════════════════════
# PHASE 2: EXPLORATORY & SPATIAL ANALYSIS
# ═══════════════════════════════════════════════════════════

# 6. Exploratory data analysis
uv run scripts/data_analysis/exploratory_data_analysis.py

# 7. Spatial analysis (creates spatial weights matrix)
uv run scripts/data_analysis/spatial_analysis.py

# ═══════════════════════════════════════════════════════════
# PHASE 3: SPATIAL ECONOMETRIC MODELING
# ═══════════════════════════════════════════════════════════

# 8. Model specification tests (determines SDM is best)
uv run scripts/data_analysis/model_specification_tests.py

# 9. Spatial Durbin Model (MAIN ANALYSIS)
uv run scripts/data_analysis/spatial_durbin_model.py
```

**Alternative: Use Makefile targets (see Makefile in project root)**

```bash
# Run entire pipeline
make all

# Or run specific phases
make preprocessing
make analysis

# Clean outputs
make clean-results
make clean-all
```

---

## Pipeline Dependencies Graph

```
data_preprocessing.py
    ↓ pm10_era5_land_era5_reanalysis_blh.parquet
    ↓ pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson
    ↓
build_panel_matrix.py
    ↓ panel_data_matrix.parquet (20 vars)
    ↓
add_elevation_data.py (optional)
    ↓ metadata_with_elevation.geojson
    ↓
multicollinearity_analysis.py
    ↓ VIF analysis, correlations, PCA
    ↓
filter_multicollinearity.py
    ↓ panel_data_matrix_filtered_for_collinearity.parquet (12 vars)
    ↓
    ├──→ exploratory_data_analysis.py
    │    └── EDA results, visualizations
    │
    └──→ spatial_analysis.py
         ↓ spatial_weights_knn6.gal + clusters
         ↓
         ├──→ model_specification_tests.py
         │    └── LRT tests, AIC/BIC comparison
         │
         └──→ spatial_durbin_model.py ⭐
              └── Global SDM + 5 cluster-specific models
                  Direct/Indirect/Endogenous spillover decomposition
```

---

## Next Steps: Interactive Maps

After completing the SDM analysis, you can generate interactive visualizations:

**Required inputs:**
- Spatial analysis results: `results/spatial_analysis/`
- Model outputs: `results/spatial_durbin_model/`
- Metadata: `pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`

**Generate interactive maps:**
```bash
uv run scripts/interactive_maps/generate_all_maps.py
```

**Outputs:**
- `assets/maps/lisa_clusters_explorer.html` (spatial clusters)
- `assets/maps/seasonal_pm10_patterns.html` (temporal patterns)

---

## Pipeline Status: ✅ COMPLETE

All pipeline stages verified:
- ✅ Phase 1: Data preprocessing complete (5 scripts)
- ✅ Phase 2: Exploratory & spatial analysis complete (2 scripts)
- ✅ Phase 3: Spatial econometric modeling complete (2 scripts)
- ✅ Interactive maps generated
- 🎯 **Analysis pipeline fully operational**

**Total Outputs:**
- 14 result files from global SDM
- 10 result files from cluster-specific SDMs (5 clusters × 2 files)
- 2 combined analysis files
- 2 visualization files (plots)
- 2 interactive HTML maps

Last updated: **9 February 2026**
