# Pipeline Execution Flow - Quick Reference

**Last Updated:** 9 February 2026

---

## 🚀 Quick Start Commands

### Complete Pipeline
```bash
make all
```

### Phase-by-Phase Execution
```bash
make preprocessing  # Phase 1: Data preparation
make analysis       # Phase 2: Exploratory & spatial analysis
make models         # Phase 3: Econometric modeling
make maps           # Phase 4: Interactive visualizations
```

---

## 📋 Detailed Execution Flow

### Phase 1: Preprocessing (5 scripts)

```bash
# Step 1: Convert CSV to Parquet
make data-preprocess
# Output: data/pm10_era5_land_era5_reanalysis_blh.parquet
#         data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson

# Step 2: Build panel matrix (20 variables)
make build-panel
# Output: data/panel_data_matrix.parquet

# Step 3: Add elevation data (optional)
make elevation
# Output: data/pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson

# Step 4: Analyze multicollinearity
make multicollinearity
# Output: results/multicollinearity_analysis/vif_analysis_*.csv
#         results/multicollinearity_analysis/recommendations_*.txt
#         assets/multicollinearity_analysis/*.png

# Step 5: Filter collinear variables (20 → 12 variables)
make filter-collinearity
# Output: data/panel_data_matrix_filtered_for_collinearity.parquet
```

---

### Phase 2: Exploratory & Spatial Analysis (2 scripts)

```bash
# Step 6: Exploratory data analysis
make eda
# Output: results/eda_analysis/eda_analysis.txt
#         assets/eda_analysis/correlation_heatmap.png
#         assets/eda_analysis/terrain_comparison_boxplots.png
#         assets/eda_analysis/temporal_seasonality_analysis.png

# Step 7: Spatial analysis (CRITICAL - creates weights matrix)
make spatial-analysis
# Output: weights/spatial_weights_knn6.gal ← REQUIRED FOR SDM
#         results/spatial_analysis/optionA_global_morans_I_by_variable.csv
#         results/spatial_analysis/optionA_lisa_results_all_variables.csv
#         results/spatial_analysis/optionC_multivariate_clusters.csv
#         results/spatial_analysis/optionC_cluster_profiles.csv
```

---

### Phase 3: Spatial Econometric Modeling (2 scripts)

```bash
# Step 8: Model specification tests
make model-tests
# Output: results/model_specification_tests/lrt_test_results.csv
#         results/model_specification_tests/information_criteria.csv
#         results/model_specification_tests/specification_test_summary.csv

# Step 9: Spatial Durbin Model (MAIN ANALYSIS)
make sdm
# Output: Global model
#         - results/spatial_durbin_model/model_summary.txt
#         - results/spatial_durbin_model/coefficients_table.csv
#         - results/spatial_durbin_model/spillover_decomposition_observations.csv
#
#         Cluster-specific models (5 regimes)
#         - results/spatial_durbin_model/cluster_0_model_summary.txt
#         - results/spatial_durbin_model/cluster_0_coefficients.csv
#         - ... (repeated for clusters 1-4)
#
#         Combined analysis
#         - results/spatial_durbin_model/regime_comparison.csv
#         - results/spatial_durbin_model/all_clusters_coefficients_combined.csv
#
#         Visualizations
#         - assets/spatial_durbin_model/coefficient_forest_plot.png
#         - assets/spatial_durbin_model/residual_qq_plot.png
```

---

### Phase 4: Interactive Maps (optional)

```bash
# Generate all maps
make maps
# Output: assets/maps/lisa_clusters_explorer.html
#         assets/maps/seasonal_pm10_patterns.html

# Or generate individual maps
make lisa-map       # LISA clusters only
make seasonal-map   # Seasonal patterns only
```

---

## 🔍 Validation & Checking

### Check Dependencies
```bash
make check-deps       # Verify all required files exist
```

**Example output:**
```
Checking pipeline dependencies...

✓ Raw data:
  ✅ PM10 data CSV
  ✅ Metadata CSV

✓ Processed data:
  ✅ PM10 parquet
  ✅ Panel matrix (20 vars)
  ✅ Filtered panel (12 vars)

✓ Critical dependencies:
  ✅ Spatial weights (KNN6)
  ✅ Atmospheric clusters
```

### Check Results
```bash
make check-results    # Count generated files
```

**Example output:**
```
Checking generated results...

Preprocessing results:
  📊 Dataset docs: 3 files
  📊 Multicollinearity: 2 files

Analysis results:
  📊 EDA: 1 files
  📊 Spatial analysis: 4 files
  📊 Model tests: 3 files
  📊 SDM results: 17 files

Visualizations:
  🎨 EDA plots: 3 files
  🎨 Spatial plots: 0 files
  🎨 SDM plots: 2 files
  🗺️  Interactive maps: 2 files
```

### Full Validation
```bash
make validate-pipeline   # Run both checks
```

---

## 🗑️ Cleaning Commands

### Selective Cleaning
```bash
make clean-results   # Remove analysis results (keep processed data)
make clean-assets    # Remove visualizations (keep data & results)
make clean-data      # Remove processed data (keep raw CSV)
make clean-weights   # Remove spatial weights matrix
```

### Complete Reset
```bash
make clean-all       # Remove EVERYTHING (requires full re-run)
```

---

## ⚡ Quick Shortcuts

### Skip Preprocessing (if already done)
```bash
make quick-analysis  # Run EDA + spatial analysis only
```

### Skip to Modeling (if preprocessing & analysis done)
```bash
make quick-models    # Run model tests + SDM only
```

---

## 📊 Pipeline Dependencies Graph

```
Raw CSV Files
    ↓
[1] data_preprocessing.py
    ↓ *.parquet, *.geojson
[2] build_panel_matrix.py
    ↓ panel_data_matrix.parquet (20 vars)
[3] add_elevation_data.py (optional)
    ↓ metadata_with_elevation.geojson
[4] multicollinearity_analysis.py
    ↓ VIF analysis
[5] filter_multicollinearity.py
    ↓ panel_data_matrix_filtered_for_collinearity.parquet (12 vars)
    ↓
    ├──→ [6] exploratory_data_analysis.py
    │    └── EDA results + plots
    │
    └──→ [7] spatial_analysis.py ⭐
         ↓ spatial_weights_knn6.gal + clusters
         ↓
         ├──→ [8] model_specification_tests.py
         │    └── LRT tests, AIC/BIC
         │
         └──→ [9] spatial_durbin_model.py ⭐⭐
              └── Global SDM + 5 cluster models
                  17 result files
                  2 visualizations

⭐ = Critical for modeling
⭐⭐ = Main analysis output
```

---

## 🎯 Critical Dependencies

### For SDM to run, you MUST have:

1. **Filtered panel data:**
   ```
   data/panel_data_matrix_filtered_for_collinearity.parquet
   ```
   - Generated by: `make filter-collinearity`
   - Contains: 12 variables (1 target + 11 predictors)

2. **Spatial weights matrix:**
   ```
   weights/spatial_weights_knn6.gal
   ```
   - Generated by: `make spatial-analysis`
   - Type: KNN6 (k=6 nearest neighbors)

3. **Atmospheric clusters:**
   ```
   results/spatial_analysis/optionC_multivariate_clusters.csv
   ```
   - Generated by: `make spatial-analysis`
   - Contains: 5 distinct atmospheric regimes

4. **Station metadata:**
   ```
   data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson
   ```
   - Generated by: `make data-preprocess`
   - Contains: Station coordinates and metadata

---

## 📝 Common Workflows

### First-Time Setup
```bash
# Complete pipeline from scratch
make all
```

### Re-run Analysis After Data Update
```bash
# Clean old results, keep data
make clean-results

# Re-run analysis & modeling
make analysis
make models
```

### Test Different Model Specifications
```bash
# Just re-run modeling
make clean-results
make models
```

### Regenerate Visualizations Only
```bash
# Clean and regenerate plots
make clean-assets
make sdm    # Regenerates plots
make maps   # Regenerates interactive maps
```

---

## 🐛 Debugging Tips

### Script fails midway?
1. Check which step failed
2. Run individual script directly:
   ```bash
   uv run scripts/data_analysis/spatial_durbin_model.py
   ```
3. Check output log in `results/spatial_durbin_model/sdm_analysis_log_*.txt`

### Missing dependencies?
```bash
make check-deps
```
Will show exactly what's missing and which command to run.

### Want to see detailed progress?
All scripts print verbose output to console and save logs to `results/`.

---

## 📚 Output File Counts

### Expected number of files after complete pipeline:

| Directory | Files | Description |
|-----------|-------|-------------|
| `data/` | 4-5 | Parquet files + GeoJSON metadata |
| `weights/` | 1 | KNN6 spatial weights |
| `results/dataset_documentation/` | 3 | Dataset info logs |
| `results/multicollinearity_analysis/` | 2 | VIF analysis + recommendations |
| `results/eda_analysis/` | 1 | EDA text log |
| `results/spatial_analysis/` | 4-5 | Moran's I, LISA, clusters |
| `results/model_specification_tests/` | 3-4 | LRT, AIC/BIC, summary |
| `results/spatial_durbin_model/` | **17** | Global + 5 clusters + combined |
| `assets/eda_analysis/` | 3-4 | Correlation, terrain, temporal plots |
| `assets/spatial_durbin_model/` | 2 | Forest plot + Q-Q plot |
| `assets/maps/` | 2 | Interactive HTML maps |

**Total:** ~45-50 files

---

## ✅ Success Criteria

After running `make all`, verify:

- [ ] No error messages in console
- [ ] `make check-deps` shows all ✅
- [ ] `make check-results` shows expected file counts
- [ ] Main results exist:
  - `results/spatial_durbin_model/model_summary.txt`
  - `results/spatial_durbin_model/all_clusters_coefficients_combined.csv`
  - `assets/spatial_durbin_model/coefficient_forest_plot.png`
  - `assets/maps/lisa_clusters_explorer.html`

---

## 🔗 Additional Resources

- **Detailed Documentation:** [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)
- **Research Objectives:** [PROJECT_PURPOSE.md](PROJECT_PURPOSE.md)
- **All Commands:** Run `make help` in terminal

---

**Last Updated:** 9 February 2026  
**Pipeline Version:** v2.0 (with regime-stratified SDM)
