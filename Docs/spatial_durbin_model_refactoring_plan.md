# Spatial Durbin Model - Refactoring Plan

**Date:** 6 February 2026  
**Purpose:** University Project - Quantifying Po Valley→Alpine PM10 Transport  
**Alignment:** Course methodology using `spreg.Panel_FE_Lag` with manual WX construction

---

## Objective

Build a **panel Spatial Durbin Model** using the full temporal dataset (21,275 observations) with all 11 multicollinearity-filtered meteorological variables, implementing SDM by manually creating spatially lagged features (WX) and feeding them into `spreg.Panel_FE_Lag`, with log-transformed PM10, standardized variables, and regime-stratified analysis to quantify Po Valley→Alpine pollution transport.

---

## Research Question

**"Quantifying the impact of transboundary atmospheric transport on Alpine PM10 levels using a Spatial Durbin Model"**

### Hypothesis
Cross-border meteorological conditions and pollution from the Po Valley (Lombardy, Veneto) significantly affect PM10 concentrations in Alpine regions (Trentino, Alto-Adige), with transport effects being stronger during specific atmospheric regimes.

---

## Data Sources

### Input Files
1. **Panel Data:** `data/panel_data_matrix_filtered_for_collinearity.parquet`
   - 21,275 observations (575 weeks × 37 stations)
   - 12 variables: 1 target (pm10) + 11 meteorological features
   - VIF-filtered to remove multicollinear variables

2. **Spatial Weights:** `weights/spatial_weights_knn6.gal`
   - K-Nearest Neighbors (k=6)
   - Row-standardized
   - 222 connections across 37 stations

3. **Regime Classifications:** `results/spatial_analysis/optionC_multivariate_clusters.csv`
   - 6 atmospheric regimes from K-Means clustering
   - Focus: Cluster 0 (Stagnation) vs. Cluster 2 (Transport Corridor)

4. **Station Metadata:** `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`
   - Station coordinates, names, regions

---

## Variables

### Target Variable
- **pm10** (μg/m³) → **log_pm10** (log-transformed for variance stabilization)

### Meteorological Features (11 variables, all VIF-filtered)

**Boundary Layer & Mixing:**
1. `blh` - Boundary Layer Height (m) - controls vertical dispersion
2. `temperature_2m` - Surface Temperature (K) - affects atmospheric stability

**Surface Winds (10m):**
3. `wind_u_10m` - Zonal wind component (m/s) - E-W transport
4. `wind_v_10m` - Meridional wind component (m/s) - N-S transport

**Upper-Level Winds (850 hPa ~1.5km, 950 hPa ~0.5km):**
5. `uwind_850` - Zonal wind at 850 hPa (m/s) - mid-level transport
6. `uwind_950` - Zonal wind at 950 hPa (m/s) - boundary layer transport
7. `Vwind_850` - Meridional wind at 850 hPa (m/s) - valley flow patterns
8. `Vwind_950` - Meridional wind at 950 hPa (m/s) - surface-level valley flow

**Humidity & Precipitation:**
9. `humidity_950` - Relative Humidity at 950 hPa (%) - near-surface moisture
10. `total_precipitation` - Weekly precipitation (mm) - wet deposition/washout

**Radiation:**
11. `solar_radiation_downwards` - Solar radiation (J/m²/week) - photochemistry proxy

---

## Methodology

### The Spatial Durbin Model (SDM)

**Mathematical Specification:**
```
y = ρWy + Xβ + WXθ + ε
```

Where:
- **y** = log(PM10) - dependent variable (21,275 × 1)
- **ρ** = spatial autoregressive coefficient (endogenous spillover)
- **Wy** = spatially lagged dependent variable (neighbor PM10)
- **X** = matrix of local meteorological conditions (21,275 × 11)
- **β** = direct effect coefficients (local impacts)
- **WX** = spatially lagged independent variables (neighbor meteorology)
- **θ** = indirect effect coefficients (neighbor impacts)
- **ε** = error term

**Manual Implementation Strategy:**
1. Create WX matrix by computing `W @ X` for each of 575 weeks
2. Combine X and WX into single feature matrix (22 columns)
3. Use `spreg.Panel_FE_Lag` to estimate ρ, β, and θ simultaneously
4. Fixed effects control for station-specific and time-specific heterogeneity

---

## Implementation Steps

### 1. Data Preparation
- Load `panel_data_matrix_filtered_for_collinearity.parquet`
- Log-transform PM10: `log_pm10 = np.log(pm10 + 1)` (add 1 to handle near-zero values)
- Standardize all 11 meteorological variables using `StandardScaler`
- Load spatial weights from `spatial_weights_knn6.gal`

### 2. Create Spatially Lagged Features (WX Matrix)
```python
# Pseudo-code
for each week t in 575 weeks:
    X_week = extract_cross_section(panel_df, week=t)  # 37 stations × 11 vars
    WX_week = W @ X_week  # Apply spatial weights
    append WX_week as lag_* columns
```

**Result:** 22 feature columns
- 11 direct: `temperature_2m`, `humidity_950`, ..., `total_precipitation`
- 11 lagged: `lag_temperature_2m`, `lag_humidity_950`, ..., `lag_total_precipitation`

### 3. Fit Panel SDM
```python
from spreg import Panel_FE_Lag

model = Panel_FE_Lag(
    y=log_pm10,           # 21,275 × 1
    x=X_and_WX,           # 21,275 × 22 (standardized)
    w=w,                  # KNN k=6 weights
    name_y=["log_pm10"],
    name_x=[...],         # All 22 feature names
    name_ds="PM10_Alpine_Panel",
    robust='white'        # Robust SE for heteroskedasticity
)
```

**Outputs:**
- ρ (rho): Spatial lag coefficient
- β coefficients (11): Direct local effects
- θ coefficients (11): Indirect neighbor effects
- Standard errors, t-statistics, p-values

### 4. Spillover Decomposition
For each observation, compute:
1. **Direct Local Effect:** `X · β`
2. **Indirect Neighbor Effect:** `WX · θ`
3. **Endogenous Spillover:** `ρ · Wy`
4. **Total Effect:** Using spatial multiplier `(I - ρW)^(-1)`

Aggregate to:
- Station-level averages (37 stations)
- Time-level patterns (575 weeks)
- Focus on 4 Trentino target stations

### 5. Regime-Stratified Analysis
- Merge cluster assignments with panel data
- Fit separate models for:
  - **Cluster 0:** Stagnation regime (6 stations)
  - **Cluster 2:** Transport corridor regime (7 stations)
- Compare ρ values: Test if ρ_Cluster2 > ρ_Cluster0

### 6. Model Diagnostics
1. **Residual Spatial Autocorrelation:** Compute Moran's I on residuals
   - Should be non-significant if model captures spatial structure
2. **Q-Q Plot:** Check normality of residuals
3. **Leverage Analysis:** Identify influential observations
4. **Coefficient Significance:** Highlight significant β and θ

---

## Expected Outputs

### Directory Structure
```
results/spatial_durbin_model_refactored/
├── sdm_analysis_log.txt                    # Verbose timestamped log
├── model_summary.txt                       # Full spreg model output
├── coefficients_table.csv                  # β and θ with p-values
├── spillover_decomposition_observation.csv # 21,275 rows
├── spillover_decomposition_stations.csv    # 37 stations aggregated
├── spillover_temporal_patterns.csv         # 575 weeks aggregated
├── regime_comparison.csv                   # Cluster 0 vs. Cluster 2
└── target_stations_spillover.csv           # 4 Trentino stations focus

assets/spatial_durbin_model_refactored/
├── spillover_timeseries_targets.png        # 4 Trentino stations over time
├── spillover_spatial_map.png               # Average spillover intensity
├── regime_coefficient_comparison.png       # ρ, β, θ across regimes
├── residual_qq_plot.png                    # Normality check
├── residual_moran_scatter.png              # Spatial autocorrelation check
└── coefficient_forest_plot.png             # β and θ with confidence intervals
```

---

## Validation Criteria

### Model Must Satisfy:
1. **ρ is significant** (p < 0.05) → Confirms cross-border transport exists
2. **Moran's I on residuals is non-significant** (p > 0.05) → No residual spatial autocorrelation
3. **At least 2-3 `lag_` variables have significant θ** → Neighbor meteorology matters
4. **ρ_Cluster2 > ρ_Cluster0** → Transport corridors show stronger spillover than stagnation zones
5. **Script completes without memory errors** → 21,275 observations processed successfully
6. **All outputs saved with timestamps** → Results reproducible and documented

---

## Interpretation to Report

### Key Findings to Document:

1. **Spatial Lag Coefficient (ρ):**
   - "A 1% increase in neighbors' PM10 leads to a ρ% increase locally"
   - If significant: "Cross-border transport is confirmed"

2. **Direct vs. Indirect Effects:**
   - **β coefficients:** How local meteorology affects local PM10
   - **θ coefficients:** How neighbor meteorology affects local PM10
   - Example: "High BLH locally reduces PM10 (β_blh < 0), but high BLH in neighbors has no effect (θ_blh ≈ 0)"

3. **Regime Heterogeneity:**
   - "Transport corridors (Cluster 2) show ρ = X, while stagnation zones (Cluster 0) show ρ = Y"
   - "This confirms that spatial spillover is regime-dependent"

4. **Policy Implications:**
   - "If `lag_wind` coefficients are significant → Local emission controls insufficient"
   - "Trentino stations receive X% of PM10 from Po Valley sources"

---

## Course Alignment

### Matching Professor's Requirements:

✓ **Research Question:** Clear and testable  
✓ **Data Description:** ERA5 + APPA/ARPA stations documented  
✓ **Exploratory Analysis:** Moran's I confirms spatial clustering  
✓ **Analysis Techniques:** Spatial Durbin Model (Slide 463)  
✓ **Software:** `spreg`, `libpysal` (Slides 178-180)  
✓ **Reproducibility:** Full script + requirements.txt  
✓ **Individual Contribution:** Manual WX construction, regime stratification  

---

## Notes

- **Why log-transform PM10?** PM10 is log-normally distributed (right-skewed), log transformation stabilizes variance and linearizes relationships
- **Why White robust SE?** Panel data exhibits heteroskedasticity across stations and time periods
- **Why manual WX?** `spreg.Panel_FE_Lag` doesn't automatically create WX terms; we construct the full Durbin model by including WX as standard features
- **Why fixed effects?** Controls for unobserved station characteristics (elevation, urbanization) and temporal shocks (synoptic events)

---

## References

- **Spatial Econometrics Course:** Slides 463 (Spatial Durbin Model specification)
- **Python Implementation:** `spreg.Panel_FE_Lag` documentation
- **Data Source:** ERA5 Reanalysis (Copernicus Climate Data Store)
- **Study Area:** Po Valley - Alpine Arc cross-border region

---

**Plan Status:** Approved for implementation (6 Feb 2026)
