# GAR4DS - Pollution Corridors Analysis 🌬️

**Spatial analysis of PM10 transport from Po Valley to Alpine regions using Panel Spatial Durbin Models**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 📋 Project Overview

This project implements a comprehensive spatial econometric analysis to quantify cross-border PM10 pollution transport from the Po Valley (Northern Italy) to Alpine regions (Trentino, Veneto). Using Panel Fixed Effects Spatial Durbin Models (SDM), we decompose pollution into:

- **Direct effects** (local meteorological conditions)
- **Indirect effects** (neighbor spillovers)
- **Endogenous spillovers** (spatial autocorrelation)

### Key Innovation

**Regime-Stratified Analysis**: Fits separate SDM models for 5 distinct atmospheric regimes to test whether the global model averages over fundamentally different physical processes (e.g., stagnation vs. transport corridor conditions).

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **uv** (Python package manager)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd gar4ds

# Install dependencies with uv
uv sync
```

### Run Complete Pipeline

```bash
# Option 1: Using Makefile (recommended)
make all

# Option 2: Run specific phases
make preprocessing   # Data preparation (5 scripts)
make analysis        # Exploratory & spatial analysis (2 scripts)
make models          # Spatial econometric models (2 scripts)
make maps            # Interactive visualizations

# Option 3: Manual execution (see Docs/PIPELINE_DOCUMENTATION.md)
uv run scripts/preprocessing/data_preprocessing.py
# ... (see documentation for full sequence)
```

---

## 📊 Pipeline Structure

```
├── Preprocessing (5 scripts)
│   ├── data_preprocessing.py          → Parquet conversion
│   ├── build_panel_matrix.py          → Panel data (20 vars)
│   ├── add_elevation_data.py          → Terrain classification
│   ├── multicollinearity_analysis.py  → VIF analysis
│   └── filter_multicollinearity.py    → Filtered data (12 vars)
│
├── Analysis (2 scripts)
│   ├── exploratory_data_analysis.py   → EDA, correlations
│   └── spatial_analysis.py            → Moran's I, LISA, weights
│
├── Models (2 scripts)
│   ├── model_specification_tests.py   → LRT, AIC, BIC
│   └── spatial_durbin_model.py        → Panel SDM (main analysis)
│
└── Interactive Maps
    ├── lisa_clusters_map.py           → Spatial clusters
    └── seasonal_patterns_map.py       → Temporal patterns
```

**Total: 9 analysis scripts + 2 visualization scripts**

---

## 🛠️ Makefile Commands

### Main Targets

| Command | Description |
|---------|-------------|
| `make help` | Show all available targets |
| `make all` | Run complete pipeline |
| `make preprocessing` | Data preparation phase |
| `make analysis` | Exploratory & spatial analysis |
| `make models` | Fit spatial econometric models |
| `make maps` | Generate interactive HTML maps |

### Individual Scripts

| Command | Description |
|---------|-------------|
| `make data-preprocess` | [1/9] Convert CSV to Parquet |
| `make build-panel` | [2/9] Create panel matrix (20 vars) |
| `make elevation` | [3/9] Add elevation data |
| `make multicollinearity` | [4/9] VIF analysis |
| `make filter-collinearity` | [5/9] Filter to 12 variables |
| `make eda` | [6/9] Exploratory data analysis |
| `make spatial-analysis` | [7/9] Spatial autocorrelation |
| `make model-tests` | [8/9] Model specification tests |
| `make sdm` | [9/9] Spatial Durbin Model |

### Cleaning Targets

| Command | Description |
|---------|-------------|
| `make clean-results` | Remove all result files (keep data) |
| `make clean-assets` | Remove visualizations |
| `make clean-data` | Remove processed data (keep raw CSV) |
| `make clean-all` | Complete reset |

### Utility Targets

| Command | Description |
|---------|-------------|
| `make check-deps` | Verify all dependencies exist |
| `make check-results` | Count generated files |
| `make validate-pipeline` | Full validation check |

---

## 📁 Directory Structure

```
gar4ds/
├── data/                          # Raw & processed data
│   ├── *.csv                      # Raw input data
│   ├── *.parquet                  # Processed panel data
│   └── *.geojson                  # Spatial metadata
├── weights/                       # Spatial weights matrices
│   └── spatial_weights_knn6.gal   # KNN6 weights (critical!)
├── results/                       # All numerical outputs
│   ├── dataset_documentation/
│   ├── multicollinearity_analysis/
│   ├── eda_analysis/
│   ├── spatial_analysis/
│   ├── model_specification_tests/
│   └── spatial_durbin_model/      # Main results
│       ├── model_summary.txt           # Global model
│       ├── coefficients_table.csv
│       ├── cluster_0_model_summary.txt # Cluster-specific
│       ├── ...
│       └── all_clusters_coefficients_combined.csv
├── assets/                        # All visualizations
│   ├── eda_analysis/
│   ├── spatial_analysis/
│   ├── spatial_durbin_model/
│   └── maps/*.html                # Interactive maps
├── scripts/
│   ├── preprocessing/             # 5 preprocessing scripts
│   ├── data_analysis/             # 4 analysis scripts
│   └── interactive_maps/          # 3 map generators
├── Docs/
│   ├── PIPELINE_DOCUMENTATION.md  # Detailed pipeline guide
│   └── PROJECT_PURPOSE.md         # Research objectives
├── Makefile                       # Pipeline automation
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

---

## 📈 Key Outputs

### Spatial Durbin Model Results

**Global Model:**
- `model_summary.txt` - Full regression output
- `coefficients_table.csv` - Direct (β) and indirect (θ) effects
- `spillover_decomposition_observations.csv` - Observation-level decomposition

**Cluster-Specific Models (5 atmospheric regimes):**
- `cluster_N_model_summary.txt` (N=0-4)
- `cluster_N_coefficients.csv` (N=0-4)

**Combined Analysis:**
- `regime_comparison.csv` - Compare ρ across clusters
- `all_clusters_coefficients_combined.csv` - All coefficients in one table

**Visualizations:**
- `coefficient_forest_plot.png` - Direct vs. indirect effects
- `residual_qq_plot.png` - Normality diagnostics

### Interactive Maps

- `assets/maps/lisa_clusters_explorer.html` - Spatial clusters
- `assets/maps/seasonal_pm10_patterns.html` - Temporal patterns

---

## 🔬 Methodology

### Model Specification (Panel SDM)

```
log(PM10ᵢₜ) = ρWyₜ + Xᵢₜβ + WXᵢₜθ + αᵢ + γₜ + εᵢₜ
```

**Where:**
- `y` = log(PM10) concentration
- `ρ` = spatial autoregressive parameter (endogenous spillover)
- `Wy` = spatially lagged PM10 (neighbor pollution)
- `X` = 11 meteorological variables
- `β` = direct effect coefficients
- `WX` = spatially lagged meteorology (neighbor conditions)
- `θ` = indirect effect coefficients (spillover)
- `αᵢ` = station fixed effects
- `γₜ` = time fixed effects
- `ε` = error term

### Variables (12 total)

**Target:**
- PM10 concentration (weekly mean)

**Meteorological Predictors (11):**
- Temperature: `temperature_2m`
- Humidity: `humidity_950`
- Boundary layer: `blh`
- Solar radiation: `solar_radiation_downwards`
- Surface winds: `wind_u_10m`, `wind_v_10m`
- Upper-level winds: `uwind_850`, `uwind_950`, `Vwind_850`, `Vwind_950`
- Precipitation: `total_precipitation`

### Spatial Weights Matrix

- **Type:** KNN6 (k=6 nearest neighbors)
- **Stations:** 37 monitoring stations
- **Row-standardized:** Yes
- **Generated by:** `spatial_analysis.py`

---

## 📚 Documentation

- **[PIPELINE_DOCUMENTATION.md](Docs/PIPELINE_DOCUMENTATION.md)** - Complete pipeline guide with inputs/outputs
- **[PROJECT_PURPOSE.md](Docs/PROJECT_PURPOSE.md)** - Research objectives and methodology

---

## 🔧 Troubleshooting

### Common Issues

**1. Missing dependencies**
```bash
make check-deps  # Check what's missing
```

**2. Pipeline fails midway**
```bash
# Run specific phase only
make quick-analysis  # Skip preprocessing
make quick-models    # Skip preprocessing & analysis
```

**3. Need to regenerate specific outputs**
```bash
make clean-results   # Remove results, keep data
make sdm             # Re-run SDM only
```

**4. Complete reset**
```bash
make clean-all       # Remove everything
make all             # Start fresh
```

---

## 📊 Data Sources

- **PM10 Monitoring Data:** ARPAV (Veneto) & APPA (Trentino)
- **Meteorological Data:** ERA5 reanalysis
  - Temperature (multiple levels)
  - Humidity (multiple levels)  
  - Wind components (U/V at multiple levels)
  - Boundary layer height
  - Solar radiation
  - Precipitation
- **Spatial Data:** Station coordinates (lat/lon)

---

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
@misc{gar4ds2026,
  author = {Miglioranza, Ettore},
  title = {GAR4DS: Spatial Analysis of PM10 Transport from Po Valley to Alpine Regions},
  year = {2026},
  howpublished = {\url{<repository-url>}}
}
```

---

## 📜 License

[Add license information]

---

## 👤 Author

**Ettore Miglioranza**

- Project: GAR4DS - Pollution Corridors Analysis
- Last Updated: 9 February 2026

---

## 🙏 Acknowledgments

- ARPAV (Regional Agency for Environmental Prevention and Protection of Veneto)
- APPA (Provincial Agency for Environment Protection - Trentino)
- ERA5 Reanalysis Data (Copernicus Climate Change Service)

---

## 📞 Support

For questions or issues:
1. Check [PIPELINE_DOCUMENTATION.md](Docs/PIPELINE_DOCUMENTATION.md)
2. Run `make help` for available commands
3. Use `make validate-pipeline` to check setup

---

**Happy analyzing! 🌬️📊**
