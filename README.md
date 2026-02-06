# GAR4DS - PM10 Spatial Analysis Pipeline

**Pollution Corridors Analysis: From Po Valley to Alpine Region**

A configurable Python pipeline for analyzing PM10 pollution patterns and cross-border transport using spatial econometrics (Spatial Durbin Model).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Usage Examples](#usage-examples)
- [Data Requirements](#data-requirements)
- [License](#license)

---

## Overview

GAR4DS analyzes PM10 pollution transport from the Po Valley (Italy's most industrialized and polluted region) to the surrounding Alpine areas. The pipeline combines:

- **ERA5 Reanalysis Data**: Meteorological variables at multiple pressure levels
- **PM10 Station Data**: Ground-based air quality measurements
- **Spatial Econometrics**: Spatial Durbin Model to quantify spillover effects

### Key Research Questions

1. How do meteorological conditions influence PM10 concentrations?
2. What is the spatial autocorrelation pattern of PM10 across stations?
3. How much PM10 pollution "spills over" from neighboring areas?
4. Are there distinct atmospheric regimes (stagnation vs. transport corridors)?

---

## Features

- **Configurable Temporal Aggregation**: Analyze data at hourly, daily, weekly, or monthly resolution
- **Modular Pipeline**: Run individual stages or the complete workflow
- **Spatial Analysis**: Moran's I, LISA clusters, KNN spatial weights
- **Spillover Decomposition**: Quantify direct vs. indirect (neighbor) effects
- **Interactive Maps**: Folium-based visualizations for exploration
- **Reproducible**: YAML configuration for all parameters

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/gar4ds.git
cd gar4ds

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Verify Installation

```bash
uv run python -m gar4ds.run_pipeline --show-config
```

---

## Quick Start

### 1. Configure the Pipeline

Edit `config/pipeline.yaml` to set your desired temporal aggregation:

```yaml
temporal:
  aggregation: "weekly"  # Options: hourly, daily, weekly, monthly
```

### 2. Run the Full Pipeline

```bash
# Using Makefile
make run-all

# Or using Python directly
uv run python -m gar4ds.run_pipeline
```

### 3. View Results

- **Results**: `results/` (CSV tables, analysis logs)
- **Figures**: `assets/` (PNG plots, HTML interactive maps)
- **Processed Data**: `data/` (Parquet panel matrices)

---

## Configuration

All pipeline settings are controlled via `config/pipeline.yaml`:

### Temporal Aggregation

```yaml
temporal:
  aggregation: "weekly"  # hourly | daily | weekly | monthly
  
  aggregation_methods:
    default: "mean"           # Default for most variables
    precipitation: "sum"      # Cumulative for precipitation
    pm10: "mean"              # Mean concentration
```

| Aggregation | Resample Rule | Use Case |
|-------------|---------------|----------|
| `hourly`    | None          | Maximum granularity, large dataset |
| `daily`     | `D`           | Daily patterns, moderate size |
| `weekly`    | `W-MON`       | **Recommended** - balances detail and size |
| `monthly`   | `MS`          | Long-term trends, smallest dataset |

### Spatial Analysis Settings

```yaml
spatial:
  knn_neighbors: 6           # K for KNN weights matrix
  
  lisa:
    permutations: 999        # Monte Carlo permutations
    significance_level: 0.05
  
  clustering:
    n_clusters: 4            # K-Means clusters
    random_state: 42
```

### Variable Selection

```yaml
variables:
  target: "pm10"
  
  keep_after_filtering:
    - pm10
    - temperature_2m
    - humidity_950
    - blh                    # Boundary Layer Height
    - solar_radiation_downwards
    - wind_u_10m
    - wind_v_10m
    - uwind_850
    - uwind_950
    - Vwind_850
    - Vwind_950
    - total_precipitation
```

---

## Pipeline Stages

The pipeline consists of 5 sequential stages:

```
┌─────────────────┐
│ 1. Preprocessing│  CSV → Parquet/GeoJSON conversion
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Filtering    │  Multicollinearity analysis & variable selection
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. Analysis     │  EDA, Spatial Analysis, Model Specification Tests
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. Modeling     │  Spatial Durbin Model estimation
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. Visualization│  Interactive maps generation
└─────────────────┘
```

### Stage 1: Preprocessing

**Purpose**: Convert raw CSV data to efficient formats and create panel matrix.

**Scripts**:
- `data_preprocessing.py` - CSV → Parquet/GeoJSON
- `build_panel_matrix.py` - Wide → Long format, temporal aggregation
- `add_elevation.py` - Fetch elevation from Open-Elevation API

**Outputs**:
- `data/pm10_era5_land_era5_reanalysis_blh.parquet`
- `data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson`
- `data/panel_data_matrix_{aggregation}.parquet`

### Stage 2: Filtering

**Purpose**: Identify and remove highly collinear variables.

**Scripts**:
- `multicollinearity_analysis.py` - VIF, correlation, PCA analysis
- `filter_multicollinearity.py` - Apply filtering decisions

**Key Findings** (from analysis):
- Temperature levels (550/850/950 hPa): VIF > 50,000 → **DROP**
- Surface pressure: VIF = 2,335 → **DROP**
- Upper humidity levels: High VIF → **DROP**

**Outputs**:
- `data/panel_data_matrix_{aggregation}_filtered.parquet` (12 variables)
- `results/multicollinearity_analysis/vif_analysis_*.csv`

### Stage 3: Analysis

**Purpose**: Validate data quality and perform spatial analysis.

**Scripts**:
- `exploratory_data_analysis.py` - Correlation analysis, terrain effects, seasonality
- `spatial_analysis.py` - Moran's I, LISA, K-Means clustering
- `model_specification_tests.py` - LRT tests (SAR vs SEM)

**Key Analyses**:
1. **Global Moran's I**: Test for spatial autocorrelation
2. **LISA Clusters**: Identify hot spots (HH) and cold spots (LL)
3. **Atmospheric Regimes**: K-Means clustering of meteorological profiles

**Outputs**:
- `results/spatial_analysis/global_morans_I_by_variable.csv`
- `results/spatial_analysis/lisa_results_pm10.csv`
- `weights/spatial_weights_knn6.gal`

### Stage 4: Modeling

**Purpose**: Estimate Spatial Durbin Model for spillover quantification.

**Model Specification**:
```
y = ρWy + Xβ + WXθ + ε

Where:
  y  = log(PM10)
  ρ  = Spatial autoregressive coefficient
  W  = Spatial weights matrix (KNN-6)
  X  = Local meteorological conditions
  WX = Spatially lagged meteorology (neighbor conditions)
  β  = Direct effects
  θ  = Indirect (spillover) effects
```

**Outputs**:
- `results/spatial_durbin_model/coefficients_table.csv`
- `results/spatial_durbin_model/spillover_decomposition.csv`
- `results/spatial_durbin_model/model_summary.txt`

### Stage 5: Visualization

**Purpose**: Generate interactive maps for data exploration.

**Maps Generated**:
1. **LISA Clusters Map**: Spatial clustering of PM10 (High-High, Low-Low, etc.)
2. **PM10 Meteorological Map**: Station-level PM10 with weather tooltip
3. **Seasonal Patterns Map**: Dominant pollution season by station

**Outputs**:
- `assets/maps/lisa_clusters_explorer.html`
- `assets/maps/pm10_meteorological_explorer.html`
- `assets/maps/seasonal_pm10_patterns.html`

---

## Project Structure

```
gar4ds/
├── config/
│   └── pipeline.yaml          # Main configuration file
│
├── data/
│   ├── pm10_era5_land_era5_reanalysis_blh.csv           # Raw data (input)
│   ├── pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv
│   ├── panel_data_matrix_{aggregation}.parquet          # Panel data
│   ├── panel_data_matrix_{aggregation}_filtered.parquet # Filtered panel
│   └── new_data/                                        # Additional data
│
├── gar4ds/                    # Python package
│   ├── __init__.py
│   ├── config.py              # Configuration loader
│   ├── run_pipeline.py        # CLI entry point
│   ├── preprocessing/         # Data processing modules
│   ├── analysis/              # Analysis modules
│   └── visualization/         # Map generation modules
│
├── scripts/                   # Legacy standalone scripts (deprecated)
│   ├── preprocessing/
│   ├── data_analysis/
│   └── interactive_maps/
│
├── results/                   # Analysis outputs
│   ├── dataset_documentation/
│   ├── eda_analysis/
│   ├── multicollinearity_analysis/
│   ├── spatial_analysis/
│   ├── model_specification_tests/
│   └── spatial_durbin_model/
│
├── assets/                    # Figures and maps
│   ├── eda_analysis/
│   ├── spatial_analysis/
│   ├── spatial_durbin_model/
│   └── maps/
│
├── weights/                   # Spatial weights files
│   └── spatial_weights_knn6.gal
│
├── Docs/                      # Documentation
├── Makefile                   # Build automation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

---

## Outputs

### Data Files

| File | Description |
|------|-------------|
| `panel_data_matrix_{agg}.parquet` | Full panel data with MultiIndex (time, station_id) |
| `panel_data_matrix_{agg}_filtered.parquet` | Filtered panel (12 variables) |
| `*_stations_metadata_with_elevation.geojson` | Station locations with elevation/terrain |

### Results CSVs

| File | Description |
|------|-------------|
| `vif_analysis_*.csv` | Variance Inflation Factors |
| `global_morans_I_by_variable.csv` | Spatial autocorrelation statistics |
| `lisa_results_pm10.csv` | Local Moran's I with cluster assignments |
| `coefficients_table.csv` | SDM regression coefficients |
| `spillover_decomposition.csv` | Direct vs indirect effects |

### Interactive Maps

| Map | Description |
|-----|-------------|
| `lisa_clusters_explorer.html` | LISA hot/cold spots |
| `pm10_meteorological_explorer.html` | PM10 with weather data |
| `seasonal_pm10_patterns.html` | Dominant pollution season |

---

## Usage Examples

### Run Full Pipeline

```bash
# Default (weekly aggregation)
make run-all

# With custom config
uv run python -m gar4ds.run_pipeline --config config/pipeline.yaml
```

### Run Specific Stages

```bash
# Only preprocessing
make preprocessing
# or
uv run python -m gar4ds.run_pipeline --stages preprocessing

# Multiple stages
uv run python -m gar4ds.run_pipeline --stages preprocessing filtering analysis
```

### Change Aggregation Level

```bash
# Quick change via Makefile
make set-daily
make set-weekly
make set-monthly

# Or edit config/pipeline.yaml directly
```

### View Current Configuration

```bash
make config
# or
uv run python -m gar4ds.run_pipeline --show-config
```

### Clean Generated Files

```bash
make clean
```

### Python API Usage

```python
from gar4ds import load_config
from gar4ds.preprocessing import run_panel_builder
from gar4ds.analysis import run_spatial_analysis

# Load configuration
config = load_config()
print(f"Aggregation: {config.temporal.aggregation}")

# Run specific module
df_panel = run_panel_builder(config)

# Access paths
print(config.get_panel_matrix_path())
print(config.get_spatial_weights_path())
```

---

## Data Requirements

### Input Files

Place these files in `data/`:

1. **Main Dataset**: `pm10_era5_land_era5_reanalysis_blh.csv`
   - Columns: `datetime`, plus columns for each variable-station combination
   - Format: `{variable}_{region}_{station_id}` (e.g., `pm10_Veneto_502604`)

2. **Station Metadata**: `pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv`
   - Required columns: `station_code`, `station_name`, `region`, `latitude`, `longitude`

### Variable Naming Convention

```
{variable}_{region}_{station_id}

Examples:
  pm10_Lombardia_ARPAL_001
  temperature_2m_Veneto_502604
  blh_Alto-Adige_AB2
```

### Regions Supported

- Alto-Adige
- Lombardia
- Trentino
- Veneto

---

## Technical Notes

### Spatial Durbin Model

The SDM captures three types of effects:

1. **Direct Effect (β)**: Impact of local conditions on local PM10
2. **Indirect Effect (θ)**: Impact of neighbor conditions on local PM10
3. **Endogenous Spillover (ρ)**: Impact of neighbor PM10 on local PM10

### Log Transformation

PM10 is log-transformed (`log(PM10 + 1)`) to:
- Stabilize variance
- Approximate normality
- Handle right-skewed distribution

### Spatial Weights

KNN-6 weights mean each station is connected to its 6 nearest neighbors, row-standardized so weights sum to 1.

---

## Troubleshooting

### "Config file not found"

Ensure you're running from the project root:
```bash
cd /path/to/gar4ds
uv run python -m gar4ds.run_pipeline
```

### "Panel matrix not found"

Run preprocessing first:
```bash
make preprocessing
```

### "Spatial weights not found"

Run spatial analysis first:
```bash
uv run python -m gar4ds.run_pipeline --stages analysis
```

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{gar4ds2026,
  title = {GAR4DS: PM10 Spatial Analysis Pipeline},
  author = {Miglioranza, Ettore},
  year = {2026},
  url = {https://github.com/your-org/gar4ds}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
