# ============================================================================
# GAR4DS - Pollution Corridors Analysis Pipeline
# ============================================================================
# Complete Makefile for reproducible analysis execution
# Author: Ettore Miglioranza
# Last Updated: 9 February 2026
# ============================================================================

.PHONY: all help preprocessing analysis models maps clean clean-results clean-assets clean-all check-deps

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# HELP & DOCUMENTATION
# ============================================================================

help: ## Show this help message
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "  GAR4DS - Pollution Corridors Analysis Pipeline"
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Main Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' | \
		grep -E "(all|preprocessing|analysis|models|maps|clean)"
	@echo ""
	@echo "Individual Script Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[90m%-20s\033[0m %s\n", $$1, $$2}' | \
		grep -v -E "(all|preprocessing|analysis|models|maps|clean|help|check)"
	@echo ""
	@echo "Cleaning Targets:"
	@grep -E '^clean.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[33m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Utility Targets:"
	@grep -E '^check.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# MAIN PIPELINE TARGETS
# ============================================================================

all: preprocessing analysis models maps ## Run complete pipeline (preprocessing → analysis → models → maps)
	@echo "✅ Complete pipeline executed successfully!"
	@echo ""
	@echo "Check outputs in:"
	@echo "  • results/       (all numerical results)"
	@echo "  • assets/        (all visualizations)"
	@echo "  • assets/maps/   (interactive HTML maps)"
	@echo ""

preprocessing: data-preprocess build-panel elevation multicollinearity filter-collinearity ## Run all preprocessing steps
	@echo "✅ Preprocessing phase complete!"

analysis: eda spatial-analysis ## Run exploratory and spatial analysis
	@echo "✅ Analysis phase complete!"

models: model-tests sdm ## Run model specification tests and fit Spatial Durbin Model
	@echo "✅ Modeling phase complete!"

maps: generate-maps ## Generate interactive HTML maps
	@echo "✅ Interactive maps generated!"

# ============================================================================
# PREPROCESSING SCRIPTS
# ============================================================================

data-preprocess: ## [1/9] Convert raw CSV to efficient formats
	@echo "▶ Running data preprocessing..."
	uv run scripts/preprocessing/data_preprocessing.py

build-panel: ## [2/9] Build panel data matrix (20 variables)
	@echo "▶ Building panel matrix..."
	uv run scripts/preprocessing/build_panel_matrix.py

elevation: ## [3/9] Add elevation data to station metadata
	@echo "▶ Adding elevation data..."
	uv run scripts/preprocessing/add_elevation_data.py

multicollinearity: ## [4/9] Analyze multicollinearity (VIF, PCA)
	@echo "▶ Analyzing multicollinearity..."
	uv run scripts/preprocessing/multicollinearity_analysis.py

filter-collinearity: ## [5/9] Filter variables based on VIF analysis (12 variables)
	@echo "▶ Filtering collinear variables..."
	uv run scripts/preprocessing/filter_multicollinearity.py

# ============================================================================
# ANALYSIS SCRIPTS
# ============================================================================

eda: ## [6/9] Exploratory data analysis (correlations, seasonality, terrain)
	@echo "▶ Running exploratory data analysis..."
	uv run scripts/data_analysis/exploratory_data_analysis.py

spatial-analysis: ## [7/9] Spatial autocorrelation & weights matrix (KNN6)
	@echo "▶ Running spatial analysis..."
	uv run scripts/data_analysis/spatial_analysis.py

model-tests: ## [8/9] Model specification tests (LRT, AIC, BIC)
	@echo "▶ Running model specification tests..."
	uv run scripts/data_analysis/model_specification_tests.py

sdm: ## [9/9] Spatial Durbin Model with regime-stratified analysis
	@echo "▶ Fitting Spatial Durbin Model (global + 5 cluster-specific models)..."
	uv run scripts/data_analysis/spatial_durbin_model.py

# ============================================================================
# INTERACTIVE MAPS
# ============================================================================

generate-maps: ## Generate all interactive HTML maps
	@echo "▶ Generating interactive maps..."
	uv run scripts/interactive_maps/generate_all_maps.py

lisa-map: ## Generate LISA clusters map only
	@echo "▶ Generating LISA clusters map..."
	uv run scripts/interactive_maps/lisa_clusters_map.py

seasonal-map: ## Generate seasonal patterns map only
	@echo "▶ Generating seasonal patterns map..."
	uv run scripts/interactive_maps/seasonal_patterns_map.py

# ============================================================================
# CLEANING TARGETS
# ============================================================================

clean-results: ## Remove all result files (keeps processed data)
	@echo "🗑️  Removing results..."
	rm -rf results/dataset_documentation/*
	rm -rf results/multicollinearity_analysis/*
	rm -rf results/eda_analysis/*
	rm -rf results/spatial_analysis/*
	rm -rf results/model_specification_tests/*
	rm -rf results/spatial_durbin_model/*
	@echo "✅ Results cleaned!"

clean-assets: ## Remove all visualizations and maps
	@echo "🗑️  Removing visualizations..."
	rm -rf assets/eda_analysis/*
	rm -rf assets/spatial_analysis/*
	rm -rf assets/spatial_durbin_model/*
	rm -rf assets/maps/*.html
	@echo "✅ Assets cleaned!"

clean-data: ## Remove processed data files (keeps raw CSV)
	@echo "🗑️  Removing processed data..."
	rm -f data/*.parquet
	rm -f data/*.geojson
	rm -f data/pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson
	@echo "⚠️  Raw CSV files preserved!"
	@echo "✅ Processed data cleaned!"

clean-weights: ## Remove spatial weights matrix
	@echo "🗑️  Removing spatial weights..."
	rm -f weights/*.gal
	@echo "✅ Weights cleaned!"

clean-all: clean-results clean-assets clean-data clean-weights ## Remove ALL generated files (complete reset)
	@echo "✅ Complete cleanup done!"
	@echo "⚠️  You'll need to run the full pipeline to regenerate outputs."

# ============================================================================
# UTILITY TARGETS
# ============================================================================

check-deps: ## Check if all required dependencies exist
	@echo "Checking pipeline dependencies..."
	@echo ""
	@echo "✓ Raw data:"
	@test -f data/pm10_era5_land_era5_reanalysis_blh.csv && echo "  ✅ PM10 data CSV" || echo "  ❌ PM10 data CSV (MISSING)"
	@test -f data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv && echo "  ✅ Metadata CSV" || echo "  ❌ Metadata CSV (MISSING)"
	@echo ""
	@echo "✓ Processed data:"
	@test -f data/pm10_era5_land_era5_reanalysis_blh.parquet && echo "  ✅ PM10 parquet" || echo "  ⚠️  PM10 parquet (run 'make data-preprocess')"
	@test -f data/panel_data_matrix.parquet && echo "  ✅ Panel matrix (20 vars)" || echo "  ⚠️  Panel matrix (run 'make build-panel')"
	@test -f data/panel_data_matrix_filtered_for_collinearity.parquet && echo "  ✅ Filtered panel (12 vars)" || echo "  ⚠️  Filtered panel (run 'make filter-collinearity')"
	@echo ""
	@echo "✓ Critical dependencies:"
	@test -f weights/spatial_weights_knn6.gal && echo "  ✅ Spatial weights (KNN6)" || echo "  ⚠️  Spatial weights (run 'make spatial-analysis')"
	@test -f results/spatial_analysis/optionC_multivariate_clusters.csv && echo "  ✅ Atmospheric clusters" || echo "  ⚠️  Clusters (run 'make spatial-analysis')"
	@echo ""

check-results: ## Check which result files exist
	@echo "Checking generated results..."
	@echo ""
	@echo "Preprocessing results:"
	@ls -1 results/dataset_documentation/ 2>/dev/null | wc -l | xargs printf "  📊 Dataset docs: %s files\n"
	@ls -1 results/multicollinearity_analysis/ 2>/dev/null | wc -l | xargs printf "  📊 Multicollinearity: %s files\n"
	@echo ""
	@echo "Analysis results:"
	@ls -1 results/eda_analysis/ 2>/dev/null | wc -l | xargs printf "  📊 EDA: %s files\n"
	@ls -1 results/spatial_analysis/ 2>/dev/null | wc -l | xargs printf "  📊 Spatial analysis: %s files\n"
	@ls -1 results/model_specification_tests/ 2>/dev/null | wc -l | xargs printf "  📊 Model tests: %s files\n"
	@ls -1 results/spatial_durbin_model/ 2>/dev/null | wc -l | xargs printf "  📊 SDM results: %s files\n"
	@echo ""
	@echo "Visualizations:"
	@ls -1 assets/eda_analysis/*.png 2>/dev/null | wc -l | xargs printf "  🎨 EDA plots: %s files\n"
	@ls -1 assets/spatial_analysis/*.png 2>/dev/null | wc -l | xargs printf "  🎨 Spatial plots: %s files\n"
	@ls -1 assets/spatial_durbin_model/*.png 2>/dev/null | wc -l | xargs printf "  🎨 SDM plots: %s files\n"
	@ls -1 assets/maps/*.html 2>/dev/null | wc -l | xargs printf "  🗺️  Interactive maps: %s files\n"
	@echo ""

list-outputs: ## List all generated output files with timestamps
	@echo "Generated outputs (most recent first):"
	@echo ""
	@find results -type f -name "*.txt" -o -name "*.csv" 2>/dev/null | xargs ls -lt 2>/dev/null | head -20

validate-pipeline: check-deps check-results ## Run full validation (dependencies + results)
	@echo ""
	@echo "✅ Pipeline validation complete!"

# ============================================================================
# QUICK START TARGETS
# ============================================================================

quick-analysis: filter-collinearity eda spatial-analysis ## Skip preprocessing, run analysis only (requires processed data)
	@echo "✅ Quick analysis complete!"

quick-models: model-tests sdm ## Skip preprocessing & analysis, run models only (requires spatial weights)
	@echo "✅ Quick modeling complete!"

# ============================================================================
# PHASED EXECUTION (for debugging)
# ============================================================================

phase1: preprocessing ## Alias for preprocessing phase
phase2: analysis ## Alias for analysis phase
phase3: models ## Alias for modeling phase

# ============================================================================
# NOTES
# ============================================================================

# Pipeline execution order:
# 1. make preprocessing  (5 scripts)
# 2. make analysis       (2 scripts)
# 3. make models         (2 scripts)
# 4. make maps           (optional)
#
# Or simply: make all
#
# For selective cleaning:
# - make clean-results   (keep data, remove analysis results)
# - make clean-assets    (keep data, remove visualizations)
# - make clean-all       (complete reset)
#
# Dependencies are automatically created by targets
# No need to manually create directories
