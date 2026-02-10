# ============================================================================
# GAR4DS - Pollution Corridors Analysis Pipeline
# ============================================================================
# Complete Makefile for reproducible analysis execution
# Author: Ettore Miglioranza
# Last Updated: 10 February 2026
# ============================================================================

.PHONY: all help preprocessing analysis models maps clean clean-results clean-assets clean-all check-deps
.PHONY: set-daily set-weekly set-monthly show-frequency all-daily all-weekly all-monthly

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
	@echo "⏱️  Current frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")"
	@echo ""
	@echo "Frequency Configuration:"
	@grep -E '^(set-|show-|all-).*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[35m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Main Targets:"
	@grep -E '^(all|preprocessing|analysis|models|maps|clean-).*:.*?## .*$$' $(MAKEFILE_LIST) | \
		grep -v -E '^(all-|set-|show-)' | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Individual Script Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[90m%-20s\033[0m %s\n", $$1, $$2}' | \
		grep -v -E "(all|preprocessing|analysis|models|maps|clean|help|check|set-|show-)"
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
# TEMPORAL FREQUENCY CONFIGURATION
# ============================================================================

show-frequency: ## Show current temporal aggregation frequency
	@echo "⏱️  Current temporal frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")"
	@echo ""
	@echo "Available frequencies:"
	@echo "  • daily   - 148,666 observations (4,018 days × 37 stations)"
	@echo "  • weekly  - 21,275 observations (575 weeks × 37 stations)"
	@echo "  • monthly - 4,884 observations (132 months × 37 stations)"
	@echo ""
	@echo "To change frequency: make set-daily | set-weekly | set-monthly"

set-daily: ## Set temporal frequency to DAILY aggregation
	@echo "⏱️  Setting frequency to DAILY..."
	@sed -i.bak "s/TEMPORAL_FREQUENCY = '.*'/TEMPORAL_FREQUENCY = 'daily'/" scripts/config.py && rm scripts/config.py.bak
	@echo "✅ Frequency set to: daily"
	@echo "   Run 'make build-panel' to create panel_data_matrix_daily.parquet"

set-weekly: ## Set temporal frequency to WEEKLY aggregation (default)
	@echo "⏱️  Setting frequency to WEEKLY..."
	@sed -i.bak "s/TEMPORAL_FREQUENCY = '.*'/TEMPORAL_FREQUENCY = 'weekly'/" scripts/config.py && rm scripts/config.py.bak
	@echo "✅ Frequency set to: weekly"
	@echo "   Run 'make build-panel' to create panel_data_matrix_weekly.parquet"

set-monthly: ## Set temporal frequency to MONTHLY aggregation
	@echo "⏱️  Setting frequency to MONTHLY..."
	@sed -i.bak "s/TEMPORAL_FREQUENCY = '.*'/TEMPORAL_FREQUENCY = 'monthly'/" scripts/config.py && rm scripts/config.py.bak
	@echo "✅ Frequency set to: monthly"
	@echo "   Run 'make build-panel' to create panel_data_matrix_monthly.parquet"

# ============================================================================
# MAIN PIPELINE TARGETS
# ============================================================================

all: ## Run complete pipeline with current frequency setting
	@echo "⏱️  Running pipeline with frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")"
	@echo "   (Use 'make set-daily/weekly/monthly' to change)"
	@echo ""
	$(MAKE) preprocessing
	$(MAKE) analysis
	$(MAKE) models
	$(MAKE) maps
	@echo "✅ Complete pipeline executed successfully!"
	@echo ""
	@echo "Check outputs in:"
	@echo "  • results/       (all numerical results)"
	@echo "  • assets/        (all visualizations)"
	@echo "  • assets/maps/   (interactive HTML maps)"
	@echo ""

all-daily: set-daily all ## Run complete pipeline with DAILY frequency
	@echo "✅ Daily pipeline complete!"

all-weekly: set-weekly all ## Run complete pipeline with WEEKLY frequency
	@echo "✅ Weekly pipeline complete!"

all-monthly: set-monthly all ## Run complete pipeline with MONTHLY frequency
	@echo "✅ Monthly pipeline complete!"

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

build-panel: ## [2/9] Build panel data matrix with current frequency setting
	@echo "▶ Building panel matrix with frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")"
	uv run scripts/preprocessing/build_panel_matrix.py

build-panel-daily: set-daily build-panel ## Build panel data matrix with DAILY aggregation
	@echo "✅ Panel matrix created: panel_data_matrix_daily.parquet"

build-panel-weekly: set-weekly build-panel ## Build panel data matrix with WEEKLY aggregation
	@echo "✅ Panel matrix created: panel_data_matrix_weekly.parquet"

build-panel-monthly: set-monthly build-panel ## Build panel data matrix with MONTHLY aggregation
	@echo "✅ Panel matrix created: panel_data_matrix_monthly.parquet"

elevation: ## [3/9] Add elevation data to station metadata
	@echo "▶ Adding elevation data..."
	uv run scripts/preprocessing/add_elevation_data.py

multicollinearity: ## [4/9] Analyze multicollinearity (VIF, PCA) with current frequency
	@echo "▶ Analyzing multicollinearity [frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")]..."
	uv run scripts/preprocessing/multicollinearity_analysis.py

filter-collinearity: ## [5/9] Filter variables based on VIF analysis with current frequency
	@echo "▶ Filtering collinear variables [frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")]..."
	uv run scripts/preprocessing/filter_multicollinearity.py

# ============================================================================
# ANALYSIS SCRIPTS
# ============================================================================

eda: ## [6/9] Exploratory data analysis with current frequency
	@echo "▶ Running exploratory data analysis [frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")]..."
	uv run scripts/data_analysis/exploratory_data_analysis.py

spatial-analysis: ## [7/9] Spatial autocorrelation & weights matrix (KNN6) with current frequency
	@echo "▶ Running spatial analysis [frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")]..."
	uv run scripts/data_analysis/spatial_analysis.py

model-tests: ## [8/9] Model specification tests (LRT, AIC, BIC) with current frequency
	@echo "▶ Running model specification tests [frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")]..."
	uv run scripts/data_analysis/model_specification_tests.py

sdm: ## [9/9] Spatial Durbin Model with current frequency
	@echo "▶ Fitting Spatial Durbin Model [frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")]..."
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
	@echo "⏱️  Current frequency: $$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//")"
	@echo ""
	@echo "✓ Raw data:"
	@test -f data/pm10_era5_land_era5_reanalysis_blh.csv && echo "  ✅ PM10 data CSV" || echo "  ❌ PM10 data CSV (MISSING)"
	@test -f data/pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv && echo "  ✅ Metadata CSV" || echo "  ❌ Metadata CSV (MISSING)"
	@echo ""
	@echo "✓ Processed data (frequency-specific):"
	@test -f data/pm10_era5_land_era5_reanalysis_blh.parquet && echo "  ✅ PM10 parquet" || echo "  ⚠️  PM10 parquet (run 'make data-preprocess')"
	@FREQ=$$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//"); \
		test -f data/panel_data_matrix_$$FREQ.parquet && echo "  ✅ Panel matrix ($$FREQ)" || echo "  ⚠️  Panel matrix ($$FREQ) - run 'make build-panel'"
	@FREQ=$$(grep '^TEMPORAL_FREQUENCY' scripts/config.py | sed "s/.*= '//;s/'.*//"); \
		test -f data/panel_data_matrix_filtered_for_collinearity_$$FREQ.parquet && echo "  ✅ Filtered panel ($$FREQ)" || echo "  ⚠️  Filtered panel ($$FREQ) - run 'make filter-collinearity'"
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
