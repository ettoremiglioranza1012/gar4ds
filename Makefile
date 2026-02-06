.PHONY: clean run run-all preprocessing filtering analysis modeling visualization config help

# ============================================================================
# CLEANING
# ============================================================================

clean:
	@echo "Cleaning results/..."
	@find results -mindepth 1 -delete 2>/dev/null || true
	@mkdir -p results
	
	@echo "Cleaning data/ (keeping original CSVs and new_data/)..."
	@find data -maxdepth 1 -type f ! -name "pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv" ! -name "pm10_era5_land_era5_reanalysis_blh.csv" -delete 2>/dev/null || true
	
	@echo "Cleaning assets/..."
	@find assets -mindepth 1 -delete 2>/dev/null || true
	@mkdir -p assets
	
	@echo "Removing .gal files from weights/..."
	@rm -f weights/*.gal
	
	@echo "Clean complete."

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

# Run full pipeline with current config
run-all:
	@echo "Running full pipeline..."
	uv run python -m gar4ds.run_pipeline

# Run individual stages
preprocessing:
	@echo "Running preprocessing stage..."
	uv run python -m gar4ds.run_pipeline --stages preprocessing

filtering:
	@echo "Running filtering stage..."
	uv run python -m gar4ds.run_pipeline --stages filtering

analysis:
	@echo "Running analysis stage..."
	uv run python -m gar4ds.run_pipeline --stages analysis

modeling:
	@echo "Running modeling stage..."
	uv run python -m gar4ds.run_pipeline --stages modeling

visualization:
	@echo "Running visualization stage..."
	uv run python -m gar4ds.run_pipeline --stages visualization

# ============================================================================
# CONFIGURATION
# ============================================================================

config:
	@echo "Current pipeline configuration:"
	@uv run python -m gar4ds.run_pipeline --show-config

# Set aggregation level (creates modified config)
set-daily:
	@echo "Setting aggregation to DAILY..."
	@sed -i '' 's/aggregation: ".*"/aggregation: "daily"/' config/pipeline.yaml
	@echo "Config updated. Run 'make run-all' to execute."

set-weekly:
	@echo "Setting aggregation to WEEKLY..."
	@sed -i '' 's/aggregation: ".*"/aggregation: "weekly"/' config/pipeline.yaml
	@echo "Config updated. Run 'make run-all' to execute."

set-monthly:
	@echo "Setting aggregation to MONTHLY..."
	@sed -i '' 's/aggregation: ".*"/aggregation: "monthly"/' config/pipeline.yaml
	@echo "Config updated. Run 'make run-all' to execute."

# ============================================================================
# LEGACY SCRIPTS (for compatibility)
# ============================================================================

run-legacy-preprocessing:
	@echo "Running legacy preprocessing scripts..."
	uv run scripts/preprocessing/data_preprocessing.py
	uv run scripts/preprocessing/build_panel_matrix.py
	uv run scripts/preprocessing/add_elevation_data.py

run-legacy-analysis:
	@echo "Running legacy analysis scripts..."
	uv run scripts/data_analysis/exploratory_data_analysis.py
	uv run scripts/data_analysis/spatial_analysis.py

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "GAR4DS Pipeline Makefile"
	@echo "========================"
	@echo ""
	@echo "Clean targets:"
	@echo "  make clean          - Remove all generated files"
	@echo ""
	@echo "Pipeline targets:"
	@echo "  make run-all        - Run complete pipeline"
	@echo "  make preprocessing  - Run preprocessing stage only"
	@echo "  make filtering      - Run filtering stage only"
	@echo "  make analysis       - Run analysis stage only"
	@echo "  make modeling       - Run SDM modeling stage only"
	@echo "  make visualization  - Run visualization stage only"
	@echo ""
	@echo "Configuration:"
	@echo "  make config         - Show current configuration"
	@echo "  make set-daily      - Set aggregation to daily"
	@echo "  make set-weekly     - Set aggregation to weekly"
	@echo "  make set-monthly    - Set aggregation to monthly"
	@echo ""
	@echo "Edit config/pipeline.yaml for detailed settings."
