"""
Configuration loader for GAR4DS pipeline.

This module provides configuration management with YAML support,
validation, and path resolution.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


# Find project root by looking for config/ directory
def find_project_root() -> Path:
    """Find the project root directory by looking for config/pipeline.yaml."""
    current = Path(__file__).resolve().parent
    
    # Walk up the directory tree
    for _ in range(10):  # Max 10 levels up
        config_file = current / "config" / "pipeline.yaml"
        if config_file.exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    # Fallback: assume we're in gar4ds/ package
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"


@dataclass
class TemporalConfig:
    """Temporal aggregation settings."""
    aggregation: str = "weekly"
    aggregation_methods: Dict[str, str] = field(default_factory=lambda: {
        "default": "mean",
        "precipitation": "sum",
        "pm10": "mean"
    })
    
    @property
    def resample_rule(self) -> Optional[str]:
        """Get pandas resample rule for current aggregation."""
        rules = {
            "hourly": None,  # No resample needed
            "daily": "D",
            "weekly": "W-MON",
            "monthly": "MS"
        }
        return rules.get(self.aggregation)
    
    @property
    def time_label(self) -> str:
        """Get label for time index column."""
        labels = {
            "hourly": "datetime",
            "daily": "date",
            "weekly": "week_start",
            "monthly": "month_start"
        }
        return labels.get(self.aggregation, "datetime")
    
    def validate(self):
        """Validate aggregation setting."""
        valid_options = ["hourly", "daily", "weekly", "monthly"]
        if self.aggregation not in valid_options:
            raise ValueError(
                f"Invalid aggregation '{self.aggregation}'. "
                f"Must be one of: {valid_options}"
            )


@dataclass
class PathsConfig:
    """Path configuration with automatic resolution."""
    data_dir: str = "data"
    results_dir: str = "results"
    assets_dir: str = "assets"
    weights_dir: str = "weights"
    source: Dict[str, str] = field(default_factory=lambda: {
        "main_dataset": "pm10_era5_land_era5_reanalysis_blh.csv",
        "stations_metadata": "pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv"
    })
    
    def resolve(self, base: Path) -> "ResolvedPaths":
        """Resolve all paths relative to base directory."""
        return ResolvedPaths(
            base=base,
            data=base / self.data_dir,
            results=base / self.results_dir,
            assets=base / self.assets_dir,
            weights=base / self.weights_dir,
            source_main=base / self.data_dir / self.source["main_dataset"],
            source_metadata=base / self.data_dir / self.source["stations_metadata"]
        )


@dataclass
class ResolvedPaths:
    """Resolved absolute paths for the project."""
    base: Path
    data: Path
    results: Path
    assets: Path
    weights: Path
    source_main: Path
    source_metadata: Path
    
    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for path in [self.data, self.results, self.assets, self.weights]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class SpatialConfig:
    """Spatial analysis configuration."""
    knn_neighbors: int = 6
    lisa_permutations: int = 999
    lisa_significance: float = 0.05
    n_clusters: int = 4
    random_state: int = 42


@dataclass
class VariablesConfig:
    """Variable configuration."""
    target: str = "pm10"
    keep_after_filtering: List[str] = field(default_factory=list)
    precipitation_vars: List[str] = field(default_factory=lambda: ["total_precipitation"])


@dataclass 
class OutputConfig:
    """Output file naming configuration."""
    parquet_dataset: str = "pm10_era5_land_era5_reanalysis_blh.parquet"
    stations_geojson: str = "pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson"
    stations_geojson_elevation: str = "pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson"
    panel_matrix: str = "panel_data_matrix_{aggregation}.parquet"
    panel_matrix_filtered: str = "panel_data_matrix_{aggregation}_filtered.parquet"
    spatial_weights: str = "spatial_weights_knn{k}.gal"
    
    def format_panel_matrix(self, aggregation: str) -> str:
        """Get panel matrix filename with aggregation."""
        return self.panel_matrix.format(aggregation=aggregation)
    
    def format_panel_matrix_filtered(self, aggregation: str) -> str:
        """Get filtered panel matrix filename with aggregation."""
        return self.panel_matrix_filtered.format(aggregation=aggregation)
    
    def format_spatial_weights(self, k: int) -> str:
        """Get spatial weights filename with k parameter."""
        return self.spatial_weights.format(k=k)


class PipelineConfig:
    """
    Main configuration class for the GAR4DS pipeline.
    
    Loads configuration from YAML and provides typed access to all settings.
    
    Usage:
        config = PipelineConfig.load()  # Load from default location
        config = PipelineConfig.load("path/to/config.yaml")  # Custom path
        
        # Access settings
        print(config.temporal.aggregation)
        print(config.paths.data)
        print(config.get_panel_matrix_path())
    """
    
    def __init__(self, config_dict: Dict[str, Any], base_path: Optional[Path] = None):
        self._raw = config_dict
        self._base_path = base_path or PROJECT_ROOT
        
        # Parse temporal config
        temporal_dict = config_dict.get("temporal", {})
        self.temporal = TemporalConfig(
            aggregation=temporal_dict.get("aggregation", "weekly"),
            aggregation_methods=temporal_dict.get("aggregation_methods", {
                "default": "mean",
                "precipitation": "sum", 
                "pm10": "mean"
            })
        )
        self.temporal.validate()
        
        # Parse paths config
        paths_dict = config_dict.get("paths", {})
        paths_config = PathsConfig(
            data_dir=paths_dict.get("data_dir", "data"),
            results_dir=paths_dict.get("results_dir", "results"),
            assets_dir=paths_dict.get("assets_dir", "assets"),
            weights_dir=paths_dict.get("weights_dir", "weights"),
            source=paths_dict.get("source", {
                "main_dataset": "pm10_era5_land_era5_reanalysis_blh.csv",
                "stations_metadata": "pm10_era5_land_era5_reanalysis_blh_stations_metadata.csv"
            })
        )
        self.paths = paths_config.resolve(self._base_path)
        
        # Parse output config
        output_dict = config_dict.get("output", {})
        self.output = OutputConfig(
            parquet_dataset=output_dict.get("parquet_dataset", "pm10_era5_land_era5_reanalysis_blh.parquet"),
            stations_geojson=output_dict.get("stations_geojson", "pm10_era5_land_era5_reanalysis_blh_stations_metadata.geojson"),
            stations_geojson_elevation=output_dict.get("stations_geojson_elevation", "pm10_era5_land_era5_reanalysis_blh_stations_metadata_with_elevation.geojson"),
            panel_matrix=output_dict.get("panel_matrix", "panel_data_matrix_{aggregation}.parquet"),
            panel_matrix_filtered=output_dict.get("panel_matrix_filtered", "panel_data_matrix_{aggregation}_filtered.parquet"),
            spatial_weights=output_dict.get("spatial_weights", "spatial_weights_knn{k}.gal")
        )
        
        # Parse spatial config
        spatial_dict = config_dict.get("spatial", {})
        lisa_dict = spatial_dict.get("lisa", {})
        cluster_dict = spatial_dict.get("clustering", {})
        self.spatial = SpatialConfig(
            knn_neighbors=spatial_dict.get("knn_neighbors", 6),
            lisa_permutations=lisa_dict.get("permutations", 999),
            lisa_significance=lisa_dict.get("significance_level", 0.05),
            n_clusters=cluster_dict.get("n_clusters", 4),
            random_state=cluster_dict.get("random_state", 42)
        )
        
        # Parse variables config
        vars_dict = config_dict.get("variables", {})
        self.variables = VariablesConfig(
            target=vars_dict.get("target", "pm10"),
            keep_after_filtering=vars_dict.get("keep_after_filtering", []),
            precipitation_vars=vars_dict.get("precipitation_vars", ["total_precipitation"])
        )
        
        # Store raw sections for advanced access
        self.elevation = config_dict.get("elevation", {})
        self.visualization = config_dict.get("visualization", {})
        self.logging = config_dict.get("logging", {})
        self.model = config_dict.get("model", {})
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "PipelineConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default.
            
        Returns:
            PipelineConfig instance
        """
        if config_path is None:
            path = DEFAULT_CONFIG_PATH
        else:
            path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Determine base path (parent of config/ directory)
        base_path = path.parent.parent
        
        return cls(config_dict, base_path)
    
    # ==================== Convenience Methods ====================
    
    def get_panel_matrix_path(self) -> Path:
        """Get full path to panel matrix file."""
        filename = self.output.format_panel_matrix(self.temporal.aggregation)
        return self.paths.data / filename
    
    def get_panel_matrix_filtered_path(self) -> Path:
        """Get full path to filtered panel matrix file."""
        filename = self.output.format_panel_matrix_filtered(self.temporal.aggregation)
        return self.paths.data / filename
    
    def get_parquet_path(self) -> Path:
        """Get full path to parquet dataset."""
        return self.paths.data / self.output.parquet_dataset
    
    def get_stations_geojson_path(self, with_elevation: bool = False) -> Path:
        """Get full path to stations GeoJSON file."""
        if with_elevation:
            return self.paths.data / self.output.stations_geojson_elevation
        return self.paths.data / self.output.stations_geojson
    
    def get_spatial_weights_path(self) -> Path:
        """Get full path to spatial weights file."""
        filename = self.output.format_spatial_weights(self.spatial.knn_neighbors)
        return self.paths.weights / filename
    
    def get_results_subdir(self, name: str) -> Path:
        """Get path to a results subdirectory, creating if needed."""
        path = self.paths.results / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_assets_subdir(self, name: str) -> Path:
        """Get path to an assets subdirectory, creating if needed."""
        path = self.paths.assets / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_aggregation_method(self, variable: str) -> str:
        """Get aggregation method for a specific variable."""
        methods = self.temporal.aggregation_methods
        
        if variable in self.variables.precipitation_vars:
            return methods.get("precipitation", "sum")
        if variable == self.variables.target:
            return methods.get("pm10", "mean")
        return methods.get("default", "mean")
    
    def summary(self) -> str:
        """Get a summary of current configuration."""
        return f"""
GAR4DS Pipeline Configuration
=============================
Temporal Aggregation: {self.temporal.aggregation}
Time Index Label: {self.temporal.time_label}
Resample Rule: {self.temporal.resample_rule or 'None (hourly)'}

Paths:
  Data: {self.paths.data}
  Results: {self.paths.results}
  Assets: {self.paths.assets}
  Weights: {self.paths.weights}

Output Files:
  Panel Matrix: {self.output.format_panel_matrix(self.temporal.aggregation)}
  Filtered: {self.output.format_panel_matrix_filtered(self.temporal.aggregation)}
  Weights: {self.output.format_spatial_weights(self.spatial.knn_neighbors)}

Spatial Settings:
  KNN neighbors: {self.spatial.knn_neighbors}
  LISA permutations: {self.spatial.lisa_permutations}
  Clusters: {self.spatial.n_clusters}

Variables:
  Target: {self.variables.target}
  Kept after filtering: {len(self.variables.keep_after_filtering)} variables
"""


# Convenience function for quick loading
def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        PipelineConfig instance
        
    Usage:
        from gar4ds import load_config
        config = load_config()
    """
    return PipelineConfig.load(config_path)
