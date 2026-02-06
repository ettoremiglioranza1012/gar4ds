"""
GAR4DS - PM10 Spatial Analysis Pipeline
========================================

A configurable pipeline for analyzing PM10 pollution patterns
from the Po Valley to the Alpine Region.

Modules:
    - config: Configuration loading and validation
    - preprocessing: Data preprocessing and panel matrix creation
    - analysis: Spatial analysis, EDA, and model specification
    - visualization: Interactive maps and static visualizations
"""

__version__ = "1.0.0"
__author__ = "GAR4DS Team"

from .config import PipelineConfig, load_config

__all__ = [
    "PipelineConfig",
    "load_config",
    "__version__",
]
