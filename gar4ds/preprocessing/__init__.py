"""
GAR4DS Preprocessing Module
===========================

Data preprocessing and panel matrix creation.

Modules:
    - data_preprocessing: Convert CSV to efficient formats
    - build_panel_matrix: Create panel data with configurable aggregation
    - add_elevation: Add elevation data from API
    - multicollinearity: VIF analysis and filtering
"""

from .data_preprocessing import run_preprocessing
from .build_panel_matrix import run_panel_builder
from .add_elevation import run_add_elevation
from .multicollinearity_analysis import run_multicollinearity_analysis
from .filter_multicollinearity import run_filter_multicollinearity

__all__ = [
    "run_preprocessing",
    "run_panel_builder",
    "run_add_elevation",
    "run_multicollinearity_analysis",
    "run_filter_multicollinearity",
]
