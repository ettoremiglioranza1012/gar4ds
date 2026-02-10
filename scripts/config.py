"""
Temporal Configuration for Panel Data Pipeline
===============================================
Controls the temporal aggregation frequency for the entire pipeline.

Supported frequencies:
- 'daily': Daily aggregation (D)
- 'weekly': Weekly aggregation starting Monday (W-MON) - DEFAULT
- 'monthly': Monthly aggregation starting first day (MS)

To change frequency:
1. Update TEMPORAL_FREQUENCY below
2. Re-run the entire pipeline from build_panel_matrix onwards
3. Results will be saved with frequency suffix for comparison

Note: Changing frequency requires rebuilding panel matrix and re-running all analyses.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Select temporal aggregation frequency
TEMPORAL_FREQUENCY = 'daily'

# ============================================================================
# FREQUENCY SPECIFICATIONS (Do not modify)
# ============================================================================

FREQUENCY_SPECS = {
    'daily': {
        'resample_code': 'D',
        'label': 'Daily',
        'label_lower': 'daily',
        'period_column': 'date',
        'period_label': 'day',
        'period_label_plural': 'days',
        'description': 'Daily aggregation (24-hour periods)'
    },
    'weekly': {
        'resample_code': 'W-MON',
        'label': 'Weekly',
        'label_lower': 'weekly',
        'period_column': 'week_start',
        'period_label': 'week',
        'period_label_plural': 'weeks',
        'description': 'Weekly aggregation (Monday to Sunday)'
    },
    'monthly': {
        'resample_code': 'MS',
        'label': 'Monthly',
        'label_lower': 'monthly',
        'period_column': 'month_start',
        'period_label': 'month',
        'period_label_plural': 'months',
        'description': 'Monthly aggregation (calendar months)'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(frequency=None):
    """
    Get configuration for specified frequency.
    
    Args:
        frequency: One of 'daily', 'weekly', 'monthly'. 
                   If None, uses TEMPORAL_FREQUENCY global.
    
    Returns:
        dict: Configuration dictionary with all settings
    """
    if frequency is None:
        frequency = TEMPORAL_FREQUENCY
    
    if frequency not in FREQUENCY_SPECS:
        raise ValueError(
            f"Invalid frequency '{frequency}'. "
            f"Must be one of: {list(FREQUENCY_SPECS.keys())}"
        )
    
    return FREQUENCY_SPECS[frequency]


def get_file_suffix():
    """Get filename suffix for current frequency (e.g., '_weekly')"""
    return f"_{TEMPORAL_FREQUENCY}"


def get_output_path(base_name, extension='.parquet'):
    """
    Generate output path with frequency suffix.
    
    Args:
        base_name: Base filename without extension
        extension: File extension (default: .parquet)
    
    Returns:
        str: Filename with frequency suffix (e.g., 'panel_data_matrix_weekly.parquet')
    """
    return f"{base_name}{get_file_suffix()}{extension}"


# ============================================================================
# VALIDATION
# ============================================================================

# Validate configuration on import
if TEMPORAL_FREQUENCY not in FREQUENCY_SPECS:
    raise ValueError(
        f"Invalid TEMPORAL_FREQUENCY: '{TEMPORAL_FREQUENCY}'. "
        f"Must be one of: {list(FREQUENCY_SPECS.keys())}"
    )

# Print configuration on import
if __name__ != "__main__":
    print(f"✓ Temporal configuration loaded: {FREQUENCY_SPECS[TEMPORAL_FREQUENCY]['label']} aggregation")
