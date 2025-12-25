"""
Data cleaning utilities for currency, percentages, and text normalization.
"""

import pandas as pd
import numpy as np


def clean_currency(x):
    """
    Converts currency strings to float by removing '$', ',', and whitespace.
    
    Args:
    - x: Value to clean (string or numeric).
    
    Returns:
    - float: Cleaned numeric value.
    """
    return float(x.replace('$', '').replace(',', '').strip()) if isinstance(x, str) else x


def convert_percent_to_float(value):
    """
    Converts percentage strings to float decimals (e.g., '900%' -> 9.0).
    
    Args:
    - value: Percentage string or numeric value.
    
    Returns:
    - float: Decimal representation, or np.nan if invalid.
    
    Examples:
    - '50%' -> 0.5
    - '900%' -> 9.0
    - '∞' -> np.inf
    """
    # Pass through NaNs
    if pd.isna(value):
        return np.nan
    
    value = str(value).strip()
    
    # Handle '∞' symbol (infinite change)
    if value == '∞':
        return np.inf
    
    # Try to remove '%' and convert to float
    try:
        cleaned_value = value.replace('%', '').strip()
        float_value = float(cleaned_value)
        # Convert to decimal (e.g., 900% -> 9.0, 0% -> 0.0)
        return float_value / 100.0
    except ValueError:
        # In case there are other non-numeric values
        return np.nan
