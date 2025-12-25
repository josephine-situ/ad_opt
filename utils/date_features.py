"""
Temporal feature extraction: holidays, weekends, days-to-event, etc.
"""

import pandas as pd
import numpy as np
import holidays
import datetime


# Course start dates (fixed)
COURSE_START_DATES = ['2024-10-15', '2025-02-10', '2025-09-29', '2026-02-09']


def _region_to_country_code(r):
    """
    Maps region strings to holiday country codes.
    
    Args:
    - r: Region name (e.g., 'USA', 'Canada', 'CA', 'US').
    
    Returns:
    - str: Country code ('US' or 'CA'), defaults to 'US' if unrecognized.
    """
    if pd.isna(r):
        return 'US'
    s = str(r).upper()
    if 'CAN' in s or s == 'CA':
        return 'CA'
    if 'USA' in s or s == 'US':
        return 'US'
    # default to US if unknown
    return 'US'


def _get_holiday_calendars(country_codes, years=None):
    """
    Builds a dictionary of holiday calendars from a list of country codes.
    
    Args:
    - country_codes (list or pd.Series): List of country codes (e.g., ['US', 'CA']).
    - years (list): List of years to generate holidays for. If None, uses current year.
    
    Returns:
    - dict: Mapping from country code to holidays.CountryHoliday object.
    """
    if years is None:
        years = [datetime.date.today().year]
    
    holiday_calendars = {}
    for c in set(country_codes):  # Use set for unique codes
        if c not in holiday_calendars:
            try:
                holiday_calendars[c] = holidays.CountryHoliday(c, years=years)
            except Exception:
                holiday_calendars[c] = None
    return holiday_calendars


def _is_holiday(row, holiday_calendars, day_col='Day', country_col='_country_code'):
    """
    Helper function for pd.Series.apply() to check if a row's date is a holiday.
    
    Args:
    - row (pd.Series): A DataFrame row.
    - holiday_calendars (dict): Dictionary from _get_holiday_calendars().
    - day_col (str): Column name containing the date. Default 'Day'.
    - country_col (str): Column name with country code. Default '_country_code'.
    
    Returns:
    - int: 1 if holiday, 0 otherwise.
    """
    cal = holiday_calendars.get(row[country_col])
    if cal is None:
        return 0
    
    date_val = row[day_col]
    
    # Try to get .date() if it's a datetime object
    try:
        date_obj = date_val.date()
    except AttributeError:
        # Assume it's already a date object
        date_obj = date_val
    except Exception:
        # If it's not a date-like object at all
        return 0
    
    return 1 if date_obj in cal else 0


def calculate_days_to_next(d, course_start_dts=None):
    """
    Calculate days from date d to the next course start date.
    
    Args:
    - d (datetime): The reference date.
    - course_start_dts (list): List of ISO date strings for course start dates. 
      If None, uses COURSE_START_DATES.
    
    Returns:
    - int or nan: Minimum days to next course start, or nan if no future course.
    """
    if course_start_dts is None:
        course_start_dts = COURSE_START_DATES
    starts = pd.to_datetime(course_start_dts).sort_values()
    diffs = [(int((cs - d).days)) for cs in starts if (cs - d).days >= 0]
    return int(min(diffs)) if diffs else np.nan


def calculate_date_features(target_date, regions=None):
    """
    Calculate all date-related features for a target date.
    
    Args:
    - target_date (datetime or str): The target date to calculate features for.
    - regions (list): List of regions for holiday calculation. If None, defaults to ['USA'].
    
    Returns:
    - dict: Dictionary with keys: day_of_week, is_weekend, month, is_public_holiday, days_to_next_course_start
    """
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    
    if regions is None:
        regions = ['USA']
    
    # Basic date features
    features = {
        'day_of_week': target_date.strftime('%A'),
        'is_weekend': 1 if target_date.weekday() >= 5 else 0,  # 5=Saturday, 6=Sunday
        'month': target_date.month,
        'days_to_next_course_start': calculate_days_to_next(target_date),
    }
    
    # Holiday features - check if date is a holiday in any of the regions
    country_codes = [_region_to_country_code(r) for r in regions]
    holiday_calendars = _get_holiday_calendars(country_codes)
    
    is_holiday = 0
    for country_code in set(country_codes):
        cal = holiday_calendars.get(country_code)
        if cal and target_date.date() in cal:
            is_holiday = 1
            break
    
    features['is_public_holiday'] = is_holiday
    
    return features
