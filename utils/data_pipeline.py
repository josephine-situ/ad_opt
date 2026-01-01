"""
Data pipeline functions for ad optimization.
Handles loading, cleaning, merging, and preparing datasets with embeddings.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .data_cleaning import convert_percent_to_float, clean_currency
from .date_features import (
    _region_to_country_code,
    _get_holiday_calendars,
    _is_holiday,
    calculate_days_to_next,
)
from .embeddings import get_tfidf_embeddings, get_bert_embeddings_pipeline


def load_and_combine_keyword_data(data_dir="data/reports"):
    """
    Load 2024 and 2025 keyword data and combine them.
    
    Args:
    - data_dir (str): Directory containing the data files.
    
    Returns:
    - kw_df (pd.DataFrame): Combined keyword data.
    """
    print("[Step 1] Loading and combining keyword data...")
    
    # Load 2024 data
    df_2024 = pd.read_excel(f"{data_dir}/Search keyword report (by Day 2024).xlsx")
    df_2024['Day'] = pd.to_datetime(df_2024['Day'])
    
    # Load 2025 data (UTF-16, tab-separated)
    df_2025 = pd.read_csv(
        f"{data_dir}/Search keyword report (By day 2025).csv",
        sep='\t',
        encoding='utf-16',
        dtype={'Day': str}
    )
    
    # Fix dates (force US format)
    df_2025['Day'] = pd.to_datetime(df_2025['Day'], dayfirst=False, errors='coerce')
    df_2025 = df_2025.dropna(subset=['Day'])
    
    # Clean currency columns
    cols_money = ['Cost', 'Avg. CPC', 'Conv. value']
    for col in cols_money:
        if col in df_2025.columns:
            df_2025[col] = df_2025[col].apply(clean_currency).replace('--', 0).astype(float)
    
    # Clean any other currency columns (Conv. value / cost, etc.)
    currency_cols = [col for col in df_2025.columns if any(x in col.lower() for x in ['cost', 'cpc', 'value', 'bid'])]
    for col in currency_cols:
        if col not in cols_money and df_2025[col].dtype == 'object':
            try:
                df_2025[col] = df_2025[col].apply(clean_currency).replace('--', 0).astype(float)
            except:
                pass  # Skip if can't convert
    
    # Clean impressions
    df_2025['Impr.'] = df_2025['Impr.'].astype(str).str.replace(',', '').replace('--', 0).astype(float)
    
    # Clean CTR and Conv. rate (percentage columns)
    df_2025['CTR'] = df_2025['CTR'].apply(convert_percent_to_float)
    if 'Conv. rate' in df_2025.columns:
        df_2025['Conv. rate'] = df_2025['Conv. rate'].apply(convert_percent_to_float)
    
    # Combine
    kw_df = pd.concat([df_2024, df_2025], ignore_index=True).sort_values('Day')
    
    # Keep only rows where Clicks > 0
    kw_df = kw_df[kw_df['Clicks'] > 0]
    
    print(f"  Data covers: {kw_df['Day'].min()} to {kw_df['Day'].max()}")
    print(f"  Total rows: {len(kw_df)}")
    
    return kw_df


def format_keyword_data(kw_df):
    """
    Format and clean keyword data (campaigns, regions, keywords).
    
    Args:
    - kw_df (pd.DataFrame): Raw keyword data.
    
    Returns:
    - kw_df (pd.DataFrame): Formatted keyword data.
    """
    print("[Step 2] Formatting keyword data...")
    
    kw_df['Campaign'] = kw_df['Campaign'].str.replace(r'\[.*?\]', '', regex=True)
    kw_df['Region'] = kw_df['Campaign'].str.split('-').str[-1].str.strip()
    kw_df['Region'] = kw_df['Region'].replace({'USA and CA': 'USA'})
    kw_df['Keyword'] = kw_df['Keyword'].str.replace(r'["\[\]]', '', regex=True).str.lower().str.strip()
    kw_df['Day'] = pd.to_datetime(kw_df['Day'])
    
    kw_df = kw_df[['Day', 'Keyword', 'Match type', 'Region', 'Avg. CPC', 'Cost', 'Conv. value', 'Clicks']].copy()
    
    return kw_df


def extract_date_features(kw_df, course_start_dts):
    """
    Extract temporal features (day of week, weekend, holidays, etc.).
    
    Args:
    - kw_df (pd.DataFrame): Keyword data with Day column.
    - course_start_dts (list): ISO date strings for course start dates.
    
    Returns:
    - kw_df (pd.DataFrame): Data with date features added.
    """
    print("[Step 3] Extracting date features...")
    
    kw_df['day_of_week'] = kw_df['Day'].dt.day_name()
    kw_df['is_weekend'] = (kw_df['Day'].dt.weekday >= 5).astype(int)
    kw_df['month'] = kw_df['Day'].dt.month
    
    # Holiday detection
    kw_df['_country_code'] = kw_df['Region'].apply(_region_to_country_code)
    years_needed = sorted(set(kw_df['Day'].dt.year.dropna().astype(int).tolist()))
    holiday_calendars = _get_holiday_calendars(kw_df['_country_code'].unique(), years=years_needed)
    
    kw_df['is_public_holiday'] = kw_df.apply(
        lambda row: _is_holiday(row, holiday_calendars), axis=1
    )
    
    # Days to next course start
    kw_df['days_to_next_course_start'] = kw_df['Day'].apply(
        lambda d: calculate_days_to_next(d, course_start_dts)
    )
    
    kw_df.drop(columns=['_country_code'], inplace=True)
    
    return kw_df


def filter_data_by_date(kw_df, min_date='2024-11-03'):
    """
    Filter data to remove early records (based on EDA insights).
    
    Args:
    - kw_df (pd.DataFrame): Keyword data.
    - min_date (str): ISO date string for cutoff.
    
    Returns:
    - kw_df (pd.DataFrame): Filtered data.
    """
    print(f"[Step 4] Filtering data from {min_date} onwards...")
    
    kw_df = kw_df[kw_df['Day'] >= min_date].copy()
    print(f"  Rows after filter: {len(kw_df)}")
    
    return kw_df


def get_gkp_data(gkp_dir='data/gkp'):
    """
    Load and tidy Google Keyword Planner data from saved keywords stats file.
    
    Processes monthly search columns and returns all monthly data.
    Returns data with columns: Keyword, searches_YYYY_MM (for each month), 
    Competition, Competition (indexed value), Top of page bid (low range), 
    Top of page bid (high range)
    
    Args:
    - gkp_dir (str): Directory containing GKP data files.
    
    Returns:
    - gkp_df (pd.DataFrame): Tidied GKP data
    """
    print("[Step 5] Loading Google Keyword Planner data...")
    
    gkp_path = Path(gkp_dir)
    
    # Find the most recent "Saved Keywords Stats" file
    if not gkp_path.exists():
        print(f"  Warning: GKP directory not found: {gkp_path}")
        return pd.DataFrame()  # Return empty dataframe
    
    # Find files matching the pattern
    gkp_files = list(gkp_path.glob('Saved Keywords Stats*.csv'))
    
    if not gkp_files:
        print(f"  Warning: No 'Saved Keywords Stats*.csv' files found in {gkp_path}")
        return pd.DataFrame()  # Return empty dataframe
    
    # Use the most recent file (by modification time)
    gkp_file = max(gkp_files, key=lambda f: f.stat().st_mtime)
    print(f"  Found GKP file: {gkp_file.name}")
    
    # Read file, skipping header rows (first 2 rows contain metadata)
    # File is UTF-16 encoded (like the 2025 keyword report)
    gkp_df = pd.read_csv(gkp_file, sep='\t', skiprows=2, encoding='utf-16')
    
    print(f"  Loaded {len(gkp_df)} rows from {gkp_file.name}")
    
    # Clean column names (remove extra spaces)
    gkp_df.columns = gkp_df.columns.str.strip()
    
    # Remove rows with empty/null keywords
    gkp_df = gkp_df.dropna(subset=['Keyword'])
    gkp_df = gkp_df[gkp_df['Keyword'].str.strip() != '']
    
    # Find all "Searches: " columns
    search_cols = [col for col in gkp_df.columns if col.startswith('Searches:')]
    
    # Parse dates from search columns and sort to find most recent
    from datetime import datetime
    search_dates = []
    for col in search_cols:
        # Format is "Searches: Mon YYYY"
        try:
            month_year = col.replace('Searches:', '').strip()
            # Parse the month year string (using abbreviated month format %b)
            date_obj = pd.to_datetime(month_year, format='%b %Y')
            search_dates.append((date_obj, col))
        except Exception as e:
            print(f"  Warning: Could not parse date from column '{col}': {e}")
    
    # Sort by date to find most recent
    search_dates.sort(key=lambda x: x[0])
    
    # Create columns with standardized format (searches_YYYY_MM)
    if search_dates:
        
        # Rename search columns to standardized format (searches_YYYY_MM)
        search_col_mapping = {}
        for date_obj, col in search_dates:
            new_col_name = f"searches_{date_obj.strftime('%Y_%m')}"
            search_col_mapping[col] = new_col_name
            gkp_df.rename(columns={col: new_col_name}, inplace=True)
    else:
        print(f"  Warning: No 'Searches: ' columns found in GKP data")
        search_col_mapping = {}
    
    # Keep only relevant columns: Keyword, all searches columns, and competition columns
    keep_cols = ['Keyword']
    keep_cols.extend([f"searches_{date_obj.strftime('%Y_%m')}" for date_obj, _ in search_dates])
    keep_cols.extend(['Competition', 'Competition (indexed value)',
                      'Top of page bid (low range)', 'Top of page bid (high range)'])
    
    # Only keep columns that exist
    available_cols = [col for col in keep_cols if col in gkp_df.columns]
    gkp_df = gkp_df[available_cols].copy()
    
    # Remove duplicate keywords, keeping first occurrence
    gkp_df = gkp_df.drop_duplicates(subset=['Keyword'], keep='first')
    
    print(f"  After cleaning: {len(gkp_df)} rows with unique keywords")
    
    return gkp_df


def impute_missing_data(df):
    """
    Impute missing data in a dataframe.
    
    For numeric columns: fill with 0
    For categorical columns: fill with "Missing"
    
    Reports percentage of data imputed with summary table.
    
    Args:
    - df (pd.DataFrame): Data to impute
    
    Returns:
    - df (pd.DataFrame): Data with missing values filled
    """
    print("[Step 5.5] Imputing missing data...")
    
    df = df.copy()
    initial_nulls = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    
    # Create summary table
    summary_data = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        pct_missing = (null_count / len(df) * 100) if null_count > 0 else 0
        
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_type = 'numeric (0)'
                df[col].fillna(0, inplace=True)
            else:
                fill_type = 'categorical (Missing)'
                df[col].fillna('Missing', inplace=True)
            
            summary_data.append({
                'Column': col,
                'Missing %': f"{pct_missing:.2f}%",
                'Fill Type': fill_type,
                'Count': null_count
            })
    
    # Print summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n  Imputation Summary:")
        print("  " + "-" * 85)
        for _, row in summary_df.iterrows():
            print(f"  {row['Column']:<35} {row['Missing %']:>10} {row['Fill Type']:<25} (n={row['Count']})")
        print("  " + "-" * 85)
    
    final_nulls = df.isnull().sum().sum()
    pct_imputed = (initial_nulls / total_cells * 100) if initial_nulls > 0 else 0
    
    print(f"  Total: Imputed {initial_nulls} missing values ({pct_imputed:.2f}% of total data)")
    
    return df


def merge_with_ads_data(kw_df, gkp_df=None):
    """
    Merge keyword data with GKP data (keyword planner info) by Keyword and Date.
    
    Calculates time series statistics from monthly search volumes:
    - last_month_searches: searches from the most recent month
    - three_month_avg: average searches over last 3 months
    - six_month_avg: average searches over last 6 months
    - mom_change: month-over-month percent change
    - yoy_change: year-over-year percent change
    - search_trend: linear trend (positive = growing, negative = declining)
    
    Args:
    - kw_df (pd.DataFrame): Keyword data with 'Day' column.
    - gkp_df (pd.DataFrame): Google Keyword Planner data. If None, will attempt to load from data/gkp/
    
    Returns:
    - merged_df (pd.DataFrame): Merged data with time series features.
    """
    print("[Step 6] Merging with GKP data by keyword and date...")
    
    # Load GKP data if not provided
    if gkp_df is None:
        gkp_df = get_gkp_data()
    
    # Make a copy to avoid modifying original
    merged_df = kw_df.copy()
    
    # Create merge key from Day
    merged_df['year_month'] = merged_df['Day'].dt.strftime('%Y_%m')
    
    if len(gkp_df) > 0:
        gkp_df = gkp_df.copy()
        gkp_df['Keyword'] = gkp_df['Keyword'].str.lower().str.strip()
        
        # Find which search columns are available (searches_YYYY_MM format)
        search_cols = sorted([col for col in gkp_df.columns if col.startswith('searches_')])
        print(f"  Found {len(search_cols)} monthly search columns")
        
        # Merge on Keyword
        merged_df = pd.merge(merged_df, gkp_df, on='Keyword', how='left')
        
        # Calculate time series statistics for each row
        if search_cols:
            print(f"  Calculating time series statistics...")
            
            def calculate_ts_stats(row, search_cols, gkp_df_dict):
                """Calculate time series statistics for a single keyword-month combination."""
                keyword = row['Keyword']
                year_month = row['year_month']  # Format: YYYY_MM
                
                stats = {}
                
                # Find all available search values for this keyword
                if keyword in gkp_df_dict:
                    kw_searches = gkp_df_dict[keyword]  # dict of YYYY_MM -> search value
                    
                    if kw_searches:
                        sorted_months = sorted(kw_searches.keys())
                        
                        # Find months before the current row's month
                        months_before = [m for m in sorted_months if m < year_month]
                        
                        if months_before:
                            # Use the most recent month available (most recent overall, not necessarily before)
                            stats['last_month_searches'] = kw_searches[sorted_months[-1]]
                            
                            # 3-month and 6-month averages using months before current date
                            three_m = [kw_searches[m] for m in months_before[-3:]]
                            six_m = [kw_searches[m] for m in months_before[-6:]]
                            
                            stats['three_month_avg'] = sum(three_m) / len(three_m) if three_m else None
                            stats['six_month_avg'] = sum(six_m) / len(six_m) if six_m else None
                            
                            # MoM change (comparing adjacent months before current date)
                            if len(months_before) >= 2:
                                prev_month = months_before[-2]
                                curr_month = months_before[-1]
                                prev_val = kw_searches.get(prev_month, 0)
                                curr_val = kw_searches.get(curr_month, 0)
                                if prev_val > 0:
                                    # Avoid division by zero
                                    stats['mom_change'] = ((curr_val - prev_val) / prev_val) * 100
                                else:
                                    if curr_val > 0:
                                        stats['mom_change'] = 100.0
                                    else:
                                        stats['mom_change'] = 0. 
                            else:
                                stats['mom_change'] = None
                            
                            # Search trend (linear regression slope of 6 months before current date)
                            if len(six_m) >= 2:
                                import numpy as np
                                x = np.arange(len(six_m))
                                y = np.array(six_m)
                                # Simple slope calculation
                                if len(x) > 1:
                                    slope = np.polyfit(x, y, 1)[0]
                                    stats['search_trend'] = slope
                                else:
                                    stats['search_trend'] = None
                            else:
                                stats['search_trend'] = None
                
                return stats
            
            # Build a dictionary mapping keyword -> {YYYY_MM -> search_value}
            gkp_dict = {}
            for _, row in gkp_df.iterrows():
                kw = row['Keyword']
                if kw not in gkp_dict:
                    gkp_dict[kw] = {}
                
                for col in search_cols:
                    month_key = col.replace('searches_', '')
                    gkp_dict[kw][month_key] = row[col]
            
            # Calculate stats for each row
            stats_list = []
            for idx, row in merged_df.iterrows():
                stats = calculate_ts_stats(row, search_cols, gkp_dict)
                stats_list.append(stats)
            
            # Add stats columns
            for col_name in ['last_month_searches', 'three_month_avg', 'six_month_avg', 
                           'mom_change', 'search_trend']:
                merged_df[col_name] = [s.get(col_name) for s in stats_list]
            
            print(f"  Added time series statistics columns")
        
        # Identify unmatched keywords (rows where GKP data wasn't found)
        gkp_cols = [col for col in gkp_df.columns if col not in ['Keyword', 'year_month']]
        if gkp_cols:
            unmatched_mask = merged_df[gkp_cols[0]].isnull()
            unmatched_keywords = merged_df[unmatched_mask]['Keyword'].unique()
            matched_keywords = merged_df[~unmatched_mask]['Keyword'].unique()
            
            print(f"  Matched {len(matched_keywords)} unique keywords from GKP")
            if len(unmatched_keywords) > 0:
                sample = list(unmatched_keywords[:5])
                print(f"  Unmatched {len(unmatched_keywords)} keywords (no GKP data): {sample}{'...' if len(unmatched_keywords) > 5 else ''}")
    
    # # debug
    # merged_df.to_csv('check_merge.csv', index=False)

    # Clean up temporary columns
    merged_df.drop(columns=['year_month'], inplace=True)
    
    # Drop individual month search columns
    search_cols_to_drop = [col for col in merged_df.columns if col.startswith('searches_')]
    merged_df.drop(columns=search_cols_to_drop, errors='ignore', inplace=True)
    
    # Calculate EPC (Expected conversion value Per Click)
    # Avoid division by zero by using fillna
    merged_df['EPC'] = merged_df.apply(
        lambda row: row['Conv. value'] / row['Clicks'] if row['Clicks'] > 0 else None,
        axis=1
    )
    print(f"  Calculated EPC (Expected Conversion value Per Click)")
    
    # Check for NaNs introduced at this step
    ts_stat_cols = ['last_month_searches', 'three_month_avg', 'six_month_avg', 'mom_change', 'search_trend']
    ts_stat_cols = [col for col in ts_stat_cols if col in merged_df.columns]
    
    print(f"\n  NaN check after time series calculations:")
    for col in ts_stat_cols:
        nan_count = merged_df[col].isnull().sum()
        nan_pct = (nan_count / len(merged_df) * 100)
        print(f"    {col:<25} {nan_count:>6} rows ({nan_pct:>6.2f}%)")
    
    print(f"  Merged rows: {len(merged_df)}")
    
    return merged_df


def add_embeddings(cleaned_df, embedding_method='bert', n_components=50, save_models=False, model_dir='models'):
    """
    Add keyword embeddings (TF-IDF or BERT).
    
    Args:
    - cleaned_df (pd.DataFrame): Data with keywords.
    - embedding_method (str): 'tfidf' or 'bert'.
    - n_components (int): Target embedding dimensionality.
    - save_models (bool): If True, save vectorizer/SVD/normalizer for later use.
    - model_dir (str): Directory to save models. Default 'models'.
    
    Returns:
    - df (pd.DataFrame): Data with embedding columns added.
    """
    import pickle
    from pathlib import Path
    
    print(f"[Step 8] Computing {embedding_method.upper()} embeddings...")
    
    unique_keywords = cleaned_df['Keyword'].unique()
    print(f"  Processing {len(unique_keywords)} unique keywords...")
    
    if embedding_method.lower() == 'tfidf':
        embedding_df, tfidf_models = get_tfidf_embeddings(
            unique_keywords, 
            n_components=n_components,
            ngram_range=(1, 2),
            min_df=1,
            return_model=True
        )
        embedding_df.rename(columns={'text': 'Keyword'}, inplace=True)
        
        if save_models:
            model_path = Path(model_dir) / f'tfidf_pipeline_{n_components}d.pkl'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(tfidf_models, f)
            print(f"  Saved TF-IDF pipeline to {model_path}")
            
    elif embedding_method.lower() == 'bert':
        embedding_df, bert_models = get_bert_embeddings_pipeline(
            unique_keywords,
            n_components=n_components,
            model_name='all-MiniLM-L6-v2',
            batch_size=32,
            return_model=True
        )
        embedding_df.rename(columns={'text': 'Keyword'}, inplace=True)
        
        if save_models:
            model_path = Path(model_dir) / f'bert_pipeline_{n_components}d.pkl'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            # Store model name in the pipeline dict for consistent reloading
            bert_models['model_name'] = 'all-MiniLM-L6-v2'
            with open(model_path, 'wb') as f:
                pickle.dump(bert_models, f)
            print(f"  Saved BERT pipeline to {model_path}")
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    # Merge embeddings back
    df = cleaned_df.merge(embedding_df, on='Keyword', how='left')
    
    print(f"  Embeddings added with shape: {len(embedding_df)} x {len(embedding_df.columns)}")
    
    return df


def prepare_train_test_split(df, test_size=0.25, random_state=42):
    """
    Prepare training and test datasets.
    
    Args:
    - df (pd.DataFrame): Full dataset.
    - test_size (float): Test set proportion.
    - random_state (int): Random seed.
    
    Returns:
    - df_train (pd.DataFrame): Training set.
    - df_test (pd.DataFrame): Test set.
    """
    print("[Step 9] Preparing train-test split...")
    
    # Identify embedding columns
    embedding_cols = [col for col in df.columns if 'tfidf' in col or 'bert' in col]
    
    # Feature and target columns
    # Use new time series statistics columns (from merge_with_ads_data)
    feature_cols = [
        'Match type', 'Region', 'day_of_week', 'is_weekend', 'month', 
        'is_public_holiday', 'days_to_next_course_start', 
        'last_month_searches', 'three_month_avg', 'six_month_avg',
        'mom_change', 'search_trend',
        'Competition (indexed value)', 
        'Top of page bid (low range)', 'Top of page bid (high range)', 'Avg. CPC'
    ] + embedding_cols
    
    # Check for missing required columns
    missing_cols = [col for col in feature_cols if col not in df.columns]
    target_cols = ['Conv. value', 'Clicks', 'EPC']
    missing_targets = [col for col in target_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}\nAvailable columns: {list(df.columns)}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}\nAvailable columns: {list(df.columns)}")
    
    X = df[feature_cols]
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    print(f"  Train set: {len(df_train)} rows")
    print(f"  Test set: {len(df_test)} rows")
    
    return df_train, df_test


def save_outputs(df, df_train, df_test, embedding_method='bert', output_dir='data/clean'):
    """
    Save processed data to CSV files, including unique keyword embeddings.
    
    Args:
    - df (pd.DataFrame): Full processed data.
    - df_train (pd.DataFrame): Training data.
    - df_test (pd.DataFrame): Test data.
    - embedding_method (str): 'tfidf' or 'bert' (for naming).
    - output_dir (str): Output directory.
    """
    print(f"[Step 10] Saving outputs to {output_dir}/...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save full dataset
    full_output = output_path / f'ad_opt_data_{embedding_method}.csv'
    df.to_csv(full_output, index=False)
    print(f"  Saved: {full_output}")
    
    # Save train/test
    train_output = output_path / f'train_{embedding_method}.csv'
    test_output = output_path / f'test_{embedding_method}.csv'
    
    df_train.to_csv(train_output, index=False)
    df_test.to_csv(test_output, index=False)
    print(f"  Saved: {train_output}")
    print(f"  Saved: {test_output}")
    
    # Extract and save unique keyword embeddings without NAs in embedding columns
    embedding_prefix = embedding_method.lower()
    embedding_cols = [col for col in df.columns if col.startswith(f'{embedding_prefix}_')]
    
    if embedding_cols:
        # Get unique keywords with their embeddings, dropping rows with NaN in embedding columns
        # Keep rows even if they have NAs in other columns
        unique_kw_embeddings = df[['Keyword'] + embedding_cols].drop_duplicates(subset=['Keyword'])
        unique_kw_embeddings = unique_kw_embeddings.dropna(subset=embedding_cols).reset_index(drop=True)
        
        embeddings_output = output_path / f'unique_keyword_embeddings_{embedding_method}.csv'
        unique_kw_embeddings.to_csv(embeddings_output, index=False)
        print(f"  Saved: {embeddings_output} ({len(unique_kw_embeddings)} rows)")
    else:
        print(f"  Warning: No embedding columns found for method '{embedding_method}'")


def load_embeddings(embeddings_file, embedding_method='bert', keywords=None):
    """
    Load embeddings from file.
    
    Args:
    - embeddings_file (str or Path): Path to CSV with keyword embeddings.
    - embedding_method (str): 'tfidf' or 'bert'.
    - keywords (list, optional): If provided, filter to only these keywords.
    
    Returns:
    - embeddings_df (pd.DataFrame): DataFrame with columns ['Keyword', 'embedding_0', ...]
                                   without any NaN values in embedding columns.
    """
    embeddings_file = Path(embeddings_file)
    
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    print(f"Loading embeddings from {embeddings_file}...")
    df = pd.read_csv(embeddings_file)
    
    # Get embedding column names (those starting with 'tfidf_' or 'bert_')
    embedding_prefix = embedding_method.lower()
    embedding_cols = [col for col in df.columns if col.startswith(f'{embedding_prefix}_')]
    
    # Drop rows with NaN in embedding columns
    df_clean = df.dropna(subset=embedding_cols).reset_index(drop=True)
    print(f"  Loaded {len(df_clean)} rows with complete embeddings")
    
    # Filter by keywords if provided
    if keywords is not None:
        keywords_set = set(keywords)
        df_clean = df_clean[df_clean['Keyword'].isin(keywords_set)].reset_index(drop=True)
        print(f"  Filtered to {len(df_clean)} keywords")
    
    return df_clean
