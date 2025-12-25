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


def load_and_combine_keyword_data(data_dir="."):
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
        df_2025[col] = df_2025[col].apply(clean_currency).replace('--', 0).astype(float)
    
    # Clean impressions
    df_2025['Impr.'] = df_2025['Impr.'].astype(str).str.replace(',', '').replace('--', 0).astype(float)
    
    # Clean CTR
    df_2025['CTR'] = df_2025['CTR'].apply(convert_percent_to_float)
    
    # Combine
    kw_df = pd.concat([df_2024, df_2025], ignore_index=True).sort_values('Day')
    kw_df['profit'] = kw_df['Conv. value'] - kw_df['Cost']
    
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
    kw_df['Keyword'] = kw_df['Keyword'].str.replace(r'["\[\]]', '', regex=True)
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


def merge_with_ads_data(kw_df, ads_file='combined_kw_ads_data2.csv'):
    """
    Merge keyword data with ads data (keyword planner info).
    
    Args:
    - kw_df (pd.DataFrame): Keyword data.
    - ads_file (str): Path to ads data CSV.
    
    Returns:
    - merged_df (pd.DataFrame): Merged data.
    """
    print("[Step 5] Merging with ads data...")
    
    ad_data = pd.read_csv(ads_file)
    
    # Clean merge key
    kw_df['Keyword_clean'] = kw_df['Keyword'].str.lower().str.strip()
    ad_data['Keyword_clean'] = ad_data['Keyword'].str.lower().str.strip()
    
    # Remove duplicates from ad_data
    if ad_data['Keyword_clean'].duplicated().any():
        print("  Warning: Removing duplicate keywords from ads data.")
        ad_data = ad_data.drop_duplicates(subset=['Keyword_clean'])
    
    # Merge
    ad_data_to_merge = ad_data.drop(columns=['Keyword'], errors='ignore')
    merged_df = pd.merge(kw_df, ad_data_to_merge, on='Keyword_clean', how='left')
    merged_df.drop(columns=['Keyword_clean'], inplace=True)
    
    print(f"  Merged rows: {len(merged_df)}")
    
    return merged_df


def clean_ads_data(merged_df):
    """
    Drop rows with missing ads data and clean percentage columns.
    
    Args:
    - merged_df (pd.DataFrame): Merged data.
    
    Returns:
    - cleaned_df (pd.DataFrame): Cleaned data.
    """
    print("[Step 6] Cleaning ads data...")
    
    subset_cols = [
        'Avg. monthly searches',
        'Three month change',
        'YoY change',
        'Competition',
        'Competition (indexed value)',
        'Top of page bid (low range)',
        'Top of page bid (high range)'
    ]
    
    # Drop NaNs
    cleaned_df = merged_df.dropna(subset=subset_cols, how='any').copy()
    rows_dropped = merged_df.shape[0] - cleaned_df.shape[0]
    print(f"  Dropped {rows_dropped} rows with missing ads data.")
    print(f"  Rows after cleaning: {len(cleaned_df)}")
    
    # Convert percentages
    cleaned_df['Three month change'] = cleaned_df['Three month change'].apply(convert_percent_to_float)
    cleaned_df['YoY change'] = cleaned_df['YoY change'].apply(convert_percent_to_float)
    
    return cleaned_df


def add_embeddings(cleaned_df, embedding_method='tfidf', n_components=50):
    """
    Add keyword embeddings (TF-IDF or BERT).
    
    Args:
    - cleaned_df (pd.DataFrame): Data with keywords.
    - embedding_method (str): 'tfidf' or 'bert'.
    - n_components (int): Target embedding dimensionality.
    
    Returns:
    - df (pd.DataFrame): Data with embedding columns added.
    """
    print(f"[Step 7] Computing {embedding_method.upper()} embeddings...")
    
    unique_keywords = cleaned_df['Keyword'].unique()
    print(f"  Processing {len(unique_keywords)} unique keywords...")
    
    if embedding_method.lower() == 'tfidf':
        embedding_df = get_tfidf_embeddings(
            unique_keywords, 
            n_components=n_components,
            ngram_range=(1, 2),
            min_df=1
        )
        embedding_df.rename(columns={'text': 'Keyword'}, inplace=True)
    elif embedding_method.lower() == 'bert':
        embedding_df = get_bert_embeddings_pipeline(
            unique_keywords,
            n_components=n_components,
            model_name='all-MiniLM-L6-v2',
            batch_size=32
        )
        embedding_df.rename(columns={'text': 'Keyword'}, inplace=True)
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
    print("[Step 8] Preparing train-test split...")
    
    # Identify embedding columns
    embedding_cols = [col for col in df.columns if 'tfidf' in col or 'bert' in col]
    
    # Feature and target columns
    feature_cols = [
        'Match type', 'Region', 'day_of_week', 'is_weekend', 'month', 
        'is_public_holiday', 'days_to_next_course_start', 'Avg. monthly searches',
        'Three month change', 'YoY change', 'Competition (indexed value)', 
        'Top of page bid (low range)', 'Top of page bid (high range)', 'Avg. CPC'
    ] + embedding_cols
    
    X = df[feature_cols]
    y = df[['Conv. value', 'Clicks']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    print(f"  Train set: {len(df_train)} rows")
    print(f"  Test set: {len(df_test)} rows")
    
    return df_train, df_test


def save_outputs(df, df_train, df_test, embedding_method='tfidf', output_dir='data'):
    """
    Save processed data to CSV files, including unique keyword embeddings.
    
    Args:
    - df (pd.DataFrame): Full processed data.
    - df_train (pd.DataFrame): Training data.
    - df_test (pd.DataFrame): Test data.
    - embedding_method (str): 'tfidf' or 'bert' (for naming).
    - output_dir (str): Output directory.
    """
    print(f"[Step 9] Saving outputs to {output_dir}/...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
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


def load_embeddings(embeddings_file, embedding_method='tfidf', keywords=None):
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
