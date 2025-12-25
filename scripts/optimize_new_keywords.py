#!/usr/bin/env python3
"""
Complete end-to-end pipeline for new keyword bid optimization.

This script:
 1. Computes new keywords (combined - unique)
 2. Generates TF-IDF and BERT embeddings for new keywords
 3. Creates keyword feature matrix from embeddings + ads data
 4. Runs linear programming optimization using bid_optimization.optimize_bids_embedded()

Outputs:
 - `clean_data/new_keyword_embeddings_tfidf.csv`
 - `clean_data/new_keyword_embeddings_bert.csv`
 - `opt_results/optimized_bids_new_keywords_{embedding_method}.csv`

Usage:
    python3 scripts/optimize_new_keywords.py --embedding-method bert --budget 400 --max-bid 50
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_tfidf_embeddings, get_bert_embeddings_pipeline

# Import optimization functions from bid_optimization
from scripts.bid_optimization import (
    optimize_bids_embedded,
    extract_solution,
    load_weights_from_csv,
)

# Check for required libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not found. Install with: pip install gurobipy")
    print("Note: Gurobi requires a valid license.")
    sys.exit(1)


def normalize(s):
    if pd.isna(s):
        return ''
    return str(s).strip().lower()


def generate_embeddings_for_new_keywords(root, embedding_method):
    """
    Generate embeddings for new keywords (combined - unique).
    
    Returns:
        DataFrame with embeddings
    """
    combined_file = root / 'raw_data' / 'combined_kw_ads_data2.csv'
    unique_file = root / 'raw_data' / 'unique_keywords.csv'

    if not combined_file.exists() or not unique_file.exists():
        print("ERROR: Could not find source files to compute new keywords.")
        print(f"Looked for: {combined_file}, {unique_file}")
        sys.exit(1)

    combined = pd.read_csv(combined_file)
    unique = pd.read_csv(unique_file)

    combined_kw = set(normalize(k) for k in combined['Keyword'].dropna().unique())
    unique_kw = set(normalize(k) for k in unique['Keyword'].dropna().unique())

    new_norm = combined_kw - unique_kw
    keywords = [k for k in combined['Keyword'].dropna().unique() if normalize(k) in new_norm]

    if len(keywords) == 0:
        print("No new keywords found. Nothing to embed.")
        return None

    print(f"Generating embeddings for {len(keywords)} keywords")

    # Generate embeddings
    embedding_df = None
    if embedding_method == 'tfidf':
        try:
            embedding_df = get_tfidf_embeddings(keywords, n_components=50, ngram_range=(1, 2), min_df=1)
            embedding_df = embedding_df.rename(columns={'text': 'Keyword'})
            cols = ['Keyword'] + [c for c in embedding_df.columns if c != 'Keyword']
            embedding_df = embedding_df[cols]
            print(f"Generated TF-IDF embeddings for {len(embedding_df)} keywords")
        except Exception as e:
            print("ERROR generating TF-IDF embeddings:", e)
            raise
    elif embedding_method == 'bert':
        try:
            embedding_df = get_bert_embeddings_pipeline(keywords, n_components=50, model_name='all-MiniLM-L6-v2', batch_size=32)
            embedding_df = embedding_df.rename(columns={'text': 'Keyword'})
            cols = ['Keyword'] + [c for c in embedding_df.columns if c != 'Keyword']
            embedding_df = embedding_df[cols]
            print(f"Generated BERT embeddings for {len(embedding_df)} keywords")
        except Exception as e:
            print("ERROR generating BERT embeddings:", e)
            print("If this is an import/model error, try: pip install sentence-transformers transformers")
            raise

    return embedding_df


def load_weights(embedding_method='bert', models_dir='models'):
    """Load model weights from CSV files (wrapper around bid_optimization function)."""
    return load_weights_from_csv(embedding_method, models_dir)


def create_feature_matrix(embedding_df, ads_df, embedding_method='bert', target_day=None):
    """
    Create feature matrix by merging embeddings with ads data.
    
    Args:
        embedding_df: DataFrame with embeddings
        ads_df: DataFrame with ads data
        embedding_method: 'bert' or 'tfidf'
    
    Returns:
        DataFrame with keywords and their features
    """
    print(f"Creating feature matrix...")
    regions = ["USA", "A", "B", "C"]
    match_types = ["Broad match", "Exact match", "Phrase match"]
    
    # Merge embeddings with ads data on keyword
    emb_df = embedding_df.copy()
    ads_copy = ads_df.copy()
    
    # Convert percentage strings to numeric
    def convert_percentage(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            # Remove '%' and any other non-numeric characters
            cleaned = val.rstrip('%').strip()
            if cleaned == '?' or cleaned == '' or cleaned.lower() == 'nan':
                return 0.0
            try:
                return float(cleaned) / 100.0
            except ValueError:
                return 0.0
        try:
            return float(val) / 100.0 if val > 1 else float(val)
        except (ValueError, TypeError):
            return 0.0
    
    ads_copy['Three month change'] = ads_copy['Three month change'].apply(convert_percentage)
    ads_copy['YoY change'] = ads_copy['YoY change'].apply(convert_percentage)
    
    # Fill NaN values with 0 for numeric columns
    ads_copy['Competition'] = ads_copy['Competition'].fillna(0.0)
    ads_copy['Competition (indexed value)'] = ads_copy['Competition (indexed value)'].fillna(0.0)
    ads_copy['Top of page bid (low range)'] = ads_copy['Top of page bid (low range)'].fillna(0.0)
    ads_copy['Top of page bid (high range)'] = ads_copy['Top of page bid (high range)'].fillna(0.0)
    ads_copy['Avg. monthly searches'] = ads_copy['Avg. monthly searches'].fillna(0.0)
    
    emb_df['_norm'] = emb_df['Keyword'].apply(normalize)
    ads_copy['_norm'] = ads_copy['Keyword'].apply(normalize)
    
    merged = emb_df.merge(ads_copy, on='_norm', how='inner', suffixes=('_emb', '_ads'))
    print(f"Matched {len(merged)} keywords with ads data")
    
    if len(merged) == 0:
        print("ERROR: No keywords matched between embeddings and ads data!")
        sys.exit(1)
    
    # Use keyword from embeddings
    merged['Keyword'] = merged['Keyword_emb']
    
    # Extract embedding columns
    embedding_prefix = embedding_method.lower()
    embedding_cols = [col for col in merged.columns if col.startswith(f'{embedding_prefix}_')]
    
    # Select feature columns (embeddings + all relevant ads features including temporal)
    feature_cols = [
        'Keyword', 'Competition', 'Competition (indexed value)',
        'Top of page bid (low range)', 'Top of page bid (high range)',
        'Avg. monthly searches', 'Three month change', 'YoY change'
    ] + embedding_cols
    
    feature_matrix = merged[[c for c in feature_cols if c in merged.columns]].copy()
    
    # Add temporal features that might be needed by the model
    if 'days_to_next_course_start' not in feature_matrix.columns:
        feature_matrix['days_to_next_course_start'] = 30.0  # Default value
    
    # Add 'month' column if not present (derived from day of month or defaulted)
    if 'month' not in feature_matrix.columns:
        feature_matrix['month'] = 12  # December as default
    
    # Rename only specific columns to match training data format exactly
    # Based on inspection of weight files, some features use spaces, some use underscores
    rename_map = {
        'Competition (indexed value)': 'Competition_(indexed_value)',
        'Top of page bid (low range)': 'Top_of_page_bid_(low_range)',
        'Top of page bid (high range)': 'Top_of_page_bid_(high_range)',
        'Avg. monthly searches': 'Avg_ monthly searches',  # Note: space preserved after 'Avg_'
        'Avg. CPC': 'Avg_ CPC',  # Note: space preserved after 'Avg_'
        # Don't rename these - they use spaces in the model:
        # 'Three month change', 'YoY change'
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in feature_matrix.columns:
            feature_matrix = feature_matrix.rename(columns={old_name: new_name})
    
    # Now replace remaining dots with underscores (for generic handling)
    feature_matrix.columns = feature_matrix.columns.str.replace('.', '_', regex=False)
    
    # Make sure all numeric columns are actually numeric
    numeric_cols = feature_matrix.select_dtypes(exclude=['object']).columns
    for col in numeric_cols:
        feature_matrix[col] = pd.to_numeric(feature_matrix[col], errors='coerce').fillna(0.0)
    
    print(f"Feature matrix: {len(feature_matrix)} keywords x {len(feature_matrix.columns)} features")

    combinations = []
    for kw in feature_matrix['Keyword']:
        for region in regions:
            for match in match_types:
                combinations.append({
                    'Keyword': kw,
                    'Region': region,
                    'Match type': match,
                })
    
    combo_df = pd.DataFrame(combinations)
    combo_df['Day'] = target_day
    print(f"  Created {len(combo_df)} keyword-region-match combinations")
        
    # Merge using asof on date, exact match on categorical columns
    result = combo_df.merge(
        feature_matrix,
        on='Keyword',
        how='left'
    )
    
    # If target_day != latest date in data, we need to adjust date features
    from utils.date_features import calculate_date_features
    
    target_date = pd.to_datetime(datetime.today())
    date_features = calculate_date_features(target_date, regions=regions)
    
    print(f"  Adjusting date features to today ({target_date.date()})...")
    result['day_of_week'] = date_features['day_of_week']
    result['is_weekend'] = date_features['is_weekend']
    result['month'] = date_features['month']
    result['is_public_holiday'] = date_features['is_public_holiday']
    result['days_to_next_course_start'] = date_features['days_to_next_course_start']
    
    print(f"  Adjusted temporal features for {target_date.date()}")
    
    # Extract the order information for rows that actually made it into the feature matrix
    # (BEFORE dropping Day/Keyword columns so we still have access to them)
    keyword_idx_list = []
    region_list = []
    match_list = []
    
    for _, row in result.iterrows():
        kw_idx = feature_matrix[feature_matrix['Keyword'] == row['Keyword']].index[0]
        keyword_idx_list.append(kw_idx)
        region_list.append(row['Region'])
        match_list.append(row['Match type'])
    
    # Drop day column
    result = result.drop(columns=['Day', 'Keyword'])

    # Reset index
    result.reset_index(drop=True, inplace=True)
    
    # Convert column names: replace dots with underscores to match model feature names
    result.columns = result.columns.str.replace('.', '_', regex=False)
    
    # One-hot encode categorical columns (if any)
    categorical_cols = result.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"  One-hot encoding categorical columns: {categorical_cols}")
        result = pd.get_dummies(result, columns=categorical_cols, drop_first=False)
    
    # Convert all columns to float for numeric operations
    result = result.astype(float)
    
    print(f"  Final feature matrix shape: {result.shape}")
    print(f"  Columns: {result.columns.tolist()[:15]}...")
    
    return result, keyword_idx_list, region_list, match_list





def main():
    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Optimize new keyword bids end-to-end.")
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='bert',
        choices=['tfidf', 'bert'],
        help='Embedding method to use (default: bert)'
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=400,
        help='Total budget for bids (default: 400)'
    )
    parser.add_argument(
        '--max-bid',
        type=float,
        default=50.0,
        help='Maximum individual bid (default: 50.0)'
    )
    parser.add_argument(
        '--target-day',
        type=str,
        required=True,
        help='Target day for optimization (format: YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    target_day = args.target_day
    embedding_method = args.embedding_method

    print("=" * 70)
    print("New Keyword Bid Optimization Pipeline")
    print("=" * 70)
    print(f"Embedding method: {embedding_method}")
    print("=" * 70)

    # Step 1: Generate embeddings
    print("\n[Step 1] Generating embeddings...")
    embedding_df = generate_embeddings_for_new_keywords(root, embedding_method)
    
    if embedding_df is None:
        print("Failed to generate embeddings")
        sys.exit(1)
    
    # Save embeddings
    out_embedding = root / 'clean_data' / f'new_keyword_embeddings_{embedding_method}.csv'
    out_embedding.parent.mkdir(parents=True, exist_ok=True)
    embedding_df.to_csv(out_embedding, index=False)
    print(f"Saved embeddings: {out_embedding}")

    # Step 2: Load ads data and create feature matrix
    print(f"\n[Step 2] Creating feature matrix...")
    ads_file = root / 'raw_data' / 'combined_kw_ads_data2.csv'
    ads_df = pd.read_csv(ads_file)
    
    feature_matrix, keyword_idx_list, region_list, match_list = create_feature_matrix(embedding_df, ads_df, embedding_method, target_day=target_day)
    feature_matrix['days_to_next_course_start'] = 30.0  # Placeholder
    # Step 3: Load model weights
    print(f"\n[Step 3] Loading model weights...")
    weights_dict = load_weights(embedding_method, root / 'models')

    # Step 4: Run optimization using optimize_bids_embedded from bid_optimization.py
    print(f"\n[Step 4] Running bid optimization with embedded LR constraints...")
    model, b, z, y, f_eff, g_eff = optimize_bids_embedded(
        feature_matrix,
        weights_dict,
        budget=args.budget,
        max_bid=args.max_bid
    )
    
    # Extract solution
    if model.status == 2 or model.status == 9:  # OPTIMAL or TIME_LIMIT
        output_dir = root / 'opt_results'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'optimized_bids_new_keywords_{embedding_method}.csv'
        
        bids_df = extract_solution(model, b, z, y, f_eff, g_eff, embedding_df, keyword_idx_list, region_list, match_list, X=feature_matrix, weights_dict=weights_dict)
        
        # Save results
        bids_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        print(f"\nTop 10 keywords by bid:")
        print(bids_df.head(10).to_string(index=False))
    else:
        print(f"Optimization failed with status {model.status}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ New keyword bid optimization completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
