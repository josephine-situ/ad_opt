"""
Data Preparation Pipeline for Ad Optimization
==============================================
Loads, cleans, and preprocesses keyword and ads data with embeddings.
Supports both TF-IDF and BERT embeddings for keyword representations.

Usage:
    python tidy_get_data.py --embedding-method tfidf
    python tidy_get_data.py --embedding-method bert
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_and_combine_keyword_data,
    format_keyword_data,
    extract_date_features,
    filter_data_by_date,
    merge_with_ads_data,
    clean_ads_data,
    add_embeddings,
    prepare_train_test_split,
    save_outputs,
)
from utils.date_features import COURSE_START_DATES


def main():
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for ad optimization."
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='tfidf',
        choices=['tfidf', 'bert'],
        help='Embedding method: tfidf or bert (default: tfidf)'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=50,
        help='Number of embedding dimensions (default: 50)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='clean_data',
        help='Output directory for processed data (default: clean_data)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='raw_data',
        help='Input data directory (default: raw_data)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Data Preparation Pipeline for Ad Optimization")
    print("=" * 70)
    print(f"Embedding method: {args.embedding_method}")
    print(f"N components: {args.n_components}")
    print("=" * 70)
    
    try:
        # Pipeline
        kw_df = load_and_combine_keyword_data(args.data_dir)
        kw_df = format_keyword_data(kw_df)
        kw_df = extract_date_features(kw_df, COURSE_START_DATES)
        kw_df = filter_data_by_date(kw_df)
        merged_df = merge_with_ads_data(kw_df, f"{args.data_dir}/combined_kw_ads_data2.csv")
        cleaned_df = clean_ads_data(merged_df)
        df = add_embeddings(cleaned_df, embedding_method=args.embedding_method, n_components=args.n_components)
        
        # Remove rows with NaN values before splitting
        print("Removing rows with NaN values...")
        df = df.dropna()
        print(f"  Data after NaN removal: {len(df)} rows")
        
        df_train, df_test = prepare_train_test_split(df)
        save_outputs(df, df_train, df_test, embedding_method=args.embedding_method, output_dir=args.output_dir)
        
        print("=" * 70)
        print("✓ Pipeline completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
