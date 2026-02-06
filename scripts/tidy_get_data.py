"""
Data Preparation Pipeline for Ad Optimization
==============================================
Loads, cleans, and preprocesses keyword and ads data with embeddings.
Supports both TF-IDF and BERT embeddings for keyword representations.

Usage:
    python scripts/tidy_get_data.py --embedding-method tfidf
    python scripts/tidy_get_data.py --embedding-method bert
    python scripts/tidy_get_data.py --force-reload # Force full recompute if you updated source data

LLMs:
    request gpus with salloc -p pi_dbertsim --gpus=1
    python scripts/tidy_get_data.py --course sys_eng --embedding-method llm # Use LLM-based relevance scoring
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_and_combine_keyword_data,
    format_keyword_data,
    get_date_features,
    filter_data_by_date,
    get_gkp_data,
    impute_missing_data,
    merge_with_ads_data,
    add_embeddings,
    prepare_train_test_split,
    save_outputs,
    setup_tee_logging,
)
from utils.date_features import COURSE_START_DATES_MAP, COURSE_MIN_DATES


def load_or_cache(func, cache_path, force_reload=False, *args, **kwargs):
    """
    Load data from cache if exists, otherwise compute and cache.
    
    Args:
        func: Function to call if cache doesn't exist
        cache_path: Path to parquet cache file
        force_reload: If True, ignore cache and recompute
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Loaded or computed dataframe
    """
    cache_path = Path(cache_path)
    
    if cache_path.exists() and not force_reload:
        try:
            print(f"  [Cache] Loading from {cache_path.name}")
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"  [Warning] Cache corrupted ({type(e).__name__}), rebuilding...")
            cache_path.unlink()  # Delete corrupted cache
    
    print(f"  [Computing] Running {func.__name__}...")
    result = func(*args, **kwargs)
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    result.to_parquet(cache_path)
    print(f"  [Saved] Cached to {cache_path.name}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Data preparation pipeline for ad optimization."
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='bert',
        choices=['tfidf', 'bert', 'llm'],
        help='Embedding method: tfidf, bert, or llm (default: bert). LLM uses Prometheus for relevance scoring.'
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
        default=None,
        help='Output directory for processed data (default: data/{course}/clean)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Input data directory (default: data/{course}/reports)'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force reload from source files, skip all caches'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help=(
            "Log file path. Default: logs/tidy_get_data_<timestamp>.log. "
            "Set to empty string '' to disable file logging."
        ),
    )
    parser.add_argument(
        '--course',
        type=str,
        default='gen_ai',
        help='Course name (default: gen_ai)'
    )
    
    args = parser.parse_args()

    # Determine paths based on course
    base_dir = Path(f'data/{args.course}')
    if args.output_dir is None:
        args.output_dir = str(base_dir / 'clean')
    if args.data_dir is None:
        args.data_dir = str(base_dir / 'reports')

    # Tee stdout/stderr to a log file (plus console).
    log_path = setup_tee_logging(
        log_file=args.log_file,
        default_log_dir='logs',
        default_log_prefix=f'tidy_get_data_{args.course}',
    )
    if log_path is not None:
        print(f"[Logging] Tee output to {log_path}")
    
    print("=" * 70)
    print("Data Preparation Pipeline for Ad Optimization")
    print("=" * 70)
    print(f"Course: {args.course}")
    print(f"Embedding method: {args.embedding_method}")
    print(f"N components: {args.n_components}")
    print(f"Force reload: {args.force_reload}")
    print("=" * 70)
    
    try:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        cache_path = base_dir / 'cache'
        cache_path.mkdir(exist_ok=True, parents=True)
        
        # Pipeline with automatic caching at each time-consuming step
        print("\n[Step 1] Load and combine keyword data...")
        kw_df = load_or_cache(
            load_and_combine_keyword_data,
            cache_path / 'step1_combined.parquet',
            args.force_reload,
            args.data_dir
        )
        
        print("\n[Step 2] Format keyword data...")
        kw_df = format_keyword_data(kw_df)
        
        print("\n[Step 3] Extract date features...")
        kw_df = load_or_cache(
            get_date_features,
            cache_path / 'step3_features.parquet',
            args.force_reload,
            kw_df,
            COURSE_START_DATES_MAP.get(args.course, [])
        )
        
        print("\n[Step 4] Filter data by date...")
        kw_df = load_or_cache(
            filter_data_by_date,
            cache_path / 'step4_filtered.parquet',
            args.force_reload,
            kw_df,
            COURSE_MIN_DATES.get(args.course, '2024-11-03')
        )
        
        print("\n[Step 5] Load GKP data...")
        gkp_df = get_gkp_data(gkp_dir=base_dir / 'gkp')
        
        print("\n[Step 5.5] Impute missing data...")
        kw_df = impute_missing_data(kw_df)
        gkp_df = impute_missing_data(gkp_df)
        
        print("\n[Step 6] Merge with GKP data and calculate search stats...")
        merged_df = load_or_cache(
            merge_with_ads_data,
            cache_path / 'step6_merged.parquet',
            args.force_reload,
            kw_df,
            gkp_df,
            use_fuzzy_matching=True,
            drop_unmatched_gkp=True,
            unmatched_print_limit=200,
        )
        
        print("\n[Step 6.5] Remove outlier rows...")
        cleaned_df = load_or_cache(
            lambda df: df[df['Avg. CPC'] < 50] if 'Avg. CPC' in df.columns else df,  # Example: remove rows with CPC >= 10,000
            cache_path / 'step6_5_cleaned.parquet',
            args.force_reload,
            merged_df
        )  
        
        print(f"\n[Step 7] Add {args.embedding_method.upper()} {'scores' if args.embedding_method == 'llm' else 'embeddings'}...")
        df = load_or_cache(
            add_embeddings,
            cache_path / f'step8_embeddings_{args.embedding_method}.parquet',
            args.force_reload,
            cleaned_df,
            args.embedding_method,
            args.n_components,
            True,  # save_models=True
            args.output_dir,  # model_dir
            args.course,  # course for LLM scoring
            str(cache_path),  # cache_dir for LLM scores
        )
        
        # Remove rows with NaN values before splitting
        print("\n[Step 8] Removing rows with NaN values...")
        df = df.dropna()
        print(f"  Data after NaN removal: {len(df)} rows")
        
        print("\n[Step 9] Preparing train-test split and saving outputs...")
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
