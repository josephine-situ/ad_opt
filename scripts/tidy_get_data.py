"""
Data Preparation Pipeline for Ad Optimization
==============================================
Loads, cleans, and preprocesses keyword and ads data with embeddings.
Supports both TF-IDF and BERT embeddings for keyword representations.

Usage:
    python scripts/tidy_get_data.py --embedding-method tfidf
    python scripts/tidy_get_data.py --embedding-method bert
    python scripts/tidy_get_data.py --force-reload # Force full recompute if you updated source data
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
    extract_date_features,
    filter_data_by_date,
    get_gkp_data,
    impute_missing_data,
    merge_with_ads_data,
    add_embeddings,
    prepare_train_test_split,
    save_outputs,
    setup_tee_logging,
)
from utils.date_features import COURSE_START_DATES
from utils.data_pipeline import MIN_DATE_CUTOFF

from utils.keyword_diversity_models import (
    train_or_load_and_predict,
    DEFAULT_ENTROPY_COL,
    DEFAULT_HHI_COL,
)


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
        choices=['tfidf', 'bert'],
        help='Embedding method: tfidf or bert (default: bert)'
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
        default='data/clean',
        help='Output directory for processed data (default: data/clean)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/reports',
        help='Input data directory (default: data/reports)'
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
            "Path to write a copy of console output. If omitted, writes to logs/tidy_get_data_<timestamp>.log. "
            "Set to empty string to disable file logging."
        )
    )

    parser.add_argument(
        '--diversity-mode',
        type=str,
        default='training',
        choices=['off', 'training', 'prod'],
        help=(
            "Whether to add stacked keyword diversity features (entropy/HHI). "
            "'training' generates keyword-grouped OOF predictions; 'prod' uses saved models. "
            "Default: training."
        )
    )
    parser.add_argument(
        '--diversity-n-splits',
        type=int,
        default=5,
        help="Number of GroupKFold splits for diversity OOF predictions (default: 5)"
    )
    
    args = parser.parse_args()

    log_path = setup_tee_logging(
        log_file=args.log_file,
        default_log_prefix='tidy_get_data',
    )
    if log_path is not None:
        print(f"[Log] Writing output to {log_path}")
    
    print("=" * 70)
    print("Data Preparation Pipeline for Ad Optimization")
    print("=" * 70)
    print(f"Embedding method: {args.embedding_method}")
    print(f"N components: {args.n_components}")
    print(f"Force reload: {args.force_reload}")
    print(f"Diversity mode: {args.diversity_mode}")
    print("=" * 70)
    
    try:
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        cache_path = Path('data/cache')
        cache_path.mkdir(exist_ok=True, parents=True)
        
        # Pipeline with automatic caching at each time-consuming step
        print("\n[Step 1] Load and combine keyword data...")
        diversity_df = None
        if args.diversity_mode != 'off':
            # Cache the two outputs separately (parquet can't store tuple directly).
            # NOTE: cache names include "grouped" to avoid reusing older caches from the pre-grouping loader.
            main_cache = cache_path / 'step1_combined_main_grouped.parquet'
            div_cache = cache_path / 'step1_combined_diversity_ungrouped.parquet'

            if main_cache.exists() and div_cache.exists() and not args.force_reload:
                print(f"  [Cache] Loading from {main_cache.name} and {div_cache.name}")
                kw_df = pd.read_parquet(main_cache)
                diversity_df = pd.read_parquet(div_cache)
            else:
                print("  [Computing] Running load_and_combine_keyword_data (with diversity frames)...")
                kw_df, diversity_df = load_and_combine_keyword_data(args.data_dir, return_diversity_frames=True)
                main_cache.parent.mkdir(exist_ok=True, parents=True)
                kw_df.to_parquet(main_cache)
                diversity_df.to_parquet(div_cache)
                print(f"  [Saved] Cached to {main_cache.name} and {div_cache.name}")
        else:
            kw_df = load_or_cache(
                load_and_combine_keyword_data,
                # NOTE: cache name includes "grouped" to avoid reusing older caches from the pre-grouping loader.
                cache_path / 'step1_combined_grouped.parquet',
                args.force_reload,
                args.data_dir
            )
        
        print("\n[Step 2] Format keyword data...")
        # `load_and_combine_keyword_data` now already normalizes Keyword and derives Region.
        # We keep this step for compatibility and to enforce the expected column set.
        kw_df = format_keyword_data(kw_df)
        
        print("\n[Step 3] Extract date features...")
        kw_df = load_or_cache(
            extract_date_features,
            cache_path / 'step3_features.parquet',
            args.force_reload,
            kw_df,
            COURSE_START_DATES
        )
        
        print("\n[Step 4] Filter data by date...")
        kw_df = load_or_cache(
            filter_data_by_date,
            cache_path / 'step4_filtered.parquet',
            args.force_reload,
            kw_df
        )

        # Keep diversity target data on the same date window as the main pipeline.
        if args.diversity_mode != 'off' and diversity_df is not None:
            print("\n[Step 4b] Filter diversity data by date...")
            diversity_df = filter_data_by_date(diversity_df, min_date=MIN_DATE_CUTOFF)
            print(f"  Diversity rows after filter: {len(diversity_df)}")
        
        print("\n[Step 5] Load GKP data...")
        gkp_df = get_gkp_data()
        
        print("\n[Step 5.5] Impute missing data...")
        kw_df = impute_missing_data(kw_df)
        gkp_df = impute_missing_data(gkp_df)
        
        print("\n[Step 6] Merge with GKP data...")
        merged_df = load_or_cache(
            merge_with_ads_data,
            cache_path / 'step6_merged.parquet',
            args.force_reload,
            kw_df,
            gkp_df
        )
        
        cleaned_df = merged_df  # Imputation already handled, no additional cleaning needed
        
        print(f"\n[Step 7] Add {args.embedding_method.upper()} embeddings...")
        df = load_or_cache(
            add_embeddings,
            cache_path / f'step8_embeddings_{args.embedding_method}.parquet',
            args.force_reload,
            cleaned_df,
            args.embedding_method,
            args.n_components,
            True,  # save_models=True
            args.output_dir  # model_dir
        )

        # Optional: Add stacked keyword diversity features (entropy/HHI) from search-term report.
        if args.diversity_mode != 'off':
            print(f"\n[Step 7.5] Adding keyword diversity features ({args.diversity_mode})...")
            if diversity_df is None:
                raise RuntimeError("diversity_mode is enabled but diversity_df was not loaded")

            diversity_preds = train_or_load_and_predict(
                main_df=df,
                search_term_df=diversity_df,
                embedding_method=args.embedding_method,
                n_components=args.n_components,
                mode=args.diversity_mode,
                model_dir=args.output_dir,
                n_splits=args.diversity_n_splits,
            )

            df = df.merge(diversity_preds, on='Keyword', how='left')

            # Ensure columns exist and are numeric.
            for col in (DEFAULT_ENTROPY_COL, DEFAULT_HHI_COL):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Keep pipeline robust; missing predictions get 0.0.
                    df[col] = df[col].fillna(0.0)

        # Add interaction terms: competition (indexed value) x match type indicators
        print("\n[Step 7.6] Adding competition × match-type interaction feature...")
        if 'Competition (indexed value)' in df.columns and 'Match type' in df.columns:
            comp = pd.to_numeric(df['Competition (indexed value)'], errors='coerce')
            is_exact = (df['Match type'] == 'Exact match').astype(int)
            is_phrase = (df['Match type'] == 'Phrase match').astype(int)
            is_broad = (df['Match type'] == 'Broad match').astype(int)

            df['competition_x_is_exact'] = (comp * is_exact).fillna(0.0)
            df['competition_x_is_phrase'] = (comp * is_phrase).fillna(0.0)
            df['competition_x_is_broad'] = (comp * is_broad).fillna(0.0)
        else:
            # Create the column for downstream consistency even if source cols are missing.
            df['competition_x_is_exact'] = 0.0
            df['competition_x_is_phrase'] = 0.0
            df['competition_x_is_broad'] = 0.0
        
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
