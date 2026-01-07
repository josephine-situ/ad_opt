#!/usr/bin/env python3
"""
Compare Semrush Keywords with Existing Keywords

This script:
1. Loads new keywords from gkp/semrush_new_kws.csv
2. Loads existing broad-match search terms from gkp/broad_match_search_terms.csv
3. Loads existing keywords from the data pipeline
4. Compares and marks keywords as 'new', 'existing searches', or 'existing'
4. Removes duplicates (prioritizes 'existing' if both)
5. Outputs results to gkp folder
6. Prints summary statistics

Usage:
    python scripts/compare_keywords.py
    python scripts/compare_keywords.py --input gkp/semrush_new_kws.csv --output gkp/keywords_classified.csv
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Optional
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_and_combine_keyword_data, format_keyword_data, setup_tee_logging


MAX_KEYWORD_LEN = 80
MAX_KEYWORD_WORDS = 10


def _word_count(s: str) -> int:
    # Count whitespace-separated tokens, ignoring empty fragments.
    return len([w for w in str(s).strip().split() if w])


def _is_valid_keyword_string(s: str) -> bool:
    if len(s) > MAX_KEYWORD_LEN:
        return False
    if _word_count(s) > MAX_KEYWORD_WORDS:
        return False
    return True


def _read_csv_with_fallbacks(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV robustly across common encodings.

    Tries utf-8 first, then cp1252 (common on Windows exports), then latin1.
    """
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # Re-raise with context
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to read CSV: {path}")


def normalize_keyword(kw):
    """Normalize keyword for comparison (lowercase, stripped, no extra whitespace)."""
    if pd.isna(kw):
        return ''
    return str(kw).strip().lower()


def clean_keyword(kw):
    """Clean keyword by removing brackets and filtering out special characters.
    
    - Removes [...] patterns
    - Removes keywords with special characters (: / . but allows apostrophes)
    - Returns None if keyword should be filtered out
    """
    if pd.isna(kw):
        return None
    
    kw = str(kw).strip()
    
    # Remove [...] patterns
    kw = re.sub(r'\[.*?\]', '', kw).strip()

    # Filter out overly-long keywords (often junk / export artifacts)
    if not _is_valid_keyword_string(kw):
        return None
    
    # Check for special characters (: / . but allow apostrophes)
    # Allow letters, numbers, spaces, and apostrophes
    if not re.match(r"^[a-zA-Z0-9\s']*$", kw):
        return None
    
    # Return None if empty after cleaning
    if not kw or kw.isspace():
        return None
    
    return kw


def main():
    parser = argparse.ArgumentParser(
        description="Compare Semrush keywords with existing keywords in the pipeline."
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/gkp/semrush_new_kws.csv',
        help='Input CSV file with new keywords (default: gkp/semrush_new_kws.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/gkp/keywords_classified.csv',
        help='Output CSV file with classified keywords (default: data/gkp/keywords_classified.csv)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/reports',
        help='Directory containing keyword reports (default: data/reports)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help=(
            "Path to write a copy of console output. If omitted, writes to logs/compare_keywords_<timestamp>.log. "
            "Set to empty string to disable file logging."
        )
    )
    parser.add_argument(
        '--search-terms',
        type=str,
        default='data/gkp/broad_match_search_terms.csv',
        help=(
            "CSV file containing existing broad-match search terms "
            "(default: data/gkp/broad_match_search_terms.csv)"
        )
    )
    
    args = parser.parse_args()

    log_path = setup_tee_logging(
        log_file=args.log_file,
        default_log_prefix='compare_keywords',
    )
    if log_path is not None:
        print(f"[Log] Writing output to {log_path}")
    
    print("=" * 70)
    print("Keyword Comparison: Semrush vs Existing")
    print("=" * 70)
    
    # Check if input file exists
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"\nERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Load new keywords from Semrush
    print(f"\n[Step 1] Loading new keywords from {args.input}...")
    try:
        semrush_df = pd.read_csv(input_file)
        print(f"  Loaded {len(semrush_df)} rows from {args.input}")
        print(f"  Columns: {list(semrush_df.columns)}")
    except Exception as e:
        print(f"  ERROR reading {args.input}: {e}")
        sys.exit(1)
    
    # Determine keyword column name
    # Try common names: 'Keyword', 'keyword', 'Kw', etc.
    kw_col = None
    for col_name in ['Keyword', 'keyword', 'Kw', 'kw', 'KEYWORD']:
        if col_name in semrush_df.columns:
            kw_col = col_name
            break
    
    if kw_col is None:
        print(f"  ERROR: Could not find keyword column. Available columns: {list(semrush_df.columns)}")
        sys.exit(1)
    
    print(f"  Using keyword column: '{kw_col}'")
    
    # Extract keywords, clean, and normalize
    print(f"\n[Step 2] Cleaning and normalizing new keywords...")
    semrush_kws_raw = semrush_df[kw_col].dropna().unique()
    
    # Apply cleaning function
    semrush_kws_cleaned = [clean_keyword(kw) for kw in semrush_kws_raw]
    filtered_out = [kw for kw in semrush_kws_raw if clean_keyword(kw) is None]
    semrush_kws_cleaned = [kw for kw in semrush_kws_cleaned if kw is not None]  # Filter out None values
    
    # Normalize and remove duplicates
    semrush_kws_norm = set(normalize_keyword(kw) for kw in semrush_kws_cleaned)
    semrush_kws_norm.discard('')  # Remove empty strings

    semrush_before_len = len(semrush_kws_norm)
    semrush_kws_norm = {kw for kw in semrush_kws_norm if _is_valid_keyword_string(kw)}
    semrush_len_removed = semrush_before_len - len(semrush_kws_norm)
    
    print(f"  Found {len(semrush_kws_raw)} unique raw keywords")
    print(f"  {len(semrush_kws_cleaned)} keywords after cleaning (removed invalid characters/brackets)")
    if filtered_out:
        print(f"  Filtered out {len(filtered_out)} keywords with special characters:")
        for kw in sorted(filtered_out)[:10]:
            print(f"    - {kw}")
        if len(filtered_out) > 10:
            print(f"    ... and {len(filtered_out) - 10} more")
    print(f"  {len(semrush_kws_norm)} unique keywords after normalization")
    if semrush_len_removed:
        print(
            f"  Filtered out {semrush_len_removed} keywords longer than {MAX_KEYWORD_LEN} chars "
            f"or more than {MAX_KEYWORD_WORDS} words"
        )

    # Load existing search terms (broad match) if available
    print(f"\n[Step 3] Loading existing search terms from {args.search_terms}...")
    search_terms_file = Path(args.search_terms)
    search_terms_norm = set()
    if search_terms_file.exists():
        try:
            search_terms_df = _read_csv_with_fallbacks(search_terms_file)
        except Exception as e:
            print(f"  ERROR reading {args.search_terms}: {e}")
            sys.exit(1)

        if 'Search term' not in search_terms_df.columns:
            print(
                "  ERROR: Expected column 'Search term' in "
                f"{args.search_terms}. Available columns: {list(search_terms_df.columns)}"
            )
            sys.exit(1)

        raw_terms = search_terms_df['Search term'].dropna().unique()
        cleaned_terms = [clean_keyword(t) for t in raw_terms]
        cleaned_terms = [t for t in cleaned_terms if t is not None]
        search_terms_norm = set(normalize_keyword(t) for t in cleaned_terms)
        search_terms_norm.discard('')

        st_before_len = len(search_terms_norm)
        search_terms_norm = {kw for kw in search_terms_norm if _is_valid_keyword_string(kw)}
        st_len_removed = st_before_len - len(search_terms_norm)

        print(f"  Found {len(raw_terms)} unique raw search terms")
        print(f"  {len(search_terms_norm)} unique search terms after cleaning/normalization")
        if st_len_removed:
            print(
                f"  Filtered out {st_len_removed} search terms longer than {MAX_KEYWORD_LEN} chars "
                f"or more than {MAX_KEYWORD_WORDS} words"
            )
    else:
        print(f"  WARNING: Search terms file not found, continuing without it: {search_terms_file}")
    
    # Load existing keywords
    print(f"\n[Step 4] Loading and formatting existing keywords from {args.data_dir}...")
    try:
        existing_df = load_and_combine_keyword_data(args.data_dir)
        # Format keywords to remove [] and ""
        existing_df = format_keyword_data(existing_df)
        existing_kws_raw = existing_df['Keyword'].dropna().unique()
        existing_kws_norm = set(normalize_keyword(kw) for kw in existing_kws_raw)
        existing_kws_norm.discard('')  # Remove empty strings

        ex_before_len = len(existing_kws_norm)
        existing_kws_norm = {kw for kw in existing_kws_norm if _is_valid_keyword_string(kw)}
        ex_len_removed = ex_before_len - len(existing_kws_norm)
        print(f"  Found {len(existing_kws_raw)} unique existing keywords (raw, after format_keyword_data)")
        print(f"  {len(existing_kws_norm)} unique keywords after normalization")
        if ex_len_removed:
            print(
                f"  Filtered out {ex_len_removed} existing keywords longer than {MAX_KEYWORD_LEN} chars "
                f"or more than {MAX_KEYWORD_WORDS} words"
            )
    except FileNotFoundError as e:
        print(f"  ERROR: Could not load existing keywords: {e}")
        sys.exit(1)
    
    # Classify keywords
    print(f"\n[Step 5] Classifying keywords...")
    # Precedence for overlaps: existing > existing searches > new
    all_existing_kws = existing_kws_norm
    existing_search_terms = search_terms_norm - all_existing_kws
    new_kws = semrush_kws_norm - all_existing_kws - search_terms_norm

    overlap_existing = semrush_kws_norm & all_existing_kws
    overlap_search_terms = (semrush_kws_norm & search_terms_norm) - all_existing_kws

    print(f"  New keywords (Semrush only): {len(new_kws)}")
    print(f"  Existing search terms (added): {len(existing_search_terms)}")
    print(f"  All existing keywords: {len(all_existing_kws)}")
    print(f"  Overlap: Semrush ∩ existing: {len(overlap_existing)}")
    print(f"  Overlap: Semrush ∩ search terms (not existing): {len(overlap_search_terms)}")
    
    # Create output dataframe
    print(f"\n[Step 6] Creating output file...")
    output_rows = []
    
    # Add new keywords (in Semrush but not existing)
    for kw in sorted(new_kws):
        output_rows.append({
            'Keyword': kw,
            'Origin': 'new'
        })

    # Add existing search terms (broad-match search terms not already in existing)
    for kw in sorted(existing_search_terms):
        output_rows.append({
            'Keyword': kw,
            'Origin': 'existing searches'
        })
    
    # Add all existing keywords
    for kw in sorted(all_existing_kws):
        output_rows.append({
            'Keyword': kw,
            'Origin': 'existing'
        })
    
    output_df = pd.DataFrame(output_rows)
    
    # Create output directory if needed
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save output
    output_df.to_csv(output_file, index=False)
    print(f"  Saved {len(output_df)} keywords to {args.output}")
    
    # Print summary using Origin column
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    new_count = (output_df['Origin'] == 'new').sum()
    existing_search_count = (output_df['Origin'] == 'existing searches').sum()
    existing_count = (output_df['Origin'] == 'existing').sum()
    print(f"  New keywords:      {new_count:6d}")
    print(f"  Existing searches: {existing_search_count:6d}")
    print(f"  Existing keywords: {existing_count:6d}")
    print(f"  Total:             {len(output_df):6d}")
    print(f"\n  Output file: {output_file.resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
