#!/usr/bin/env python3
"""
Compare Semrush Keywords with Existing Keywords

This script:
1. Loads new keywords from data/gkp/semrush_new_kws.csv
2. Loads existing search terms from data/reports/Search keyword - search terms.csv
3. Loads existing keywords from the data pipeline
4. Compares and marks keywords as 'new', 'existing searches', or 'existing'
5. Removes duplicates (prioritizes 'existing', then 'existing searches', then 'new')
6. Outputs results to data/gkp
7. Prints summary statistics (tee-logged)

Usage:
    python scripts/compare_keywords.py
    python scripts/compare_keywords.py --input data/gkp/semrush_new_kws.csv --output data/gkp/keywords_classified.csv
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Optional
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_and_combine_keyword_data, format_keyword_data
from utils import setup_tee_logging


MAX_KEYWORD_LEN = 80
MAX_KEYWORD_WORDS = 10


def normalize_keyword(kw):
    """Normalize keyword for comparison (lowercase, stripped, no extra whitespace)."""
    if pd.isna(kw):
        return ''
    # Collapse internal whitespace too.
    return ' '.join(str(kw).strip().lower().split())


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
    
    # Check for special characters (: / . but allow apostrophes)
    # Allow letters, numbers, spaces, and apostrophes
    if not re.match(r"^[a-zA-Z0-9\s']*$", kw):
        return None
    
    # Return None if empty after cleaning
    if not kw or kw.isspace():
        return None
    
    return kw


def _passes_len_filters(kw: str) -> bool:
    kw_s = str(kw).strip()
    if not kw_s:
        return False
    if len(kw_s) > MAX_KEYWORD_LEN:
        return False
    if len(kw_s.split()) > MAX_KEYWORD_WORDS:
        return False
    return True


def clean_and_filter_keyword(kw) -> Optional[str]:
    """Clean + apply max-length/word filters.

    Returns:
        Cleaned keyword string, or None if it should be excluded.
    """

    cleaned = clean_keyword(kw)
    if cleaned is None:
        return None
    cleaned = ' '.join(str(cleaned).split())
    if not _passes_len_filters(cleaned):
        return None
    return cleaned


def _find_csv_header_row(path: Path, header_prefix: str = "Day,") -> int:
    """Return the 0-based row index where the real CSV header starts.

    Google Ads exports often include a title line and date range line before the header.
    """

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for idx, line in enumerate(f):
            if line.strip().startswith(header_prefix):
                return idx
            if idx >= 50:
                break
    raise ValueError(
        f"Could not find header row starting with '{header_prefix}' in {path}. "
        "If this is not a Google Ads export, pass a different file or adjust parsing."
    )


def load_search_terms_report(path: Path) -> pd.DataFrame:
    header_idx = _find_csv_header_row(path, header_prefix="Day,")
    return pd.read_csv(path, skiprows=header_idx, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Semrush keywords with existing keywords in the pipeline."
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/gkp/semrush_new_kws.csv',
        help='Input CSV file with new keywords (default: data/gkp/semrush_new_kws.csv)'
    )
    parser.add_argument(
        '--search-terms',
        type=str,
        default='data/reports/Search keyword - search terms.csv',
        help='Google Ads export of search terms (default: data/reports/Search keyword - search terms.csv)'
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
            "Log file path. Default: logs/compare_keywords_<timestamp>.log. "
            "Set to empty string '' to disable file logging."
        ),
    )
    
    args = parser.parse_args()

    log_path = setup_tee_logging(
        log_file=args.log_file,
        default_log_dir='logs',
        default_log_prefix='compare_keywords',
    )
    if log_path is not None:
        print(f"[Logging] Tee output to {log_path}")
    
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

    semrush_kws_cleaned = []
    filtered_special = []
    filtered_len = []
    for kw in semrush_kws_raw:
        cleaned_basic = clean_keyword(kw)
        if cleaned_basic is None:
            filtered_special.append(kw)
            continue
        cleaned = ' '.join(str(cleaned_basic).split())
        if not _passes_len_filters(cleaned):
            filtered_len.append(cleaned)
            continue
        semrush_kws_cleaned.append(cleaned)
    
    # Normalize and remove duplicates
    semrush_kws_norm = set(normalize_keyword(kw) for kw in semrush_kws_cleaned)
    semrush_kws_norm.discard('')  # Remove empty strings
    
    print(f"  Found {len(semrush_kws_raw)} unique raw keywords")
    print(f"  {len(semrush_kws_cleaned)} keywords after cleaning + filters")
    if filtered_special:
        print(f"  Filtered out {len(filtered_special)} (invalid chars/brackets). Example:")
        for kw in sorted(set(map(str, filtered_special)))[:5]:
            print(f"    - {kw}")
    if filtered_len:
        print(f"  Filtered out {len(filtered_len)} (>{MAX_KEYWORD_LEN} chars or >{MAX_KEYWORD_WORDS} words). Example:")
        for kw in sorted(set(map(str, filtered_len)))[:5]:
            print(f"    - {kw}")
    print(f"  {len(semrush_kws_norm)} unique keywords after normalization")
    
    # Load existing search terms
    print(f"\n[Step 3] Loading existing search terms from {args.search_terms}...")
    search_terms_file = Path(args.search_terms)
    if not search_terms_file.exists():
        print(f"\nERROR: Search terms file not found: {search_terms_file}")
        sys.exit(1)

    try:
        st_df = load_search_terms_report(search_terms_file)
        required_cols = {"Search term", "Search keyword match type"}
        missing = required_cols - set(st_df.columns)
        if missing:
            raise KeyError(
                f"Search terms report missing columns: {sorted(missing)}. "
                f"Found columns: {list(st_df.columns)}"
            )

        search_terms_raw = st_df["Search term"].dropna().unique()

        search_terms_cleaned = []
        search_terms_filtered = []
        for kw in search_terms_raw:
            cleaned = clean_and_filter_keyword(kw)
            if cleaned is None:
                search_terms_filtered.append(kw)
                continue
            search_terms_cleaned.append(cleaned)

        search_terms_norm = set(normalize_keyword(x) for x in search_terms_cleaned)
        search_terms_norm.discard("")

        print(f"  Loaded {len(st_df)} rows")
        print(f"  {len(search_terms_raw)} unique search terms (raw)")
        print(f"  {len(search_terms_cleaned)} search terms after cleaning + filters")
        if search_terms_filtered:
            print(f"  Filtered out {len(search_terms_filtered)} search terms (invalid/too long). Example:")
            for kw in sorted(set(map(str, search_terms_filtered)))[:5]:
                print(f"    - {kw}")
        print(f"  {len(search_terms_norm)} unique search terms (normalized)")
    except Exception as e:
        print(f"  ERROR reading {args.search_terms}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load existing keywords
    print(f"\n[Step 4] Loading and formatting existing keywords from {args.data_dir}...")
    try:
        existing_df = load_and_combine_keyword_data(args.data_dir)
        # Format keywords to remove [] and ""
        existing_df = format_keyword_data(existing_df)
        existing_kws_raw = existing_df['Keyword'].dropna().unique()
        existing_kws_cleaned = []
        existing_filtered_len = []
        for kw in existing_kws_raw:
            kw_s = ' '.join(str(kw).split())
            if not _passes_len_filters(kw_s):
                existing_filtered_len.append(kw_s)
                continue
            existing_kws_cleaned.append(kw_s)

        existing_kws_norm = set(normalize_keyword(kw) for kw in existing_kws_cleaned)
        existing_kws_norm.discard('')  # Remove empty strings
        print(f"  Found {len(existing_kws_raw)} unique existing keywords (raw, after format_keyword_data)")
        if existing_filtered_len:
            print(f"  Filtered out {len(existing_filtered_len)} existing keywords (>{MAX_KEYWORD_LEN} chars or >{MAX_KEYWORD_WORDS} words)")
        print(f"  {len(existing_kws_norm)} unique keywords after normalization")
    except FileNotFoundError as e:
        print(f"  ERROR: Could not load existing keywords: {e}")
        sys.exit(1)
    
    # Classify keywords (dedupe priority: existing > existing searches > new)
    print(f"\n[Step 5] Classifying keywords and de-duplicating...")
    all_keywords = existing_kws_norm | search_terms_norm | semrush_kws_norm

    def origin_for(kw: str) -> str:
        if kw in existing_kws_norm:
            return 'existing'
        if kw in search_terms_norm:
            return 'existing searches'
        return 'new'

    output_rows = [{'Keyword': kw, 'Origin': origin_for(kw)} for kw in sorted(all_keywords)]
    
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
    new_count = int((output_df['Origin'] == 'new').sum())
    existing_searches_count = int((output_df['Origin'] == 'existing searches').sum())
    existing_count = int((output_df['Origin'] == 'existing').sum())

    print(f"  Existing keywords:        {existing_count:6d}")
    print(f"  Existing searches:        {existing_searches_count:6d}")
    print(f"  New keywords:             {new_count:6d}")
    print(f"  Total (deduped union):    {len(output_df):6d}")

if __name__ == '__main__':
    main()
