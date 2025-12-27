#!/usr/bin/env python3
"""
Compare Semrush Keywords with Existing Keywords

This script:
1. Loads new keywords from gkp/semrush_new_kws.csv
2. Loads existing keywords from the data pipeline
3. Compares and marks keywords as 'new' or 'existing'
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
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_and_combine_keyword_data, format_keyword_data


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
    
    args = parser.parse_args()
    
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
    
    print(f"  Found {len(semrush_kws_raw)} unique raw keywords")
    print(f"  {len(semrush_kws_cleaned)} keywords after cleaning (removed invalid characters/brackets)")
    if filtered_out:
        print(f"  Filtered out {len(filtered_out)} keywords with special characters:")
        for kw in sorted(filtered_out)[:10]:
            print(f"    - {kw}")
        if len(filtered_out) > 10:
            print(f"    ... and {len(filtered_out) - 10} more")
    print(f"  {len(semrush_kws_norm)} unique keywords after normalization")
    
    # Load existing keywords
    print(f"\n[Step 3] Loading and formatting existing keywords from {args.data_dir}...")
    try:
        existing_df = load_and_combine_keyword_data(args.data_dir)
        # Format keywords to remove [] and ""
        existing_df = format_keyword_data(existing_df)
        existing_kws_raw = existing_df['Keyword'].dropna().unique()
        existing_kws_norm = set(normalize_keyword(kw) for kw in existing_kws_raw)
        existing_kws_norm.discard('')  # Remove empty strings
        print(f"  Found {len(existing_kws_raw)} unique existing keywords (raw, after format_keyword_data)")
        print(f"  {len(existing_kws_norm)} unique keywords after normalization")
    except FileNotFoundError as e:
        print(f"  ERROR: Could not load existing keywords: {e}")
        sys.exit(1)
    
    # Classify keywords
    print(f"\n[Step 4] Classifying keywords...")
    # Keywords in Semrush that are NOT in existing
    new_kws = semrush_kws_norm - existing_kws_norm
    # Keywords that appear in both (will be marked as existing)
    overlap_kws = semrush_kws_norm & existing_kws_norm
    # All existing keywords (whether or not they're in Semrush)
    all_existing_kws = existing_kws_norm
    
    print(f"  New keywords (in Semrush only): {len(new_kws)}")
    print(f"  Overlapping keywords (in both): {len(overlap_kws)}")
    print(f"  All existing keywords: {len(all_existing_kws)}")
    
    # Create output dataframe
    print(f"\n[Step 5] Creating output file...")
    output_rows = []
    
    # Add new keywords (in Semrush but not existing)
    for kw in sorted(new_kws):
        output_rows.append({
            'Keyword': kw,
            'Origin': 'new'
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
    existing_count = (output_df['Origin'] == 'existing').sum()
    print(f"  New keywords:      {new_count:6d}")
    print(f"  Existing keywords: {existing_count:6d}")
    print(f"  Total:             {len(output_df):6d}")
    print(f"\n  Output file: {output_file.resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
