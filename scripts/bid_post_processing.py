"""
Post-processing for bid optimization results.

1. Calculate bid adjustments based on conversion rates for different segments.
   Bid adjustments = (segment conversion rate / average conversion rate) - 1
   These are applied as percentage adjustments to base bids in Google Ads.

2. Add calculated bid column to optimization result files.
   bid = Optimal Cost / Gurobi Pred over Base * multiplier

Segments supported:
- Hour of Day (Ad Schedule): -90% to +900%
- Device (Mobile, Tablet, Desktop): -100% to +900%
- Location (Country/State): -90% to +900%
- Age (Demographics): -90% to +900%

Input files:
    opt_results/<course>/bids/optimized_costs.csv     (from optimization.py)
    data/<course>/reports/bid_adj/hod_clicks.csv       [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/hod_conv.csv         [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/device_clicks.csv    [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/device_conv.csv      [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/loc_clicks.csv       [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/loc_conv.csv         [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/age_clicks.csv       [FROM GOOGLE ADS REPORTS]
    data/<course>/reports/bid_adj/age_conv.csv         [FROM GOOGLE ADS REPORTS]

Output files:
    opt_results/<course>/bid_adjustments/bid_adj_hour_of_day.csv
    opt_results/<course>/bid_adjustments/bid_adj_device.csv
    opt_results/<course>/bid_adjustments/bid_adj_location.csv
    opt_results/<course>/bid_adjustments/bid_adj_age.csv
    opt_results/<course>/bid_adjustments/bid_adjustments_table.tex
    opt_results/<course>/bids/optimized_costs.csv          (updated with Bid column)
    opt_results/<course>/bids/example_bids.tex
    opt_results/<course>/bids/daily_budget.csv
    opt_results/<course>/bids/daily_budget.tex

Examples:
    # Calculate bids and bid adjustments for gen_ai course
    python scripts/bid_post_processing.py --course gen_ai
    
    # Only add bid column to bids files with custom multiplier
    python scripts/bid_post_processing.py --course gen_ai --bid-multiplier 1.5 --skip-adjustments
    
    # Process a specific file or directory
    python scripts/bid_post_processing.py --course gen_ai --bids-path opt_results/gen_ai/bids/optimized_costs.csv --skip-adjustments
    python scripts/bid_post_processing.py --course gen_ai --bids-path opt_results/gen_ai/backtests --skip-adjustments

    # Use experiment name instead of bids-path
    python scripts/bid_post_processing.py --course ml --exp-name exp107_max_conv --budget 353 --skip-adjustments

    # Only calculate bid adjustments (skip adding bid column)
    python scripts/bid_post_processing.py --course gen_ai --skip-bids

Bids path resolution (mutually exclusive):
    --exp-name  -> opt_results/<course>/backtests/<exp-name>/budget_<budget>/bids
    --bids-path -> user-specified path (file or directory)
    (default)   -> opt_results/<course>/bids

Estimated run time (HP Spectre x360, i7-1065G7 @ 1.30 GHz, 4C/8T, 16 GB RAM, no discrete GPU):
    <1 min per course
"""

import argparse
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_pipeline import _extract_region_from_campaign

# Full set of match types and regions used in optimization
MATCH_TYPES = ['Exact match', 'Phrase match', 'Broad match']
REGIONS = ['USA', 'A', 'B']

# Bid adjustment limits by category (as decimal, e.g., -0.9 for -90%)
BID_ADJUSTMENT_LIMITS = {
    'hour': {'min': -0.90, 'max': 9.00},      # -90% to +900%
    'device': {'min': -1.00, 'max': 9.00},    # -100% to +900%
    'location': {'min': -0.90, 'max': 9.00},  # -90% to +900%
    'age': {'min': -0.90, 'max': 9.00},       # -90% to +900%
}


def parse_clicks_value(value):
    """Parse click values that may have commas (e.g., '1,234' -> 1234)."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return int(str(value).replace(',', ''))


def clean_string_column(series: pd.Series) -> pd.Series:
    """Clean string column by stripping whitespace. Handles non-string columns safely."""
    if series.dtype == 'object':
        return series.str.strip()
    return series


def group_hours(hour: int) -> str:
    """Group hours into 4-hour bins: 0-4, 4-8, 8-12, etc."""
    if pd.isna(hour):
        return None
    hour = int(hour)
    start = (hour // 4) * 4
    end = start + 4
    return f"{start} - {end}"


def load_bid_adj_report(clicks_file: Path, conv_file: Path, segment_col: str,
                        extra_conv_cols=None,
                        fallback_segment_col: str = None) -> tuple[pd.DataFrame, str]:
    """
    Load and merge clicks and conversions reports.
    
    Args:
        clicks_file: Path to the clicks CSV file
        conv_file: Path to the conversions CSV file
        segment_col: Name of the segment column (e.g., 'Age', 'Device', 'Hour of the day')
        extra_conv_cols: Additional conversion columns to sum into total Conversions
                         (e.g., ['Cross-device conv.'])
        fallback_segment_col: Fallback column name if segment_col is not found
    
    Returns:
        Tuple of (DataFrame with Campaign, segment, Clicks, and All conv. columns,
                  resolved segment column name)
    """
    # Read clicks file
    clicks_df = pd.read_csv(clicks_file)
    clicks_df['Clicks'] = clicks_df['Clicks'].apply(parse_clicks_value)
    
    # Read conversions file
    conv_df = pd.read_csv(conv_file)
    
    # Resolve segment column: use fallback if primary not found
    if segment_col not in clicks_df.columns and fallback_segment_col and fallback_segment_col in clicks_df.columns:
        print(f"    Column '{segment_col}' not found, using fallback '{fallback_segment_col}'")
        segment_col = fallback_segment_col
    
    # Clean segment column (remove newlines from Device names)
    if segment_col in clicks_df.columns:
        clicks_df[segment_col] = clean_string_column(clicks_df[segment_col])
    if segment_col in conv_df.columns:
        conv_df[segment_col] = clean_string_column(conv_df[segment_col])
    
    # Sum extra conversion columns into Conversions (e.g., Cross-device conv.)
    if extra_conv_cols:
        for col in extra_conv_cols:
            if col in conv_df.columns:
                conv_df['All conv.'] = conv_df['All conv.'] + conv_df[col].fillna(0)
    
    # Aggregate clicks by Campaign and segment
    clicks_agg = clicks_df.groupby(['Campaign', segment_col])['Clicks'].sum().reset_index()
    
    # Aggregate conversions (may have multiple conversion actions)
    if 'Conversion action' in conv_df.columns:
        conv_df = conv_df.drop(columns=['Conversion action'])
    conv_agg = conv_df.groupby(['Campaign', segment_col])['All conv.'].sum().reset_index()
    
    # Merge clicks and conversions
    merged = clicks_agg.merge(conv_agg, on=['Campaign', segment_col], how='left')
    merged['All conv.'] = pd.to_numeric(merged['All conv.'], errors='coerce').fillna(0)
    
    return merged, segment_col


def calculate_bid_adjustments(
    df: pd.DataFrame,
    segment_col: str,
    category: str,
    min_clicks: int = 1000
) -> pd.DataFrame:
    """
    Calculate bid adjustments based on conversion rates.
    
    Bid adjustment = (segment conversion rate / average conversion rate) - 1
    
    Args:
        df: DataFrame with Campaign, segment, Clicks, All conv.
        segment_col: Name of the segment column
        category: Category for limit application ('hour', 'device', 'location', 'age')
        min_clicks: Minimum clicks threshold for applying bid adjustment
    
    Returns:
        DataFrame with bid adjustments per segment-region combination
    """
    # Extract region from campaign name using utility function
    df = df.copy()
    df['Region'] = df['Campaign'].apply(_extract_region_from_campaign)

    # Filter out region C
    df = df[df['Region'] != 'C'].copy()

    if category == 'location':
        # Don't calculate USA (since not split by state)
        df = df[df['Region'] != 'USA'].copy()
    
    # Aggregate by Region and Segment
    agg_df = df.groupby(['Region', segment_col]).agg({
        'Clicks': 'sum',
        'All conv.': 'sum'
    }).reset_index()
    
    # Calculate conversion rate for each segment-region combo
    agg_df['ConversionRate'] = np.where(
        agg_df['Clicks'] > 0,
        agg_df['All conv.'] / agg_df['Clicks'],
        0
    )
    
    # Calculate average conversion rate per region
    region_totals = agg_df.groupby('Region').agg({
        'Clicks': 'sum',
        'All conv.': 'sum'
    }).reset_index()
    region_totals['AvgConversionRate'] = np.where(
        region_totals['Clicks'] > 0,
        region_totals['All conv.'] / region_totals['Clicks'],
        0
    )
    region_totals = region_totals[['Region', 'AvgConversionRate']]
    
    # Merge average conversion rate
    agg_df = agg_df.merge(region_totals, on='Region', how='left')
    
    # Calculate bid adjustment: (segment rate / avg rate) - 1
    # E.g., if segment rate is 1.5x the average, adjustment is +50%
    agg_df['BidAdjustment'] = np.where(
        agg_df['AvgConversionRate'] > 0,
        (agg_df['ConversionRate'] / agg_df['AvgConversionRate']) - 1,
        0  # No adjustment if no baseline conversion rate
    )
    
    # Apply minimum clicks threshold
    agg_df['BidAdjustment'] = np.where(
        agg_df['Clicks'] >= min_clicks,
        agg_df['BidAdjustment'],
        np.nan  # NaN indicates no adjustment should be applied
    )
    
    # Apply bid adjustment limits
    limits = BID_ADJUSTMENT_LIMITS.get(category, {'min': -0.90, 'max': 9.00})
    agg_df['BidAdjustment'] = agg_df['BidAdjustment'].clip(
        lower=limits['min'],
        upper=limits['max']
    )
    
    # Convert to percentage for display
    agg_df['BidAdjustmentPct'] = agg_df['BidAdjustment'] * 100
    
    # Format for Google Ads (e.g., "+30%" or "-50%")
    def format_adjustment(val):
        if pd.isna(val):
            return "No adjustment (insufficient data)"
        pct = val * 100
        if pct >= 0:
            return f"+{pct:.0f}%"
        return f"{pct:.0f}%"
    
    agg_df['GoogleAdsFormat'] = agg_df['BidAdjustment'].apply(format_adjustment)
    
    # Sort by absolute adjustment (descending)
    agg_df['AbsBidAdjustment'] = agg_df['BidAdjustment'].abs()
    agg_df = agg_df.sort_values('AbsBidAdjustment', ascending=False)
    agg_df = agg_df.drop(columns=['AbsBidAdjustment'])
    
    return agg_df


def process_bid_adjustments(base_dir: Path, min_clicks: int = 1000, file_suffix: str = "") -> dict:
    """
    Process all bid adjustment segments for a course.
    
    Args:
        base_dir: Base data directory (e.g., data/gen_ai)
        min_clicks: Minimum clicks threshold
        file_suffix: Suffix appended to report filenames before the extension
                     (e.g., "_7d" reads hod_clicks_7d.csv instead of hod_clicks.csv)
    
    Returns:
        Dictionary with bid adjustments for each segment type
    """
    bid_adj_dir = base_dir / 'reports' / 'bid_adj'
    
    if not bid_adj_dir.exists():
        raise FileNotFoundError(f"Bid adjustment directory not found: {bid_adj_dir}")
    
    results = {}
    
    # Hour of Day (grouped into 4-hour bins)
    hod_clicks = bid_adj_dir / f'hod_clicks{file_suffix}.csv'
    hod_conv = bid_adj_dir / f'hod_conv{file_suffix}.csv'
    if hod_clicks.exists() and hod_conv.exists():
        print("  Processing Hour of Day adjustments (grouped 0-4, 4-8, etc.)...")
        df, _ = load_bid_adj_report(hod_clicks, hod_conv, 'Hour of the day')
        df['Hour Group'] = df['Hour of the day'].apply(group_hours)
        df = df.groupby(['Campaign', 'Hour Group']).agg({
            'Clicks': 'sum',
            'All conv.': 'sum'
        }).reset_index()
        results['hour_of_day'] = calculate_bid_adjustments(
            df, 'Hour Group', 'hour', min_clicks
        )
    
    # Device
    device_clicks = bid_adj_dir / f'device_clicks{file_suffix}.csv'
    device_conv = bid_adj_dir / f'device_conv{file_suffix}.csv'
    if device_clicks.exists() and device_conv.exists():
        print("  Processing Device adjustments...")
        df, _ = load_bid_adj_report(device_clicks, device_conv, 'Device',
                                    extra_conv_cols=['Cross-device conv.'])
        results['device'] = calculate_bid_adjustments(
            df, 'Device', 'device', min_clicks
        )
    
    # Location
    loc_clicks = bid_adj_dir / f'loc_clicks{file_suffix}.csv'
    loc_conv = bid_adj_dir / f'loc_conv{file_suffix}.csv'
    if loc_clicks.exists() and loc_conv.exists():
        print("  Processing Location adjustments...")
        df, loc_col = load_bid_adj_report(
            loc_clicks, loc_conv, 'Targeted location',
            fallback_segment_col='Country/Territory (User location)'
        )
        results['location'] = calculate_bid_adjustments(
            df, loc_col, 'location', min_clicks
        )
    
    # Age
    age_clicks = bid_adj_dir / f'age_clicks{file_suffix}.csv'
    age_conv = bid_adj_dir / f'age_conv{file_suffix}.csv'
    if age_clicks.exists() and age_conv.exists():
        print("  Processing Age adjustments...")
        df, _ = load_bid_adj_report(age_clicks, age_conv, 'Age')
        results['age'] = calculate_bid_adjustments(
            df, 'Age', 'age', min_clicks
        )
    
    return results


def save_bid_adjustments(results: dict, output_dir: Path):
    """Save bid adjustment results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for segment_type, df in results.items():
        output_file = output_dir / f'bid_adj_{segment_type}.csv'
        df.to_csv(output_file, index=False)
        print(f"  Saved {segment_type} adjustments to {output_file}")



def add_bid_column_to_file(
    file_path: Path,
    bid_multiplier: float = 1.0,
    output_dir: Path = None,
    keywords_path: Path = None,
) -> pd.DataFrame:
    """
    Add 'Bid' and 'Status' columns to a bids file.

    When *keywords_path* is provided the output contains one row for every
    keyword x match type x region combination from the classified keyword list.
    Keywords not selected by the optimiser receive Optimal Cost = 0 and
    Status = PAUSED.  Selected keywords get Status = ENABLED.

    Bid = Optimal Cost / Gurobi Pred over Base * multiplier

    Args:
        file_path: Path to the bids CSV file (from optimization.py).
        bid_multiplier: Multiplier for bid calculation (default: 1.0).
        output_dir: Output directory (if None, overwrites original).
        keywords_path: Path to keywords_classified.csv.  When provided the
            output will contain every keyword x match-type x region.

    Returns:
        DataFrame with Bid and Status columns added.
    """
    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['Keyword', 'Region', 'Match type', 'Origin', 'Optimal Cost', 'Gurobi Pred over Base'])

    # Check if required columns exist
    if 'Optimal Cost' not in df.columns:
        print(f"  Warning: 'Optimal Cost' column not found in {file_path.name}")
        return df

    # --- Expand to full keyword roster if keywords_classified provided ---
    if keywords_path is not None and keywords_path.exists():
        kw_df = pd.read_csv(keywords_path)
        all_keywords = kw_df['Keyword'].tolist()
        full_roster = pd.DataFrame(
            list(product(all_keywords, REGIONS, MATCH_TYPES)),
            columns=['Keyword', 'Region', 'Match type'],
        )
        # Bring in Origin column
        full_roster = full_roster.merge(
            kw_df[['Keyword', 'Origin']], on='Keyword', how='left',
        )
        # Left-join optimisation results onto the full roster
        merge_cols = ['Keyword', 'Region', 'Match type']
        df = full_roster.merge(df, on=merge_cols, how='left', suffixes=('', '_opt'))
        # If Origin existed in both, keep the roster version
        if 'Origin_opt' in df.columns:
            df = df.drop(columns=['Origin_opt'])
        print(f"  Expanded to {len(df)} rows using {keywords_path.name}")

    # Calculate bid: Optimal Cost / Gurobi Pred over Base * multiplier
    if 'Gurobi Pred over Base' in df.columns:
        denom = df['Gurobi Pred over Base']
    else:
        denom = pd.Series(0.0, index=df.index)
        print(f"  Warning: 'Gurobi Pred over Base' not found in {file_path.name}; Bid set to 0")

    df['Bid'] = np.where(
        denom > 0.0001,
        df['Optimal Cost'] / denom * bid_multiplier,
        0.0,
    )

    # Status: ENABLED when bid > 0, PAUSED otherwise
    df['Status'] = np.where(df['Bid'] > 0, 'ENABLED', 'PAUSED')

    # Sort: ENABLED keywords first (by Optimal Cost desc, then Bid desc)
    df = df.sort_values(
        ['Optimal Cost', 'Bid'], ascending=[False, False],
    ).reset_index(drop=True)

    # Determine output path
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / file_path.name
    else:
        output_file = file_path

    df.to_csv(output_file, index=False)
    enabled = (df['Status'] == 'ENABLED').sum()
    print(f"  Saved {output_file}  ({enabled} ENABLED / {len(df)} total)")

    return df


def process_bids(
    bids_path: Path, 
    bid_multiplier: float = 1.0,
    output_dir: Path = None,
    keywords_path: Path = None,
):
    """
    Process bids file(s) - can be a single file or a directory.
    
    Args:
        bids_path: Path to a bids CSV file or directory containing them
        bid_multiplier: Multiplier for bid calculation
        output_dir: Output directory (if None, overwrites original files)
        keywords_path: Path to keywords_classified.csv (passed to
            add_bid_column_to_file so that every keyword gets a row).
    """
    if not bids_path.exists():
        print(f"Warning: Bids path not found: {bids_path}")
        return

    if bids_path.is_file():
        # Process single file
        print(f"  Processing single bids file: {bids_path.name}")
        add_bid_column_to_file(bids_path, bid_multiplier, output_dir,
                               keywords_path=keywords_path)
        return
    
    # It's a directory
    # Find all bids files
    bids_files = list(bids_path.rglob("optimized_costs*.csv"))
    
    if not bids_files:
        print(f"  No optimized_costs*.csv files found in {bids_path}")
        return
    
    print(f"  Found {len(bids_files)} bids file(s) to process in {bids_path}")
    
    for file_path in bids_files:
        add_bid_column_to_file(file_path, bid_multiplier, output_dir,
                               keywords_path=keywords_path)


def generate_daily_budget_csv(
    bids_path: Path,
    output_dir: Path,
):
    """
    Generate a daily budget summary CSV.

    The daily budget is the sum of Optimal Cost for each (Region, Match type)
    combination.

    Args:
        bids_path: Path to a single CSV or a directory with optimized_costs*.csv
        output_dir: Directory to write the CSV into
    """
    if bids_path.is_file():
        first_file = bids_path
    else:
        files = sorted(bids_path.rglob("optimized_costs*.csv"))
        if not files:
            print(f"  No optimized_costs*.csv files found in {bids_path}")
            budget_df = pd.DataFrame(columns=['Region', 'Match type', 'Daily Budget'])
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / 'daily_budget.csv'
            budget_df.to_csv(csv_path, index=False)
            print(f"  Saved empty daily budget CSV to {csv_path}")
            return
        first_file = files[0]

    print(f"  Generating daily budget CSV from {first_file.name} ...")
    try:
        df = pd.read_csv(first_file)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=['Region', 'Match type', 'Optimal Cost'])

    if df.empty:
        budget_df = pd.DataFrame(columns=['Region', 'Match type', 'Daily Budget'])
    else:
        budget_df = (
            df.groupby(['Region', 'Match type'])['Optimal Cost']
            .sum()
            .reset_index()
            .rename(columns={'Optimal Cost': 'Daily Budget'})
            .sort_values(['Region', 'Match type'])
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'daily_budget.csv'
    budget_df.to_csv(csv_path, index=False)
    print(f"  Saved daily budget CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-processing for bid optimization (Google Ads upload). "
                    "For presentation (.tex) outputs see bid_presentation.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--course',
        choices=['gen_ai', 'ml', 'sys_eng', 'sys_think'],
        help='Course to process (gen_ai, ml, sys_eng, or sys_think)'
    )
    parser.add_argument(
        '--all-courses',
        action='store_true',
        help='Process all courses'
    )
    parser.add_argument(
        '--min-clicks',
        type=int,
        default=5000,
        help='Minimum clicks threshold for bid adjustment (default: 5000)'
    )
    parser.add_argument(
        '--bid-multiplier',
        type=float,
        default=1.0,
        help='Multiplier for bid calculation: bid = Optimal Cost / Gurobi Pred over Base * multiplier (default: 1.0)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Base data directory (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for bid adjustments (default: opt_results/<course>/bid_adjustments)'
    )
    parser.add_argument(
        '--bids-path',
        type=str,
        help='Custom bids file or directory to process (default: opt_results/<course>/bids)'
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        help='Experiment name (resolves to opt_results/<course>/backtests/<exp-name>/budget_<budget>/bids). '
             'Use instead of --bids-path for backtest experiments.'
    )
    parser.add_argument(
        '--budget',
        type=int,
        help='Budget value for experiment directory (e.g., 353). Required when --exp-name is used.'
    )
    parser.add_argument(
        '--skip-adjustments',
        action='store_true',
        help='Skip bid adjustment calculation (only add bid column)'
    )
    parser.add_argument(
        '--skip-bids',
        action='store_true',
        help='Skip adding bid column (only calculate bid adjustments)'
    )
    
    args = parser.parse_args()
    
    # Validate --exp-name / --bids-path mutual exclusivity and budget requirement
    if args.exp_name and args.bids_path:
        parser.error("--exp-name and --bids-path are mutually exclusive")
    if args.exp_name and args.budget is None:
        parser.error("--budget is required when --exp-name is used")

    # Determine courses to process
    if args.all_courses:
        courses = ['gen_ai', 'ml', 'sys_eng']
    elif args.course:
        courses = [args.course]
    else:
        parser.error("Either --course or --all-courses must be specified")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_base = project_root / args.data_dir
    
    for course in courses:
        print(f"\n{'='*60}")
        print(f"Processing: {course.upper()}")
        print('='*60)

        keywords_path = data_base / course / 'gkp' / 'keywords_classified.csv'
        
        # --- Bid Adjustments ---
        if not args.skip_adjustments:
            base_dir = data_base / course
            
            if not base_dir.exists():
                print(f"Warning: Data directory not found: {base_dir}")
            else:
                try:
                    # Process all segments
                    results = process_bid_adjustments(base_dir, args.min_clicks)
                    
                    if results:
                        # Determine output directory
                        if args.output_dir:
                            output_dir = Path(args.output_dir)
                        else:
                            output_dir = project_root / 'opt_results' / course / 'bid_adjustments'
                        
                        # Save results
                        save_bid_adjustments(results, output_dir)
                        
                        # Print summary
                        print(f"\n  --- Bid Adjustments Summary ---")
                        for segment_type, df in results.items():
                            valid_adjustments = df['BidAdjustment'].notna().sum()
                            total_combos = len(df)
                            print(f"    {segment_type}: {valid_adjustments}/{total_combos} combos with sufficient data")
                    else:
                        print(f"  No bid adjustment data found for {course}")
                        
                except Exception as e:
                    print(f"  Error processing bid adjustments for {course}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # --- Resolve bids path ---
        if args.exp_name:
            bids_path = (
                project_root / 'opt_results' / course / 'backtests'
                / args.exp_name / f'budget_{args.budget}' / 'bids'
            )
        elif args.bids_path:
            bids_path = Path(args.bids_path)
        else:
            bids_path = project_root / 'opt_results' / course / 'bids'

        # --- Add Bid + Status columns to bids files ---
        if not args.skip_bids:
            print(f"\n  --- Processing Bids Files (multiplier={args.bid_multiplier}) ---")
            process_bids(bids_path, args.bid_multiplier,
                         keywords_path=keywords_path)

        # --- Daily Budget CSV ---
        if bids_path.exists():
            if args.exp_name:
                tables_dir = (
                    project_root / 'opt_results' / course / 'backtests'
                    / args.exp_name
                )
            elif args.output_dir:
                tables_dir = Path(args.output_dir)
            else:
                tables_dir = project_root / 'opt_results' / course / 'bids'

            generate_daily_budget_csv(bids_path, tables_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()