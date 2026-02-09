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
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_cleaning import clean_currency
from utils.data_pipeline import _extract_region_from_campaign

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
    """Group hours into 3-hour bins: 0-2, 3-5, 6-8, etc."""
    if pd.isna(hour):
        return None
    hour = int(hour)
    start = (hour // 3) * 3
    end = start + 2
    return f"{start} - {end}"


def load_bid_adj_report(clicks_file: Path, conv_file: Path, segment_col: str,
                        extra_conv_cols=None) -> pd.DataFrame:
    """
    Load and merge clicks and conversions reports.
    
    Args:
        clicks_file: Path to the clicks CSV file
        conv_file: Path to the conversions CSV file
        segment_col: Name of the segment column (e.g., 'Age', 'Device', 'Hour of the day')
        extra_conv_cols: Additional conversion columns to sum into total Conversions
                         (e.g., ['Cross-device conv.'])
    
    Returns:
        DataFrame with Campaign, segment, Clicks, and Conversions columns
    """
    # Read clicks file (skip header rows)
    clicks_df = pd.read_csv(clicks_file, skiprows=2)
    clicks_df['Clicks'] = clicks_df['Clicks'].apply(parse_clicks_value)
    
    # Read conversions file (skip header rows)
    conv_df = pd.read_csv(conv_file, skiprows=2)
    
    # Clean segment column (remove newlines from Device names)
    if segment_col in clicks_df.columns:
        clicks_df[segment_col] = clean_string_column(clicks_df[segment_col])
    if segment_col in conv_df.columns:
        conv_df[segment_col] = clean_string_column(conv_df[segment_col])
    
    # Sum extra conversion columns into Conversions (e.g., Cross-device conv.)
    if extra_conv_cols:
        for col in extra_conv_cols:
            if col in conv_df.columns:
                conv_df['Conversions'] = conv_df['Conversions'] + conv_df[col].fillna(0)
    
    # Aggregate clicks by Campaign and segment
    clicks_agg = clicks_df.groupby(['Campaign', segment_col])['Clicks'].sum().reset_index()
    
    # Aggregate conversions (may have multiple conversion actions)
    if 'Conversion action' in conv_df.columns:
        conv_df = conv_df.drop(columns=['Conversion action'])
    conv_agg = conv_df.groupby(['Campaign', segment_col])['Conversions'].sum().reset_index()
    
    # Merge clicks and conversions
    merged = clicks_agg.merge(conv_agg, on=['Campaign', segment_col], how='left')
    merged['Conversions'] = merged['Conversions'].fillna(0)
    
    return merged


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
        df: DataFrame with Campaign, segment, Clicks, Conversions
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
        'Conversions': 'sum'
    }).reset_index()
    
    # Calculate conversion rate for each segment-region combo
    agg_df['ConversionRate'] = np.where(
        agg_df['Clicks'] > 0,
        agg_df['Conversions'] / agg_df['Clicks'],
        0
    )
    
    # Calculate average conversion rate per region
    region_totals = agg_df.groupby('Region').agg({
        'Clicks': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    region_totals['AvgConversionRate'] = np.where(
        region_totals['Clicks'] > 0,
        region_totals['Conversions'] / region_totals['Clicks'],
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


def process_bid_adjustments(base_dir: Path, min_clicks: int = 1000) -> dict:
    """
    Process all bid adjustment segments for a course.
    
    Args:
        base_dir: Base data directory (e.g., data/gen_ai)
        min_clicks: Minimum clicks threshold
    
    Returns:
        Dictionary with bid adjustments for each segment type
    """
    bid_adj_dir = base_dir / 'reports' / 'bid_adj'
    
    if not bid_adj_dir.exists():
        raise FileNotFoundError(f"Bid adjustment directory not found: {bid_adj_dir}")
    
    results = {}
    
    # Hour of Day (grouped into 3-hour bins)
    hod_clicks = bid_adj_dir / 'hod_clicks.csv'
    hod_conv = bid_adj_dir / 'hod_conv.csv'
    if hod_clicks.exists() and hod_conv.exists():
        print("  Processing Hour of Day adjustments (grouped 0-2, 3-5, etc.)...")
        df = load_bid_adj_report(hod_clicks, hod_conv, 'Hour of the day')
        # Group hours into 3-hour bins
        df['Hour Group'] = df['Hour of the day'].apply(group_hours)
        # Aggregate by the new hour groups
        df = df.groupby(['Campaign', 'Hour Group']).agg({
            'Clicks': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        results['hour_of_day'] = calculate_bid_adjustments(
            df, 'Hour Group', 'hour', min_clicks
        )
    
    # Device
    device_clicks = bid_adj_dir / 'device_clicks.csv'
    device_conv = bid_adj_dir / 'device_conv.csv'
    if device_clicks.exists() and device_conv.exists():
        print("  Processing Device adjustments...")
        df = load_bid_adj_report(device_clicks, device_conv, 'Device',
                                extra_conv_cols=['Cross-device conv.'])
        results['device'] = calculate_bid_adjustments(
            df, 'Device', 'device', min_clicks
        )
    
    # Location
    loc_clicks = bid_adj_dir / 'loc_clicks.csv'
    loc_conv = bid_adj_dir / 'loc_conv.csv'
    if loc_clicks.exists() and loc_conv.exists():
        print("  Processing Location adjustments...")
        df = load_bid_adj_report(loc_clicks, loc_conv, 'Targeted location')
        results['location'] = calculate_bid_adjustments(
            df, 'Targeted location', 'location', min_clicks
        )
    
    # Age
    age_clicks = bid_adj_dir / 'age_clicks.csv'
    age_conv = bid_adj_dir / 'age_conv.csv'
    if age_clicks.exists() and age_conv.exists():
        print("  Processing Age adjustments...")
        df = load_bid_adj_report(age_clicks, age_conv, 'Age')
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


def generate_latex_table(results: dict, output_file: Path, min_clicks: int, top_n: int = 10):
    """
    Generate a LaTeX table showing top increases and decreases in bid adjustments.
    
    Args:
        results: Dictionary with bid adjustments for each segment type
        output_file: Path to save the LaTeX table
        min_clicks: Minimum clicks threshold used
        top_n: Number of top increases/decreases to show
    """
    # Map segment types to display names
    segment_display_names = {
        'hour_of_day': 'Hour of Day',
        'device': 'Device',
        'location': 'Location',
        'age': 'Age'
    }
    
    # Get the segment column name for each type
    segment_col_names = {
        'hour_of_day': 'Hour Group',
        'device': 'Device',
        'location': 'Targeted location',
        'age': 'Age'
    }
    
    # Combine all results into one DataFrame
    all_rows = []
    for segment_type, df in results.items():
        if df is None or df.empty:
            continue
        
        segment_col = segment_col_names.get(segment_type, segment_type)
        display_name = segment_display_names.get(segment_type, segment_type)
        
        # Only include rows with valid bid adjustments
        valid_df = df[df['BidAdjustment'].notna()].copy()
        
        for _, row in valid_df.iterrows():
            all_rows.append({
                'Region': row['Region'],
                'Segment': display_name,
                'Value': row[segment_col],
                'Clicks': row['Clicks'],
                'Conversions': row['Conversions'],
                'BidAdjustment': row['BidAdjustment']
            })
    
    if not all_rows:
        print("  No valid bid adjustments to generate LaTeX table")
        return
    
    combined_df = pd.DataFrame(all_rows)
    
    # Count non-zero adjustments
    non_zero_count = (abs(combined_df['BidAdjustment']) >= 1e-6).sum()
    
    # Sort for top increases and decreases
    sorted_df = combined_df.sort_values('BidAdjustment', ascending=False)
    top_increases = sorted_df.head(top_n)
    top_decreases = sorted_df.tail(top_n).iloc[::-1]  # Reverse to show most negative first
    
    def format_row(row):
        """Format a single row for LaTeX."""
        region = row['Region']
        segment = row['Segment']
        value = str(row['Value'])
        clicks = f"{int(row['Clicks']):,}"
        conversions = f"{row['Conversions']:.1f}"
        adj_pct = row['BidAdjustment'] * 100
        if adj_pct >= 0:
            adj_str = f"+{adj_pct:.0f}\\%"
        else:
            adj_str = f"{adj_pct:.0f}\\%"
        return f"{region} & {segment} & {value} & {clicks} & {conversions} & {adj_str} \\\\"
    
    # Build LaTeX table
    latex_lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\tiny % Sets text size small for single-slide fit",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Segment} & \textbf{Value} & \textbf{Clicks} & \textbf{Purchases} & \textbf{Bid Adjustment} \\",
        r"\midrule",
        r"\multicolumn{6}{c}{\textit{\textbf{Top " + str(top_n) + r" Increases}}} \\",
        r"\midrule",
    ]
    
    for _, row in top_increases.iterrows():
        latex_lines.append(format_row(row))
    
    latex_lines.extend([
        r"\midrule",
        r"\multicolumn{6}{c}{\textit{\textbf{Top " + str(top_n) + r" Decreases}}} \\",
        r"\midrule",
    ])
    
    for _, row in top_decreases.iterrows():
        latex_lines.append(format_row(row))
    
    latex_lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{\textit{Note: total of " + str(non_zero_count) + r" non-zero bid adjustments.}} \\",
        r"\multicolumn{6}{l}{\textit{Bid adjustments are only calculated where we have more than " + f"{min_clicks:,}" + r" clicks.}} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"  Saved LaTeX table to {output_file}")


def add_bid_column_to_file(
    file_path: Path, 
    bid_multiplier: float = 1.3,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Add a 'Bid' column to a bids file.
    
    Bid = Optimal Cost / Gurobi Pred over Base * multiplier
    
    Args:
        file_path: Path to the bids CSV file
        bid_multiplier: Multiplier for bid calculation (default: 1.3)
        output_dir: Output directory (if None, overwrites original)
    
    Returns:
        DataFrame with Bid column added
    """
    df = pd.read_csv(file_path)
    
    # Check if required columns exist
    if 'Optimal Cost' not in df.columns:
        print(f"  Warning: 'Optimal Cost' column not found in {file_path.name}")
        return df
    
    if 'Gurobi Pred over Base' not in df.columns:
        print(f"  Warning: 'Gurobi Pred over Base' column not found in {file_path.name}")
        return df
    
    # Calculate bid: Optimal Cost / Gurobi Pred over Base * multiplier
    # Handle division by zero or near-zero
    df['Bid'] = np.where(
        df['Gurobi Pred over Base'] > 0.0001,  # Avoid division by very small numbers
        df['Optimal Cost'] / df['Gurobi Pred over Base'] * bid_multiplier,
        0.0
    )
    
    # Determine output path
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / file_path.name
    else:
        output_file = file_path
    
    df.to_csv(output_file, index=False)
    print(f"  Added Bid column to {output_file}")
    
    return df


def process_bids(
    bids_path: Path, 
    bid_multiplier: float = 1.3,
    output_dir: Path = None
):
    """
    Process bids file(s) - can be a single file or a directory.
    
    Args:
        bids_path: Path to a bids CSV file or directory containing them
        bid_multiplier: Multiplier for bid calculation
        output_dir: Output directory (if None, overwrites original files)
    """
    if not bids_path.exists():
        print(f"Warning: Bids path not found: {bids_path}")
        return

    if bids_path.is_file():
        # Process single file
        print(f"  Processing single bids file: {bids_path.name}")
        add_bid_column_to_file(bids_path, bid_multiplier, output_dir)
        return
    
    # It's a directory
    # Find all bids files
    bids_files = list(bids_path.rglob("optimized_costs*.csv"))
    
    if not bids_files:
        print(f"  No optimized_costs*.csv files found in {bids_path}")
        return
    
    print(f"  Found {len(bids_files)} bids file(s) to process in {bids_path}")
    
    for file_path in bids_files:
        add_bid_column_to_file(file_path, bid_multiplier, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Post-processing for bid optimization: calculate bid adjustments and add bid column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Calculate bid adjustments for gen_ai course
            python scripts/bid_post_processing.py --course gen_ai
            
            # Calculate with custom minimum clicks threshold
            python scripts/bid_post_processing.py --course ml --min-clicks 500
            
            # Process all courses
            python scripts/bid_post_processing.py --all-courses
            
            # Only add bid column to bids files with custom multiplier
            python scripts/bid_post_processing.py --course gen_ai --bid-multiplier 1.5 --skip-adjustments
            
            # Process a specific file or directory
            python scripts/bid_post_processing.py --course gen_ai --bids-path opt_results/gen_ai/bids/optimized_costs.csv --skip-adjustments
            python scripts/bid_post_processing.py --course gen_ai --bids-path opt_results/gen_ai/backtests --skip-adjustments

            # Only calculate bid adjustments (skip adding bid column)
            python scripts/bid_post_processing.py --course gen_ai --skip-bids
        """
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
        default=1.3,
        help='Multiplier for bid calculation: bid = Optimal Cost / Gurobi Pred over Base * multiplier (default: 1.3)'
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
                        
                        # Generate LaTeX table
                        latex_file = output_dir / 'bid_adjustments_table.tex'
                        generate_latex_table(results, latex_file, args.min_clicks)
                        
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
        
        # --- Add Bid Column to Bids Files ---
        if not args.skip_bids:
            if args.bids_path:
                bids_path = Path(args.bids_path)
            else:
                bids_path = project_root / 'opt_results' / course / 'bids'
            
            print(f"\n  --- Processing Bids Files (multiplier={args.bid_multiplier}) ---")
            process_bids(bids_path, args.bid_multiplier)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
