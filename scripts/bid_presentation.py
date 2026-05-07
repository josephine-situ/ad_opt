"""
Presentation (.tex) outputs for bid optimization results.

Generates LaTeX tables for slides / reports:
  - Bid adjustments summary table (top increases & decreases)
  - Example optimized bids table
  - Daily budget table

Input files:
    opt_results/<course>/bid_adjustments/bid_adj_*.csv  (from bid_post_processing.py)
    opt_results/<course>/bids/optimized_costs.csv       (from bid_post_processing.py)

Output files:
    opt_results/<course>/bid_adjustments/bid_adjustments_table.tex
    opt_results/<course>/bids/example_bids.tex
    opt_results/<course>/bids/daily_budget.tex
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_latex_table(results: dict, output_file: Path, min_clicks: int, top_n: int = 10):
    """
    Generate a LaTeX table showing top increases and decreases in bid adjustments.

    Args:
        results: Dictionary with bid adjustments for each segment type
        output_file: Path to save the LaTeX table
        min_clicks: Minimum clicks threshold used
        top_n: Number of top increases/decreases to show
    """
    segment_display_names = {
        'hour_of_day': 'Hour of Day',
        'device': 'Device',
        'location': 'Location',
        'age': 'Age',
    }
    segment_col_names = {
        'hour_of_day': 'Hour Group',
        'device': 'Device',
        'location': 'Targeted location',
        'age': 'Age',
    }

    all_rows = []
    for segment_type, df in results.items():
        if df is None or df.empty:
            continue

        segment_col = segment_col_names.get(segment_type, segment_type)
        if segment_col not in df.columns:
            for col in df.columns:
                if col not in (
                    'Region', 'Clicks', 'All conv.', 'ConversionRate',
                    'AvgConversionRate', 'BidAdjustment', 'BidAdjustmentPct',
                    'BidAdjustmentFormatted', 'Campaign',
                ):
                    segment_col = col
                    break
        display_name = segment_display_names.get(segment_type, segment_type)

        valid_df = df[df['BidAdjustment'].notna()].copy()

        for _, row in valid_df.iterrows():
            all_rows.append({
                'Region': row['Region'],
                'Segment': display_name,
                'Value': row[segment_col],
                'Clicks': row['Clicks'],
                'Purchases': row['All conv.'],
                'BidAdjustment': row['BidAdjustment'],
            })

    if not all_rows:
        print("  No valid bid adjustments to generate LaTeX table")
        return

    combined_df = pd.DataFrame(all_rows)
    non_zero_count = (abs(combined_df['BidAdjustment']) >= 1e-6).sum()

    sorted_df = combined_df.sort_values('BidAdjustment', ascending=False)
    top_increases = sorted_df.head(top_n)
    top_decreases = sorted_df.tail(top_n).iloc[::-1]

    def format_row(row):
        region = row['Region']
        segment = row['Segment']
        value = str(row['Value'])
        clicks = f"{int(row['Clicks']):,}"
        conversions = f"{row['Purchases']:.1f}"
        adj_pct = row['BidAdjustment'] * 100
        if adj_pct >= 0:
            adj_str = f"+{adj_pct:.0f}\\%"
        else:
            adj_str = f"{adj_pct:.0f}\\%"
        return f"{region} & {segment} & {value} & {clicks} & {conversions} & {adj_str} \\\\"

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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"  Saved LaTeX table to {output_file}")


def generate_example_bid_table(
    bids_path: Path,
    output_dir: Path,
    bid_multiplier: float = 1.0,
    top_n: int = 20,
):
    """
    Generate an example output bid table (.tex) from the first bids file found.

    The table is sorted by daily budget (Optimal Cost) descending, then base bid
    descending, and shows the top *top_n* rows with columns:
    Keyword, Region, Match Type, Base Bid.

    Args:
        bids_path: Path to a single CSV or a directory containing optimized_costs*.csv
        output_dir: Directory to write the .tex file into
        bid_multiplier: Multiplier used when the Bid column needs to be computed
        top_n: Number of rows to include
    """
    if bids_path.is_file():
        first_file = bids_path
    else:
        files = sorted(bids_path.rglob("optimized_costs*.csv"))
        if not files:
            print(f"  No optimized_costs*.csv files found in {bids_path}")
            return
        first_file = files[0]

    print(f"  Generating example bid table from {first_file.name} ...")
    df = pd.read_csv(first_file)

    # Ensure a Bid column exists
    if 'Bid' not in df.columns:
        if 'Gurobi Pred over Base' in df.columns:
            df['Bid'] = np.where(
                df['Gurobi Pred over Base'] > 0.0001,
                df['Optimal Cost'] / df['Gurobi Pred over Base'] * bid_multiplier,
                0.0,
            )
        else:
            df['Bid'] = df['Optimal Cost']

    # Only show ENABLED keywords in the example table
    if 'Status' in df.columns:
        df = df[df['Status'] == 'ENABLED']

    # Sort by Optimal Cost desc, then Bid desc
    df = df.sort_values(
        ['Optimal Cost', 'Bid'], ascending=[False, False]
    ).head(top_n)

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"\textbf{Keyword} & \textbf{Region} & \textbf{Match Type} & \textbf{Base Bid (\$)} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        kw = str(row['Keyword']).replace('&', r'\&').replace('_', r'\_')
        region = str(row['Region'])
        mt = str(row['Match type'])
        bid = f"{row['Bid']:.2f}"
        lines.append(f"{kw} & {region} & {mt} & {bid} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Example optimized bids (top " + str(top_n) + r", sorted by daily budget then base bid).}",
        r"\end{table}",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / 'example_bids.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved example bid table to {tex_path}")


def generate_daily_budget_table_tex(
    bids_path: Path,
    output_dir: Path,
):
    """
    Generate a daily budget summary LaTeX table (.tex).

    The daily budget is the sum of Optimal Cost for each (Region, Match type)
    combination.

    Args:
        bids_path: Path to a single CSV or a directory with optimized_costs*.csv
        output_dir: Directory to write the .tex file into
    """
    if bids_path.is_file():
        first_file = bids_path
    else:
        files = sorted(bids_path.rglob("optimized_costs*.csv"))
        if not files:
            print(f"  No optimized_costs*.csv files found in {bids_path}")
            return
        first_file = files[0]

    print(f"  Generating daily budget LaTeX table from {first_file.name} ...")
    df = pd.read_csv(first_file)

    budget_df = (
        df.groupby(['Region', 'Match type'])['Optimal Cost']
        .sum()
        .reset_index()
        .rename(columns={'Optimal Cost': 'Daily Budget'})
        .sort_values(['Region', 'Match type'])
    )

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"\textbf{Region} & \textbf{Match Type} & \textbf{Daily Budget (\$)} \\",
        r"\midrule",
    ]

    for _, row in budget_df.iterrows():
        region = str(row['Region'])
        mt = str(row['Match type'])
        budget = f"{row['Daily Budget']:.2f}"
        lines.append(f"{region} & {mt} & {budget} \\\\")

    total = budget_df['Daily Budget'].sum()
    lines.extend([
        r"\midrule",
        f"\\textbf{{Total}} & & \\textbf{{{total:.2f}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{Daily budget by region and match type.}",
        r"\end{table}",
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / 'daily_budget.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved daily budget LaTeX to {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate presentation (.tex) tables for bid optimization results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python scripts/bid_presentation.py --course gen_ai
            python scripts/bid_presentation.py --all-courses
            python scripts/bid_presentation.py --course ml --bids-path opt_results/ml/bids/optimized_costs.csv
        """,
    )

    parser.add_argument(
        '--course',
        choices=['gen_ai', 'ml', 'sys_eng', 'sys_think'],
        help='Course to process',
    )
    parser.add_argument('--all-courses', action='store_true', help='Process all courses')
    parser.add_argument(
        '--bids-path', type=str,
        help='Custom bids file or directory (default: opt_results/<course>/bids)',
    )
    parser.add_argument(
        '--exp-name', type=str,
        help='Experiment name (resolves to opt_results/<course>/backtests/<exp-name>/budget_<budget>/bids)',
    )
    parser.add_argument('--budget', type=int, help='Budget value for experiment directory')
    parser.add_argument(
        '--bid-multiplier', type=float, default=1.0,
        help='Multiplier for bid calculation (default: 1.0)',
    )
    parser.add_argument(
        '--min-clicks', type=int, default=5000,
        help='Minimum clicks threshold for bid adjustment display (default: 5000)',
    )

    args = parser.parse_args()

    if args.exp_name and args.bids_path:
        parser.error("--exp-name and --bids-path are mutually exclusive")
    if args.exp_name and args.budget is None:
        parser.error("--budget is required when --exp-name is used")

    if args.all_courses:
        courses = ['gen_ai', 'ml', 'sys_eng']
    elif args.course:
        courses = [args.course]
    else:
        parser.error("Either --course or --all-courses must be specified")

    project_root = Path(__file__).parent.parent

    for course in courses:
        print(f"\n{'='*60}")
        print(f"Generating presentation tables: {course.upper()}")
        print('='*60)

        # --- Bid Adjustments LaTeX ---
        adj_dir = project_root / 'opt_results' / course / 'bid_adjustments'
        adj_csvs = sorted(adj_dir.glob('bid_adj_*.csv')) if adj_dir.exists() else []
        if adj_csvs:
            results = {}
            for csv_path in adj_csvs:
                segment_type = csv_path.stem.replace('bid_adj_', '')
                results[segment_type] = pd.read_csv(csv_path)
            latex_file = adj_dir / 'bid_adjustments_table.tex'
            generate_latex_table(results, latex_file, args.min_clicks)
        else:
            print("  No bid adjustment CSVs found; skipping LaTeX table.")

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

        if not bids_path.exists():
            print(f"  Bids path not found: {bids_path}")
            continue

        # --- Output directory for tables ---
        if args.exp_name:
            tables_dir = (
                project_root / 'opt_results' / course / 'backtests'
                / args.exp_name
            )
        else:
            tables_dir = project_root / 'opt_results' / course / 'bids'

        print(f"\n  --- Generating Example Tables ---")
        generate_example_bid_table(bids_path, tables_dir, bid_multiplier=args.bid_multiplier)
        generate_daily_budget_table_tex(bids_path, tables_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
