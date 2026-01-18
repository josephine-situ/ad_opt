"""
To run: python scripts/analyze_backtest_results.py --x-max 10 20 --alpha 0.025 0.05
This script analyzes backtest results from the backtest_daily.py script. It aggregates daily evaluation metrics across different
(x_max, alpha) parameter combinations, computes average performance metrics, and generates a summary CSV and LaTeX table.
"""
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def generate_latex_table(summary_df):
    df = summary_df.copy()

    # 1. Format Data (Manually adding \% for LaTeX safety)
    # We use escape=False later, so we must write \% explicitly here.
    df['x_max'] = df['x_max'].map('{:g}'.format)
    df['alpha'] = df['alpha'].map('{:g}'.format)
    
    # Format Metrics
    formatters = {
        'avg clicks (opt)': '{:,.1f}',
        'avg cost (opt)': '{:,.2f}',
        'clicks/$ (opt)': '{:,.3f}',
        'avg n kws (opt)': '{:,.0f}',
        'avg clicks (act)': '{:,.1f}',
        'avg cost (act)': '{:,.2f}',
        'clicks/$ (act)': '{:,.3f}',
        'avg n kws (act)': '{:,.0f}',
    }
    for col, fmt in formatters.items():
        if col in df.columns:
            df[col] = df[col].map(fmt.format)

    # Handle Improvements: Multiply by 100 AND add the LaTeX escape slash for %
    for col in ['improvement in clicks', 'improvement in clicks/$']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)

    # 2. restructure Columns for Alignment
    # We make x_max and alpha regular columns with an empty top header ('').
    # This forces them to sit on the bottom header row, aligned with metrics.
    col_mapping = [
        ('x_max',                   ('', r'$x_{max}$')),
        ('alpha',                   ('', r'$\alpha$')),  # Greek letter
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', 'Clicks/\$')),    # Escape the $ sign
        ('avg n kws (opt)',         ('Opt', 'Kws')),
        ('avg clicks (act)',        ('Act', 'Clicks')),
        ('avg cost (act)',          ('Act', 'Cost')),
        ('clicks/$ (act)',          ('Act', 'Clicks/\$')),
        ('avg n kws (act)',         ('Act', 'Kws')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in clicks/$', ('Improvement', 'Clicks/\$'))
    ]

    # Reorder and Rename
    # Only keep columns that actually exist in your dataframe
    existing_cols = [old for old, new in col_mapping if old in df.columns]
    df = df[existing_cols]
    df.columns = pd.MultiIndex.from_tuples([new for old, new in col_mapping if old in df.columns])

    # 3. Generate LaTeX
    # index=False removes the side index, treating x_max/alpha as data columns
    latex_body = df.to_latex(
        index=False,
        escape=False,        # vital for $\alpha$ and \% to render as code
        multicolumn_format='c',
        column_format='rr' + 'r' * (len(df.columns) - 2) # Right align everything
    )

    # 4. Inject Custom CMIDRULES
    # Pandas default rules are often messy. We manually replace the top rule.
    # We want lines spanning: Opt (cols 3-6), Act (cols 7-10), Imp (cols 11-12)
    # (Note: LaTeX column count starts at 1)
    
    # We look for the standard rule produced by pandas and replace it
    # Usually pandas puts \toprule ... \midrule. We want to insert cmidrules before the midrule.
    
    # Find the header row end to inject lines underneath
    # A generic way to find the header row in pandas latex output:
    header_fix = r'\cmidrule(lr){3-6} \cmidrule(lr){7-10} \cmidrule(lr){11-12}'
    
    # Injecting the cmidrules:
    # We split by the first occurrence of the metric names row
    split_token = r"& Clicks &" 
    if split_token in latex_body:
        # Insert the lines right before the row with "Clicks"
        parts = latex_body.split(split_token)
        # We reconstruct the string adding the cmidrules before the second header row
        # Note: This is a string hack, but effective for fixed table structures
        latex_body = latex_body.replace(r'\\', r'\\' + '\n' + header_fix, 1)

    # Clean up the empty top header for x_max and alpha (Pandas leaves extra & symbols)
    # This step is optional but makes the code cleaner
    
    # Final Wrap
    latex_output = "\\resizebox{\\textwidth}{!}{\n" + latex_body + "\n}"
    
    return latex_output

def main():
    p = argparse.ArgumentParser()
    # Accept same args to locate folders, though we might iterate all if not provided
    p.add_argument("--x-max", type=float, nargs='+', default=[50])
    p.add_argument("--alpha", type=float, nargs='+', default=[1.0])
    args = p.parse_args()
    
    base_results_dir = Path("opt_results/backtests")
    
    summary_rows = []
    
    for xm in args.x_max:
        for al in args.alpha:
            run_dir = base_results_dir / f"xmax_{xm}_alpha_{al}"
            if not run_dir.exists():
                print(f"Skipping {run_dir}, does not exist")
                continue
                
            # Find all daily_eval files
            eval_files = list(run_dir.glob("daily_eval*.csv"))
            if not eval_files:
                print(f"No eval files found in {run_dir}")
                continue
                
            df_list = []
            for f in eval_files:
                try:
                    df_list.append(pd.read_csv(f))
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            
            if not df_list:
                continue
                
            full_df = pd.concat(df_list, ignore_index=True)

            # Calculate metrics
            # Use pred_opt - pred_opt_base and pred_act - pred_base
            avg_clicks_opt = full_df["t_Clicks_OptCost"].mean()
            avg_cost_opt = full_df["Opt_Cost"].mean()
            # Clicks/$ = Total Clicks / Total Cost
            clicks_per_dollar_opt = full_df["t_Clicks_OptCost"].sum() / full_df["Opt_Cost"].sum() if full_df["Opt_Cost"].sum() > 0 else 0
            avg_n_kws_opt = full_df["N_Opt"].mean()
            
            # Use the same model to compare opt vs actual, so use the pred clicks = pred_act - pred_base for the actual costs
            avg_clicks_act = full_df["t_Clicks_ActCost"].mean()
            avg_cost_act = full_df["Act_Cost"].mean()
            clicks_per_dollar_act = full_df["t_Clicks_ActCost"].sum() / full_df["Act_Cost"].sum() if full_df["Act_Cost"].sum() > 0 else 0
            avg_n_kws_act = full_df["N_Obs"].mean()
            
            # Improvement
            imp_clicks = (avg_clicks_opt - avg_clicks_act) / avg_clicks_act if avg_clicks_act > 0 else 0
            imp_c_d = (clicks_per_dollar_opt - clicks_per_dollar_act) / clicks_per_dollar_act if clicks_per_dollar_act > 0 else 0
            
            summary_rows.append({
                "x_max": xm,
                "alpha": al,
                "avg clicks (opt)": avg_clicks_opt,
                "avg cost (opt)": avg_cost_opt,
                "clicks/$ (opt)": clicks_per_dollar_opt,
                "avg n kws (opt)": avg_n_kws_opt,
                "avg clicks (act)": avg_clicks_act,
                "avg cost (act)": avg_cost_act,
                "clicks/$ (act)": clicks_per_dollar_act,
                "avg n kws (act)": avg_n_kws_act,
                "improvement in clicks": imp_clicks,
                "improvement in clicks/$": imp_c_d
            })

    if not summary_rows:
        print("No results found.")
        return

    summary_df = pd.DataFrame(summary_rows)
    
    # Save CSV
    out_csv = base_results_dir / "backtest_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Summary saved to {out_csv}")
    
    # LaTeX
    latex_code = generate_latex_table(summary_df)
    
    out_tex = base_results_dir / "backtest_summary.tex"
    with open(out_tex, "w") as f:
        f.write(latex_code)
    print(f"LaTeX table saved to {out_tex}")
    print("\n" + latex_code)

if __name__ == "__main__":
    main()
