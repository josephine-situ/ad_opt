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
    # summary_df has columns: x_max, alpha, and the metrics
    # grouping by x_max/alpha or just listing them?
    # User asked for "Multirows may be helpul here (group all opt, and group all act, group improvement)."
    
    # We want a table like:
    # Config | Opt | | | | Act | | | | Imp |
    # x | a | Clicks | Cost | C/$ | N Kws | Clicks | Cost | C/$ | N Kws | Clicks % | C/$ %
    
    # Let's format the numbers first
    df = summary_df.copy()
    
    mapping = {
        'avg clicks (opt)': ('Opt', 'Clicks'),
        'avg cost (opt)': ('Opt', 'Cost'),
        'clicks/$ (opt)': ('Opt', 'Clicks/$'),
        'avg n kws (opt)': ('Opt', 'N Kws'),
        'avg clicks (act)': ('Act', 'Clicks'),
        'avg cost (act)': ('Act', 'Cost'),
        'clicks/$ (act)': ('Act', 'Clicks/$'),
        'avg n kws (act)': ('Act', 'N Kws'),
        'improvement in clicks': ('Improvement', 'Clicks %'),
        'improvement in clicks/$': ('Improvement', 'Clicks/$ %')
    }
    
    # Rounding and formatting
    for col in df.columns:
        if 'clicks (opt)' in col or 'clicks (act)' in col:
            df[col] = df[col].map('{:,.1f}'.format)
        elif 'cost' in col:
            df[col] = df[col].map('{:,.2f}'.format)
        elif 'n kws' in col:
            df[col] = df[col].map('{:,.0f}'.format)
        elif 'clicks/$' in col and 'improvement' not in col:
            df[col] = df[col].map('{:,.3f}'.format)
        elif 'improvement' in col:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
            
    # create multiindex columns
    cols = []
    for c in df.columns:
        if c in ['x_max', 'alpha']:
            cols.append((c, ''))
        else:
            cols.append(mapping.get(c, (c, '')))
            
    df.columns = pd.MultiIndex.from_tuples(cols)
    
    latex = df.to_latex(index=False, multirow=True, escape=False, sparklines=True)
    return latex

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
