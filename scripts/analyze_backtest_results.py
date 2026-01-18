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
    # --- 0. Sort Data ---
    df = summary_df.copy()
    df = df.sort_values(by=['x_max', 'alpha'], ascending=[True, True], na_position='last')

    # --- 1. Extract Baseline Data for Note ---
    act_vals = {
        'clicks': df['avg clicks (act)'].iloc[0],
        'cost': df['avg cost (act)'].iloc[0],
        'cpc': df['clicks/$ (act)'].iloc[0],
        'kws': df['avg n kws (act)'].iloc[0]
    }
    
    # Create the Footer Note String
    # \footnotesize makes it slightly smaller, \textit puts it in italics (optional)
    note_text = (
        r"\multicolumn{8}{l}{\footnotesize \textbf{Actual values:} "
        f"Clicks: {act_vals['clicks']:,.1f}, "
        f"Cost: \\${act_vals['cost']:,.2f}, "
        f"Clicks/\\$: {act_vals['cpc']:.3f}, "
        f"Kws: {act_vals['kws']:.0f}.}}"
    )

    # --- 2. Format Data ---
    x_max_formatted = []
    previous_val = -99999
    
    for val in df['x_max']:
        if pd.isna(val):
            current_val = np.inf
            display_str = r'$\infty$'
        else:
            current_val = val
            display_str = '{:g}'.format(val)
            
        if current_val == previous_val and len(x_max_formatted) > 0:
            x_max_formatted.append('') 
        else:
            x_max_formatted.append(display_str)
            previous_val = current_val
            
    df['x_max'] = x_max_formatted
    df['alpha'] = df['alpha'].map('{:g}'.format)

    formatters = {
        'avg clicks (opt)': '{:,.1f}',
        'avg cost (opt)': '{:,.2f}',
        'clicks/$ (opt)': '{:,.3f}',
        'avg n kws (opt)': '{:,.0f}',
    }
    for col, fmt in formatters.items():
        if col in df.columns:
            df[col] = df[col].map(fmt.format)

    for col in ['improvement in clicks', 'improvement in clicks/$']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)

    # --- 3. Restructure Columns ---
    col_mapping = [
        ('x_max',                   ('', r'$x_{max}$')),
        ('alpha',                   ('', r'$\alpha$')),
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', 'Clicks/\$')),
        ('avg n kws (opt)',         ('Opt', 'Kws')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in clicks/$', ('Improvement', 'Clicks/\$'))
    ]

    existing_cols = [old for old, new in col_mapping if old in df.columns]
    df = df[existing_cols]
    df.columns = pd.MultiIndex.from_tuples([new for old, new in col_mapping if old in df.columns])

    # --- 4. Generate Raw Tabular Body ---
    latex_body = df.to_latex(
        index=False,
        escape=False,
        multicolumn_format='c',
        column_format='rrccccrr'
    )

    # --- 5. Inject Custom CMIDRULES ---
    # Add mid-rules for headers
    target_string = r'\\ \multicolumn{4}{c}{Opt} & \multicolumn{2}{c}{Improvement} \\'
    replacement = target_string + '\n' + r'\cmidrule(lr){3-6} \cmidrule(lr){7-8}'
    
    if target_string in latex_body:
        latex_body = latex_body.replace(target_string, replacement)
    else:
        # Fallback injection
        lines = latex_body.split('\n')
        for i, line in enumerate(lines):
            if 'Opt' in line and 'Improvement' in line and r'\\' in line:
                lines.insert(i+1, r'\cmidrule(lr){3-6} \cmidrule(lr){7-8}')
                break
        latex_body = '\n'.join(lines)

    # --- 6. Inject Footer Note ---
    # We insert the note row right after \bottomrule but before \end{tabular}
    # This keeps it inside the resizebox logic.
    if r'\bottomrule' in latex_body:
        latex_body = latex_body.replace(r'\bottomrule', r'\bottomrule' + '\n' + note_text)

    # --- 7. Final Wrap ---
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{latex_body}"
        "}\n"
        "\\end{table}"
    )
    
    return final_latex

def main():

    def float_or_none(value):
        """Helper to parse command line args as float or None"""
        if value.lower() == "none":
            return None
        return float(value)

    p = argparse.ArgumentParser()
    # Accept same args to locate folders, though we might iterate all if not provided
    p.add_argument("--x-max", type=float_or_none, nargs='+', default=[50])
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
