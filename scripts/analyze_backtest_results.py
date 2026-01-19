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

    # --- 1. Identify "Best" Row (Before Formatting) ---
    # We identify the index label of the row with the max clicks improvement
    best_idx = df['improvement in clicks'].idxmax()

    # --- 2. Extract Baseline Data for Note ---
    act_vals = {
        'clicks': df['avg clicks (act)'].iloc[0],
        'se_clicks': df.get('se clicks (act)', pd.Series([0]*len(df))).iloc[0],
        'cost': df['avg cost (act)'].iloc[0],
        'se_cost': df.get('se cost (act)', pd.Series([0]*len(df))).iloc[0],
        'cpc': df['clicks/$ (act)'].iloc[0],
        'kws': df['avg n kws (act)'].iloc[0],
        'se_kws': df.get('se n kws (act)', pd.Series([0]*len(df))).iloc[0],
    }
    
    # helper for formatting mean +/- se
    def fmt_mse(mean, se, decimals=1, prefix="", suffix=""):
        return f"{prefix}{mean:,.{decimals}f} \\pm {se:,.{decimals}f}{suffix}"

    note_row = (
        r"\multicolumn{8}{l}{\scriptsize \textbf{Actual values:} "
        f"Clicks: {fmt_mse(act_vals['clicks'], act_vals['se_clicks'])}, "
        f"Cost: {fmt_mse(act_vals['cost'], act_vals['se_cost'], 2, prefix='\\$')}, "
        f"Clicks/\\$: {act_vals['cpc']:.3f}, "
        f"Kws: {fmt_mse(act_vals['kws'], act_vals['se_kws'], 0)}."
        "}"
    )

    # --- 3. Format Data (Strings, Percentages, Special Chars) ---
    # It is crucial to do this BEFORE adding \textbf so we don't break formatters
    
    # x_max: Infinity handling
    df['x_max'] = df['x_max'].apply(lambda x: r'$\infty$' if pd.isna(x) else f'{x:g}')
    # Alpha
    df['alpha'] = df['alpha'].map('{:g}'.format)

    # Numeric Metrics
    # Simple formatters for non-SD columns
    simple_formatters = {
        'clicks/$ (opt)': '{:,.3f}',
    }
    for col, fmt in simple_formatters.items():
        if col in df.columns:
            df[col] = df[col].map(fmt.format)

    # Combined Mean +/- SE formatters
    # We construct the string manually using the se columns
    if 'se clicks (opt)' in df.columns:
        df['avg clicks (opt)'] = df.apply(lambda row: f"{row['avg clicks (opt)']:,.1f} $\\pm$ {row['se clicks (opt)']:,.1f}", axis=1)
    elif 'avg clicks (opt)' in df.columns:
        df['avg clicks (opt)'] = df['avg clicks (opt)'].map('{:,.1f}'.format)
        
    if 'se cost (opt)' in df.columns:
        df['avg cost (opt)'] = df.apply(lambda row: f"{row['avg cost (opt)']:,.2f} $\\pm$ {row['se cost (opt)']:,.2f}", axis=1)
    elif 'avg cost (opt)' in df.columns:
        df['avg cost (opt)'] = df['avg cost (opt)'].map('{:,.2f}'.format)

    if 'se n kws (opt)' in df.columns:
        df['avg n kws (opt)'] = df.apply(lambda row: f"{row['avg n kws (opt)']:,.0f} $\\pm$ {row['se n kws (opt)']:,.0f}", axis=1)
    elif 'avg n kws (opt)' in df.columns:
        df['avg n kws (opt)'] = df['avg n kws (opt)'].map('{:,.0f}'.format)

    # Percentage Metrics (escape % manually since we use escape=False later)
    for col in ['improvement in clicks', 'improvement in clicks/$']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)

    # --- 4. Apply Bolding to WHOLE Row ---
    # We iterate through ALL columns for the identified best index
    for col in df.columns:
        # Check if the column exists to be safe
        if col in df.columns:
            current_val = df.at[best_idx, col]
            df.at[best_idx, col] = f"\\textbf{{{current_val}}}"

    # --- 5. Define Columns ---
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

    # --- 6. Generate Raw Tabular Content ---
    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        multicolumn_format='c',
        column_format='rrccccrr'
    )

    # --- 7. Inject Header Rules ---
    target_header = r'\\ \multicolumn{4}{c}{Opt} & \multicolumn{2}{c}{Improvement} \\'
    header_rules = r'\cmidrule(lr){3-6} \cmidrule(lr){7-8}'
    
    if target_header in latex_tabular:
        latex_tabular = latex_tabular.replace(target_header, target_header + '\n' + header_rules)
    else:
        latex_tabular = latex_tabular.replace(r'\\', r'\\' + '\n' + header_rules, 1)

    # --- 8. Inject Footer Note ---
    if r'\bottomrule' in latex_tabular:
        latex_tabular = latex_tabular.replace(
            r'\bottomrule', 
            r'\midrule' + '\n' + note_row + r' \\' + '\n' + r'\bottomrule'
        )

    # --- 9. Final Assembly ---
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{!}{0.4\\textheight}{%\n"
        f"{latex_tabular}"
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
            se_clicks_opt = full_df["t_Clicks_OptCost"].sem()
            avg_cost_opt = full_df["Opt_Cost"].mean()
            se_cost_opt = full_df["Opt_Cost"].sem()
            # Clicks/$ = Total Clicks / Total Cost
            clicks_per_dollar_opt = full_df["t_Clicks_OptCost"].sum() / full_df["Opt_Cost"].sum() if full_df["Opt_Cost"].sum() > 0 else 0
            avg_n_kws_opt = full_df["N_Opt"].mean()
            se_n_kws_opt = full_df["N_Opt"].sem()
            
            # Use the same model to compare opt vs actual, so use the pred clicks = pred_act - pred_base for the actual costs
            avg_clicks_act = full_df["t_Clicks_ActCost"].mean()
            se_clicks_act = full_df["t_Clicks_ActCost"].sem()
            avg_cost_act = full_df["Act_Cost"].mean()
            se_cost_act = full_df["Act_Cost"].sem()
            clicks_per_dollar_act = full_df["t_Clicks_ActCost"].sum() / full_df["Act_Cost"].sum() if full_df["Act_Cost"].sum() > 0 else 0
            avg_n_kws_act = full_df["N_Obs"].mean()
            se_n_kws_act = full_df["N_Obs"].sem()
            
            # Improvement
            imp_clicks = (avg_clicks_opt - avg_clicks_act) / avg_clicks_act if avg_clicks_act > 0 else 0
            imp_c_d = (clicks_per_dollar_opt - clicks_per_dollar_act) / clicks_per_dollar_act if clicks_per_dollar_act > 0 else 0
            
            summary_rows.append({
                "x_max": xm,
                "alpha": al,
                "avg clicks (opt)": avg_clicks_opt,
                "se clicks (opt)": se_clicks_opt,
                "avg cost (opt)": avg_cost_opt,
                "se cost (opt)": se_cost_opt,
                "clicks/$ (opt)": clicks_per_dollar_opt,
                "avg n kws (opt)": avg_n_kws_opt,
                "se n kws (opt)": se_n_kws_opt,
                "avg clicks (act)": avg_clicks_act,
                "se clicks (act)": se_clicks_act,
                "avg cost (act)": avg_cost_act,
                "se cost (act)": se_cost_act,
                "clicks/$ (act)": clicks_per_dollar_act,
                "avg n kws (act)": avg_n_kws_act,
                "se n kws (act)": se_n_kws_act,
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
