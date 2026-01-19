"""
To run: python scripts/analyze_backtest_results.py --exp-name "experiment_v1"
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

    # --- 1. Identify "Best" Row ---
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
    
    # <--- FIX: Added $ around \pm to prevent "Missing $" error in footer
    def fmt_mse(mean, se, decimals=1, prefix="", suffix=""):
        return f"{prefix}{mean:,.{decimals}f} $\\pm$ {se:,.{decimals}f}{suffix}"

    # 1. Define the string with the backslash outside the f-string
    dollar_prefix = r'\$' 

    note_row = (
        r"\multicolumn{8}{l}{\scriptsize \textbf{Actual values:} "
        f"Clicks: {fmt_mse(act_vals['clicks'], act_vals['se_clicks'])}, "
        # 2. Use the variable inside the function call
        f"Cost: {fmt_mse(act_vals['cost'], act_vals['se_cost'], 2, prefix=dollar_prefix)}, "
        # Note: The backslash below is fine because it is NOT inside curly braces
        f"Clicks/\\$: {act_vals['cpc']:.3f}, "
        f"Kws: {fmt_mse(act_vals['kws'], act_vals['se_kws'], 0)}."
        "}"
    )

    # --- 3. Format Data ---
    # Convert to object to handle mixed types (strings + numbers) without warnings
    df = df.astype(object)

    # Format x_max (handle infinity and None)
    def fmt_xmax(x):
        s = str(x).lower()
        if pd.isna(x) or s in ['inf', 'infinity', 'none', 'nan']:
            return r'$\infty$'
        try:
            return f'{float(x):g}'
        except ValueError:
            return str(x)

    df['x_max'] = df['x_max'].apply(fmt_xmax)
    df['alpha'] = df['alpha'].map('{:g}'.format)

    # Simple numeric formatters
    simple_formatters = {'clicks/$ (opt)': '{:,.3f}'}
    for col, fmt in simple_formatters.items():
        if col in df.columns:
            df[col] = df[col].map(fmt.format)

    # Combined Mean +/- SE formatters (Already includes $ around \pm)
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

    # Percentage Metrics
    for col in ['improvement in clicks', 'improvement in clicks/$']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)

    # --- 4. Apply Bolding ---
    # Iterate columns and apply bold wrapper
    for col in df.columns:
        current_val = df.at[best_idx, col]
        df.at[best_idx, col] = f"\\textbf{{{str(current_val)}}}"

    # --- 5. Define Columns ---
    col_mapping = [
        ('x_max',                   ('', r'$x_{max}$')),
        ('alpha',                   ('', r'$\alpha$')),
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', r'Clicks/\$')),
        ('avg n kws (opt)',         ('Opt', 'Kws')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in clicks/$', ('Improvement', r'Clicks/\$'))
    ]

    # Select and rename columns
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

    # --- 7. Manual Booktabs Injection ---
    # Replace default \hline with booktabs commands
    latex_tabular = latex_tabular.replace(r'\hline', r'\toprule', 1)
    if latex_tabular.strip().endswith(r'\hline'):
        latex_tabular = latex_tabular.strip()[:-6] + r'\bottomrule'

    # --- 8. Inject Header Rules ---
    # Due to pandas version differences, exact header string might vary.
    # We will try to find the row with 'Opt' and inject cmidrules relative to it or blindly replace if exact match.
    
    # Try Regex replacement for robustness?
    # Or just simpler replacement of the top structure.
    # The header usually looks like:
    # & & \multicolumn{4}{c}{Opt} & \multicolumn{2}{c}{Improvement} \\
    #
    # Let's rebuild the header manually to be safe.
    
    lines = latex_tabular.split('\n')
    new_lines = []
    
    header_replaced = False
    
    # We expect the first few lines to contain the grouped header
    # We will look for the line with "Opt" and "Improvement"
    
    for i, line in enumerate(lines):
        if "Opt" in line and "Improvement" in line and not header_replaced:
            # Reconstruct this line exactly as we want
            new_lines.append(r' &  & \multicolumn{4}{c}{Opt} & \multicolumn{2}{c}{Improvement} \\')
            new_lines.append(r'\cmidrule(lr){3-6} \cmidrule(lr){7-8}')
            header_replaced = True
        elif "Clicks" in line and "Cost" in line and "Kws" in line:
            # This is the second header row (column names)
            # Ensure it has midrule after
            new_lines.append(line)
            new_lines.append(r'\midrule')
        else:
            new_lines.append(line)
            
    latex_tabular = '\n'.join(new_lines)

    # --- 9. Inject Footer Note ---
    if r'\bottomrule' in latex_tabular:
        latex_tabular = latex_tabular.replace(
            r'\bottomrule', 
            r'\midrule' + '\n' + note_row + r' \\' + '\n' + r'\bottomrule'
        )

    # --- 10. Final Assembly ---
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{!}{0.2\\textheight}{%\n"
        f"{latex_tabular}"
        "}\n"
        "\\end{table}"
    )
    
    return final_latex

def main():

    p = argparse.ArgumentParser()

    # Accept same args to locate folders, though we might iterate all if not provided
    p.add_argument("--exp-name", default="backtests", help="Experiment name")
    args = p.parse_args()
    
    base_results_dir = Path(f"opt_results/backtests/{args.exp_name}")
    eval_csv = base_results_dir / "evaluation_results.csv"

    if not eval_csv.exists():
        print(f"Evaluation results not found at {eval_csv}")
        return

    # Read CSV, keeping 'None' as string if present, or better yet, reading as is
    # Using keep_default_na=False might mess up other cols, so let's just fillna after
    full_results = pd.read_csv(eval_csv)
    
    # Fill NaN x_max with "None" string or float('inf') to ensure it is not dropped by groupby
    # Also handle string "None" or "inf"
    full_results['x_max'] = full_results['x_max'].astype(str).replace({'nan': 'None', 'inf': 'None', 'None': 'None'})
    
    # Map back to float where possible for sorting/display logic later?
    # Actually, keep as stable string "None" or number, or let groupby handle mixed types?
    # Better to normalize to either float (with inf) or keep mixed.
    # Groupby dropna=False is easiest.
    
    summary_rows = []
    
    # Filter based on args if needed (logic to match None/inf is tricky with pandas, so maybe just group by)
    # If args.x_max/alpha are provided, we could filter, but let's just process what is in the file
    # Or strict filtering?
    # Let's group by x_max and alpha
    
    # Clean up x_max for grouping (handle infs/None)
    # full_results['x_max_grp'] = full_results['x_max'].apply(lambda x: float('inf') if str(x).lower() == 'inf' else (None if pd.isna(x) or str(x).lower() == 'none' else float(x)))

    # Only include request x_max/alpha if needed? Or just show all.
    # The original script iterated logic.
    grouped = full_results.groupby(['x_max', 'alpha'], dropna=False)

    for (xm, al), df_group in grouped:
        # Check if this combo is in args (optional)
        # For now, let's process all present in the file
        
        avg_clicks_opt = df_group["t_Clicks_OptCost"].mean()
        se_clicks_opt = df_group["t_Clicks_OptCost"].sem()
        avg_cost_opt = df_group["Opt_Cost"].mean()
        se_cost_opt = df_group["Opt_Cost"].sem()
        
        clicks_per_dollar_opt = df_group["t_Clicks_OptCost"].sum() / df_group["Opt_Cost"].sum() if df_group["Opt_Cost"].sum() > 0 else 0
        avg_n_kws_opt = df_group.get("N_Opt", pd.Series([0]*len(df_group))).mean() # N_Opt might not be in CSV if I missed adding it backtest_eval
        se_n_kws_opt = df_group.get("N_Opt", pd.Series([0]*len(df_group))).sem()

        avg_clicks_act = df_group["t_Clicks_ActCost"].mean()
        se_clicks_act = df_group["t_Clicks_ActCost"].sem()
        avg_cost_act = df_group["Act_Cost"].mean()
        se_cost_act = df_group["Act_Cost"].sem()
        clicks_per_dollar_act = df_group["t_Clicks_ActCost"].sum() / df_group["Act_Cost"].sum() if df_group["Act_Cost"].sum() > 0 else 0
        
        avg_n_kws_act = df_group.get("N_Obs", pd.Series([0]*len(df_group))).mean() # N_Obs was added
        se_n_kws_act = df_group.get("N_Obs", pd.Series([0]*len(df_group))).sem()

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
