"""
To run: python scripts/analyze_backtest_results.py --exp-name "experiment_v1"
This script analyzes backtest results from the backtest_daily.py script. It aggregates daily evaluation metrics across different
(x_max, alpha) parameter combinations, computes average performance metrics, and generates a summary CSV and LaTeX table.
"""
from sys import prefix
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def generate_performance_table(summary_df):
    # Create a copy to avoid SettingWithCopyWarning on the original df
    df = summary_df.copy()
    df = df.sort_values(by=['Budget'], ascending=[True], na_position='last')

    # Identify "Best" Row (max improvement in clicks)
    best_idx = df['improvement in clicks'].idxmax()

    # --- FORMATTING ---
    df['Budget'] = df['Budget'].apply(lambda x: f"{x:.0f}")
    
    if 'se clicks (opt)' in df.columns:
        df['avg clicks (opt)'] = df.apply(lambda row: f"{row['avg clicks (opt)']:,.1f} $\\pm$ {row['se clicks (opt)']:,.1f}", axis=1)
    
    if 'se cost (opt)' in df.columns:
        df['avg cost (opt)'] = df.apply(lambda row: f"{row['avg cost (opt)']:,.2f} $\\pm$ {row['se cost (opt)']:,.2f}", axis=1)

    if 'se kws (opt)' in df.columns:
        df['avg kws (opt)'] = df.apply(lambda row: f"{row['avg kws (opt)']:,.0f} $\\pm$ {row['se kws (opt)']:,.0f}", axis=1)

    # Percentage Metrics
    for col in ['improvement in clicks', 'improvement in clicks/$']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
            
    # Simple numeric
    if 'clicks/$ (opt)' in df.columns:
        df['clicks/$ (opt)'] = df['clicks/$ (opt)'].map('{:,.3f}'.format)

    # --- APPLY BOLDING ---
    # Fix for Warning: Convert all data to object/string type BEFORE injecting bold strings
    df = df.astype(object)
    
    for col in df.columns:
        current_val = df.at[best_idx, col]
        df.at[best_idx, col] = f"\\textbf{{{str(current_val)}}}"

    # --- COLUMN MAPPING ---
    col_mapping = [
        ('Budget',                  ('', 'Budget')),
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', r'Clicks/\$')),
        ('avg kws (opt)',           ('Opt', 'Kws')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in clicks/$', ('Improvement', r'Clicks/\$'))
    ]

    existing_cols = [old for old, new in col_mapping if old in df.columns]
    df = df[existing_cols]
    df.columns = pd.MultiIndex.from_tuples([new for old, new in col_mapping if old in df.columns])

    # --- DYNAMIC FORMATTING LOGIC ---
    # 1. Count dynamic columns to calculate spans and format string
    opt_cols_count = len([c for c in df.columns if c[0] == 'Opt'])
    imp_cols_count = len([c for c in df.columns if c[0] == 'Improvement'])
    
    # Calculate total columns: 1 (Budget) + Opt columns + Improvement columns
    total_cols = 1 + opt_cols_count + imp_cols_count
    
    # Create dynamic column format (e.g., 'lccccc')
    dyn_col_format = 'l' + 'c' * (total_cols - 1)

    # --- GENERATE LATEX ---
    
    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        multicolumn_format='c',
        column_format=dyn_col_format
    )
    
    latex_tabular = latex_tabular.replace(r'\hline', r'\toprule', 1)
    if latex_tabular.strip().endswith(r'\hline'):
        latex_tabular = latex_tabular.strip()[:-6] + r'\bottomrule'
        
    dollar_prefix = r'\$' 

    def fmt_mse(mean, se, decimals=1, prefix="", suffix=""):
        return f"{prefix}{mean:,.{decimals}f} $\\pm$ {se:,.{decimals}f}{suffix}"

    # Extract Baseline Data for Note
    act_vals = {
        'clicks': summary_df['avg clicks (act)'].iloc[0] if 'avg clicks (act)' in summary_df.columns else 0,
        'se_clicks': summary_df.get('se clicks (act)', pd.Series([0]*len(summary_df))).iloc[0],
        'cost': summary_df['avg cost (act)'].iloc[0] if 'avg cost (act)' in summary_df.columns else 0,
        'se_cost': summary_df.get('se cost (act)', pd.Series([0]*len(summary_df))).iloc[0],
        'cpc': summary_df['clicks/$ (act)'].iloc[0] if 'clicks/$ (act)' in summary_df.columns else 0,
        'kws': summary_df['avg n kws (act)'].iloc[0] if 'avg n kws (act)' in summary_df.columns else 0,
        'se_kws': summary_df.get('se n kws (act)', pd.Series([0]*len(summary_df))).iloc[0],
    }

    note_row = (
        fr"\multicolumn{{{total_cols}}}{{l}}{{\scriptsize \textbf{{Actual values:}} "
        f"Clicks: {fmt_mse(act_vals['clicks'], act_vals['se_clicks'])}, "
        f"Cost: {fmt_mse(act_vals['cost'], act_vals['se_cost'], 2, prefix=dollar_prefix)}, "
        f"Clicks/\\$: {act_vals['cpc']:.3f}, "
        f"Kws: {fmt_mse(act_vals['kws'], act_vals['se_kws'], 0)}."
        "}"
    )

    # --- DYNAMIC HEADER INJECTION ---
    lines = latex_tabular.split('\n')
    new_lines = []
    header_replaced = False
    
    # Construct the dynamic header row
    header_row = r' ' 
    if opt_cols_count > 0:
        header_row += fr'& \multicolumn{{{opt_cols_count}}}{{c}}{{Opt}} '
    if imp_cols_count > 0:
        header_row += fr'& \multicolumn{{{imp_cols_count}}}{{c}}{{Improvement}} '
    header_row += r'\\'

    # Construct dynamic cmidrules
    # Budget is col 1. Opt starts at 2.
    cmid_row = ''
    current_col = 2
    if opt_cols_count > 0:
        cmid_row += fr'\cmidrule(lr){{{current_col}-{current_col + opt_cols_count - 1}}} '
        current_col += opt_cols_count
    if imp_cols_count > 0:
        cmid_row += fr'\cmidrule(lr){{{current_col}-{current_col + imp_cols_count - 1}}}'

    for line in lines:
        if "Opt" in line and "Improvement" in line and not header_replaced:
            new_lines.append(header_row)
            new_lines.append(cmid_row)
            header_replaced = True
        elif "Budget" in line and "Clicks" in line:
            new_lines.append(line)
            new_lines.append(r'\midrule')
        elif "\\bottomrule" in line:
             new_lines.append(line)
             new_lines.append(note_row + r' \\')
        else:
            new_lines.append(line)
            
    lines_joined = '\n'.join(new_lines)
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{lines_joined}"
        "}\n"
        "\\end{table}"
    )
    return final_latex

def generate_regional_table(share_df):
    df = share_df.copy()
    df['Budget'] = df['Budget'].astype(str) # Can sort properly with 'Actual' entry
    df = df.sort_values(by=['Budget'], ascending=True)

    # Format Percentages
    for col in ['Share USA', 'Share A', 'Share B']:
        df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
    
    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        column_format='lrrr'
    )
    
    latex_tabular = latex_tabular.replace(r'\hline', r'\toprule', 1)
    if latex_tabular.strip().endswith(r'\hline'):
        latex_tabular = latex_tabular.strip()[:-6] + r'\bottomrule'

    # Add midrule under header
    lines = latex_tabular.split('\n')
    new_lines = []
    header_found = False
    for line in lines:
        new_lines.append(line)
        if "Share USA" in line and not header_found:
             new_lines.append(r'\midrule')
             header_found = True
    
    lines_joined = '\n'.join(new_lines)
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{lines_joined}"
        "}\n"
        "\\end{table}"
    )
    return final_latex

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="backtests", help="Experiment name")
    args = p.parse_args()
    
    base_results_dir = Path(f"opt_results/backtests/{args.exp_name}")
    eval_csv = base_results_dir / "evaluation_results.csv"

    if not eval_csv.exists():
        print(f"Evaluation results not found at {eval_csv}")
        return

    full_results = pd.read_csv(eval_csv)
    
    summary_rows = []
    regional_rows = []
    
    grouped = full_results.groupby('Budget', dropna=False)

    # Calculate Actuals (GLOBAL - average across all data, assumed constant across budgets implicitly)
    # Deduplicate by Day to avoid inflating N with multiple budgets/params per day
    act_df = full_results.drop_duplicates(subset=['Day']) if 'Day' in full_results.columns else full_results

    avg_clicks_act = act_df["t_Clicks_ActCost"].mean()
    se_clicks_act = act_df["t_Clicks_ActCost"].sem()
    avg_cost_act = act_df["Act_Cost"].mean()
    se_cost_act = act_df["Act_Cost"].sem()
    avg_kws_act = act_df["N_Obs"].mean() if "N_Obs" in act_df.columns else 0
    se_kws_act = act_df["N_Obs"].sem() if "N_Obs" in act_df.columns else 0

    clicks_per_dollar_act = act_df["t_Clicks_ActCost"].sum() / act_df["Act_Cost"].sum() if act_df["Act_Cost"].sum() > 0 else 0
    
    # Actual Regional Share
    total_act_cost = act_df["Act_Cost"].sum()
    act_share_usa = act_df["Act_Cost_USA"].sum() / total_act_cost if total_act_cost > 0 else 0
    act_share_a = act_df["Act_Cost_A"].sum() / total_act_cost if total_act_cost > 0 else 0
    act_share_b = act_df["Act_Cost_B"].sum() / total_act_cost if total_act_cost > 0 else 0
    
    # Add Actual row to regional rows first
    regional_rows.append({
        "Budget": "Actual",
        "Share USA": act_share_usa,
        "Share A": act_share_a,
        "Share B": act_share_b
    })

    for budget, df_group in grouped:
        
        # 1. Performance Metrics
        avg_clicks_opt = df_group["t_Clicks_OptCost"].mean()
        se_clicks_opt = df_group["t_Clicks_OptCost"].sem()
        avg_cost_opt = df_group["Opt_Cost"].mean()
        se_cost_opt = df_group["Opt_Cost"].sem()
        
        clicks_per_dollar_opt = df_group["t_Clicks_OptCost"].sum() / df_group["Opt_Cost"].sum() if df_group["Opt_Cost"].sum() > 0 else 0
        avg_kws_opt = df_group["N_Opt"].mean()
        se_kws_opt = df_group["N_Opt"].sem()
        
        clicks_per_dollar_opt = df_group["t_Clicks_OptCost"].sum() / df_group["Opt_Cost"].sum() if df_group["Opt_Cost"].sum() > 0 else 0
        
        # Improvements vs Actual
        imp_clicks = (avg_clicks_opt - avg_clicks_act) / avg_clicks_act if avg_clicks_act > 0 else 0
        imp_c_d = (clicks_per_dollar_opt - clicks_per_dollar_act) / clicks_per_dollar_act if clicks_per_dollar_act > 0 else 0

        summary_rows.append({
            "Budget": budget,
            "avg clicks (opt)": avg_clicks_opt,
            "se clicks (opt)": se_clicks_opt,
            "avg cost (opt)": avg_cost_opt,
            "se cost (opt)": se_cost_opt,
            "clicks/$ (opt)": clicks_per_dollar_opt,
            "avg kws (opt)": avg_kws_opt,
            "se kws (opt)": se_kws_opt,
            "improvement in clicks": imp_clicks,
            "improvement in clicks/$": imp_c_d,
            "avg clicks (act)": avg_clicks_act,
            "se clicks (act)": se_clicks_act,
            "avg cost (act)": avg_cost_act,
            "se cost (act)": se_cost_act,
            "clicks/$ (act)": clicks_per_dollar_act,
            "avg n kws (act)": avg_kws_act,
            "se n kws (act)": se_kws_act
        })
        
        # 2. Regional Share Metrics
        total_opt_cost_grp = df_group["Opt_Cost"].sum()
        share_opt_usa = df_group["Opt_Cost_USA"].sum() / total_opt_cost_grp if total_opt_cost_grp > 0 else 0
        share_opt_a = df_group["Opt_Cost_A"].sum() / total_opt_cost_grp if total_opt_cost_grp > 0 else 0
        share_opt_b = df_group["Opt_Cost_B"].sum() / total_opt_cost_grp if total_opt_cost_grp > 0 else 0
        
        regional_rows.append({
            "Budget": budget,
            "Share USA": share_opt_usa,
            "Share A": share_opt_a,
            "Share B": share_opt_b
        })

    if not summary_rows:
        print("No results found.")
        return

    # --- Output Performance Table ---
    summary_df = pd.DataFrame(summary_rows)
    out_csv = base_results_dir / "backtest_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    
    latex_perf = generate_performance_table(summary_df)
    out_tex = base_results_dir / "backtest_summary.tex"
    with open(out_tex, "w") as f:
        f.write(latex_perf)
    print(f"\nPerformance Table saved to {out_tex}")
    print(latex_perf)
    
    # --- Output Regional Table ---
    regional_df = pd.DataFrame(regional_rows)
    out_reg_csv = base_results_dir / "regional_cost_shares.csv"
    regional_df.to_csv(out_reg_csv, index=False)
    
    latex_reg = generate_regional_table(regional_df)
    out_reg_tex = base_results_dir / "regional_cost_shares.tex"
    with open(out_reg_tex, "w") as f:
        f.write(latex_reg)
    print(f"\nRegional Table saved to {out_reg_tex}")
    print(latex_reg)

if __name__ == "__main__":
    main()
