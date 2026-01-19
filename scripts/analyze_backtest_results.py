"""
To run: python scripts/analyze_backtest_results.py --exp-name "experiment_v1"
This script analyzes backtest results from the backtest_daily.py script. It aggregates daily evaluation metrics across different
(x_max, alpha) parameter combinations, computes average performance metrics, and generates a summary CSV and LaTeX table.
"""
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

def generate_performance_table(summary_df):
    df = summary_df.copy()
    df = df.sort_values(by=['Budget'], ascending=[True], na_position='last')

    # Identify "Best" Row (max improvement in clicks)
    best_idx = df['improvement in clicks'].idxmax()

    # Formatters
    df['Budget'] = df['Budget'].apply(lambda x: f"{x:.0f}")
    
    if 'se clicks (opt)' in df.columns:
        df['avg clicks (opt)'] = df.apply(lambda row: f"{row['avg clicks (opt)']:,.1f} $\\pm$ {row['se clicks (opt)']:,.1f}", axis=1)
    
    if 'se cost (opt)' in df.columns:
        df['avg cost (opt)'] = df.apply(lambda row: f"{row['avg cost (opt)']:,.2f} $\\pm$ {row['se cost (opt)']:,.2f}", axis=1)

    # Percentage Metrics
    for col in ['improvement in clicks', 'improvement in clicks/$']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
            
    # Simple numeric
    if 'clicks/$ (opt)' in df.columns:
        df['clicks/$ (opt)'] = df['clicks/$ (opt)'].map('{:,.3f}'.format)

    # Apply Bolding
    for col in df.columns:
        current_val = df.at[best_idx, col]
        df.at[best_idx, col] = f"\\textbf{{{str(current_val)}}}"

    col_mapping = [
        ('Budget',                  ('', 'Budget')),
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', r'Clicks/\$')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in clicks/$', ('Improvement', r'Clicks/\$'))
    ]

    existing_cols = [old for old, new in col_mapping if old in df.columns]
    df = df[existing_cols]
    df.columns = pd.MultiIndex.from_tuples([new for old, new in col_mapping if old in df.columns])

    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        multicolumn_format='c',
        column_format='lccccc'
    )
    
    latex_tabular = latex_tabular.replace(r'\hline', r'\toprule', 1)
    if latex_tabular.strip().endswith(r'\hline'):
        latex_tabular = latex_tabular.strip()[:-6] + r'\bottomrule'
        
    # Header injection
    lines = latex_tabular.split('\n')
    new_lines = []
    header_replaced = False
    
    for line in lines:
        if "Opt" in line and "Improvement" in line and not header_replaced:
            new_lines.append(r' & \multicolumn{3}{c}{Opt} & \multicolumn{2}{c}{Improvement} \\')
            new_lines.append(r'\cmidrule(lr){2-4} \cmidrule(lr){5-6}')
            header_replaced = True
        elif "Budget" in line and "Clicks" in line:
            new_lines.append(line)
            new_lines.append(r'\midrule')
        else:
            new_lines.append(line)
            
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Optimization Performance by Budget}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{'\n'.join(new_lines)}"
        "}\n"
        "\\end{table}"
    )
    return final_latex

def generate_regional_table(share_df):
    df = share_df.copy()
    df = df.sort_values(by=['Budget'], ascending=True)

    # Format Percentages
    for col in ['Share USA', 'Share A', 'Share B']:
        df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
        
    df['Budget'] = df['Budget'].astype(str)
    
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
             
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Regional Cost Share vs Budget}\n"
        f"{'\n'.join(new_lines)}\n"
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
    
    grouped = full_results.groupby(['Budget'], dropna=False)

    # Calculate Actuals (GLOBAL - average across all data, assumed constant across budgets implicitly)
    avg_clicks_act = full_results["t_Clicks_ActCost"].mean()
    # se_clicks_act = full_results["t_Clicks_ActCost"].sem()
    avg_cost_act = full_results["Act_Cost"].mean()
    # se_cost_act = full_results["Act_Cost"].sem()
    clicks_per_dollar_act = full_results["t_Clicks_ActCost"].sum() / full_results["Act_Cost"].sum() if full_results["Act_Cost"].sum() > 0 else 0
    
    # Actual Regional Share
    total_act_cost = full_results["Act_Cost"].sum()
    act_share_usa = full_results["Act_Cost_USA"].sum() / total_act_cost
    act_share_a = full_results["Act_Cost_A"].sum() / total_act_cost
    act_share_b = full_results["Act_Cost_B"].sum() / total_act_cost
    
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
            "improvement in clicks": imp_clicks,
            "improvement in clicks/$": imp_c_d
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
