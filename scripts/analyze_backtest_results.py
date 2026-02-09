"""
Example usage: python scripts/analyze_backtest_results.py --course gen_ai --exp-name exp1
Analyzes backtest results from the backtest_eval.py script. Generates a summary csv and LaTeX table of performance metrics
and regional cost shares across different budget levels.
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
    
    df['avg clicks (opt)'] = df.apply(lambda row: f"{row['avg clicks (opt)']:,.1f} $\\pm$ {row['se clicks (opt)']:,.1f}", axis=1)
    df['avg purch (opt)'] = df.apply(lambda row: f"{row['avg purch (opt)']:,.2f} $\\pm$ {row['se purch (opt)']:,.2f}", axis=1)
    df['avg cost (opt)'] = df.apply(lambda row: f"{row['avg cost (opt)']:,.2f} $\\pm$ {row['se cost (opt)']:,.2f}", axis=1)
    df['avg kws (opt)'] = df.apply(lambda row: f"{row['avg kws (opt)']:,.0f} $\\pm$ {row['se kws (opt)']:,.0f}", axis=1)

    # Percentage Metrics
    for col in ['improvement in clicks', 'improvement in clicks/$', 'improvement in purch']:
        if col in df.columns:
            df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
            
    # Simple numeric
    if 'clicks/$ (opt)' in df.columns:
        df['clicks/$ (opt)'] = df['clicks/$ (opt)'].map('{:,.3f}'.format)

    # --- APPLY BOLDING ---
    # Fix for Warning: Convert all data to object/string type BEFORE injecting bold strings
    df = df.astype(object)
    
    # Bold best improvement in clicks
    if 'improvement in clicks' in df.columns:
        best_idx = df['improvement in clicks'].str.rstrip(r'\%').astype(float).idxmax()
        for col in df.columns:
            current_val = df.at[best_idx, col]
            df.at[best_idx, col] = f"\\textbf{{{str(current_val)}}}"

    # --- COLUMN MAPPING ---
    # Note: Added Purchases
    col_mapping = [
        ('Budget',                  ('', 'Budget')),
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg purch (opt)',         ('Opt', 'Purch')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', r'Clicks/\$')),
        ('avg kws (opt)',           ('Opt', 'Kws')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in purch',    ('Improvement', 'Purch')),
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
        'clicks': summary_df['avg clicks (act)'].iloc[0],
        'se_clicks': summary_df['se clicks (act)'].iloc[0],
        'purch': summary_df['avg purch (act)'].iloc[0] if 'avg purch (act)' in summary_df.columns else 0,
        'se_purch': summary_df['se purch (act)'].iloc[0] if 'se purch (act)' in summary_df.columns else 0,
        'cost': summary_df['avg cost (act)'].iloc[0],
        'se_cost': summary_df['se cost (act)'].iloc[0],
        'cpc': summary_df['clicks/$ (act)'].iloc[0],
        'kws': summary_df['avg n kws (act)'].iloc[0],
        'se_kws': summary_df['se n kws (act)'].iloc[0],
    }

    note_row = (
        fr"\multicolumn{{{total_cols}}}{{l}}{{\scriptsize \textbf{{Actual values:}} "
        f"Clicks: {fmt_mse(act_vals['clicks'], act_vals['se_clicks'])}, "
        f"Purch: {fmt_mse(act_vals['purch'], act_vals['se_purch'], 2)}, "
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

def generate_share_table(share_df, categories, display_renames=None):
    """Generic function to generate a LaTeX share breakdown table.
    
    Used for regional, origin, and match type percentage breakdowns.
    
    Args:
        share_df: DataFrame with 'Budget' column and columns like 'Spend {cat}', 'Clicks {cat}', etc.
        categories: List of category names as they appear in column names.
        display_renames: Optional dict mapping column category names to shorter display names.
    """
    df = share_df.copy()
    df['Budget'] = df['Budget'].astype(str)
    
    def sorter(val):
        if val == "Actual": return -1
        try: return float(val)
        except: return 999999
    
    df['sort_key'] = df['Budget'].apply(sorter)
    df = df.sort_values(by=['sort_key']).drop(columns=['sort_key'])

    metrics = ['Spend', 'Clicks', 'Purch']
    col_order = ['Budget']
    for cat in categories:
        for metric in metrics:
            col_order.append(f'{metric} {cat}')
        
    df = df[[c for c in col_order if c in df.columns]]

    # Apply display renames
    if display_renames:
        new_cols = []
        for c in df.columns:
            new_c = c
            for old_name, new_name in display_renames.items():
                if old_name in c:
                    new_c = c.replace(old_name, new_name)
                    break
            new_cols.append(new_c)
        df.columns = new_cols

    # Format percentages
    for col in df.columns:
        if col != 'Budget':
             df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
             
    # Create MultiIndex for LaTeX: (Category, Metric)
    tuples = []
    for c in df.columns:
        if c == 'Budget': 
            tuples.append(('', 'Budget'))
        else:
            parts = c.split(' ', 1)  # Split on first space only
            metric = parts[0]
            cat = parts[1] if len(parts) > 1 else ''
            tuples.append((cat, metric))
            
    df.columns = pd.MultiIndex.from_tuples(tuples)
    
    col_format = 'l' + 'c' * (len(df.columns) - 1)
    
    latex = df.to_latex(
        index=False, 
        column_format=col_format, 
        multicolumn_format='c', 
        escape=False
    )
    latex = latex.replace(r'\hline', r'\toprule', 1)
    if latex.strip().endswith(r'\hline'):
        latex = latex.strip()[:-6] + r'\bottomrule'
        
    # Inject Cmidrules
    lines = latex.split('\n')
    new_lines = []
    header_idx = -1
    
    for i, line in enumerate(lines):
        if "multicolumn" in line:
            header_idx = i
            break
            
    if header_idx != -1:
        level0 = df.columns.get_level_values(0)
        unique_cats = [x for x in level0.unique() if x != '']
        
        cmid_str = ""
        current_col = 2  # 1-based index, Budget is col 1
        
        for cat in unique_cats:
            count = sum(1 for x in level0 if x == cat)
            end_col = current_col + count - 1
            cmid_str += fr"\cmidrule(lr){{{current_col}-{end_col}}} "
            current_col += count
            
        for i, line in enumerate(lines):
            new_lines.append(line)
            if i == header_idx:
                new_lines.append(cmid_str)
    else:
        new_lines = lines

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

def generate_country_table(exp_name, budget, course="gen_ai", top_n=10):
    # Load all daily country files for this budget
    run_dir = Path(f"opt_results/{course}/backtests/{exp_name}/budget_{budget}")
    if not run_dir.exists():
        return None
        
    country_files = list(run_dir.glob("country_breakdown_*.csv"))
    if not country_files:
        return None
        
    dfs = []
    for f in country_files:
        try:
            d = pd.read_csv(f)
            # Ensure columns exist (backwards compatibility or if file is old)
            for col in ['Opt_Clicks', 'Act_Clicks', 'Opt_Spend', 'Act_Spend']:
                 if col not in d.columns: d[col] = 0.0
            dfs.append(d)
        except:
            pass
            
    if not dfs:
        return None
        
    full_df = pd.concat(dfs)
    
    # Aggregation
    agg_df = full_df.groupby(['Location', 'Region']).agg(
        Opt_Purch_Mean=('Opt_Purchases', 'mean'),
        Opt_Purch_SE=('Opt_Purchases', 'sem'),
        Act_Purch_Mean=('Act_Purchases', 'mean'),
        Act_Purch_SE=('Act_Purchases', 'sem'),
        
        Opt_Click_Mean=('Opt_Clicks', 'mean'),
        Opt_Click_SE=('Opt_Clicks', 'sem'),
        Act_Click_Mean=('Act_Clicks', 'mean'),
        Act_Click_SE=('Act_Clicks', 'sem'),
        
        Opt_Spend_Mean=('Opt_Spend', 'mean'),
        Opt_Spend_SE=('Opt_Spend', 'sem'),
        Act_Spend_Mean=('Act_Spend', 'mean'),
        Act_Spend_SE=('Act_Spend', 'sem'),
    ).reset_index()
    
    # Sort by Opt Purch Mean Desc
    agg_df = agg_df.sort_values(by='Opt_Purch_Mean', ascending=False).head(top_n)
    
    # Format Function
    def fmt(mean, se, decimals=1):
        return f"{mean:,.{decimals}f} $\\pm$ {se:,.{decimals}f}"

    # Create formatted columns
    agg_df[('Opt', 'Spend')] = agg_df.apply(lambda r: fmt(r['Opt_Spend_Mean'], r['Opt_Spend_SE'], 2), axis=1)
    agg_df[('Opt', 'Clicks')] = agg_df.apply(lambda r: fmt(r['Opt_Click_Mean'], r['Opt_Click_SE'], 1), axis=1)
    agg_df[('Opt', 'Purch')] = agg_df.apply(lambda r: fmt(r['Opt_Purch_Mean'], r['Opt_Purch_SE'], 2), axis=1)
    
    agg_df[('Act', 'Spend')] = agg_df.apply(lambda r: fmt(r['Act_Spend_Mean'], r['Act_Spend_SE'], 2), axis=1)
    agg_df[('Act', 'Clicks')] = agg_df.apply(lambda r: fmt(r['Act_Click_Mean'], r['Act_Click_SE'], 1), axis=1)
    agg_df[('Act', 'Purch')] = agg_df.apply(lambda r: fmt(r['Act_Purch_Mean'], r['Act_Purch_SE'], 2), axis=1)

    # Select columns
    cols = [('Opt', 'Spend'), ('Opt', 'Clicks'), ('Opt', 'Purch'), 
            ('Act', 'Spend'), ('Act', 'Clicks'), ('Act', 'Purch')]
    
    # We need Location and Region too
    # For MultiIndex, we can handle index differently or flatten.
    # Let's clean up dataframe for export
    
    # Create a clean DF with MultiIndex Columns
    out_df = agg_df.set_index(['Location', 'Region'])[cols]
    out_df.columns = pd.MultiIndex.from_tuples(cols)
    out_df = out_df.reset_index()
    
    # The columns are now (Location, ''), (Region, ''), (Opt, Spend), ...
    # Wait, reset_index flattens or makes it weird with existing MultiIndex columns.
    # Better approach: construct tuples for all columns.
    
    final_cols = []
    final_cols.append(('', 'Location'))
    final_cols.append(('', 'Region'))
    final_cols.extend(cols)
    
    out_df.columns = pd.MultiIndex.from_tuples(final_cols)
    
    col_format = 'll' + 'c'*6
    latex = out_df.to_latex(index=False, column_format=col_format, multicolumn_format='c', escape=False)
    latex = latex.replace(r'\hline', r'\toprule', 1)
    if latex.strip().endswith(r'\hline'):
        latex = latex.strip()[:-6] + r'\bottomrule'
        
    # Inject Cmidrules
    lines = latex.split('\n')
    new_lines = []
    header_idx = -1
    
    for i, line in enumerate(lines):
        if "multicolumn" in line:
            header_idx = i
            break
            
    if header_idx != -1:
        # Columns: Loc(1), Reg(2), Opt(3-5), Act(6-8)
        cmid_str = r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}"
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            if i == header_idx:
                new_lines.append(cmid_str)
    else:
        new_lines = lines
        
    lines_joined = '\n'.join(new_lines)
        
    caption_budget = f"{budget}" if budget != 'Actual' else "Actual"
    
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Top %d Countries by Predicted Conversions (Budget %s)}\n"
        "\\resizebox{\\textwidth}{!}{%%\n"
        f"{lines_joined}"
        "}\n"
        "\\end{table}"
    ) % (top_n, caption_budget)
    
    return final_latex

def generate_stability_table(stability_df):
    df = stability_df.copy()
    
    # Sort by Budget
    df['Budget'] = df['Budget'].apply(lambda x: f"{float(x):.0f}")
    df['sort_key'] = df['Budget'].astype(float)
    df = df.sort_values('sort_key').drop(columns=['sort_key'])
    
    # Format Function
    def fmt(mean, se, decimals=1, suffix=""):
        # Handle percent scaling if needed (input is fraction or percent?)
        # Scripts output: "pct_change" is fraction. "pct_new_keywords" is fraction.
        # So we mul by 100 for display.
        mean_pct = mean * 100
        se_pct = se * 100
        return f"{mean_pct:,.{decimals}f}{suffix} $\\pm$ {se_pct:,.{decimals}f}{suffix}"
        
    df['Avg Cost Change'] = df.apply(lambda r: fmt(r['avg_cost_change'], r['se_cost_change'], 1, r'\%'), axis=1)
    df['New Keywords'] = df.apply(lambda r: fmt(r['avg_new_kws'], r['se_new_kws'], 1, r'\%'), axis=1)
    
    # Select columns
    out_df = df[['Budget', 'Avg Cost Change', 'New Keywords']].copy()
    
    # LaTeX
    latex = out_df.to_latex(index=False, column_format='lcc', escape=False)
    latex = latex.replace(r'\hline', r'\toprule', 1)
    if latex.strip().endswith(r'\hline'):
        latex = latex.strip()[:-6] + r'\bottomrule'
        
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Keywords Stability Metrics (Avg $\\pm$ SE)}\n"
        f"{latex}"
        "\\end{table}"
    )
    return final_latex

def compute_share_row(source_df, categories, prefix, infix, totals, display_names=None):
    """Compute percentage share row for a given breakdown dimension.
    
    Args:
        source_df: DataFrame containing the breakdown columns.
        categories: List of category values (e.g., ['USA', 'A', 'B']).
        prefix: Column prefix ('Opt' or 'Act').
        infix: Column infix ('Region', 'Origin', 'Match').
        totals: Dict with keys 'cost', 'clicks', 'conv', 'purch' for denominators.
        display_names: Optional dict mapping category values to display names.
    
    Returns:
        Dict with keys like 'Spend {display_cat}', 'Clicks {display_cat}', etc.
    """
    row = {}
    metric_map = {
        'Spend': ('Cost', totals.get('cost', 0)),
        'Clicks': ('Clicks', totals.get('clicks', 0)),
        'Purch': ('Purch', totals.get('purch', 0)),
    }
    for cat in categories:
        display_cat = display_names.get(cat, cat) if display_names else cat
        for display_metric, (col_metric, total) in metric_map.items():
            col = f"{prefix}_{col_metric}_{infix}_{cat}"
            if col in source_df.columns and total > 0:
                row[f"{display_metric} {display_cat}"] = source_df[col].sum() / total
            else:
                row[f"{display_metric} {display_cat}"] = 0
    return row

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", default="backtests", help="Experiment name")
    p.add_argument("--course", default="gen_ai", help="Course name")
    args = p.parse_args()
    
    base_results_dir = Path(f"opt_results/{args.course}/backtests/{args.exp_name}")
    eval_csv = base_results_dir / "evaluation_results.csv"

    if not eval_csv.exists():
        print(f"Evaluation results not found at {eval_csv}")
        return

    full_results = pd.read_csv(eval_csv)
    
    # Handle different evaluation formats
    if 'Budget' not in full_results.columns:
        print(f"Available columns: {list(full_results.columns)}")
        # Check for old format with x_max/alpha columns
        if 'x_max' in full_results.columns or 'alpha' in full_results.columns:
            print("Detected old evaluation format with x_max/alpha columns.")
            print("Please re-run backtest_eval.py to generate the new format with Budget column.")
            print("The new format includes regional breakdowns, conversions, and purchases.")
            return
        else:
            print("Warning: 'Budget' column not found. Assuming single budget scenario.")
            # If there's no Budget column, assume a single budget scenario
            full_results['Budget'] = 'N/A'
    
    summary_rows = []
    regional_rows = []
    origin_rows = []
    match_type_rows = []
    stability_rows = []
    
    grouped = full_results.groupby('Budget', dropna=False)

    # Calculate Actuals (GLOBAL)
    act_df = full_results.drop_duplicates(subset=['Day']) if 'Day' in full_results.columns else full_results

    avg_clicks_act = act_df["t_Clicks_ActCost"].mean()
    se_clicks_act = act_df["t_Clicks_ActCost"].sem()
    avg_purch_act = act_df["Act_Purch"].mean() if "Act_Purch" in act_df.columns else 0
    se_purch_act = act_df["Act_Purch"].sem() if "Act_Purch" in act_df.columns else 0
    avg_cost_act = act_df["Act_Cost"].mean()
    se_cost_act = act_df["Act_Cost"].sem()

    avg_kws_act = act_df["N_Obs"].mean() if "N_Obs" in act_df.columns else 0
    se_kws_act = act_df["N_Obs"].sem() if "N_Obs" in act_df.columns else 0

    clicks_per_dollar_act = act_df["t_Clicks_ActCost"].sum() / act_df["Act_Cost"].sum() if act_df["Act_Cost"].sum() > 0 else 0
    
    # Actual Share Breakdowns (Regional, Origin, Match Type)
    total_act_cost = act_df["Act_Cost"].sum()
    total_act_purch = act_df["Act_Purch"].sum() if "Act_Purch" in act_df.columns else 0
    total_act_clicks = act_df["t_Clicks_ActCost"].sum()
    
    act_totals = {'cost': total_act_cost, 'clicks': total_act_clicks, 'purch': total_act_purch}
    
    # Breakdown dimensions
    regions = ['USA', 'A', 'B']
    origin_keys = ['new', 'existing', 'existing searches']
    origin_display = {k: k.capitalize() for k in origin_keys}
    match_types = ['Exact match', 'Phrase match', 'Broad match']
    
    # Actual Regional Share
    act_reg_row = {"Budget": "Actual"}
    act_reg_row.update(compute_share_row(act_df, regions, 'Act', 'Region', act_totals))
    regional_rows.append(act_reg_row)
    
    # Actual Origin Share
    act_orig_row = {"Budget": "Actual"}
    act_orig_row.update(compute_share_row(act_df, origin_keys, 'Act', 'Origin', act_totals, origin_display))
    origin_rows.append(act_orig_row)
    
    # Actual Match Type Share
    act_mt_row = {"Budget": "Actual"}
    act_mt_row.update(compute_share_row(act_df, match_types, 'Act', 'Match', act_totals))
    match_type_rows.append(act_mt_row)

    for budget, df_group in grouped:
        
        # 1. Performance Metrics
        avg_clicks_opt = df_group["t_Clicks_OptCost"].mean()
        se_clicks_opt = df_group["t_Clicks_OptCost"].sem()
        avg_cost_opt = df_group["Opt_Cost"].mean()
        se_cost_opt = df_group["Opt_Cost"].sem()
        avg_purch_opt = df_group['Opt_Purch'].mean() if 'Opt_Purch' in df_group.columns else 0
        se_purch_opt = df_group['Opt_Purch'].sem() if 'Opt_Purch' in df_group.columns else 0

        avg_kws_opt = df_group["N_Opt"].mean()
        se_kws_opt = df_group["N_Opt"].sem()
        
        # Stability Metrics
        if "Avg_Cost_Change" in df_group.columns:
            avg_cost_change = df_group["Avg_Cost_Change"].mean()
            se_cost_change = df_group["Avg_Cost_Change"].sem()
        else:
            avg_cost_change, se_cost_change = 0, 0
            
        if "Pct_New_Keywords" in df_group.columns:
            avg_new_kws = df_group["Pct_New_Keywords"].mean()
            se_new_kws = df_group["Pct_New_Keywords"].sem()
        else:
            avg_new_kws, se_new_kws = 0, 0
            
        stability_rows.append({
            "Budget": budget,
            "avg_cost_change": avg_cost_change,
            "se_cost_change": se_cost_change,
            "avg_new_kws": avg_new_kws,
            "se_new_kws": se_new_kws
        })
        
        clicks_per_dollar_opt = df_group["t_Clicks_OptCost"].sum() / df_group["Opt_Cost"].sum() if df_group["Opt_Cost"].sum() > 0 else 0
        
        # Improvements vs Actual
        imp_clicks = (avg_clicks_opt - avg_clicks_act) / avg_clicks_act if avg_clicks_act > 0 else 0
        imp_c_d = (clicks_per_dollar_opt - clicks_per_dollar_act) / clicks_per_dollar_act if clicks_per_dollar_act > 0 else 0
        imp_purch = (avg_purch_opt - avg_purch_act) / avg_purch_act if avg_purch_act > 0 else 0

        summary_rows.append({
            "Budget": budget,
            "avg clicks (opt)": avg_clicks_opt,
            "se clicks (opt)": se_clicks_opt,
            "avg cost (opt)": avg_cost_opt,
            "se cost (opt)": se_cost_opt,
            "avg purch (opt)": avg_purch_opt,
            "se purch (opt)": se_purch_opt,
            "clicks/$ (opt)": clicks_per_dollar_opt,
            "avg kws (opt)": avg_kws_opt,
            "se kws (opt)": se_kws_opt,
            "improvement in clicks": imp_clicks,
            "improvement in clicks/$": imp_c_d,
            "improvement in purch": imp_purch,
            "avg clicks (act)": avg_clicks_act,
            "se clicks (act)": se_clicks_act,
            "avg cost (act)": avg_cost_act,
            "se cost (act)": se_cost_act,
            "avg purch (act)": avg_purch_act,
            "se purch (act)": se_purch_act,
            "clicks/$ (act)": clicks_per_dollar_act,
            "avg n kws (act)": avg_kws_act,
            "se n kws (act)": se_kws_act
        })
        
        # 2. Share Breakdown Metrics (Regional, Origin, Match Type)
        total_opt_cost_grp = df_group["Opt_Cost"].sum()
        total_opt_purch_grp = df_group["Opt_Purch"].sum() if "Opt_Purch" in df_group.columns else 0
        total_opt_clicks_grp = df_group["t_Clicks_OptCost"].sum()
        
        opt_totals = {'cost': total_opt_cost_grp, 'clicks': total_opt_clicks_grp, 'purch': total_opt_purch_grp}
        
        # Regional
        reg_row = {"Budget": budget}
        reg_row.update(compute_share_row(df_group, regions, 'Opt', 'Region', opt_totals))
        regional_rows.append(reg_row)
        
        # Origin
        orig_row = {"Budget": budget}
        orig_row.update(compute_share_row(df_group, origin_keys, 'Opt', 'Origin', opt_totals, origin_display))
        origin_rows.append(orig_row)
        
        # Match Type
        mt_row = {"Budget": budget}
        mt_row.update(compute_share_row(df_group, match_types, 'Opt', 'Match', opt_totals))
        match_type_rows.append(mt_row)

    if not summary_rows:
        print("No results found.")
        return

    # --- Output Performance Table ---
    summary_df = pd.DataFrame(summary_rows)
    out_csv = base_results_dir / "backtest_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    
    # --- Output Stability Table ---
    if stability_rows and any(r['avg_cost_change'] != 0 for r in stability_rows):
        stability_df = pd.DataFrame(stability_rows)
        out_stab_tex = base_results_dir / "stability_metrics.tex"
        latex_stab = generate_stability_table(stability_df)
        with open(out_stab_tex, "w") as f:
            f.write(latex_stab)
        print(f"\nStability Table saved to {out_stab_tex}")
        print(latex_stab)
    
    latex_perf = generate_performance_table(summary_df)
    out_tex = base_results_dir / "backtest_summary.tex"
    with open(out_tex, "w") as f:
        f.write(latex_perf)
    print(f"\nPerformance Table saved to {out_tex}")
    print(latex_perf)
    
    # --- Output Regional Table ---
    regional_df = pd.DataFrame(regional_rows)
    out_reg_csv = base_results_dir / "regional_breakdown.csv"
    regional_df.to_csv(out_reg_csv, index=False)
    
    latex_reg = generate_share_table(regional_df, regions)
    out_reg_tex = base_results_dir / "regional_breakdown.tex"
    with open(out_reg_tex, "w") as f:
        f.write(latex_reg)
    print(f"\nRegional Table saved to {out_reg_tex}")
    print(latex_reg)

    # --- Output Origin Table ---
    origin_df = pd.DataFrame(origin_rows)
    out_orig_csv = base_results_dir / "origin_breakdown.csv"
    origin_df.to_csv(out_orig_csv, index=False)
    
    latex_orig = generate_share_table(origin_df, 
                                       [k.capitalize() for k in origin_keys],
                                       display_renames={'Existing searches': 'ExSearches'})
    out_orig_tex = base_results_dir / "origin_breakdown.tex"
    with open(out_orig_tex, "w") as f:
        f.write(latex_orig)
    print(f"\nOrigin Table saved to {out_orig_tex}")
    print(latex_orig)

    # --- Output Match Type Table ---
    match_type_df = pd.DataFrame(match_type_rows)
    out_mt_csv = base_results_dir / "match_type_breakdown.csv"
    match_type_df.to_csv(out_mt_csv, index=False)
    
    latex_mt = generate_share_table(match_type_df, match_types,
                                     display_renames={'Exact match': 'Exact', 'Phrase match': 'Phrase', 'Broad match': 'Broad'})
    out_mt_tex = base_results_dir / "match_type_breakdown.tex"
    with open(out_mt_tex, "w") as f:
        f.write(latex_mt)
    print(f"\nMatch Type Table saved to {out_mt_tex}")
    print(latex_mt)

    # --- Output Country Table (Best Budget) ---
    # Find best budget based on clicks improvement
    if 'improvement in clicks' in summary_df.columns:
         best_row = summary_df.loc[summary_df['improvement in clicks'].idxmax()]
         best_budget = best_row['Budget']
         
         latex_country = generate_country_table(args.exp_name, int(best_budget), args.course)
         if latex_country:
             out_country_tex = base_results_dir / "country_breakdown.tex"
             with open(out_country_tex, "w") as f:
                 f.write(latex_country)
             print(f"\nCountry Table (Budget {best_budget}) saved to {out_country_tex}")
             print(latex_country)
         else:
             print("\nCould not generate country table (files missing?)")

if __name__ == "__main__":
    main()
