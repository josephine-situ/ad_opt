"""
Example usage: python scripts/analyze_backtest_results.py --exp-name exp1
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
    df['avg conv (opt)'] = df.apply(lambda row: f"{row['avg conv (opt)']:,.1f} $\\pm$ {row['se conv (opt)']:,.1f}", axis=1)
    df['avg cost (opt)'] = df.apply(lambda row: f"{row['avg cost (opt)']:,.2f} $\\pm$ {row['se cost (opt)']:,.2f}", axis=1)
    df['avg kws (opt)'] = df.apply(lambda row: f"{row['avg kws (opt)']:,.0f} $\\pm$ {row['se kws (opt)']:,.0f}", axis=1)

    # Percentage Metrics
    for col in ['improvement in clicks', 'improvement in clicks/$', 'improvement in conv']:
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
    # Note: Added Conversions
    col_mapping = [
        ('Budget',                  ('', 'Budget')),
        ('avg clicks (opt)',        ('Opt', 'Clicks')),
        ('avg conv (opt)',          ('Opt', 'Conv')),
        ('avg cost (opt)',          ('Opt', 'Cost')),
        ('clicks/$ (opt)',          ('Opt', r'Clicks/\$')),
        ('avg kws (opt)',           ('Opt', 'Kws')),
        ('improvement in clicks',   ('Improvement', 'Clicks')),
        ('improvement in conv',     ('Improvement', 'Conv')),
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
        'conv': summary_df['avg conv (act)'].iloc[0],
        'se_conv': summary_df['se conv (act)'].iloc[0],
        'cost': summary_df['avg cost (act)'].iloc[0],
        'se_cost': summary_df['se cost (act)'].iloc[0],
        'cpc': summary_df['clicks/$ (act)'].iloc[0],
        'kws': summary_df['avg n kws (act)'].iloc[0],
        'se_kws': summary_df['se n kws (act)'].iloc[0],
    }

    note_row = (
        fr"\multicolumn{{{total_cols}}}{{l}}{{\scriptsize \textbf{{Actual values:}} "
        f"Clicks: {fmt_mse(act_vals['clicks'], act_vals['se_clicks'])}, "
        f"Conv: {fmt_mse(act_vals['conv'], act_vals['se_conv'])}, "
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
    df['Budget'] = df['Budget'].astype(str)
    
    def sorter(val):
        if val == "Actual": return -1
        try: return float(val)
        except: return 999999
    
    df['sort_key'] = df['Budget'].apply(sorter)
    df = df.sort_values(by=['sort_key']).drop(columns=['sort_key'])

    regions = ['USA', 'A', 'B']
    col_order = ['Budget']
    for r in regions:
        col_order.append(f'Spend {r}')
        col_order.append(f'Clicks {r}')
        col_order.append(f'Conv {r}')
        
    df = df[[c for c in col_order if c in df.columns]]

    # Format
    for col in df.columns:
        if col != 'Budget':
             df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)
             
    # Create MultiIndex for LaTeX
    # Tuples: (Region, Metric)
    tuples = []
    for c in df.columns:
        if c == 'Budget': 
            tuples.append(('', 'Budget'))
        else:
            # c is "Spend USA" -> Type=Spend, Reg=USA. We want Reg on top.
            parts = c.split(' ')
            tuples.append((parts[1], parts[0])) # (Region, Type)
            
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
    
    # Locate header row with multicolumns (regions)
    for i, line in enumerate(lines):
        if "USA" in line and "multicolumn" in line:
            header_idx = i
            break
            
    if header_idx != -1:
        # Construct cmidrules
        # Budget is col 1. Regions start at 2.
        # Assuming fixed 3 cols per region if present, but we should count.
        # df.columns is MultiIndex. 
        # Level 0 is Region.
        level0 = df.columns.get_level_values(0)
        unique_regions = [x for x in level0.unique() if x != '']
        
        cmid_str = ""
        current_col = 2 # 1-based index, Budget is 1
        
        for reg in unique_regions:
            count = sum(1 for x in level0 if x == reg)
            end_col = current_col + count - 1
            cmid_str += fr"\cmidrule(lr){{{current_col}-{end_col}}} "
            current_col += count
        
        # Insert cmid_str after the header line
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

def generate_origin_table(share_df):
    df = share_df.copy()
    df['Budget'] = df['Budget'].astype(str)
    
    def sorter(val):
        if val == "Actual": return -1
        try: return float(val)
        except: return 999999
    
    df['sort_key'] = df['Budget'].apply(sorter)
    df = df.sort_values(by=['sort_key']).drop(columns=['sort_key'])
    
    # Clean column names to ensure they look like "Spend New", "Clicks New" etc.
    # We used "Spend {org_key.capitalize()}" previously (e.g., Spend New, Spend Existing).
    # map old generic names if any remain? No, we changed the generator.
    
    origins = ['New', 'Existing', 'Existing searches'] # Capitalized as per generator
    # Note: generator uses org_key.capitalize(). 'existing searches' -> 'Existing searches'
    
    col_order = ['Budget']
    for original_org in ['new', 'existing', 'existing searches']:
        org = original_org.capitalize()
        col_order.append(f'Spend {org}')
        col_order.append(f'Clicks {org}')
        col_order.append(f'Conv {org}')
    
    df = df[[c for c in col_order if c in df.columns]]
    
    # Renaming for display if needed? 'Existing searches' is long.
    # Maybe rename 'Existing searches' -> 'ExSearches' in columns?
    new_cols = []
    for c in df.columns:
        if 'Existing searches' in c:
            new_cols.append(c.replace('Existing searches', 'ExSearches'))
        else:
            new_cols.append(c)
    df.columns = new_cols
    
    for col in df.columns:
        if col != 'Budget':
             df[col] = (df[col] * 100).map('{:,.1f}\\%'.format)

    # MultiIndex
    tuples = []
    for c in df.columns:
        if c == 'Budget': 
            tuples.append(('', 'Budget'))
        else:
            # c is "Spend New" -> Type=Spend, Org=New. We want Org on top.
            parts = c.split(' ') # ['Spend', 'New']
            metric = parts[0]
            org = " ".join(parts[1:]) 
            tuples.append((org, metric))
            
    df.columns = pd.MultiIndex.from_tuples(tuples)

    col_format = 'l' + 'c' * (len(df.columns) - 1)
    
    latex = df.to_latex(index=False, column_format=col_format, multicolumn_format='c', escape=False)
    latex = latex.replace(r'\hline', r'\toprule', 1)
    if latex.strip().endswith(r'\hline'):
        latex = latex.strip()[:-6] + r'\bottomrule'
    
    # Inject Cmidrules
    lines = latex.split('\n')
    new_lines = []
    header_idx = -1
    
    # Locate header row with multicolumns
    for i, line in enumerate(lines):
        if "multicolumn" in line:
            header_idx = i
            break
            
    if header_idx != -1:
        level0 = df.columns.get_level_values(0)
        unique_orgs = [x for x in level0.unique() if x != '']
        
        cmid_str = ""
        current_col = 2
        
        for org in unique_orgs:
            count = sum(1 for x in level0 if x == org)
            end_col = current_col + count - 1
            cmid_str += fr"\cmidrule(lr){{{current_col}-{end_col}}} "
            current_col += count
            
        for i, line in enumerate(lines):
            new_lines.append(line)
            if i == header_idx:
                new_lines.append(cmid_str)
    else:
        new_lines = lines # fallback

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

def generate_country_table(exp_name, budget, top_n=10):
    # Load all daily country files for this budget
    run_dir = Path(f"opt_results/backtests/{exp_name}/budget_{budget}")
    if not run_dir.exists():
        return None
        
    country_files = list(run_dir.glob("country_breakdown_*.csv"))
    if not country_files:
        return None
        
    dfs = []
    for f in country_files:
        try:
            d = pd.read_csv(f)
            dfs.append(d)
        except:
            pass
            
    if not dfs:
        return None
        
    full_df = pd.concat(dfs)
    
    # Aggregation
    # Group by Location, Region. Calculate Mean and SE for Act and Opt Conv
    agg_df = full_df.groupby(['Location', 'Region']).agg(
        Opt_Mean=('Opt_Conversions', 'mean'),
        Opt_SE=('Opt_Conversions', 'sem'),
        Act_Mean=('Act_Conversions', 'mean'),
        Act_SE=('Act_Conversions', 'sem')
    ).reset_index()
    
    # Sort by Opt Mean Desc
    agg_df = agg_df.sort_values(by='Opt_Mean', ascending=False).head(top_n)
    
    # Format
    agg_df['Opt Conv'] = agg_df.apply(lambda r: f"{r['Opt_Mean']:.1f} $\\pm$ {r['Opt_SE']:.1f}", axis=1)
    agg_df['Act Conv'] = agg_df.apply(lambda r: f"{r['Act_Mean']:.1f} $\\pm$ {r['Act_SE']:.1f}", axis=1)
    
    out_df = agg_df[['Location', 'Region', 'Opt Conv', 'Act Conv']]
    
    col_format = 'llcc'
    latex = out_df.to_latex(index=False, column_format=col_format, escape=False)
    latex = latex.replace(r'\hline', r'\toprule', 1)
    if latex.strip().endswith(r'\hline'):
        latex = latex.strip()[:-6] + r'\bottomrule'
        
    final_latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Top %d Countries by Predicted Conversions (Budget %s)}\n"
        "\\resizebox{\\textwidth}{!}{%%\n"
        f"{latex}"
        "}\n"
        "\\end{table}"
    ) % (top_n, budget)
    
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
    origin_rows = []
    
    grouped = full_results.groupby('Budget', dropna=False)

    # Calculate Actuals (GLOBAL)
    act_df = full_results.drop_duplicates(subset=['Day']) if 'Day' in full_results.columns else full_results

    avg_clicks_act = act_df["t_Clicks_ActCost"].mean()
    se_clicks_act = act_df["t_Clicks_ActCost"].sem()
    avg_conv_act = act_df["Act_Conv"].mean()
    se_conv_act = act_df["Act_Conv"].sem()
    avg_cost_act = act_df["Act_Cost"].mean()
    se_cost_act = act_df["Act_Cost"].sem()

    avg_kws_act = act_df["N_Obs"].mean() if "N_Obs" in act_df.columns else 0
    se_kws_act = act_df["N_Obs"].sem() if "N_Obs" in act_df.columns else 0

    clicks_per_dollar_act = act_df["t_Clicks_ActCost"].sum() / act_df["Act_Cost"].sum() if act_df["Act_Cost"].sum() > 0 else 0
    
    # Actual Regional Share (Spend & Conv & Clicks)
    total_act_cost = act_df["Act_Cost"].sum()
    total_act_conv = act_df["Act_Conv"].sum() if "Act_Conv" in act_df.columns else 0
    # Clicks are sum of t_Clicks_ActCost (Lift)
    total_act_clicks = act_df["t_Clicks_ActCost"].sum()
    
    act_reg_row = {"Budget": "Actual"}
    for reg in ['USA', 'A', 'B']:
        # Support both old and new column naming for robustness
        col_cost = f"Act_Cost_Region_{reg}"
        if col_cost not in act_df.columns: col_cost = f"Act_Cost_{reg}"
        
        col_conv = f"Act_Conv_Region_{reg}" # Only new schema has this
        col_click = f"Act_Clicks_Region_{reg}"
        
        # Spend Share
        if col_cost in act_df.columns:
            act_reg_row[f"Spend {reg}"] = act_df[col_cost].sum() / total_act_cost if total_act_cost > 0 else 0
        else:
            act_reg_row[f"Spend {reg}"] = 0
            
        # Conv Share
        if col_conv in act_df.columns:
            act_reg_row[f"Conv {reg}"] = act_df[col_conv].sum() / total_act_conv if total_act_conv > 0 else 0
        else:
             act_reg_row[f"Conv {reg}"] = 0
             
        # Clicks Share
        if col_click in act_df.columns:
             act_reg_row[f"Clicks {reg}"] = act_df[col_click].sum() / total_act_clicks if total_act_clicks > 0 else 0
        else:
             act_reg_row[f"Clicks {reg}"] = 0

    regional_rows.append(act_reg_row)
    
    # Actual Origin Share
    act_orig_row = {"Budget": "Actual"}
    # Map origins to readable names
    origin_map = {'new': 'Share New', 'existing': 'Share Existing', 'existing searches': 'Share ExSearches'}
    for org_key, org_col in origin_map.items():
        # Cost
        col_cost = f"Act_Cost_Origin_{org_key}"
        if col_cost not in act_df.columns: col_cost = f"Act_Cost_{org_key}" # heuristic
        if col_cost not in act_df.columns and org_key == 'existing searches': col_cost = "Act_Cost_existing_searches"
        
        # Clicks & Conv (New)
        col_click = f"Act_Clicks_Origin_{org_key}"
        col_conv = f"Act_Conv_Origin_{org_key}"

        if col_cost in act_df.columns:
            act_orig_row[f"Spend {org_key.capitalize()}"] = act_df[col_cost].sum() / total_act_cost if total_act_cost > 0 else 0
        else:
             act_orig_row[f"Spend {org_key.capitalize()}"] = 0
             
        if col_click in act_df.columns:
             act_orig_row[f"Clicks {org_key.capitalize()}"] = act_df[col_click].sum() / total_act_clicks if total_act_clicks > 0 else 0
        else:
             act_orig_row[f"Clicks {org_key.capitalize()}"] = 0
             
        if col_conv in act_df.columns:
             act_orig_row[f"Conv {org_key.capitalize()}"] = act_df[col_conv].sum() / total_act_conv if total_act_conv > 0 else 0
        else:
             act_orig_row[f"Conv {org_key.capitalize()}"] = 0
            
    origin_rows.append(act_orig_row)

    for budget, df_group in grouped:
        
        # 1. Performance Metrics
        avg_clicks_opt = df_group["t_Clicks_OptCost"].mean()
        se_clicks_opt = df_group["t_Clicks_OptCost"].sem()
        avg_cost_opt = df_group["Opt_Cost"].mean()
        se_cost_opt = df_group["Opt_Cost"].sem()
        avg_conv_opt = df_group['Opt_Conv'].mean()
        se_conv_opt = df_group['Opt_Conv'].sem()

        avg_kws_opt = df_group["N_Opt"].mean()
        se_kws_opt = df_group["N_Opt"].sem()
        
        clicks_per_dollar_opt = df_group["t_Clicks_OptCost"].sum() / df_group["Opt_Cost"].sum() if df_group["Opt_Cost"].sum() > 0 else 0
        
        # Improvements vs Actual
        imp_clicks = (avg_clicks_opt - avg_clicks_act) / avg_clicks_act if avg_clicks_act > 0 else 0
        imp_c_d = (clicks_per_dollar_opt - clicks_per_dollar_act) / clicks_per_dollar_act if clicks_per_dollar_act > 0 else 0
        imp_conv = (avg_conv_opt - avg_conv_act) / avg_conv_act if avg_conv_act > 0 else 0

        summary_rows.append({
            "Budget": budget,
            "avg clicks (opt)": avg_clicks_opt,
            "se clicks (opt)": se_clicks_opt,
            "avg cost (opt)": avg_cost_opt,
            "se cost (opt)": se_cost_opt,
            "avg conv (opt)": avg_conv_opt,
            "se conv (opt)": se_conv_opt,
            "clicks/$ (opt)": clicks_per_dollar_opt,
            "avg kws (opt)": avg_kws_opt,
            "se kws (opt)": se_kws_opt,
            "improvement in clicks": imp_clicks,
            "improvement in clicks/$": imp_c_d,
            "improvement in conv": imp_conv,
            "avg clicks (act)": avg_clicks_act,
            "se clicks (act)": se_clicks_act,
            "avg cost (act)": avg_cost_act,
            "se cost (act)": se_cost_act,
            "avg conv (act)": avg_conv_act,
            "se conv (act)": se_conv_act,
            "clicks/$ (act)": clicks_per_dollar_act,
            "avg n kws (act)": avg_kws_act,
            "se n kws (act)": se_kws_act
        })
        
        # 2. Regional Share Metrics
        total_opt_cost_grp = df_group["Opt_Cost"].sum()
        total_opt_conv_grp = df_group["Opt_Conv"].sum() if "Opt_Conv" in df_group.columns else 0
        total_opt_clicks_grp = df_group["t_Clicks_OptCost"].sum()
        
        reg_row = {"Budget": budget}
        
        for reg in ['USA', 'A', 'B']:
            col_cost = f"Opt_Cost_Region_{reg}"
            if col_cost not in df_group.columns: col_cost = f"Opt_Cost_{reg}"
            
            col_conv = f"Opt_Conv_Region_{reg}"
            col_click = f"Opt_Clicks_Region_{reg}"
            
            # Spend
            if col_cost in df_group.columns:
                reg_row[f"Spend {reg}"] = df_group[col_cost].sum() / total_opt_cost_grp if total_opt_cost_grp > 0 else 0
            else:
                 reg_row[f"Spend {reg}"] = 0
                 
            # Conv
            if col_conv in df_group.columns:
                reg_row[f"Conv {reg}"] = df_group[col_conv].sum() / total_opt_conv_grp if total_opt_conv_grp > 0 else 0
            else:
                reg_row[f"Conv {reg}"] = 0
                
            # Clicks
            if col_click in df_group.columns:
                reg_row[f"Clicks {reg}"] = df_group[col_click].sum() / total_opt_clicks_grp if total_opt_clicks_grp > 0 else 0
            else:
                reg_row[f"Clicks {reg}"] = 0
                
        regional_rows.append(reg_row)
        
        # 3. Origin Share Metrics
        orig_row = {"Budget": budget}
        for org_key, org_col in origin_map.items():
            # Cost
            col_cost = f"Opt_Cost_Origin_{org_key}"
            if col_cost not in df_group.columns: col_cost = f"Opt_Cost_{org_key}"
            if col_cost not in df_group.columns and org_key == 'existing searches': col_cost = "Opt_Cost_existing_searches"
            
            # Clicks & Conv
            col_click = f"Opt_Clicks_Origin_{org_key}"
            col_conv = f"Opt_Conv_Origin_{org_key}"

            if col_cost in df_group.columns:
                orig_row[f"Spend {org_key.capitalize()}"] = df_group[col_cost].sum() / total_opt_cost_grp if total_opt_cost_grp > 0 else 0
            else:
                 orig_row[f"Spend {org_key.capitalize()}"] = 0
            
            if col_click in df_group.columns:
                 orig_row[f"Clicks {org_key.capitalize()}"] = df_group[col_click].sum() / total_opt_clicks_grp if total_opt_clicks_grp > 0 else 0
            else:
                 orig_row[f"Clicks {org_key.capitalize()}"] = 0
                 
            if col_conv in df_group.columns:
                 orig_row[f"Conv {org_key.capitalize()}"] = df_group[col_conv].sum() / total_opt_conv_grp if total_opt_conv_grp > 0 else 0
            else:
                 orig_row[f"Conv {org_key.capitalize()}"] = 0
        
        origin_rows.append(orig_row)

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
    out_reg_csv = base_results_dir / "regional_breakdown.csv"
    regional_df.to_csv(out_reg_csv, index=False)
    
    latex_reg = generate_regional_table(regional_df)
    out_reg_tex = base_results_dir / "regional_breakdown.tex"
    with open(out_reg_tex, "w") as f:
        f.write(latex_reg)
    print(f"\nRegional Table saved to {out_reg_tex}")
    print(latex_reg)

    # --- Output Origin Table ---
    origin_df = pd.DataFrame(origin_rows)
    out_orig_csv = base_results_dir / "origin_breakdown.csv"
    origin_df.to_csv(out_orig_csv, index=False)
    
    latex_orig = generate_origin_table(origin_df)
    out_orig_tex = base_results_dir / "origin_breakdown.tex"
    with open(out_orig_tex, "w") as f:
        f.write(latex_orig)
    print(f"\nOrigin Table saved to {out_orig_tex}")
    print(latex_orig)

    # --- Output Country Table (Best Budget) ---
    # Find best budget based on clicks improvement
    if 'improvement in clicks' in summary_df.columns:
         best_row = summary_df.loc[summary_df['improvement in clicks'].idxmax()]
         best_budget = best_row['Budget']
         
         latex_country = generate_country_table(args.exp_name, int(best_budget))
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
