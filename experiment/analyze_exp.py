import pandas as pd
import numpy as np

# 1. Load the datasets
budget_df = pd.read_csv('experiment/Experiment - Budget comparison.csv', skiprows=2)
cpc_df = pd.read_csv('experiment/Experiment - CPC comparison.csv', skiprows=2)

# 2. Extract Region from the Campaign Name 
budget_df['Region'] = budget_df['Campaign'].apply(lambda x: str(x).split(' - ')[2])
cpc_df['Region'] = cpc_df['Campaign'].apply(lambda x: str(x).split(' - ')[2])

# ==========================================
# PART 1: BUDGET vs SPEND ANALYSIS
# ==========================================
budget_summary = budget_df.groupby(['Region', 'Search keyword match type']).agg(
    Total_Budget=('Budget', 'sum'),
    Total_Spend=('Cost', 'sum')
).reset_index()

budget_summary['Spend_vs_Budget'] = budget_summary['Total_Spend'] - budget_summary['Total_Budget']
budget_summary['Spend_Status'] = np.where(budget_summary['Spend_vs_Budget'] > 0, 'Over Budget', 'Under Budget')

# ==========================================
# PART 2: ACTUAL CPC vs MAX CPC ANALYSIS
# ==========================================
cpc_summary = cpc_df.groupby(['Region', 'Search keyword match type']).agg(
    Avg_Max_CPC=('Keyword max CPC', 'mean'),
    Total_Cost=('Cost', 'sum'),
    Total_Clicks=('Clicks', 'sum')
).reset_index()

# True Actual Avg CPC = Total Cost / Total Clicks
cpc_summary['Actual_Avg_CPC'] = cpc_summary['Total_Cost'] / cpc_summary['Total_Clicks']
cpc_summary['CPC_vs_Max'] = cpc_summary['Actual_Avg_CPC'] - cpc_summary['Avg_Max_CPC']
cpc_summary['CPC_Status'] = np.where(cpc_summary['CPC_vs_Max'] > 0, 'Over Max CPC', 'Under Max CPC')

cpc_summary = cpc_summary.drop(columns=['Total_Cost', 'Total_Clicks'])

# ==========================================
# PART 3: ZERO CPC ANALYSIS (% of Actual CPC = 0 when Max CPC > 0)
# ==========================================
# Filter for rows where Max CPC is > 0
df_filtered = cpc_df[cpc_df['Keyword max CPC'] > 0].copy()

# Create bins (quartiles) for Keyword max CPC to group them into 4 quantiles
df_filtered['Max_CPC_Bin'] = pd.qcut(df_filtered['Keyword max CPC'], q=4, duplicates='drop')

# Flag rows where Actual CPC is 0 (either Avg CPC is 0 or Cost is 0)
df_filtered['Is_Actual_CPC_Zero'] = (df_filtered['Avg. CPC'] == 0) | (df_filtered['Cost'] == 0)

# 3A. Zero CPC grouped by Max CPC Quantile Bins
bin_summary = df_filtered.groupby('Max_CPC_Bin').agg(
    Total_Keywords_Logged=('Search keyword', 'count'),
    Zero_CPC_Count=('Is_Actual_CPC_Zero', 'sum')
).reset_index()
bin_summary['%_Zero_CPC'] = (bin_summary['Zero_CPC_Count'] / bin_summary['Total_Keywords_Logged']) * 100

# 3B. Zero CPC grouped by Region and Match Type
region_zero_cpc_summary = df_filtered.groupby(['Region', 'Search keyword match type']).agg(
    Total_Keywords_Logged=('Search keyword', 'count'),
    Zero_CPC_Count=('Is_Actual_CPC_Zero', 'sum')
).reset_index()
region_zero_cpc_summary['%_Zero_CPC'] = (region_zero_cpc_summary['Zero_CPC_Count'] / region_zero_cpc_summary['Total_Keywords_Logged']) * 100

# ==========================================
# PART 4: MERGE & EXPORT RESULTS
# ==========================================
# Save the standalone summary files
budget_summary.to_csv('experiment/budget_analysis_summary.csv', index=False)
bin_summary.to_csv('experiment/zero_cpc_by_bins.csv', index=False)

# Merge the standard CPC metrics with the Region Zero-CPC metrics for a master view
combined_cpc_view = pd.merge(
    cpc_summary, 
    region_zero_cpc_summary, 
    on=['Region', 'Search keyword match type'], 
    how='left'
)
combined_cpc_view.to_csv('experiment/combined_cpc_insights.csv', index=False)

print("--- Budget Summary ---")
print(budget_summary)
print("\n--- Combined CPC Insights (Max CPC vs Actual vs % Zero) ---")
print(combined_cpc_view)
print("\n--- Zero CPC Summary by Quantile Bins ---")
print(bin_summary)