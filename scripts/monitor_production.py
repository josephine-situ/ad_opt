import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

from utils.ads_reporting import generate_hod_clicks_and_conversion_report
from utils.google_ads_api import GoogleAdsClient
from scripts.bid_post_processing import calculate_bid_adjustments, group_hours, load_bid_adj_report

from config import COURSE_CONFIG

def monitor_production(course="sys_think", actuals_lag=1, prediction_lag=2, base_date=None, 
                       google_ads_yaml=None, customer_id=None):
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "data" / course / "reports"
    bids_dir = base_dir / "opt_results" / course / "bids"
    bid_adj_dir = base_dir / "opt_results" / course / "bid_adjustments"
    analysis_dir = base_dir / "opt_results" / "analysis" / course
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Date Alignment
    if base_date:
        today = pd.to_datetime(base_date)
    else:
        today = datetime.now()
        
    actuals_date = (today - timedelta(days=actuals_lag)).strftime('%Y-%m-%d')
    start_date_7d = (today - timedelta(days=actuals_lag + 7)).strftime('%Y-%m-%d')
    
    print(f"Running Monitoring for {course}")
    print(f"Eval Date (Actuals): {actuals_date} (Lag: {actuals_lag} days)")
    print(f"Prediction generated: {(today - timedelta(days=prediction_lag)).strftime('%Y-%m-%d')} (Lag: {prediction_lag} days)")

    # 2. Data Loading
    raw_path = reports_dir / "Search keyword - raw input to models.csv"
    if not raw_path.exists():
        print(f"Error: Raw data not found at {raw_path}")
        return
        
    raw_df = pd.read_csv(raw_path)
    raw_df['Day'] = pd.to_datetime(raw_df['Day'])
    
    # TODO: Add prediction persistence in optimization.py so we can load e.g., optimized_costs_{date}.csv
    preds_path = bids_dir / "optimized_costs.csv"
    if not preds_path.exists():
        print(f"Error: Predictions not found at {preds_path}")
        return
        
    preds_df = pd.read_csv(preds_path)
    
    # 3. Filter Actuals
    actuals_df = raw_df[raw_df['Day'] == pd.to_datetime(actuals_date)].copy()
    if 'Campaign' in actuals_df.columns:
        exp_label = COURSE_CONFIG[course].get('exp_label', 'Experiment')
        actuals_df = actuals_df[actuals_df['Campaign'].str.contains(exp_label, case=False, na=False)]
    
    match_mapping = {
        'Broad': 'Broad match',
        'Phrase': 'Phrase match',
        'Exact': 'Exact match'
    }
    actuals_df['Match type'] = actuals_df['Search keyword match type'].map(match_mapping).fillna(actuals_df['Search keyword match type'])
    actuals_df['Keyword'] = actuals_df['Search keyword']
    
    actuals_grouped = actuals_df.groupby(['Keyword', 'Match type']).agg({
        'Clicks': 'sum',
        'Cost': 'sum',
    }).reset_index()
    actuals_grouped.rename(columns={'Clicks': 'Actual Clicks', 'Cost': 'Actual Cost'}, inplace=True)
    
    merged = pd.merge(preds_df, actuals_grouped, on=['Keyword', 'Match type'], how='left')
    merged['Actual Clicks'] = merged['Actual Clicks'].fillna(0)
    merged['Actual Cost'] = merged['Actual Cost'].fillna(0)
    
    # 4. Core Metrics & Error Attribution
    pred_col = 'Gurobi Pred over Base'
    mask_eval = merged[pred_col].notna()
    if mask_eval.sum() > 0:
        y_true = merged.loc[mask_eval, 'Actual Clicks']
        y_pred = merged.loc[mask_eval, pred_col]
        
        mse = mean_squared_error(y_true, y_pred)
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan
        bias = np.mean(y_true - y_pred)
    else:
        mse = r2 = bias = np.nan
        
    merged['Residual Error'] = merged['Actual Clicks'] - merged[pred_col]
    
    total_pred_clicks = merged[pred_col].sum()
    total_actual_clicks = merged['Actual Clicks'].sum()
    total_pred_cost = merged['Optimal Cost'].sum()
    total_actual_cost = merged['Actual Cost'].sum()
    
    pred_cpc = total_pred_cost / total_pred_clicks if total_pred_clicks > 0 else 0
    actual_cpc = total_actual_cost / total_actual_clicks if total_actual_clicks > 0 else 0
    
    high_error = merged.copy()
    high_error['Abs Error'] = high_error['Residual Error'].abs()
    
    segment_cols = [col for col in ['Region', 'Match type', 'Origin'] if col in high_error.columns]
    if segment_cols:
        segment_summary = high_error.groupby(segment_cols).agg({
            'Residual Error': 'mean',
            'Abs Error': 'mean',
            'Keyword': 'count'
        }).rename(columns={'Keyword': 'Keyword Count'})
    else:
        segment_summary = pd.DataFrame()
        
    # 5. Bid Adjustment Effectiveness
    ideal_adj = None
    if google_ads_yaml and customer_id:
        print(f"Fetching HOD data from {start_date_7d} to {actuals_date}...")
        try:
            client = GoogleAdsClient.load_from_storage(google_ads_yaml)
            generate_hod_clicks_and_conversion_report(
                client, customer_id, course, start_date_7d, actuals_date, output_suffix="_7d"
            )
            report_dir = base_dir / "data" / course / "reports" / "bid_adj"
            hod_clicks = report_dir / 'hod_clicks_7d.csv'
            hod_conv = report_dir / 'hod_conv_7d.csv'
            
            if hod_clicks.exists() and hod_conv.exists():
                df, _ = load_bid_adj_report(hod_clicks, hod_conv, 'Hour of the day')
                df['Hour Group'] = df['Hour of the day'].apply(group_hours)
                df = df.groupby(['Campaign', 'Hour Group']).agg({
                    'Clicks': 'sum',
                    'All conv.': 'sum'
                }).reset_index()
                ideal_adj = calculate_bid_adjustments(df, 'Hour Group', 'hour', min_clicks=0) # use 0 to see all segments
        except Exception as e:
            print(f"Error fetching/calculating ideal bid adjustments: {e}")
            
    # Load actual adjustments
    # TODO: Prediction persistence should save out timestamped bid adjustments too.
    # For now, reading the latest output as the example actuals.
    actual_adj_path = bid_adj_dir / "bid_adj_hour_of_day.csv"
    actual_adj = None
    if actual_adj_path.exists():
        actual_adj = pd.read_csv(actual_adj_path)

    # Output to text file
    report_path = analysis_dir / f"production_report_{actuals_date}.txt"
    with open(report_path, "w") as f:
        f.write(f"Production Monitoring Report - {course}\n")
        f.write("="*50 + "\n")
        f.write(f"Target Date (Actuals): {actuals_date}\n")
        f.write(f"Predictions from: {(today - timedelta(days=prediction_lag)).strftime('%Y-%m-%d')}\n")
        f.write("\n--- Core Accuracy & Bias Metrics ---\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"R²:   {r2:.4f}\n")
        f.write(f"Bias: {bias:.4f} (Positive = Actuals higher than predicted, Negative = Actuals lower than predicted)\n")
        f.write("\n--- Aggregate Cost & CPC ---\n")
        f.write(f"Predicted Total Clicks: {total_pred_clicks:.2f} | Actual: {total_actual_clicks:.2f}\n")
        f.write(f"Predicted Total Cost:   ${total_pred_cost:.2f} | Actual: ${total_actual_cost:.2f}\n")
        f.write(f"Predicted CPC:          ${pred_cpc:.2f} | Actual: ${actual_cpc:.2f}\n")
        
        f.write("\n--- Segments with High Average Errors (Actual Clicks - Pred Clicks) ---\n")
        if not segment_summary.empty:
            top_segments = segment_summary.sort_values('Abs Error', ascending=False).head(10)
            f.write(top_segments.to_string())
        else:
            f.write("No segment data available.\n")
            
        f.write("\n\n--- Bid Adjustment Effectiveness ---\n")
        if ideal_adj is not None and not ideal_adj.empty:
            f.write("Ideal Bid Adjustments (rolling 7-day):\n")
            f.write(ideal_adj[['Region', 'Hour Group', 'BidAdjustment']].head(10).to_string(index=False) + "\n\n")
        elif google_ads_yaml:
            f.write("Ideal Bid Adjustments could not be calculated (no data/error).\n\n")
        else:
            f.write("Ideal Bid Adjustments not calculated (--google_ads_yaml not provided).\n\n")
            
        if actual_adj is not None and not actual_adj.empty:
            f.write("Actual Applied Bid Adjustments:\n")
            f.write(actual_adj[['Region', 'Hour Group', 'BidAdjustment']].head(10).to_string(index=False) + "\n")
        else:
            f.write("No actual applied bid adjustments found.\n")
            
        f.write("\n\nTODO: Ensure optimization.py persists 'optimized_costs.csv' dynamically.\n")
        f.write("TODO: Ensure bid adjustments are timestamped/persisted for accurate historical comparisons.\n")
        
    print(f"Monitoring report written to {report_path}")
    
    csv_path = analysis_dir / f"production_merged_{actuals_date}.csv"
    merged.to_csv(csv_path, index=False)
    print(f"Merged dataset saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", type=str, default="sys_think")
    parser.add_argument("--actuals_lag", type=int, default=1, help="Days ago for actuals")
    parser.add_argument("--prediction_lag", type=int, default=2, help="Days ago when predictions were made")
    parser.add_argument("--base_date", type=str, default=None, help="Base date for the run (YYYY-MM-DD), mostly for testing")
    parser.add_argument("--google_ads_yaml", type=str, default=None, help="Path to google-ads.yaml for fetching HOD data")
    parser.add_argument("--customer_id", type=str, default=None, help="Google Ads customer ID")
    args = parser.parse_args()
    
    monitor_production(args.course, args.actuals_lag, args.prediction_lag, args.base_date, 
                       args.google_ads_yaml, args.customer_id)
