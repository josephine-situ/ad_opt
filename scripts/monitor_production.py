import pandas as pd
import numpy as np
import argparse
import json
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
    
    # Load predicted optimized costs for the prediction date (prediction_lag)
    pred_date_str = (today - timedelta(days=prediction_lag)).strftime('%Y%m%d')
    preds_path = bids_dir / f"optimized_costs_{pred_date_str}.csv"
    if not preds_path.exists():
        print(f"Error: Predictions not found at {preds_path}")
        return

    preds_df = pd.read_csv(preds_path)
    # Load model metadata corresponding to the prediction date (prediction_lag)
    in_sample_r2 = float('nan')
    train_shap_cost = float('nan')
    train_pfi_cost = float('nan')
    pred_dt = (today - timedelta(days=prediction_lag))
    pred_date_str = pred_dt.strftime('%Y%m%d')
    model_meta_dir = base_dir / "models" / course
    chosen_meta = model_meta_dir / f"metadata_{pred_date_str}.json"
    if not chosen_meta.exists():
        print(f"Error: Metadata not found at {chosen_meta}")
        return
    try:
        with open(chosen_meta, 'r') as mf:
            meta = json.load(mf)
        in_sample_r2 = float(meta.get('in_sample_r2', float('nan')))
        train_shap_cost = float(meta.get('shap_cost', float('nan')))
        train_pfi_cost = float(meta.get('pfi_cost', float('nan')))
    except Exception as e:
        print(f"Warning: could not read metadata {chosen_meta}: {e}")
    
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

    # Compute out-of-sample Permutation Feature Importance (PFI) for 'Cost' on production data
    prod_pfi_cost = float('nan')
    try:
        import joblib
        from sklearn.inspection import permutation_importance

        # Find the model file corresponding to the prediction date under models/{course}/
        models_dir = Path(__file__).resolve().parent.parent / 'models' / course
        model = None
        chosen_model = models_dir / f"xgb_clicks_model_bert_{pred_date_str}.joblib"
        if not chosen_model.exists():
            chosen_model = models_dir / f"xgb_clicks_model_llm_{pred_date_str}.joblib"
        if not chosen_model.exists():
            raise FileNotFoundError(f"No model joblib found for prediction date {pred_date_str} under {models_dir}")
        model = joblib.load(chosen_model)

        # Identify feature columns to pass to pipeline
        feature_cols = None
        if hasattr(model, 'feature_names_in_'):
            feature_cols = list(model.feature_names_in_)
        elif hasattr(model, 'named_steps') and hasattr(model, 'predict'):
            # attempt to infer from merged columns: prefer numeric predictors
            possible = [c for c in merged.columns if c not in ['Keyword', 'Match type', 'Region', 'Origin', 'Optimal Cost', 'Gurobi Pred', 'Gurobi Pred over Base', 'Actual Clicks', 'Actual Cost', 'Residual Error', 'Abs Error']]
            feature_cols = [c for c in possible if merged[c].dtype.kind in 'fi']
        else:
            feature_cols = [c for c in merged.columns if merged[c].dtype.kind in 'fi']

        if feature_cols:
            X_prod = merged[feature_cols].fillna(0)
            y_prod = merged['Actual Clicks'].fillna(0)

            # Compute permutation importance using the pipeline directly
            pfi = permutation_importance(model, X_prod, y_prod, n_repeats=5, random_state=42, n_jobs=1)

            # Look for a 'Cost' feature in the importances
            cost_candidates = ['Cost', 'Actual Cost', 'Optimal Cost', 'cost']
            for cand in cost_candidates:
                if cand in feature_cols:
                    idx = feature_cols.index(cand)
                    prod_pfi_cost = float(pfi.importances_mean[idx])
                    break
            else:
                # if exact cost not found, try substring match
                for i, nm in enumerate(feature_cols):
                    if 'cost' in nm.lower():
                        prod_pfi_cost = float(pfi.importances_mean[i])
                        break
    except Exception as e:
        print(f"Warning: could not compute production PFI: {e}")
    
    segment_cols = [col for col in ['Region', 'Match type', 'Origin'] if col in high_error.columns]
    if segment_cols:
        segment_summary = high_error.groupby(segment_cols).agg({
            'Residual Error': 'mean',
            'Abs Error': 'mean',
            'Keyword': 'count'
        }).rename(columns={'Keyword': 'Keyword Count'})
    else:
        segment_summary = pd.DataFrame()
    # (removed) merging average bid metrics — segment-level bid analysis is handled below
        
    # Segment analysis by Top of page bid (high range)
    if 'Top of page bid (high range)' in high_error.columns:
        high_error['Top Page Bid Bucket'] = pd.qcut(high_error['Top of page bid (high range)'], q=5, duplicates='drop')
        top_bid_summary = high_error.groupby('Top Page Bid Bucket', observed=True).agg({
            'Residual Error': 'mean',
            'Abs Error': 'mean',
            'Keyword': 'count',
            'Top of page bid (high range)': 'mean'
        }).rename(columns={'Keyword': 'Keyword Count', 'Top of page bid (high range)': 'Avg Top Bid (High)'})
    else:
        top_bid_summary = pd.DataFrame()

    # Segment analysis by First page CPC
    if 'first_page_bid' in high_error.columns:
        # Convert back numeric if it isn't, and impute so qcut doesn't fail on all NaNs
        numeric_bid = pd.to_numeric(high_error['first_page_bid'], errors='coerce')
        if numeric_bid.notna().sum() > 0:
            high_error['First Page CPC Bucket'] = pd.qcut(numeric_bid, q=5, duplicates='drop')
            first_page_summary = high_error.groupby('First Page CPC Bucket', observed=True).agg({
                'Residual Error': 'mean',
                'Abs Error': 'mean',
                'Keyword': 'count',
            }).rename(columns={'Keyword': 'Keyword Count'})
            first_page_summary['Avg First Page CPC'] = high_error.groupby('First Page CPC Bucket', observed=True)['first_page_bid'].mean()
        else:
            first_page_summary = pd.DataFrame()
    else:
        first_page_summary = pd.DataFrame()
        
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
        f.write(f"In-sample R²: {in_sample_r2:.4f}\n")
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
            
        f.write("\n\n--- Error by Top of page bid (high range) ---\n")
        if not top_bid_summary.empty:
            f.write(top_bid_summary.to_string())
        else:
            f.write("No Top of page bid data available.\n")
            
        f.write("\n\n--- Error by First page CPC ---\n")
        if not first_page_summary.empty:
            f.write(first_page_summary.to_string())
        else:
            f.write("No First page CPC data available.\n")
            
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
            
        f.write("\n\nTODO: Ensure optimization.py persists dated 'optimized_costs_YYYYMMDD.csv' files consistently.\n")
        f.write("TODO: Ensure bid adjustments are timestamped/persisted for accurate historical comparisons.\n")
        
    print(f"Monitoring report written to {report_path}")
    
    csv_path = analysis_dir / f"production_merged_{actuals_date}.csv"
    # Keep only ENABLED keywords in production merged output
    prod_df = merged.copy()
    enabled_mask = None
    if 'Status' in prod_df.columns:
        enabled_mask = prod_df['Status'].astype(str).str.upper() == 'ENABLED'
    elif 'Bid' in prod_df.columns:
        enabled_mask = prod_df['Bid'].astype(float) > 0
    elif 'Optimal Cost' in prod_df.columns:
        enabled_mask = prod_df['Optimal Cost'].astype(float) > 5e-4
    else:
        # If no enabled indicator, default to keeping rows with predicted clicks > 0
        if 'Gurobi Pred' in prod_df.columns:
            enabled_mask = prod_df['Gurobi Pred'] > 0
        else:
            enabled_mask = pd.Series([True] * len(prod_df), index=prod_df.index)

    prod_df = prod_df[enabled_mask].copy()

    # Attach modeling metrics to each row for downstream analysis
    prod_df['In_Sample_R2'] = in_sample_r2
    prod_df['Train_SHAP'] = train_shap_cost
    prod_df['Train_PFI'] = train_pfi_cost
    prod_df['Prod_PFI'] = prod_pfi_cost

    prod_df.to_csv(csv_path, index=False)
    print(f"Production merged (ENABLED rows) saved to {csv_path} ({len(prod_df)} rows)")

    # Append to daily history (includes train and production PFI/SHAP)
    history_file = analysis_dir / "daily_metrics_history.csv"
    new_record = pd.DataFrame([{
        'Date': actuals_date,
        'MSE': mse, # For yesterday's actual vs predictions
        'R2': r2,
        'In_Sample_R2': in_sample_r2, # From latest model metadata
        'Train_SHAP': train_shap_cost,
        'Train_PFI': train_pfi_cost,
        'Prod_PFI': prod_pfi_cost,
        'Bias': bias,
        'Pred_Clicks': total_pred_clicks,
        'Actual_Clicks': total_actual_clicks,
        'Pred_Cost': total_pred_cost,
        'Actual_Cost': total_actual_cost,
        'Pred_CPC': pred_cpc,
        'Actual_CPC': actual_cpc
    }])

    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, new_record], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')
        history.sort_values('Date', inplace=True)
        # calculate rolling averages
        history['Rolling_MSE_7d'] = history['MSE'].rolling(7, min_periods=1).mean()
        history['Rolling_Bias_7d'] = history['Bias'].rolling(7, min_periods=1).mean()
        if 'In_Sample_R2' in history.columns:
            history['Rolling_In_Sample_R2_7d'] = history['In_Sample_R2'].rolling(7, min_periods=1).mean()
        if 'Train_SHAP' in history.columns:
            history['Rolling_Train_SHAP_7d'] = history['Train_SHAP'].rolling(7, min_periods=1).mean()
        if 'Train_PFI' in history.columns:
            history['Rolling_Train_PFI_7d'] = history['Train_PFI'].rolling(7, min_periods=1).mean()
        if 'Prod_PFI' in history.columns:
            history['Rolling_Prod_PFI_7d'] = history['Prod_PFI'].rolling(7, min_periods=1).mean()
    else:
        history = new_record
        history['Rolling_MSE_7d'] = history['MSE']
        history['Rolling_Bias_7d'] = history['Bias']
        history['Rolling_In_Sample_R2_7d'] = history['In_Sample_R2']
        history['Rolling_Train_SHAP_7d'] = history['Train_SHAP']
        history['Rolling_Train_PFI_7d'] = history['Train_PFI']
        history['Rolling_Prod_PFI_7d'] = history['Prod_PFI']

    history.to_csv(history_file, index=False)
    print(f"Appended metrics to {history_file}")

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
