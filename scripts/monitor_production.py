import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, r2_score

from scripts.bid_post_processing import process_bid_adjustments

from config import COURSE_CONFIG
from utils.data_pipeline import get_model_feature_columns

# Display metadata for bid-adjustment effectiveness report sections.
# Keys match those returned by process_bid_adjustments().
SEGMENT_REPORT_CONFIG = {
    'hour_of_day': ('Hour of Day', 'Hour Group', 'bid_adj_hour_of_day.csv'),
    'device':      ('Device',      'Device',     'bid_adj_device.csv'),
    'location':    ('Location',    'Targeted location', 'bid_adj_location.csv'),
    'age':         ('Age',         'Age',        'bid_adj_age.csv'),
}


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return None


def _write_adjustment_section(f, title: str, ideal_df: pd.DataFrame | None, actual_df: pd.DataFrame | None, segment_col: str):
    f.write(f"\n\n--- {title} Bid Adjustment Effectiveness ---\n")

    if ideal_df is not None and not ideal_df.empty:
        ideal_view = ideal_df.copy()
        if 'BidAdjustment' in ideal_view.columns:
            ideal_view = ideal_view[ideal_view['BidAdjustment'].notna()].copy()
        if not ideal_view.empty:
            if 'BidAdjustment' in ideal_view.columns:
                ideal_view = ideal_view.reindex(ideal_view['BidAdjustment'].abs().sort_values(ascending=False).index)
            cols = [c for c in ['Region', segment_col, 'Clicks', 'All conv.', 'BidAdjustment'] if c in ideal_view.columns]
            f.write('Ideal Bid Adjustments (top 10 by absolute adjustment):\n')
            f.write(ideal_view[cols].head(10).to_string(index=False) + '\n\n')
        else:
            f.write('Ideal Bid Adjustments: no rows with calculable adjustments.\n\n')
    else:
        f.write('Ideal Bid Adjustments not available.\n\n')

    if actual_df is not None and not actual_df.empty:
        actual_view = actual_df.copy()
        if 'BidAdjustment' in actual_view.columns:
            actual_view = actual_view[actual_view['BidAdjustment'].notna()].copy()
        if not actual_view.empty:
            if 'BidAdjustment' in actual_view.columns:
                actual_view = actual_view.reindex(actual_view['BidAdjustment'].abs().sort_values(ascending=False).index)
            cols = [c for c in ['Region', segment_col, 'Clicks', 'All conv.', 'BidAdjustment'] if c in actual_view.columns]
            f.write('Actual Applied Bid Adjustments (top 10 by absolute adjustment):\n')
            f.write(actual_view[cols].head(10).to_string(index=False) + '\n')
        else:
            f.write('Actual Applied Bid Adjustments: no rows with calculable adjustments.\n')
    else:
        f.write('No actual applied bid adjustments found.\n')

    if ideal_df is not None and actual_df is not None and not ideal_df.empty and not actual_df.empty:
        if 'Region' in ideal_df.columns and segment_col in ideal_df.columns and 'Region' in actual_df.columns and segment_col in actual_df.columns:
            compare = ideal_df[['Region', segment_col, 'BidAdjustment']].merge(
                actual_df[['Region', segment_col, 'BidAdjustment']],
                on=['Region', segment_col],
                how='inner',
                suffixes=('_ideal', '_actual'),
            )
            if not compare.empty:
                compare['Adjustment Diff'] = compare['BidAdjustment_actual'] - compare['BidAdjustment_ideal']
                compare['Abs Diff'] = compare['Adjustment Diff'].abs()
                sign_match = (
                    np.sign(compare['BidAdjustment_actual'].fillna(0))
                    == np.sign(compare['BidAdjustment_ideal'].fillna(0))
                ).mean()
                f.write('\nComparison summary:\n')
                f.write(
                    f"Matched rows: {len(compare):,} | Mean abs diff: {compare['Abs Diff'].mean():.4f} | "
                    f"Mean signed diff: {compare['Adjustment Diff'].mean():.4f} | Sign match: {sign_match:.1%}\n"
                )
                f.write('Largest adjustment gaps (top 10):\n')
                f.write(
                    compare.sort_values('Abs Diff', ascending=False)[
                        ['Region', segment_col, 'BidAdjustment_ideal', 'BidAdjustment_actual', 'Adjustment Diff']
                    ].head(10).to_string(index=False) + '\n'
                )
            else:
                f.write('\nComparison summary: no overlapping rows between ideal and actual adjustments.\n')
        else:
            f.write('\nComparison summary unavailable because the segment key columns were missing.\n')

def monitor_production(course="sys_think", lag=1, base_date=None):
    base_dir = Path(__file__).resolve().parent.parent
    reports_dir = base_dir / "data" / course / "reports"
    bids_dir = base_dir / "opt_results" / course / "bids"
    bid_adj_dir = base_dir / "opt_results" / course / "bid_adjustments"
    analysis_dir = base_dir / "opt_results" / "analysis" / course
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[Step 5] Production Monitoring — {course}")
    print(f"{'='*70}")
    
    # 1. Date Alignment
    if base_date:
        today = pd.to_datetime(base_date)
    else:
        today = datetime.now()
        
    target_dt = today - timedelta(days=lag)
    target_date = target_dt.strftime('%Y-%m-%d')
    target_date_str = target_dt.strftime('%Y%m%d')
    
    print(f"Running Monitoring for {course}")
    print(f"Eval Date: {target_date} (Lag: {lag} days)")

    # 2. Data Loading
    # Load the full training dataset and filter to the target date for actuals.
    # bert_* columns are dropped because the model uses full raw BERT embeddings
    # (saved in optimized_costs) rather than the SVD-reduced ones stored here.
    base_data_path = base_dir / "data" / course / "clean" / "ad_opt_data_bert.csv"
    if not base_data_path.exists():
        print(f"Error: base data file not found at {base_data_path}")
        return
    all_data_df = pd.read_csv(base_data_path)
    all_data_df['Day'] = pd.to_datetime(all_data_df['Day'])
    bert_cols = [c for c in all_data_df.columns if c.startswith('bert_')]
    all_data_df = all_data_df.drop(columns=bert_cols)
    actuals_df = all_data_df[
        all_data_df['Day'].dt.normalize() == pd.to_datetime(target_date)
    ].copy()
    if actuals_df.empty:
        print(f"Warning: no actuals found in {base_data_path} for {target_date}. "
              "Metrics requiring actuals will be NaN.")
    
    # Load predicted optimized costs for the target date.
    preds_path = bids_dir / f"optimized_costs_{target_date_str}.csv"
    if not preds_path.exists():
        print(f"Error: Predictions not found at {preds_path}")
        return

    preds_df = pd.read_csv(preds_path, low_memory=False)
    # Load model metadata corresponding to the target date.
    in_sample_r2 = float('nan')
    train_shap_cost = float('nan')
    train_pfi_cost = float('nan')
    model_meta_dir = base_dir / "models" / course
    chosen_meta = model_meta_dir / f"metadata_{target_date_str}.json"
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
    
    # 3. Merge actuals into the production predictions.
    # Normalise match-type values so both sides use the long form
    # (e.g. "Broad match") that the optimiser outputs.
    _mt_map = {
        'broad': 'Broad match', 'exact': 'Exact match', 'phrase': 'Phrase match',
        'broad match': 'Broad match', 'exact match': 'Exact match', 'phrase match': 'Phrase match',
    }
    for df in (preds_df, actuals_df):
        if 'Match type' in df.columns:
            df['Match type'] = df['Match type'].str.strip().str.lower().map(_mt_map).fillna(df['Match type'])

    # Pull only the columns we need from actuals to avoid column conflicts with
    # the feature columns now stored in preds_df (e.g. 'Cost' is a model feature).
    actuals_keep = ['Keyword', 'Region', 'Match type', 'Clicks', 'Cost']
    if 'first_page_bid' in actuals_df.columns:
        actuals_keep.append('first_page_bid')
    actuals_slim = (
        actuals_df[[c for c in actuals_keep if c in actuals_df.columns]]
        .rename(columns={'Clicks': 'Actual Clicks', 'Cost': 'Actual Cost'})
    )

    merged = pd.merge(preds_df, actuals_slim, on=['Keyword', 'Region', 'Match type'], how='left')

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
    
    # Filter to active/enabled keywords for segment analysis
    active_mask = pd.Series([True] * len(merged), index=merged.index)
    if 'Status' in merged.columns:
        active_mask &= merged['Status'].astype(str).str.upper() == 'ENABLED'
    if 'Bid' in merged.columns:
        active_mask &= pd.to_numeric(merged['Bid'], errors='coerce').fillna(0) > 0

    high_error = merged.loc[active_mask & mask_eval].copy()
    high_error['Abs Error'] = high_error['Residual Error'].abs()

    # Compute out-of-sample Permutation Feature Importance (PFI) for 'Cost' on production data
    prod_pfi_cost = float('nan')
    try:
        import joblib
        from sklearn.inspection import permutation_importance

        # Find the model file corresponding to the prediction date under models/{course}/
        models_dir = Path(__file__).resolve().parent.parent / 'models' / course
        model = None
        chosen_model = models_dir / f"xgb_clicks_model_bert_{target_date_str}.joblib"
        if not chosen_model.exists():
            chosen_model = models_dir / f"xgb_clicks_model_llm_{target_date_str}.joblib"
        if not chosen_model.exists():
            raise FileNotFoundError(f"No model joblib found for prediction date {target_date_str} under {models_dir}")
        model = joblib.load(chosen_model)

        # Feature columns are now stored directly in preds_df (and thus merged)
        # by extract_solution, so PFI can be computed without a separate actuals
        # file containing embeddings.
        if hasattr(model, 'feature_names_in_'):
            feature_cols = [c for c in model.feature_names_in_ if c in merged.columns]
        else:
            feature_cols = [c for c in get_model_feature_columns(merged) if c in merged.columns]

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
            'Keyword': 'nunique'
        }).rename(columns={'Keyword': 'Keyword Count'})
    else:
        segment_summary = pd.DataFrame()

    # Segment analysis by model uncertainty (Feature Space Distance)
    if 'Feature Space Distance' in high_error.columns and high_error['Feature Space Distance'].notna().sum() > 0:
        high_error['Feature Dist Bucket'] = pd.qcut(
            high_error['Feature Space Distance'].dropna(), q=5, duplicates='drop'
        ).reindex(high_error.index)
        feature_dist_summary = high_error.dropna(subset=['Feature Dist Bucket']).groupby('Feature Dist Bucket', observed=True).agg({
            'Residual Error': 'mean',
            'Abs Error': 'mean',
            'Keyword': 'nunique',
            'Feature Space Distance': 'mean',
        }).rename(columns={'Keyword': 'Keyword Count', 'Feature Space Distance': 'Avg Feature Dist'})
    else:
        feature_dist_summary = pd.DataFrame()

    # Segment analysis by model uncertainty (Leaf Uncertainty)
    if 'Leaf Uncertainty' in high_error.columns and high_error['Leaf Uncertainty'].notna().sum() > 0:
        high_error['Leaf Uncert Bucket'] = pd.qcut(
            high_error['Leaf Uncertainty'].dropna(), q=5, duplicates='drop'
        ).reindex(high_error.index)
        leaf_uncert_summary = high_error.dropna(subset=['Leaf Uncert Bucket']).groupby('Leaf Uncert Bucket', observed=True).agg({
            'Residual Error': 'mean',
            'Abs Error': 'mean',
            'Keyword': 'nunique',
            'Leaf Uncertainty': 'mean',
        }).rename(columns={'Keyword': 'Keyword Count', 'Leaf Uncertainty': 'Avg Leaf Uncertainty'})
    else:
        leaf_uncert_summary = pd.DataFrame()

    # Segment analysis by Top of page bid (high range)
    if 'Top of page bid (high range)' in high_error.columns:
        high_error['Top Page Bid Bucket'] = pd.qcut(high_error['Top of page bid (high range)'], q=5, duplicates='drop')
        top_bid_summary = high_error.groupby('Top Page Bid Bucket', observed=True).agg({
            'Residual Error': 'mean',
            'Abs Error': 'mean',
            'Keyword': 'nunique',
            'Top of page bid (high range)': 'mean'
        }).rename(columns={'Keyword': 'Keyword Count', 'Top of page bid (high range)': 'Avg Top Bid (High)'})
    else:
        top_bid_summary = pd.DataFrame()

    # Segment analysis by First page CPC
    first_page_col = None
    if 'first_page_bid' in high_error.columns:
        first_page_col = 'first_page_bid'
    elif 'First page CPC' in high_error.columns:
        first_page_col = 'First page CPC'
    if first_page_col:
        numeric_bid = pd.to_numeric(high_error[first_page_col], errors='coerce')
        if numeric_bid.notna().sum() > 0:
            high_error['First Page CPC Bucket'] = pd.qcut(numeric_bid, q=5, duplicates='drop')
            first_page_summary = high_error.groupby('First Page CPC Bucket', observed=True).agg({
                'Residual Error': 'mean',
                'Abs Error': 'mean',
                'Keyword': 'nunique',
            }).rename(columns={'Keyword': 'Keyword Count'})
            first_page_summary['Avg First Page CPC'] = high_error.groupby('First Page CPC Bucket', observed=True)[first_page_col].mean()
        else:
            first_page_summary = pd.DataFrame()
    else:
        first_page_summary = pd.DataFrame()
        
    # 5. Bid Adjustment Effectiveness (ideal from *_7d.csv; pull via pull_input_data ads_reports)
    data_dir = base_dir / "data" / course
    try:
        ideal_adj = process_bid_adjustments(data_dir, min_clicks=0, file_suffix="_7d")
        if not ideal_adj:
            print("Warning: no 7d bid adjustment data found; run pull_input_data.py with ads_reports.")
    except FileNotFoundError:
        print("Warning: bid_adj directory not found; run pull_input_data.py with ads_reports.")
        ideal_adj = {}

    actual_adj = {}
    for segment_type, (_display, _col, actual_file) in SEGMENT_REPORT_CONFIG.items():
        actual_adj[segment_type] = _load_csv_if_exists(bid_adj_dir / actual_file)

    # Output to text file
    report_path = analysis_dir / f"production_report_{target_date}.txt"
    with open(report_path, "w") as f:
        f.write(f"Production Monitoring Report - {course}\n")
        f.write("="*50 + "\n")
        f.write(f"Target Date: {target_date}\n")
        f.write(f"In-sample R2: {in_sample_r2:.4f}\n")
        f.write("\n--- Core Accuracy & Bias Metrics ---\n")
        f.write(f"MSE:  {mse:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
        f.write(f"Bias: {bias:.4f} (Positive = Actuals higher than predicted, Negative = Actuals lower than predicted)\n")
        f.write("\n--- Aggregate Cost & CPC ---\n")
        f.write(f"Predicted Total Clicks: {total_pred_clicks:.2f} | Actual: {total_actual_clicks:.2f}\n")
        f.write(f"Predicted Total Cost:   ${total_pred_cost:.2f} | Actual: ${total_actual_cost:.2f}\n")
        f.write(f"Predicted CPC:          ${pred_cpc:.2f} | Actual: ${actual_cpc:.2f}\n")
        
        f.write("\n--- Segments sorted by average errors (Actual Clicks - Pred Clicks) ---\n")
        if not segment_summary.empty:
            top_segments = segment_summary.sort_values('Abs Error', ascending=False)
            f.write(top_segments.to_string())
        else:
            f.write("No segment data available.\n")
            
        f.write("\n\n--- Error by Feature Space Distance (model uncertainty) ---\n")
        if not feature_dist_summary.empty:
            f.write(feature_dist_summary.to_string())
        else:
            f.write("No Feature Space Distance data available.\n")

        f.write("\n\n--- Error by Leaf Uncertainty (model uncertainty) ---\n")
        if not leaf_uncert_summary.empty:
            f.write(leaf_uncert_summary.to_string())
        else:
            f.write("No Leaf Uncertainty data available.\n")

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
            
        for segment_type, (display_name, segment_col, _actual_file) in SEGMENT_REPORT_CONFIG.items():
            _write_adjustment_section(
                f,
                display_name,
                ideal_adj.get(segment_type),
                actual_adj.get(segment_type),
                segment_col,
            )
        
    print(f"Monitoring report written to {report_path}")
    
    csv_path = analysis_dir / f"production_merged_{target_date}.csv"
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
    prod_df.to_csv(csv_path, index=False)
    print(f"Production merged (ENABLED rows) saved to {csv_path} ({len(prod_df)} rows)")

    # Append to daily history (includes train and production PFI/SHAP)
    history_file = analysis_dir / "daily_metrics_history.csv"
    new_record = pd.DataFrame([{
        'Date': target_date,
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
    print(f"Monitoring complete for {course}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", type=str, default="sys_think")
    parser.add_argument("--lag", type=int, default=1, help="Days ago for the target date")
    parser.add_argument("--base_date", type=str, default=None, help="Base date for the run (YYYY-MM-DD), mostly for testing")
    args = parser.parse_args()

    monitor_production(args.course, args.lag, args.base_date)
