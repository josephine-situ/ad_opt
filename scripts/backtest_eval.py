"""
Evaluates backtest results using the best model trained on full data.
Outputs an evaluation summary and adds a predicted clicks column to existing optimized costs files.
Separated from backtest_daily.py to allow re-evaluation and cross-validation.

Example Usage:
    python scripts/backtest_eval.py --course gen_ai --exp-name exp1 --masked
"""

import pandas as pd
import argparse
import sys
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, KFold
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.modeling import _to_float32_csr, train_best_model
from scripts.backtest_daily import feature_matrix_cached, select_keywords
from utils.data_pipeline import format_keyword_data, get_conversion_rates, get_purchase_conversion_rate
from utils.date_features import COURSE_START_DATES_MAP
from config import COURSE_CONFIG

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--course", default="gen_ai", help="Course name")
    p.add_argument("--start", default="2025-12-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--day", default=None)
    
    # Updated to loop over budget instead
    p.add_argument("--budget", type=float, nargs='+', default=None)

    p.add_argument("--exp-name", default="exp4", help="Experiment name")
    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", action="store_true", help="Use masked data as new keywords for testing")
    p.add_argument("--embedding-method", default="bert", choices=["bert", "llm"], help="Embedding method: bert or llm (default: bert)")
    
    args = p.parse_args()
    if args.budget is None:
        args.budget = COURSE_CONFIG[args.course]['budgets']
    return args


def main():
    args = get_args()
    
    base_dir = Path(f"data/{args.course}")

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    
    if args.day is not None:
        opt_days = [pd.to_datetime(args.day)]
    else:
        opt_days = list(pd.date_range(start=start_dt, end=end_dt, freq="D"))

    # Load Base Data based on embedding method
    embedding_method = args.embedding_method
    data_file = base_dir / f"clean/ad_opt_data_{embedding_method}.csv"
    df = pd.read_csv(data_file)

    # Filter out Region C, except if course is sys_eng
    if args.course != "sys_eng":
        df = df[df["Region"] != "C"].copy() 
    df["Day"] = pd.to_datetime(df["Day"])
    
    kw_df_all = pd.read_csv(base_dir / "gkp/keywords_classified.csv")
    
    # Merge Origin into Main Data (Actuals)
    df = df.merge(kw_df_all[["Keyword", "Origin"]], on="Keyword", how="left")
    
    # Determine feature columns based on embedding method
    if embedding_method == "llm":
        embedding_cols = ["llm_relevance_score"] if "llm_relevance_score" in df.columns else []
    else:
        embedding_cols = [c for c in df.columns if c.startswith("bert_")]
    
    features = [
        "Match type",
        "Region",
        "day_of_week",
        "is_weekend",
        "month",
        "is_public_holiday",
        "days_to_next_course_start",
        "last_month_searches",
        "three_month_avg",
        "six_month_avg",
        "mom_change",
        "search_trend",
        "Competition (indexed value)",
        "Top of page bid (low range)",
        "Top of page bid (high range)",
        "Cost",
        *embedding_cols,
    ]
    
    base_results_dir = Path(f"opt_results/{args.course}/backtests/{args.exp_name}")
    eval_models_dir = Path(f"opt_results/{args.course}/eval_models")
    models_dir = Path(f"models/{args.course}/backtests/{args.exp_name}") # For reading hist metrics
    
    eval_models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = base_results_dir / "cache"
    
    # Load Hist Metrics if available
    hist_metrics = {}
    hist_metrics_file = models_dir / "hist_model_metrics.csv"
    if hist_metrics_file.exists():
        hm_df = pd.read_csv(hist_metrics_file)
        hm_df["Day"] = pd.to_datetime(hm_df["Day"])
        for _, row in hm_df.iterrows():
            hist_metrics[row["Day"].date()] = row.to_dict()
    
    # 1. Train/Get Best Evaluation Model on FULL data
    eval_model_path = eval_models_dir / "eval_model_full.joblib"
    eval_metrics_path = eval_models_dir / "eval_metrics_full.joblib"
    
    if eval_model_path.exists():
        print(f"Loading cached full eval model")
        model = joblib.load(eval_model_path)
        e_metrics = joblib.load(eval_metrics_path) if eval_metrics_path.exists() else {}
    else:
        print(f"Training full eval model...")
        # Use a fixed date for random state or max date
        model, params, cv_mse, train_mse, train_r2, _ = train_best_model(df, features, df["Day"].max().date())
        joblib.dump(model, eval_model_path)
        e_metrics = {"CV_MSE": cv_mse, "Train_MSE": train_mse, "Train_R2": train_r2, "Best_Params": params}
        joblib.dump(e_metrics, eval_metrics_path)

    # Get observed clicks, conversion rate by location
    loc_df = get_conversion_rates(base_dir=base_dir)
    
    # Get purchase conversion rates by location
    purch_loc_df = get_purchase_conversion_rate(by_reg=False, base_dir=base_dir)

    eval_summary_rows = []
    
    for day in opt_days:
        print(f"Evaluating Day: {day.date()}")
        obs = df[df["Day"] == day].copy()
            
        # 2. Evaluate Actuals
        obs_zero = obs.copy()
        obs_zero["Cost"] = 0.0
        
        pred_act_clicks = model.predict(obs[features])
        pred_act_base = model.predict(obs_zero[features])
        print(f"  Actual Clicks Prediction: {pred_act_clicks.sum():.2f}, Base Prediction: {pred_act_base.sum():.2f}, Observed: {obs['Clicks'].sum():.2f}")
        val_act_diff = pred_act_clicks - pred_act_base
        val_act_clicks = val_act_diff.sum()
        
        # Prepare Actuals DF for country breakdown
        obs_breakdown = obs[["Keyword", "Region", "Origin", "Cost"]].copy()
        obs_breakdown["t_Clicks_ActCost"] = val_act_diff
        
        act_cost = obs["Cost"].sum()
        real_act_clicks = obs["Clicks"].sum()
        
        # 3. For each param combo
        seed = int(day.strftime('%Y%m%d'))
        kw_df_daily, keywords, new_keywords = select_keywords(kw_df_all, args.keywords_n, args.masked, seed=seed)
        
        # Reconstruct feature matrix for this day/seed
        X_base = feature_matrix_cached(
            keywords=keywords, 
            opt_date=day, 
            cache_dir=cache_dir,
            base_dir=base_dir,
            course_start_dts=COURSE_START_DATES_MAP.get(args.course, [])
        )
        
        for b in args.budget:
                
            run_dir = base_results_dir / f"budget_{int(b)}"
            bids_dir = run_dir / "bids"
            bids_file = bids_dir / f"optimized_costs_{day.date()}.csv"
            act_file = bids_dir / f"actual_costs_{day.date()}.csv"
            
            if not bids_file.exists():
                print(f"File not found: {bids_file}")
                continue

            sol = pd.read_csv(bids_file)
            
            # Merge with X_base to get features
            X_day = X_base.merge(
                sol[["Keyword", "Region", "Match type", "Origin", "Optimal Cost"]], 
                on=["Keyword", "Region", "Match type"],
                how="right" 
            )
            
            X_day["Cost"] = X_day["Optimal Cost"]

            # Predict Opt
            pred_opt = model.predict(X_day[features])
            
            # Baseline (Cost = 0)
            X_day_zero = X_day.copy()
            X_day_zero["Cost"] = 0.0
            pred_opt_base = model.predict(X_day_zero[features])
            print(f"  Budget {b}: Opt Clicks Prediction: {pred_opt.sum():.2f}, Base Prediction: {pred_opt_base.sum():.2f}")
            
            # Metrics
            pred_opt_lift = pred_opt - pred_opt_base
            
            val_opt_clicks = pred_opt_lift.sum()
            val_opt_cost = X_day["Optimal Cost"].sum()

            # --- New Metrics: Stability & Turnover ---
            prev_day = day - pd.Timedelta(days=1)
            prev_bids_file = bids_dir / f"optimized_costs_{prev_day.date()}.csv"
            
            avg_cost_change = np.nan
            pct_new_keywords = np.nan
            
            if prev_bids_file.exists():
                prev_sol = pd.read_csv(prev_bids_file)
                
                # Identify "active" keywords (Cost > 0)
                # Assuming "Optimal Cost" is the column
                curr_active = sol[sol["Optimal Cost"] > 1e-6].copy()
                prev_active = prev_sol[prev_sol["Optimal Cost"] > 1e-6].copy()
                
                # Set of active keys: Keyword, Region, Match type
                key_cols = ["Keyword", "Region", "Match type"]
                
                # Create set of tuples for easy set operations
                curr_keys = set(curr_active[key_cols].itertuples(index=False, name=None))
                prev_keys = set(prev_active[key_cols].itertuples(index=False, name=None))
                
                # % New Keywords: Present at t, but not t-1
                if len(curr_keys) > 0:
                    new_keys = curr_keys - prev_keys
                    pct_new_keywords = len(new_keys) / len(curr_keys)
                else:
                    pct_new_keywords = 0.0 # or np.nan if no active keywords
                    
                # % Change in Cost for keywords present at t-1
                # We need to join prev_active with sol (current costs, even if 0)
                # But strict interpretation "present at t-1" means we take all from prev_active
                # and look at their cost in current sol.
                
                # Rename cols for merge
                prev_active_sub = prev_active[key_cols + ["Optimal Cost"]].rename(columns={"Optimal Cost": "Prev_Cost"})
                curr_sub = sol[key_cols + ["Optimal Cost"]].rename(columns={"Optimal Cost": "Curr_Cost"})
                
                merged_cost = prev_active_sub.merge(curr_sub, on=key_cols, how="inner")
                
                # absolute cost change
                if len(merged_cost) > 0:
                    merged_cost["pct_change"] = abs((merged_cost["Curr_Cost"] - merged_cost["Prev_Cost"]) / merged_cost["Prev_Cost"])
                    avg_cost_change = merged_cost["pct_change"].mean()

            # --- Calculate Predicted Conversions ---
            # Add lift to X_day for grouping
            X_day["t_Clicks_OptCost"] = pred_opt_lift
            
            # Group clicks/lift by Region
            opt_clicks_region = X_day.groupby('Region')['t_Clicks_OptCost'].sum().reset_index()
            act_clicks_region = obs_breakdown.groupby('Region')['t_Clicks_ActCost'].sum().reset_index()
            
            # Group cost by Region
            opt_cost_region = X_day.groupby('Region')['Optimal Cost'].sum().reset_index().rename(columns={'Optimal Cost': 'Opt_Cost_Reg'})
            act_cost_region = obs_breakdown.groupby('Region')['Cost'].sum().reset_index().rename(columns={'Cost': 'Act_Cost_Reg'})

            # Group clicks/lift by Region AND Origin (for detailed conversion calc)
            opt_clicks_reg_org = X_day.groupby(['Region', 'Origin'])['t_Clicks_OptCost'].sum().reset_index()
            act_clicks_reg_org = obs_breakdown.groupby(['Region', 'Origin'])['t_Clicks_ActCost'].sum().reset_index()
            
            # Merge with loc_df
            # loc_df: [Location, Region, Click_prop, Conv_rate]
            
            # Use loc_df as base to ensure all locations are present
            df_country = loc_df.merge(opt_clicks_region, on='Region', how='left').rename(columns={'t_Clicks_OptCost': 'Opt_Clicks_Reg'})
            df_country = df_country.merge(act_clicks_region, on='Region', how='left').rename(columns={'t_Clicks_ActCost': 'Act_Clicks_Reg'})
            
            df_country = df_country.merge(opt_cost_region, on='Region', how='left')
            df_country = df_country.merge(act_cost_region, on='Region', how='left')

            df_country = df_country.fillna(0) # IMPORTANT for safety

            df_country['Opt_Conversions'] = df_country['Opt_Clicks_Reg'] * df_country['Click_prop'] * df_country['Conv_rate']
            df_country['Act_Conversions'] = df_country['Act_Clicks_Reg'] * df_country['Click_prop'] * df_country['Conv_rate']

            df_country['Opt_Clicks'] = df_country['Opt_Clicks_Reg'] * df_country['Click_prop']
            df_country['Act_Clicks'] = df_country['Act_Clicks_Reg'] * df_country['Click_prop']
            
            df_country['Opt_Spend'] = df_country['Opt_Cost_Reg'] * df_country['Click_prop']
            df_country['Act_Spend'] = df_country['Act_Cost_Reg'] * df_country['Click_prop']

            # --- Calculate Purchases ---
            # Merge purchase rates with country data
            df_country_purch = df_country.merge(
                purch_loc_df[['Location', 'Region', 'Purch_rate']], 
                on=['Location', 'Region'], 
                how='left'
            )
            df_country_purch['Purch_rate'] = df_country_purch['Purch_rate'].fillna(0)
            
            df_country_purch['Opt_Purchases'] = df_country_purch['Opt_Clicks'] * df_country_purch['Purch_rate']
            df_country_purch['Act_Purchases'] = df_country_purch['Act_Clicks'] * df_country_purch['Purch_rate']
            
            val_opt_purch = df_country_purch['Opt_Purchases'].sum()
            val_act_purch = df_country_purch['Act_Purchases'].sum()

            # Calculate Origin Conversions
            # We distribute the Region+Origin clicks to countries in that region
            def calc_origin_conversions(clicks_df, click_col):
                 # clicks_df: [Region, Origin, click_col]
                 m = clicks_df.merge(loc_df, on='Region', how='left')
                 m['Predicted_Conv'] = m[click_col] * m['Click_prop'] * m['Conv_rate']
                 return m.groupby('Origin')['Predicted_Conv'].sum().to_dict()
            
            def calc_origin_purchases(clicks_df, click_col):
                 # clicks_df: [Region, Origin, click_col]
                 m = clicks_df.merge(purch_loc_df, on='Region', how='left')
                 m['Purch_rate'] = m['Purch_rate'].fillna(0)
                 m['Predicted_Purch'] = m[click_col] * m['Click_prop'] * m['Purch_rate']
                 return m.groupby('Origin')['Predicted_Purch'].sum().to_dict()
            
            opt_conv_origin_map = calc_origin_conversions(opt_clicks_reg_org, 't_Clicks_OptCost')
            act_conv_origin_map = calc_origin_conversions(act_clicks_reg_org, 't_Clicks_ActCost')
            
            opt_purch_origin_map = calc_origin_purchases(opt_clicks_reg_org, 't_Clicks_OptCost')
            act_purch_origin_map = calc_origin_purchases(act_clicks_reg_org, 't_Clicks_ActCost')
            
            # Save Country breakdown to run_dir (include purchases)
            breakdown_file = run_dir / f"country_breakdown_{day.strftime('%Y-%m-%d')}.csv"
            df_country_out = df_country.copy()
            df_country_out['Opt_Purchases'] = df_country_purch['Opt_Purchases']
            df_country_out['Act_Purchases'] = df_country_purch['Act_Purchases']
            df_country_out[['Location', 'Region', 'Opt_Conversions', 'Act_Conversions', 'Opt_Purchases', 'Act_Purchases', 'Opt_Clicks', 'Act_Clicks', 'Opt_Spend', 'Act_Spend']].to_csv(breakdown_file, index=False)

            val_opt_conv = df_country['Opt_Conversions'].sum()
            val_act_conv = df_country['Act_Conversions'].sum()
            
            # Update Bids File with Eval Metric
            pred_diffs = pd.DataFrame({
                "Keyword": X_day["Keyword"],
                "Region": X_day["Region"],
                "Match type": X_day["Match type"],
                "t_Clicks_OptCost": pred_opt_lift
            })
            
            if "t_Clicks_OptCost" in sol.columns:
                sol = sol.drop(columns=["t_Clicks_OptCost"])
            
            sol_updated = sol.merge(pred_diffs, on=["Keyword", "Region", "Match type"], how="left")
            sol_updated.to_csv(bids_file, index=False)
            
            # Expected clicks (T-1) from optimizer
            if "Gurobi Pred over Base" in sol.columns:
                    tm1_Clicks_OptCost = sol["Gurobi Pred over Base"].sum()
            elif "t_Clicks_OptCost" in sol.columns:
                    tm1_Clicks_OptCost = 0.0 
            else:
                    tm1_Clicks_OptCost = 0.0
                    
            # Update Actual File with Eval Metric
            if act_file.exists():
                act_df_existing = pd.read_csv(act_file)
                if "t_Clicks_ActCost" in act_df_existing.columns:
                        pass
                else:
                        obs_out = obs[["Keyword", "Region", "Match type", "Cost", "Clicks"]].copy()
                        obs_out["t_Clicks_ActCost"] = val_act_diff
                        obs_out.to_csv(act_file, index=False)
            
            # Hist metrics
            hm = hist_metrics.get(day.date(), {})
            
            # --- Dynamic Compilation of Summary Row ---
            row_dict = {
                "Day": day.date(),
                "Budget": int(b),
                "t_Clicks_OptCost": val_opt_clicks, 
                "t_Clicks_ActCost": val_act_clicks, 
                "tm1_Clicks_OptCost": tm1_Clicks_OptCost, 
                "Actual_Clicks": real_act_clicks,
                "Opt_Cost": val_opt_cost,
                "Act_Cost": act_cost,
                "Opt_Conv": val_opt_conv,
                "Act_Conv": val_act_conv,
                "Opt_Purch": val_opt_purch,
                "Act_Purch": val_act_purch,
                "N_Opt": len(X_day),
                "N_Obs": len(obs),
                "Avg_Cost_Change": avg_cost_change,
                "Pct_New_Keywords": pct_new_keywords,
                
                # Eval Model Metrics
                "Eval_CV_MSE": e_metrics.get("CV_MSE"),
                "Eval_Train_MSE": e_metrics.get("Train_MSE"),
                "Eval_Train_R2": e_metrics.get("Train_R2"),
                "Eval_Best_Params": str(e_metrics.get("Best_Params")),
                
                # Hist Model Metrics
                "Hist_MSE": hm.get("Hist_MSE"),
                "Hist_R2": hm.get("Hist_R2"),
                "Hist_Bias": hm.get("Hist_Bias")
            }
            
            # Add Region/Origin Breakdowns Dynamically
            # Regions
            all_regions = set(X_day['Region'].unique()) | set(obs['Region'].unique())
            for reg in all_regions:
                row_dict[f"Opt_Cost_Region_{reg}"] = X_day[X_day['Region'] == reg]['Optimal Cost'].sum()
                row_dict[f"Act_Cost_Region_{reg}"] = obs[obs['Region'] == reg]['Cost'].sum()

                # Clicks (Lift)
                row_dict[f"Opt_Clicks_Region_{reg}"] = X_day[X_day['Region'] == reg]['t_Clicks_OptCost'].sum()
                row_dict[f"Act_Clicks_Region_{reg}"] = obs_breakdown[obs_breakdown['Region'] == reg]['t_Clicks_ActCost'].sum()

                # Regional Conversions
                row_dict[f"Opt_Conv_Region_{reg}"] = df_country[df_country['Region'] == reg]['Opt_Conversions'].sum()
                row_dict[f"Act_Conv_Region_{reg}"] = df_country[df_country['Region'] == reg]['Act_Conversions'].sum()

                # Regional Purchases
                row_dict[f"Opt_Purch_Region_{reg}"] = df_country_purch[df_country_purch['Region'] == reg]['Opt_Purchases'].sum()
                row_dict[f"Act_Purch_Region_{reg}"] = df_country_purch[df_country_purch['Region'] == reg]['Act_Purchases'].sum()

            # Origins
            all_origins = ['new', 'existing', 'existing searches'] # Enforce standard origins
            for org in all_origins:
                # Cost
                row_dict[f"Opt_Cost_Origin_{org}"] = X_day[X_day['Origin'] == org]['Optimal Cost'].sum()
                row_dict[f"Act_Cost_Origin_{org}"] = obs[obs['Origin'] == org]['Cost'].sum()
                
                # Clicks (Lift)
                row_dict[f"Opt_Clicks_Origin_{org}"] = X_day[X_day['Origin'] == org]['t_Clicks_OptCost'].sum()
                row_dict[f"Act_Clicks_Origin_{org}"] = obs_breakdown[obs_breakdown['Origin'] == org]['t_Clicks_ActCost'].sum()
                
                # Conversions (Calculated above)
                row_dict[f"Opt_Conv_Origin_{org}"] = opt_conv_origin_map.get(org, 0.0)
                row_dict[f"Act_Conv_Origin_{org}"] = act_conv_origin_map.get(org, 0.0)
                
                # Purchases (Calculated above)
                row_dict[f"Opt_Purch_Origin_{org}"] = opt_purch_origin_map.get(org, 0.0)
                row_dict[f"Act_Purch_Origin_{org}"] = act_purch_origin_map.get(org, 0.0)
            
            eval_summary_rows.append(row_dict)
                
    # Save results
    if eval_summary_rows:
        res_df = pd.DataFrame(eval_summary_rows)
        out_path = base_results_dir / "evaluation_results.csv"
        res_df.to_csv(out_path, index=False)
        print(f"Saved evaluation results to {out_path}")

if __name__ == "__main__":
    main()
