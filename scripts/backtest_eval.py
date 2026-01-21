"""
Evaluates backtest results using the best model trained on full data.
Outputs an evaluation summary and adds a predicted clicks column to existing optimized costs files.
Separated from backtest_daily.py to allow re-evaluation and cross-validation.

Example Usage:
    python scripts/backtest_eval.py --exp-name exp1 --masked
"""

import pandas as pd
import argparse
import sys
import os
import joblib
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

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--day", default=None)
    
    # Updated to loop over budget instead
    p.add_argument("--budget", type=float, nargs='+', default=[300, 350, 400, 450, 500, 550])

    p.add_argument("--exp-name", default="exp4", help="Experiment name")
    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", action="store_true", help="Use masked data as new keywords for testing")
    
    args = p.parse_args()
    return args


def main():
    args = get_args()
    
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    
    if args.day is not None:
        opt_days = [pd.to_datetime(args.day)]
    else:
        opt_days = list(pd.date_range(start=start_dt, end=end_dt, freq="D"))

    # Load Base Data
    df = pd.read_csv("data/clean/ad_opt_data_bert.csv")
    df = df[df["Region"] != "C"].copy() 
    df["Day"] = pd.to_datetime(df["Day"])
    
    kw_df_all = pd.read_csv("data/gkp/keywords_classified.csv")
    
    # Merge Origin into Main Data (Actuals)
    df = df.merge(kw_df_all[["Keyword", "Origin"]], on="Keyword", how="left")
    
    bert_cols = [c for c in df.columns if c.startswith("bert_")]
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
        *bert_cols,
    ]
    
    base_results_dir = Path(f"opt_results/backtests/{args.exp_name}")
    eval_models_dir = Path("opt_results/eval_models")
    models_dir = Path(f"models/{args.exp_name}") # For reading hist metrics
    
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

    eval_summary_rows = []
    
    for day in opt_days:
        print(f"Evaluating Day: {day.date()}")
        obs = df[df["Day"] == day].copy()
            
        # 2. Evaluate Actuals
        # If we have already evaluated actuals for this day (in a previous run), we shouldn't strictly need to do it again.
        # However, it's fast. 
        obs_zero = obs.copy()
        obs_zero["Cost"] = 0.0
        
        pred_act_clicks = model.predict(obs[features])
        pred_act_base = model.predict(obs_zero[features])
        val_act_diff = pred_act_clicks - pred_act_base
        val_act_clicks = val_act_diff.sum()
        
        act_cost = obs["Cost"].sum()
        real_act_clicks = obs["Clicks"].sum()
        
        # 3. For each param combo
        seed = int(day.strftime('%Y%m%d'))
        kw_df_daily, keywords, new_keywords = select_keywords(kw_df_all, args.keywords_n, args.masked, seed=seed)
        
        # Reconstruct feature matrix for this day/seed
        X_base = feature_matrix_cached(keywords=keywords, opt_date=day, cache_dir=cache_dir)
        
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
                sol[["Keyword", "Region", "Match type", "Optimal Cost"]],
                on=["Keyword", "Region", "Match type"],
                how="right" 
            )
            
            X_day["Cost"] = X_day["Optimal Cost"].fillna(0.0)
            
            # Merge Origin into X_day
            X_day = X_day.merge(kw_df_all[["Keyword", "Origin"]], on="Keyword", how="left")

            # Predict Opt
            pred_opt = model.predict(X_day[features])
            
            # Baseline (Cost = 0)
            X_day_zero = X_day.copy()
            X_day_zero["Cost"] = 0.0
            pred_opt_base = model.predict(X_day_zero[features])
            
            val_opt_clicks = (pred_opt - pred_opt_base).sum()
            val_opt_cost = X_day["Optimal Cost"].sum()

            # Regional Breakdown
            opt_cost_usa = X_day[X_day['Region'] == 'USA']['Optimal Cost'].sum()
            opt_cost_a = X_day[X_day['Region'] == 'A']['Optimal Cost'].sum()
            opt_cost_b = X_day[X_day['Region'] == 'B']['Optimal Cost'].sum()
            
            # Origin Breakdown
            opt_cost_new = X_day[X_day['Origin'] == 'new']['Optimal Cost'].sum()
            opt_cost_existing_searches = X_day[X_day['Origin'] == 'existing searches']['Optimal Cost'].sum()
            opt_cost_existing = X_day[X_day['Origin'] == 'existing']['Optimal Cost'].sum()
            
            act_cost_usa = obs[obs['Region'] == 'USA']['Cost'].sum()
            act_cost_a = obs[obs['Region'] == 'A']['Cost'].sum()
            act_cost_b = obs[obs['Region'] == 'B']['Cost'].sum()

            # Origin Breakdown (Actuals)
            act_cost_new = obs[obs['Origin'] == 'new']['Cost'].sum()
            act_cost_existing_searches = obs[obs['Origin'] == 'existing searches']['Cost'].sum()
            act_cost_existing = obs[obs['Origin'] == 'existing']['Cost'].sum()
            
            # Update Bids File with Eval Metric
            # We want to add column `t_Clicks_OptCost` to `optimized_costs_...csv`
            pred_diffs = pd.DataFrame({
                "Keyword": X_day["Keyword"],
                "Region": X_day["Region"],
                "Match type": X_day["Match type"],
                "t_Clicks_OptCost": pred_opt - pred_opt_base
            })
            
            # Drop existing col if exists to avoid dupes
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
            # If file exists, check if it already has the column
            if act_file.exists():
                act_df_existing = pd.read_csv(act_file)
                if "t_Clicks_ActCost" in act_df_existing.columns:
                        # Skip update
                        pass
                else:
                        obs_out = obs[["Keyword", "Region", "Match type", "Cost", "Clicks"]].copy()
                        obs_out["t_Clicks_ActCost"] = val_act_diff
                        obs_out.to_csv(act_file, index=False)
            
            # Hist metrics
            hm = hist_metrics.get(day.date(), {})
            
            eval_summary_rows.append({
                "Day": day.date(),
                "Budget": int(b),
                "t_Clicks_OptCost": val_opt_clicks, 
                "t_Clicks_ActCost": val_act_clicks, 
                "tm1_Clicks_OptCost": tm1_Clicks_OptCost, 
                "Actual_Clicks": real_act_clicks,
                "Opt_Cost": val_opt_cost,
                "Act_Cost": act_cost,
                "Opt_Cost_USA": opt_cost_usa,
                "Opt_Cost_A": opt_cost_a,
                "Opt_Cost_B": opt_cost_b,
                "Opt_Cost_new": opt_cost_new,
                "Opt_Cost_existing": opt_cost_existing,
                "Opt_Cost_existing_searches": opt_cost_existing_searches,
                
                "Act_Cost_USA": act_cost_usa,
                "Act_Cost_A": act_cost_a,
                "Act_Cost_B": act_cost_b,  
                "Act_Cost_new": act_cost_new,
                "Act_Cost_existing": act_cost_existing,
                "Act_Cost_existing_searches": act_cost_existing_searches,
                          
                "N_Opt": len(X_day),
                "N_Obs": len(obs),
                
                # Eval Model Metrics
                "Eval_CV_MSE": e_metrics.get("CV_MSE"),
                "Eval_Train_MSE": e_metrics.get("Train_MSE"),
                "Eval_Train_R2": e_metrics.get("Train_R2"),
                "Eval_Best_Params": str(e_metrics.get("Best_Params")),
                
                # Hist Model Metrics
                "Hist_MSE": hm.get("Hist_MSE"),
                "Hist_R2": hm.get("Hist_R2"),
                "Hist_Bias": hm.get("Hist_Bias")
            })
                
    # Save results
    if eval_summary_rows:
        res_df = pd.DataFrame(eval_summary_rows)
        out_path = base_results_dir / "evaluation_results.csv"
        res_df.to_csv(out_path, index=False)
        print(f"Saved evaluation results to {out_path}")

if __name__ == "__main__":
    main()
