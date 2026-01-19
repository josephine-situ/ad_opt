"""
Evaluates backtest results.
Separated from backtest_daily.py to allow re-evaluation and cross-validation.

Example Usage:
    python scripts/backtest_eval.py --start 2025-12-01 --end 2025-12-31 --exp-name "exp5" --x-max None 30 60 90 --alpha 0 0.5 1 --masked
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
from scripts.prediction_modeling_tweedie import _to_float32_csr
from scripts.backtest_daily import feature_matrix_cached, select_keywords

def get_args():
    # helper for x-max parsing (handle None/inf)
    def parse_float_or_none(x):
        if str(x).lower() in ['none', 'inf', 'nan']:
            return None
        return float(x)

    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-01")
    p.add_argument("--end", default="2025-12-03")
    p.add_argument("--day", default=None)
    # Accepts list for x-max and alpha to iterate over multiple results
    p.add_argument("--x-max", nargs='+', default=["None", "30", "60", "90"])
    p.add_argument("--alpha", nargs='+', default=["0", "0.5", "1.0"])
    p.add_argument("--exp-name", default="exp4", help="Experiment name")
    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", default=True, action="store_true", help="Use masked data as new keywords for testing")
    
    args = p.parse_args()
    
    # Process x_max to ensure None or float
    args.x_max_vals = [parse_float_or_none(x) for x in args.x_max]
    args.alpha_vals = [float(x) for x in args.alpha]
    
    return args

def train_best_model(df_day, features, day_date):
    """
    Train a single best model for the day using GridSearchCV.
    Returns: pipeline, best_params, cv_score (negative MSE), in_sample_score (MSE), r2
    """
    X, y = df_day[features], df_day["Clicks"]
    cat = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(with_mean=False), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat),
        ],
        remainder="drop",
    )
    
    # Base estimator (same hyperparams range as typically used, but can be searched)
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    
    pipeline = Pipeline(
        [
            ("preprocess", pre),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", xgb_reg),
        ]
    )

    # Keep the grid small to reduce overfitting risk on small daily data
    param_grid = {
        "model__n_estimators": [5, 10, 20],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.1, 0.3],
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=int(day_date.strftime('%Y%m%d')))
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = -grid_search.best_score_ # Convert back to positive MSE
    
    # In-sample metrics on full data (best_model is already refitted on X, y)
    y_pred = best_model.predict(X)
    in_sample_mse = mean_squared_error(y, y_pred)
    in_sample_r2 = r2_score(y, y_pred)
    
    return best_model, best_params, best_cv_score, in_sample_mse, in_sample_r2

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
    cache_dir = Path("opt_results/backtests/cache")
    
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
        model, params, cv_mse, train_mse, train_r2 = train_best_model(df, features, df["Day"].max().date())
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
        
        for xm in args.x_max_vals:
            for al in args.alpha_vals:
                
                xm_str = str(xm) # None -> 'None'
                run_dir = base_results_dir / f"xmax_{xm_str}_alpha_{al}"
                bids_dir = run_dir / "bids"
                bids_file = bids_dir / f"optimized_costs_{day.date()}.csv"
                act_file = bids_dir / f"actual_costs_{day.date()}.csv"
                
                sol = pd.read_csv(bids_file)
                
                # Merge with X_base to get features
                X_day = X_base.merge(
                    sol[["Keyword", "Region", "Match type", "Optimal Cost"]],
                    on=["Keyword", "Region", "Match type"],
                    how="inner" 
                )
                
                X_day["Cost"] = X_day["Optimal Cost"].fillna(0.0)
                
                # Predict Opt
                pred_opt = model.predict(X_day[features])
                
                # Baseline (Cost = 0)
                X_day_zero = X_day.copy()
                X_day_zero["Cost"] = 0.0
                pred_opt_base = model.predict(X_day_zero[features])
                
                val_opt_clicks = (pred_opt - pred_opt_base).sum()
                val_opt_cost = X_day["Optimal Cost"].sum()
                
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
                    "x_max": xm,
                    "alpha": al,
                    "t_Clicks_OptCost": val_opt_clicks, 
                    "t_Clicks_ActCost": val_act_clicks, 
                    "tm1_Clicks_OptCost": tm1_Clicks_OptCost, 
                    "Actual_Clicks": real_act_clicks,
                    "Opt_Cost": val_opt_cost,
                    "Act_Cost": act_cost,
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
