"""Daily backtest: train daily, optimize costs, evaluate yesterday.

For each day t:
- Train model on data with Day < t
- Evaluate day t-1 using the newly trained model (model(opt-cost), model(act-cost), actual clicks)
- Optimize costs for day t
"""

from pathlib import Path
import argparse
import sys
import hashlib
import os
import numpy as np

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.optimization import create_feature_matrix, extract_solution, optimize_bids
from scripts.prediction_modeling_tweedie import _to_float32_csr


def fit_click_model(df_train: pd.DataFrame, *, features: list[str]) -> Pipeline:
    X, y = df_train[features], df_train["Clicks"]
    cat = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(with_mean=False), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat),
        ],
        remainder="drop",
    )
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=20,
        max_depth=4,
        learning_rate=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
    )

    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", model),
        ]
    )
    pipe.fit(X, y)
    return pipe


def feature_matrix_cached(*, keywords: list[str], opt_date: pd.Timestamp, cache_dir: Path) -> pd.DataFrame:
    kw_hash = hashlib.md5("|".join(sorted(keywords)).encode("utf-8")).hexdigest()[:10]
    p = cache_dir / f"feature_matrix_{kw_hash}_{opt_date.date()}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    X = create_feature_matrix(keywords, opt_date=opt_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(p)
    return X
    # return X


def in_sample_metrics(model: Pipeline, df: pd.DataFrame, *, features: list[str]) -> dict:
    y = df["Clicks"]
    yhat = model.predict(df[features])
    return {
        "MSE": float(mean_squared_error(y, yhat)),
        "R2": float(r2_score(y, yhat)),
        "Bias": float((yhat - y).mean()),
    }

def select_keywords(kw_df, keywords_n, masked):
    """ Select keywords for backtesting, optionally masking some as "new" keywords."""
    
    if masked:
        kw_df = kw_df[kw_df["Origin"] == "existing"].copy()

        # Randomly select some existing keywords to be "new" for testing
        existing_keywords = kw_df["Keyword"].tolist()
        n_new = round(0.1 * len(existing_keywords))  # For example, 10% as new
        new_keywords = np.random.choice(existing_keywords, size=n_new, replace=False)
        kw_df.loc[kw_df["Keyword"].isin(new_keywords), "Origin"] = "new"
        print(f"Selected {n_new} existing keywords as 'new' for testing. For example: {new_keywords[:5]}")
    else:
        new_keywords = None

    # Select keywords to test (if small run)
    if keywords_n is not None:
        origins = ["existing", "existing searches", "new"]
        n_per_group = max(1, keywords_n // len(origins))
        selected = []
        for origin in origins:
            selected.extend(
                kw_df[kw_df["Origin"] == origin]["Keyword"]
                .head(n_per_group)
                .tolist()
            )

        existing_set = set(selected)
        for k in kw_df["Keyword"]:
            if len(selected) >= keywords_n:
                break
            if k not in existing_set:
                selected.append(k)
        keywords = selected[: keywords_n]
    else:
        keywords = kw_df["Keyword"].tolist()

    return kw_df, keywords, new_keywords

def main():
    def float_or_none(value):
        """Helper to parse command line args as float or None"""
        if value.lower() == "none":
            return None
        return float(value)

    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-01")
    p.add_argument("--end", default="2025-12-03")
    p.add_argument("--day", default=None)
    p.add_argument("--budget", type=float, default=[307.61, 36.65, 18.38], nargs='+', help="Budgets for regions [USA, A, B]")

    # Updated this line to use the custom type
    p.add_argument("--x-max", type=float_or_none, nargs='+', default=[50]) 

    p.add_argument("--alpha", type=float, nargs='+', default=[1.0], help="Max proportion of budget to new keywords")
    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", action="store_true", help="Use masked data as new keywords for testing")

    args = p.parse_args()

    start_dt, end_dt, budget, x_max_list, alpha_list, masked, keywords_n = args.start, args.end, args.budget, args.x_max, args.alpha, args.masked, args.keywords_n
    
    df = pd.read_csv("data/clean/ad_opt_data_bert.csv")
    df = df[df["Region"] != "C"].copy()  # remove region C since no budget allocated to it
    df["Day"] = pd.to_datetime(df["Day"])

    kw_df = pd.read_csv("data/gkp/keywords_classified.csv")
    kw_df, keywords, new_keywords = select_keywords(kw_df, keywords_n, masked)

    if args.day is not None:
        opt_days = [pd.to_datetime(args.day)]
    else:
        opt_days = list(pd.date_range(start=start_dt, end=end_dt, freq="D"))

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

    models_dir = Path("models/backtests")
    base_results_dir = Path("opt_results/backtests")
    cache_dir = base_results_dir / "cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store evaluation rows for each parameter combination
    eval_results = {}
    for xm in x_max_list:
        for al in alpha_list:
            eval_results[(xm, al)] = []

    for day in opt_days:
        print(f"\n=== Day {day.date()} ===")

        # Select a new set of masked keywords each day
        kw_df, keywords, new_keywords = select_keywords(kw_df, keywords_n, masked)

        # Get observed data before filtering out keywords
        obs = df[df["Day"] == day].copy()

        # Train model on history up to t-1, excluding new keywords if masked
        if masked:
            df = df[~df['Keyword'].isin(new_keywords)].copy()
        hist = df[df["Day"] < day].copy()
        pipe = fit_click_model(hist, features=features)
        hist_m = in_sample_metrics(pipe, hist, features=features)
        model_path = models_dir / f"xgb_clicks_model_{day.date()}.joblib"
        joblib.dump(pipe, model_path)

        # Precompute feature matrix (shared across parameters)
        X_base = feature_matrix_cached(keywords=keywords, opt_date=day, cache_dir=cache_dir)

        # Create evaluation model on day t to evaluate day t-1.
        # Predicted clicks is over the baseline: model(cost=act_cost) or model(cost=opt_cost) - model(cost=0).
        eval_model = fit_click_model(obs, features=features)
        day_m = in_sample_metrics(eval_model, obs, features=features)
        pred_act = eval_model.predict(obs[features])
        act_cost = float(obs["Cost"].sum())
        act_clicks = float(obs["Clicks"].sum())

        # Calculate baseline clicks with cost=0
        obs_zero_cost = obs.copy()
        obs_zero_cost["Cost"] = 0.0
        pred_base = eval_model.predict(obs_zero_cost[features])

        # Optimize and Evaluate for each parameter combination
        for xm in x_max_list:
            for al in alpha_list:
                # Define output directory for this run
                run_dir = base_results_dir / f"xmax_{xm}_alpha_{al}"
                bids_dir = run_dir / "bids"
                bids_dir.mkdir(parents=True, exist_ok=True)

                # Optimize bids for day t
                # Copy X_base to avoid modification issues
                m, cost_vars, pred_vars, X_opt = optimize_bids(X_base.copy(), str(model_path), budget=budget, x_max=xm, kw_df=kw_df, alpha=al)
                sol = extract_solution(m, cost_vars, pred_vars, str(model_path), X_opt)

                # Optimized cost evaluation
                X_day = X_opt.merge(
                    sol[["Keyword", "Region", "Match type", "Optimal Cost"]],
                    on=["Keyword", "Region", "Match type"],
                    how="right",
                ) # right merge to keep rows with cost > 0 only
                X_day["Optimal Cost"] = X_day["Optimal Cost"].fillna(0.0)
                X_day["Cost"] = X_day["Optimal Cost"]
                pred_opt = eval_model.predict(X_day[features])
                opt_expected_clicks = float(sol["Gurobi Pred over Base"].sum())

                # Calculate baseline clicks with cost=0 for optimized set
                X_day_zero_cost = X_day.copy()
                X_day_zero_cost["Cost"] = 0.0
                pred_opt_base = eval_model.predict(X_day_zero_cost[features])

                # Add t_clicks to solution for reference
                sol["t_Clicks_OptCost"] = pred_opt - pred_opt_base
                opt_path = bids_dir / f"optimized_costs_{day.date()}.csv"
                sol.to_csv(opt_path, index=False)

                # Save actual costs for comparison
                act_path = bids_dir / f"actual_costs_{day.date()}.csv"
                obs_out = obs[["Keyword", "Region", "Match type", "Cost", "Clicks"]].copy()
                # Predict clicks for actuals using the same model
                obs_out["t_Clicks_ActCost"] = pred_act - pred_base
                obs_out.to_csv(act_path, index=False)

                # Evaluation: all over baseline
                eval_results[(xm, al)].append(
                    {
                        "Day": day.date(),
                        "t_Clicks_OptCost": float((pred_opt - pred_opt_base).sum()),
                        "t_Clicks_ActCost": float((pred_act - pred_base).sum()),
                        "tm1_Clicks_OptCost": opt_expected_clicks,
                        "Actual_Clicks": act_clicks,
                        "Opt_Cost": float(X_day["Optimal Cost"].sum()),
                        "Act_Cost": act_cost,
                        "Hist_MSE": hist_m["MSE"],
                        "Hist_R2": hist_m["R2"],
                        "Hist_Bias": hist_m["Bias"],
                        "Day_MSE": day_m["MSE"],
                        "Day_R2": day_m["R2"],
                        "Day_Bias": day_m["Bias"],
                        "N_Obs": int(len(obs)),
                        "N_Opt": int(len(X_day)),
                        "SlurmJobId": os.environ.get("SLURM_JOB_ID"),
                        "SlurmArrayTaskId": os.environ.get("SLURM_ARRAY_TASK_ID"),
                        "x_max": xm,
                        "alpha": al
                    }
                )

    # Save evaluation results for each combination
    for (xm, al), rows in eval_results.items():
        eval_df = pd.DataFrame(rows)
        run_dir = base_results_dir / f"xmax_{xm}_alpha_{al}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if args.day is not None:
             eval_path = run_dir / f"daily_eval_{pd.to_datetime(args.day).date()}.csv"
        else:
             eval_path = run_dir / "daily_eval.csv"
        
        eval_df.to_csv(eval_path, index=False)
        print(f"Evaluation for x_max={xm}, alpha={al} saved to {eval_path}")

    print(f"\nBacktest complete.")

if __name__ == "__main__":
    main()