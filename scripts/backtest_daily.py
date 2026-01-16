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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-01")
    p.add_argument("--end", default="2025-12-03")
    p.add_argument("--day", default=None)
    p.add_argument("--budget", type=float, default=400)
    p.add_argument("--keywords-n", type=int, default=None)
    args = p.parse_args()

    start_dt, end_dt, budget = args.start, args.end, args.budget

    df = pd.read_csv("data/clean/ad_opt_data_bert.csv")
    df["Day"] = pd.to_datetime(df["Day"])

    # Select keywords to test (if small run)
    kw_df = pd.read_csv("data/gkp/keywords_classified.csv")
    if args.keywords_n is not None:
        origins = ["existing", "existing searches", "new"]
        n_per_group = max(1, args.keywords_n // len(origins))
        selected = []
        for origin in origins:
            selected.extend(
                kw_df[kw_df["Origin"] == origin]["Keyword"]
                .head(n_per_group)
                .tolist()
            )

        existing_set = set(selected)
        for k in kw_df["Keyword"]:
            if len(selected) >= args.keywords_n:
                break
            if k not in existing_set:
                selected.append(k)
        keywords = selected[: args.keywords_n]
    else:
        keywords = kw_df["Keyword"].tolist()

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
    bids_dir = Path("opt_results/backtests/bids")
    cache_dir = Path("opt_results/backtests/cache")
    models_dir.mkdir(parents=True, exist_ok=True)
    bids_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    for day in opt_days:
        print(f"\n=== Day {day.date()} ===")

        # Train model on history up to t-1
        hist = df[df["Day"] < day].copy()
        pipe = fit_click_model(hist, features=features)
        hist_m = in_sample_metrics(pipe, hist, features=features)
        model_path = models_dir / f"xgb_clicks_model_{day.date()}.joblib"
        joblib.dump(pipe, model_path)

        # Optimize bids for day t, based on data from t-1
        X = feature_matrix_cached(keywords=keywords, opt_date=day, cache_dir=cache_dir)
        m, cost_vars, pred_vars = optimize_bids(X, str(model_path), budget=budget)
        sol = extract_solution(m, cost_vars, pred_vars, str(model_path), X)

        # Create evaluation model on day t to evaluate day t-1.
        # Predicted clicks is over the baseline: model(cost=act_cost) or model(cost=opt_cost) - model(cost=0).
        obs = df[df["Day"] == day].copy()
        eval_model = fit_click_model(obs, features=features)
        day_m = in_sample_metrics(eval_model, obs, features=features)
        pred_act = float(eval_model.predict(obs[features]).sum())
        act_cost = float(obs["Cost"].sum())
        act_clicks = float(obs["Clicks"].sum())

        # Calculate baseline clicks with cost=0
        obs_zero_cost = obs.copy()
        obs_zero_cost["Cost"] = 0.0
        pred_base = float(eval_model.predict(obs_zero_cost[features]).sum())

        # Optimized cost evaluation
        X_day = X.merge(
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

        # Evaluation: all over baseline
        eval_rows.append(
            {
                "Day": day.date(),
                "t_Clicks_OptCost": float((pred_opt - pred_opt_base).sum()),
                "t_Clicks_ActCost": pred_act - pred_base,
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
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    out_dir = Path("opt_results/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.day is not None:
        eval_path = out_dir / f"daily_eval_{pd.to_datetime(args.day).date()}.csv"
    else:
        eval_path = out_dir / "daily_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\nBacktest complete. Evaluation saved to {eval_path}")

if __name__ == "__main__":
    main()