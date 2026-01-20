"""Daily backtest - generate optimal solutions.

Example Usage:
    python scripts/backtest_daily.py --start 2025-12-01 --end 2025-12-31 --exp-name exp1 --masked

For each day t:
- Train model on data until t-1.
- Embed this model and optimize to find (x^t)^*
"""

from pathlib import Path
import argparse
import sys
import hashlib
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
from scripts.modeling import _to_float32_csr


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

def select_keywords(kw_df, keywords_n, masked, seed=None):
    """ Select keywords for backtesting, optionally masking some as "new" keywords."""
    
    if masked:
        kw_df = kw_df[kw_df["Origin"] == "existing"].copy()

        # Randomly select some existing keywords to be "new" for testing
        # Use a deterministic seed if provided
        rng = np.random.default_rng(seed)
        
        existing_keywords = kw_df["Keyword"].tolist()
        n_new = round(0.1 * len(existing_keywords))  # For example, 10% as new
        new_keywords = rng.choice(existing_keywords, size=n_new, replace=False)
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
    p.add_argument("--budget", type=float, nargs='+', default=[300, 350, 400, 450, 500, 550], help="Total budgets to test")

    # Deprecated/Unused parameters (kept for compatibility or remove if safe)
    p.add_argument("--x-max", type=float_or_none, nargs='+', default=[None], help="Deprecated")
    p.add_argument("--alpha", type=float, nargs='+', default=[1.0], help="Deprecated")

    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", action="store_true", help="Use masked data as new keywords for testing")
    p.add_argument("--exp-name", default="backtests", help="Experiment name for output folder")

    args = p.parse_args()

    start_dt, end_dt, budget_list, masked, keywords_n = args.start, args.end, args.budget, args.masked, args.keywords_n
    
    df = pd.read_csv("data/clean/ad_opt_data_bert.csv")
    df = df[df["Region"] != "C"].copy()  # remove region C since no budget allocated to it
    df["Day"] = pd.to_datetime(df["Day"])

    kw_df = pd.read_csv("data/gkp/keywords_classified.csv")

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

    models_dir = Path(f"models/backtests/{args.exp_name}")
    base_results_dir = Path(f"opt_results/backtests/{args.exp_name}")
    cache_dir = base_results_dir / "cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for day in opt_days:
        print(f"\n=== Day {day.date()} ===")

        # Create a deterministic seed from the date (YYYYMMDD)
        seed = int(day.strftime('%Y%m%d'))

        # Select a new set of masked keywords each day
        kw_df_daily, keywords, new_keywords = select_keywords(kw_df, keywords_n, masked, seed=seed)

        # Train model on history up to t-1, excluding new keywords if masked
        history_source = df
        if masked:
            history_source = df[~df['Keyword'].isin(new_keywords)].copy()
        hist = history_source[history_source["Day"] < day].copy()
        pipe = fit_click_model(hist, features=features)
        
        # Calculate and Save Hist Metrics
        hist_m = in_sample_metrics(pipe, hist, features=features)
        metrics_file = models_dir / "hist_model_metrics.csv"
        metrics_row = pd.DataFrame([{
            "Day": day.date(),
            "Hist_MSE": hist_m["MSE"],
            "Hist_R2": hist_m["R2"],
            "Hist_Bias": hist_m["Bias"]
        }])
        if not metrics_file.exists():
            metrics_row.to_csv(metrics_file, index=False)
        else:
            metrics_row.to_csv(metrics_file, mode='a', header=False, index=False)

        # Save training model
        model_path = models_dir / f"xgb_clicks_model_{day.date()}.joblib"
        joblib.dump(pipe, model_path)

        # Precompute feature matrix (shared across parameters)
        X_base = feature_matrix_cached(keywords=keywords, opt_date=day, cache_dir=cache_dir)

        # Optimize for each parameter combination
        for b in budget_list:
            # Define output directory for this run
            run_dir = base_results_dir / f"budget_{int(b)}"
            bids_dir = run_dir / "bids"
            bids_dir.mkdir(parents=True, exist_ok=True)

            # Check if already optimized
            opt_path = bids_dir / f"optimized_costs_{day.date()}.csv"
            if opt_path.exists():
                print(f"Skipping {day.date()} budget={b} - already exists")
                continue

            # Optimize bids for day t
            # Copy X_base to avoid modification issues
            # optimize_bids now only takes budget and kw_df
            m, cost_vars, pred_vars, X_opt = optimize_bids(X_base.copy(), str(model_path), budget=b, kw_df=kw_df_daily)
            sol = extract_solution(m, cost_vars, pred_vars, str(model_path), X_opt)

            # Save optimized costs
            if sol is not None:
                sol.to_csv(opt_path, index=False)

    print(f"\nBacktest complete.")

if __name__ == "__main__":
    main()