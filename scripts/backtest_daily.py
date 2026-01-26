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
from scripts.modeling import _to_float32_csr, train_best_model
from utils.date_features import COURSE_START_DATES_MAP


def feature_matrix_cached(*, keywords: list[str], opt_date: pd.Timestamp, cache_dir: Path, base_dir: Path, course_start_dts: list) -> pd.DataFrame:
    kw_hash = hashlib.md5("|".join(sorted(keywords)).encode("utf-8")).hexdigest()[:10]
    p = cache_dir / f"feature_matrix_{kw_hash}_{opt_date.date()}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    X = create_feature_matrix(keywords, opt_date=opt_date, course_start_dts=course_start_dts, base_dir=base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(p)
    return X
    # return X


def select_keywords(kw_df, keywords_n, masked, mask_frac=0.1, seed=None):
    """ Select keywords for backtesting, optionally masking some as "new" keywords."""
    
    if masked:
        kw_df = kw_df[kw_df["Origin"] == "existing"].copy()

        # Randomly select some existing keywords to be "new" for testing
        # Use a deterministic seed if provided
        rng = np.random.default_rng(seed)
        
        existing_keywords = kw_df["Keyword"].tolist()
        n_new = round(mask_frac * len(existing_keywords))  # mask_frac as new
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

    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", action="store_true", help="Use masked data as new keywords for testing")
    p.add_argument("--mask-frac", type=float, default=0.1, help="Fraction of keywords to mask as new")
    p.add_argument("--order-budget", action="store_true", help="Use B_{USA} >= B_{A} >= B_{B}")
    p.add_argument("--max-conv", action="store_true", help="Use max conversions objective instead of clicks")
    p.add_argument("--exp-name", default="backtests", help="Experiment name for output folder")
    p.add_argument("--course", default="gen_ai", help="Course name")

    args = p.parse_args()

    start_dt, end_dt, budget_list, masked, keywords_n, order_budget, mask_frac, max_conv = args.start, args.end, args.budget, args.masked, args.keywords_n, args.order_budget, args.mask_frac, args.max_conv
    
    base_dir = Path(f"data/{args.course}")

    df = pd.read_csv(base_dir / "clean/ad_opt_data_bert.csv")
    df = df[df["Region"] != "C"].copy()  # remove region C since no budget allocated to it
    df["Day"] = pd.to_datetime(df["Day"])

    kw_df = pd.read_csv(base_dir / "gkp/keywords_classified.csv")

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

    models_dir = Path(f"models/{args.course}/backtests/{args.exp_name}")
    base_results_dir = Path(f"opt_results/{args.course}/backtests/{args.exp_name}")
    cache_dir = base_results_dir / "cache"
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for day in opt_days:
        print(f"\n=== Day {day.date()} ===")

        # Create a deterministic seed from the date (YYYYMMDD)
        seed = int(day.strftime('%Y%m%d'))

        # Select a new set of masked keywords each day
        kw_df_daily, keywords, new_keywords = select_keywords(kw_df, keywords_n, masked, mask_frac=mask_frac, seed=seed)

        # Train model on history up to t-1, excluding new keywords if masked
        history_source = df
        if masked:
            history_source = df[~df['Keyword'].isin(new_keywords)].copy()
        hist = history_source[history_source["Day"] < day].copy()
        
        # Train best model using CV
        pipe, best_params, best_cv, hist_mse, hist_r2, hist_bias = train_best_model(hist, features=features, day_date=day)
        
        # Calculate and Save Hist Metrics
        metrics_file = models_dir / "hist_model_metrics.csv"
        metrics_row = pd.DataFrame([{
            "Day": day.date(),
            "Hist_MSE": hist_mse,
            "Hist_R2": hist_r2,
            "Hist_Bias": hist_bias,
            "CV_Score": best_cv,
            "best_params": best_params
        }])
        if not metrics_file.exists():
            metrics_row.to_csv(metrics_file, index=False)
        else:
            metrics_row.to_csv(metrics_file, mode='a', header=False, index=False)

        # Save training model
        model_path = models_dir / f"xgb_clicks_model_{day.date()}.joblib"
        joblib.dump(pipe, model_path)

        # Precompute feature matrix (shared across parameters)
        X_base = feature_matrix_cached(
            keywords=keywords, 
            opt_date=day, 
            cache_dir=cache_dir, 
            base_dir=base_dir, 
            course_start_dts=COURSE_START_DATES_MAP.get(args.course, [])
        )

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
            m, cost_vars, pred_vars, X_opt = optimize_bids(X_base.copy(), str(model_path), budget=b, kw_df=kw_df_daily, order_budget=order_budget, max_conv=max_conv)
            sol = extract_solution(m, cost_vars, pred_vars, str(model_path), X_opt)

            # Save optimized costs
            if sol is not None:
                sol.to_csv(opt_path, index=False)

    print(f"\nBacktest complete.")

if __name__ == "__main__":
    main()