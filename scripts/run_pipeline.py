"""
End-to-end bid generation pipeline for tomorrow.
=================================================

Runs the full pipeline: data preparation → model training → optimization →
bid post-processing.  Uses full BERT embeddings (no SVD) by default, matching
the ``ml/backtests/min_spend`` configuration.

Usage:
    python scripts/run_pipeline.py                           # Default: ml course
    python scripts/run_pipeline.py --course sys_think        # Single course
    python scripts/run_pipeline.py --course ml sys_think     # Multiple courses

Defaults (from opt_results/ml/backtests/min_spend/backtest_config.json):
    embedding_method  = bert
    k_policy          = 0  (full BERT, no SVD)
    order_budget      = True
    max_purch         = True
    min_spend         = 1

Pipeline steps:
    1. Data preparation   — tidy_get_data (load, clean, embed keyword data)
    2. Model training     — train XGBoost clicks model on full BERT embeddings
    3. Optimization       — embed model in Gurobi MIP and solve for tomorrow
    4. Post-processing    — compute base bids, bid adjustments, daily budgets

Input files:
    data/<course>/reports/Search keyword - raw input to models.csv
    data/<course>/gkp/Saved Keywords Stats *.csv
    data/<course>/gkp/keywords_classified.csv
    data/<course>/reports/Purchase report.csv          (max-purch mode)
    data/<course>/reports/bid_adj/*.csv                 (bid adjustments)
    config.py

Output files:
    models/<course>_xgb_clicks_model_bert.joblib        Trained model (full BERT)
    models/<course>_svd_pipeline.joblib                  SVD pipeline (normalizer only)
    opt_results/<course>/bids/optimized_costs.csv        Optimal cost allocations + Bid column
    opt_results/<course>/bids/daily_budget.csv           Campaign daily budgets
    opt_results/<course>/bid_adjustments/bid_adj_*.csv   Segment bid adjustments

Estimated run time:
    ~10-20 min per course (HP Spectre x360, i7-1065G7, 16 GB RAM)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COURSE_CONFIG
from scripts.modeling import train_best_model
from scripts.optimization import (
    create_feature_matrix,
    extract_solution,
    optimize_bids,
)
from utils.date_features import COURSE_START_DATES_MAP
from utils.embeddings import (
    fit_svd_pipeline,
    get_raw_bert_embeddings_cached,
    replace_embeddings,
)
from utils import setup_tee_logging


def run_data_preparation(course: str, use_cache: bool = False) -> None:
    """Run tidy_get_data.py as a subprocess for the given course."""
    cmd = [
        sys.executable, "scripts/tidy_get_data.py",
        "--course", course,
        "--embedding-method", "bert",
    ]
    if use_cache:
        cmd.append("--use-cache")

    print(f"\n{'='*70}")
    print(f"[Step 1] Data Preparation — {course}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        raise RuntimeError(
            f"tidy_get_data.py failed for {course} (exit code {result.returncode})"
        )


def train_model(
    course: str,
    embedding_method: str = "bert",
) -> tuple[object, dict, list[str]]:
    """Train XGBoost model on full BERT embeddings (no SVD).

    Returns:
        (model_pipeline, svd_pipeline_dict, feature_names)
    """
    print(f"\n{'='*70}")
    print(f"[Step 2] Model Training — {course} (full BERT, no SVD)")
    print(f"{'='*70}")

    base_dir = Path(f"data/{course}")
    data_file = base_dir / f"clean/ad_opt_data_{embedding_method}.csv"

    df = pd.read_csv(data_file)
    df = df[df["Region"] != "C"].copy()
    print(f"Loaded {len(df)} rows from {data_file}")

    # Base (non-embedding) features
    features_base = [
        "Match type", "Region", "day_of_week", "is_weekend", "month",
        "is_public_holiday", "days_to_next_course_start", "last_month_searches",
        "three_month_avg", "six_month_avg", "mom_change", "search_trend",
        "Competition (indexed value)", "Top of page bid (low range)",
        "Top of page bid (high range)", "Cost",
    ]

    # Get raw BERT embeddings and fit normalizer-only pipeline (k=None → no SVD)
    all_keywords = df["Keyword"].unique().tolist()
    raw_emb_cache = base_dir / "cache" / "raw_bert_embeddings.pkl"
    raw_emb_map = get_raw_bert_embeddings_cached(all_keywords, cache_path=raw_emb_cache)
    print(f"Raw BERT embeddings: {len(raw_emb_map)} keywords")

    raw_matrix = np.array([raw_emb_map[kw] for kw in all_keywords])
    svd_pipeline = fit_svd_pipeline(raw_matrix, n_components=None)
    actual_dim = svd_pipeline["n_components"]
    print(f"Embedding dim: {actual_dim} (no SVD, L2-normalised)")

    # Replace SVD-reduced embedding columns with full BERT
    df, emb_cols = replace_embeddings(df, raw_emb_map, svd_pipeline)
    features = features_base + emb_cols

    print(f"Training on {len(df)} rows, {len(features)} features …")

    pipe, best_params, cv_mse, in_mse, in_r2, in_bias = train_best_model(
        df, features=features, day_date=None,
    )

    y_var = df["Clicks"].var()
    cv_r2 = 1 - cv_mse / y_var if y_var > 0 else float("nan")

    print(f"Best params:   {best_params}")
    print(f"CV MSE / R²:   {cv_mse:.4f} / {cv_r2:.4f}")
    print(f"In-sample:     MSE={in_mse:.4f}  R²={in_r2:.4f}  Bias={in_bias:.4f}")

    # Save model and SVD pipeline
    model_path = Path(f"models/{course}_xgb_clicks_model_{embedding_method}.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Saved model → {model_path}")

    svd_path = Path(f"models/{course}_svd_pipeline.joblib")
    joblib.dump(svd_pipeline, svd_path)
    print(f"Saved SVD pipeline (normalizer) → {svd_path}")

    return pipe, svd_pipeline, features, raw_emb_map


def run_optimization(
    course: str,
    svd_pipeline: dict,
    raw_emb_map: dict,
    opt_date: datetime | None = None,
    budget: float | None = None,
    order_budget: bool = True,
    max_purch: bool = True,
    min_spend: float | None = 1.0,
    embedding_method: str = "bert",
    time_limit: float | None = None,
) -> pd.DataFrame | None:
    """Build feature matrix for *opt_date* and solve the Gurobi MIP.

    Returns:
        DataFrame of optimized_costs or None if solve failed.
    """
    if opt_date is None:
        opt_date = datetime.now() + timedelta(days=1)
    if budget is None:
        budget = calculate_daily_budget(course, opt_date=opt_date)

    print(f"\n{'='*70}")
    print(f"[Step 3] Optimization — {course}")
    print(f"{'='*70}")
    print(f"Date:         {opt_date.date()}")
    print(f"Budget:       ${budget:.2f}")
    print(f"Order budget: {order_budget}")
    print(f"Max purch:    {max_purch}")
    print(f"Min spend:    {min_spend}")
    print(f"Time limit:   {time_limit}")

    base_dir = Path(f"data/{course}")
    kw_df = pd.read_csv(base_dir / "gkp/keywords_classified.csv")
    keywords = kw_df["Keyword"].tolist()

    # Ensure raw_emb_map covers optimization keywords (may include new ones
    # not seen during training)
    missing = [kw for kw in keywords if kw not in raw_emb_map]
    if missing:
        print(f"[Info] Encoding {len(missing)} new keywords not in training data …")
        raw_emb_cache = base_dir / "cache" / "raw_bert_embeddings.pkl"
        raw_emb_map = get_raw_bert_embeddings_cached(
            list(raw_emb_map.keys()) + missing, cache_path=raw_emb_cache,
        )

    # Create feature matrix with full BERT via the SVD pipeline (normalizer)
    X = create_feature_matrix(
        keywords,
        opt_date=opt_date,
        course_start_dts=COURSE_START_DATES_MAP.get(course, []),
        base_dir=base_dir,
        embedding_method=embedding_method,
        course=course,
        raw_emb_map=raw_emb_map,
        svd_pipeline=svd_pipeline,
    )
    X = X[X["Region"] != "C"]

    # Save feature matrix for debugging
    debug_dir = Path(f"opt_results/{course}/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"feature_matrix_{opt_date.strftime('%Y-%m-%d')}.csv"
    X.to_csv(debug_path, index=False)
    print(f"[Debug] Saved feature matrix ({len(X)} rows) → {debug_path}")

    # Validate: no NaNs allowed in feature matrix
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        nan_counts = X[nan_cols].isna().sum().to_dict()
        raise ValueError(
            f"Feature matrix contains NaN values — cannot embed into Gurobi.\n"
            f"Columns with NaNs: {nan_counts}\n"
            f"Saved matrix for inspection at: {debug_path}"
        )

    model_path = f"models/{course}_xgb_clicks_model_{embedding_method}.joblib"

    model, cost_vars, pred_vars, X_opt = optimize_bids(
        X.copy(),
        model_path,
        budget=budget,
        kw_df=kw_df,
        order_budget=order_budget,
        max_purch=max_purch,
        base_dir=base_dir,
        min_spend=min_spend,
        time_limit=time_limit,
    )

    results_df = extract_solution(model, cost_vars, pred_vars, model_path, X_opt)

    if results_df is not None:
        res_dir = Path(f"opt_results/{course}/bids")
        res_dir.mkdir(parents=True, exist_ok=True)
        out_path = res_dir / "optimized_costs.csv"
        results_df.to_csv(out_path, index=False)
        print(f"Saved optimization results → {out_path}")

    return results_df


def run_post_processing(
    course: str,
    bid_multiplier: float = 1.3,
    skip_adjustments: bool = False,
) -> None:
    """Run bid_post_processing.py as a subprocess."""
    print(f"\n{'='*70}")
    print(f"[Step 4] Post-Processing — {course}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "scripts/bid_post_processing.py",
        "--course", course,
        "--bid-multiplier", str(bid_multiplier),
    ]
    if skip_adjustments:
        cmd.append("--skip-adjustments")

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        raise RuntimeError(
            f"bid_post_processing.py failed for {course} "
            f"(exit code {result.returncode})"
        )


def calculate_daily_budget(course: str, opt_date: datetime | None = None) -> float:
    from config import COURSE_CONFIG
    from pathlib import Path
    import pandas as pd
    
    config = COURSE_CONFIG.get(course, {})
    campaign_budget = config.get('campaign_budget')
    start_date_str = config.get('current_campaign_start_date')
    end_date_str = config.get('current_campaign_end_date')
    
    if campaign_budget is None or not start_date_str or not end_date_str:
        raise ValueError(f"Missing campaign_budget, current_campaign_start_date, or current_campaign_end_date for {course}. Please check config.")
        
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    if opt_date is None:
        opt_date = datetime.now()
        
    if opt_date > end_date:
        raise ValueError(f"Optimization date ({opt_date.date()}) is after campaign end_date ({end_date.date()}).")
    
    number_of_days_in_campaign = (end_date - start_date).days + 1
    if number_of_days_in_campaign <= 0:
        raise ValueError(f"Campaign start date ({start_date.date()}) is after end date ({end_date.date()}).")
        
    days_remaining = (end_date - opt_date).days + 1
    if days_remaining <= 0:
        raise ValueError(f"Days remaining is {days_remaining}, but should be > 0.")
        
    budget_used_from_report = 0.0
    raw_input_path = Path(f"data/{course}/reports/Search keyword - raw input to models.csv")
    
    if raw_input_path.exists():
        try:
            df = pd.read_csv(raw_input_path, header=0, thousands=',', engine='python')
            if 'Day' in df.columns:
                df['Day'] = pd.to_datetime(df['Day'])
                # Filter between start_date and opt_date - 1 (inclusive)
                df = df[(df['Day'] >= start_date) & (df['Day'] < opt_date)]
            if 'Campaign' in df.columns:
                df = df[df['Campaign'].str.contains('Experiment', case=False, na=False)]
            if 'Cost' in df.columns:
                budget_used_from_report = df['Cost'].fillna(0).sum()
        except Exception as e:
            print(f"[Warning] Could not read Cost from {raw_input_path}: {e}")
            
    # The additional term (1/8 * campaign budget / num days) accounts for delays in reporting (up to 3 hours).
    budget_used = budget_used_from_report + (campaign_budget / 8.0) / number_of_days_in_campaign

    if campaign_budget - budget_used <= 0:
        raise ValueError(f"[Warning] No more campaign budget remaining for {course}. Campaign Budget: ${campaign_budget:.2f}, Used: ${budget_used:.2f}")

    # / 2 is because Google Ads can spend up to 2x the daily budget in a given day.
    daily_budget = min(
        (campaign_budget - budget_used) / days_remaining,
        (campaign_budget - budget_used) / 2.0
    )
    
    return max(0.0, daily_budget)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end bid generation pipeline for tomorrow. "
            "Defaults match opt_results/ml/backtests/min_spend/backtest_config.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--course", nargs="+", default=["ml"],
        help="Course(s) to process (default: ml)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Optimization date YYYY-MM-DD (default: tomorrow)",
    )
    parser.add_argument(
        "--budget", type=float, default=None,
        help="Total daily budget (default: from config.py)",
    )
    parser.add_argument(
        "--bid-multiplier", type=float, default=1.3,
        help="Bid multiplier (default: 1.3)",
    )
    parser.add_argument(
        "--min-spend", type=float, default=1.0,
        help="Minimum spend per active keyword (default: 1.0)",
    )
    parser.add_argument(
        "--no-order-budget", action="store_true",
        help="Disable B_USA >= B_A >= B_B constraint",
    )
    parser.add_argument(
        "--no-max-purch", action="store_true",
        help="Maximize clicks instead of expected purchases",
    )
    parser.add_argument(
        "--skip-adjustments", action="store_true",
        help="Skip bid adjustment calculation",
    )
    parser.add_argument(
        "--skip-data-prep", action="store_true",
        help="Skip data preparation (use existing clean data)",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Use cached intermediate data (ignoring cache by default)",
    )
    parser.add_argument(
        "--time-limit", type=float, default=None,
        help="Time limit for Gurobi optimization in seconds (default: no limit)",
    )

    args = parser.parse_args()

    # Resolve optimization date
    if args.date:
        opt_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        opt_date = datetime.now() + timedelta(days=1)

    log_path = setup_tee_logging(
        log_file=None,
        default_log_dir="logs",
        default_log_prefix="run_pipeline",
    )

    print("=" * 70)
    print("Ad Bid Optimizer — End-to-End Pipeline")
    print("=" * 70)
    print(f"Courses:        {args.course}")
    print(f"Opt date:       {opt_date.date()}")
    print(f"Budget:         {args.budget or 'from config.py'}")
    print(f"Bid multiplier: {args.bid_multiplier}")
    print(f"Min spend:      {args.min_spend}")
    print(f"Order budget:   {not args.no_order_budget}")
    print(f"Max purchases:  {not args.no_max_purch}")
    print(f"Skip data prep: {args.skip_data_prep}")
    print(f"Log file:       {log_path}")
    print(f"Time limit:     {args.time_limit}")
    print("=" * 70)

    for course in args.course:
        print(f"\n{'#'*70}")
        print(f"#  COURSE: {course.upper()}")
        print(f"{'#'*70}")

        budget = args.budget or calculate_daily_budget(course, opt_date=opt_date)

        # ── Step 1: Data Preparation ────────────────────────────────────
        if not args.skip_data_prep:
            run_data_preparation(course, use_cache=args.use_cache)
        else:
            print(f"\n[Step 1] Skipping data preparation (--skip-data-prep)")

        # ── Step 2: Model Training (full BERT, no SVD) ──────────────────
        pipe, svd_pipeline, features, raw_emb_map = train_model(course)

        # ── Step 3: Optimization ────────────────────────────────────────
        results = run_optimization(
            course,
            svd_pipeline=svd_pipeline,
            raw_emb_map=raw_emb_map,
            opt_date=opt_date,
            budget=budget,
            order_budget=not args.no_order_budget,
            max_purch=not args.no_max_purch,
            min_spend=args.min_spend,
            time_limit=args.time_limit,
        )

        if results is None:
            print(f"\n[Warning] Optimization failed for {course}, skipping post-processing.")
            continue

        # ── Step 4: Post-Processing ─────────────────────────────────────
        run_post_processing(
            course,
            bid_multiplier=args.bid_multiplier,
            skip_adjustments=args.skip_adjustments,
        )

    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
