from __future__ import annotations
"""Daily backtest - generate optimal solutions.

Example Usage:
    python scripts/backtest_daily.py --start 2025-12-01 --end 2025-12-31 --exp-name exp1 --masked
    python scripts/backtest_daily.py --start 2025-12-01 --end 2025-12-31 --exp-name svd_sweep --k-policy 10 20 50 100

For each candidate k_policy (SVD dimensionality):
  For each day t:
  - Fit SVD(k_policy) on keywords from Day 0 … t-1  (no data leakage).
  - Train model on the re-embedded history.
  - Optimise to find (x^t)*.
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
from utils.embeddings import (
    get_raw_bert_embeddings_cached,
    fit_svd_pipeline,
    replace_embeddings,
)
from config import COURSE_CONFIG


def calculate_dynamic_daily_budget(
    *,
    campaign_budget: float,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    opt_date: pd.Timestamp,
    bids_dir: Path,
    buffer_fraction: float = 1.0 / 8.0,
) -> float:
    """Compute a day-specific budget from prior optimized spend.

    Mirrors the logic used by run_pipeline.py, but treats the spend from
    previous optimized days as the source of truth instead of the raw report.
    """
    if opt_date > end_dt:
        raise ValueError(f"Optimization date ({opt_date.date()}) is after campaign end_date ({end_dt.date()}).")

    total_days = (end_dt - start_dt).days + 1
    if total_days <= 0:
        raise ValueError(f"Campaign start date ({start_dt.date()}) is after end date ({end_dt.date()}).")

    days_remaining = (end_dt - opt_date).days + 1
    if days_remaining <= 0:
        raise ValueError(f"Days remaining is {days_remaining}, but should be > 0.")

    budget_used_from_history = 0.0
    for prior_file in sorted(bids_dir.glob("optimized_costs_*.csv")):
        try:
            prior_day_str = prior_file.stem.replace("optimized_costs_", "")
            prior_day = pd.to_datetime(prior_day_str)
        except Exception:
            continue

        if prior_day < opt_date:
            prior_df = pd.read_csv(prior_file)
            if "Optimal Cost" in prior_df.columns:
                budget_used_from_history += pd.to_numeric(prior_df["Optimal Cost"], errors="coerce").fillna(0.0).sum()

    budget_used = budget_used_from_history + (campaign_budget / 8.0) / total_days

    if campaign_budget - budget_used <= 0:
        raise ValueError(
            f"[Warning] No more campaign budget remaining. Campaign Budget: ${campaign_budget:.2f}, Used: ${budget_used:.2f}"
        )

    daily_budget = min(
        (campaign_budget - budget_used) / days_remaining,
        (campaign_budget - budget_used) / 2.0,
    )
    return max(0.0, daily_budget)


def write_empty_optimized_costs(opt_path: Path, template_df: pd.DataFrame) -> None:
    """Write a headered empty optimized-costs file for downstream scripts."""
    empty_df = template_df.head(0).copy()
    for col in ["Optimal Cost", "Gurobi Pred", "Gurobi Pred over Base", "Actual Model Pred", "Diff"]:
        if col not in empty_df.columns:
            empty_df[col] = pd.Series(dtype=float)

    opt_path.parent.mkdir(parents=True, exist_ok=True)
    empty_df.to_csv(opt_path, index=False)
    print(f"  Saved empty placeholder to {opt_path}")


def feature_matrix_cached(
    *,
    keywords: list[str],
    opt_date: pd.Timestamp,
    cache_dir: Path,
    base_dir: Path,
    course_start_dts: list,
    embedding_method: str = 'bert',
    course: str = 'gen_ai',
    raw_emb_map: dict | None = None,
    svd_pipeline: dict | None = None,
    k_policy: int | None = None,
) -> pd.DataFrame:
    """
    Create or load cached feature matrix.

    When *raw_emb_map* and *svd_pipeline* are supplied the embeddings are
    computed on-the-fly via the provided SVD pipeline (no leakage).  The
    cache key includes *k_policy* so different dimensionalities are stored
    independently.
    """
    kw_hash = hashlib.md5("|".join(sorted(keywords)).encode("utf-8")).hexdigest()[:10]
    k_suffix = f"_k{k_policy}" if k_policy is not None else ""
    p = cache_dir / f"feature_matrix_{embedding_method}_{kw_hash}_{opt_date.date()}{k_suffix}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    X = create_feature_matrix(
        keywords,
        opt_date=opt_date,
        course_start_dts=course_start_dts,
        base_dir=base_dir,
        embedding_method=embedding_method,
        course=course,
        raw_emb_map=raw_emb_map,
        svd_pipeline=svd_pipeline,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(p)
    return X


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
    p.add_argument("--budget", type=float, nargs='+', default=None, help="Total budgets to test")

    p.add_argument("--keywords-n", type=int, default=None)
    p.add_argument("--masked", action="store_true", help="Use masked data as new keywords for testing")
    p.add_argument("--mask-frac", type=float, default=0.1, help="Fraction of keywords to mask as new")
    p.add_argument("--order-budget", action="store_true", help="Use B_{USA} >= B_{A} >= B_{B}")
    p.add_argument("--max-purch", action="store_true", help="Use max purchases objective instead of clicks")
    p.add_argument("--min-spend", type=float, default=None, help="Minimum spend per active keyword (e.g. 1.0)")
    p.add_argument("--dynamic-budget", action="store_true", help="Recompute the daily budget from prior optimized spend")
    p.add_argument("--campaign-budget", type=float, default=None, help="Total campaign budget used when --dynamic-budget is set")
    p.add_argument("--exp-name", default="backtests", help="Experiment name for output folder")
    p.add_argument("--course", default="gen_ai", help="Course name")
    p.add_argument("--embedding-method", default="bert", choices=["bert", "llm"], help="Embedding method: bert or llm (default: bert)")
    p.add_argument("--k-policy", type=int, nargs="+", default=[50],
                   help="SVD component counts to sweep (default: 50). "
                        "Use 0 for full BERT embeddings (no SVD). "
                        "Each value runs a full backtest with daily SVD fitting.")

    args = p.parse_args()

    if args.dynamic_budget:
        if args.campaign_budget is None:
            raise ValueError("--campaign-budget is required when --dynamic-budget is set")
        args.budget = [args.campaign_budget]
    elif args.budget is None:
        try:
            from scripts.run_pipeline import calculate_daily_budget
            args.budget = [calculate_daily_budget(args.course)]
        except ImportError as e:
            raise ImportError("Could not import calculate_daily_budget. Please check config.") from e

    start_dt, end_dt, budget_list, masked, keywords_n, order_budget, mask_frac, max_purch = args.start, args.end, args.budget, args.masked, args.keywords_n, args.order_budget, args.mask_frac, args.max_purch
    min_spend = args.min_spend
    embedding_method = args.embedding_method
    # Map sentinel 0 → None (no SVD, full BERT embeddings)
    k_policy_list = [None if k == 0 else k for k in args.k_policy]

    base_dir = Path(f"data/{args.course}")

    # Load data based on embedding method
    data_file = base_dir / f"clean/ad_opt_data_{embedding_method}.csv"
    df = pd.read_csv(data_file)
    df = df[df["Region"] != "C"].copy()  # remove region C since no budget allocated to it
    df["Day"] = pd.to_datetime(df["Day"])

    kw_df = pd.read_csv(base_dir / "gkp/keywords_classified.csv")

    if args.day is not None:
        opt_days = [pd.to_datetime(args.day)]
    else:
        opt_days = list(pd.date_range(start=start_dt, end=end_dt, freq="D"))

    # ── Base (non-embedding) features ───────────────────────────────────
    features_base = [
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
    ]

    # For LLM embedding method, SVD sweep is not applicable
    if embedding_method == "llm":
        llm_cols = ["llm_relevance_score"] if "llm_relevance_score" in df.columns else []
        features = features_base + llm_cols
        k_policy_list = [None]  # single pass, no SVD

    # ── Pre-compute raw BERT embeddings (no leakage – BERT is frozen) ──
    raw_emb_map: dict | None = None
    if embedding_method == "bert":
        all_keywords = list(
            set(df["Keyword"].unique().tolist())
            | set(kw_df["Keyword"].unique().tolist())
        )
        raw_emb_cache = base_dir / "cache" / "raw_bert_embeddings.pkl"
        raw_emb_map = get_raw_bert_embeddings_cached(
            all_keywords, cache_path=raw_emb_cache,
        )
        print(f"Raw BERT embeddings: {len(raw_emb_map)} keywords")

    # ── Outer loop: iterate over candidate k_policy values ─────────────
    for k in k_policy_list:
        k_label = k if k is not None else "full"
        print(f"\n{'='*70}")
        print(f"  k_policy = {k_label}")
        print(f"{'='*70}")

        # Directories scoped to this k
        k_suffix = f"/k{k}" if k is not None else "/k_full"
        models_dir = Path(f"models/{args.course}/backtests/{args.exp_name}{k_suffix}")
        base_results_dir = Path(f"opt_results/{args.course}/backtests/{args.exp_name}{k_suffix}")
        cache_dir = base_results_dir / "cache"
        models_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        budget_exhausted = {b: False for b in budget_list}

        for day in opt_days:
            print(f"\n=== Day {day.date()} | k={k_label} ===")

            # Create a deterministic seed from the date (YYYYMMDD)
            seed = int(day.strftime('%Y%m%d'))

            # Select a new set of masked keywords each day
            kw_df_daily, keywords, new_keywords = select_keywords(kw_df, keywords_n, masked, mask_frac=mask_frac, seed=seed)

            # Train model on history up to t-1, excluding new keywords if masked
            history_source = df
            if masked:
                history_source = df[~df['Keyword'].isin(new_keywords)].copy()
            hist = history_source[history_source["Day"] < day].copy()

            # ── Daily SVD: fit on keywords known up to day t ────────────
            daily_svd = None
            if embedding_method == "bert" and raw_emb_map is not None:
                hist_keywords = hist["Keyword"].unique()
                raw_matrix = np.array([raw_emb_map[kw] for kw in hist_keywords])
                daily_svd = fit_svd_pipeline(raw_matrix, n_components=k)
                actual_k = daily_svd['n_components']
                print(f"  SVD fit on {len(hist_keywords)} hist keywords -> {actual_k}D")

                # Replace embedding columns in training data
                hist, emb_cols = replace_embeddings(hist, raw_emb_map, daily_svd)
                features = features_base + emb_cols
            elif embedding_method == "llm":
                # features already set above; hist already has llm columns
                pass
            else:
                raise ValueError(
                    "BERT mode requires raw_emb_map for runtime embedding replacement. "
                    "Ensure raw BERT cache is available."
                )

            # Train best model using CV
            pipe, best_params, best_cv, hist_mse, hist_r2, hist_bias = train_best_model(hist, features=features, day_date=day)

            # Calculate and Save Hist Metrics
            metrics_file = models_dir / "hist_model_metrics.csv"
            metrics_row = pd.DataFrame([{
                "Day": day.date(),
                "k_policy": k_label,
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

            # Also persist the daily SVD pipeline for eval
            if daily_svd is not None:
                svd_path = models_dir / f"svd_pipeline_{day.date()}.joblib"
                joblib.dump(daily_svd, svd_path)

            # Precompute feature matrix (shared across budgets)
            X_base = feature_matrix_cached(
                keywords=keywords,
                opt_date=day,
                cache_dir=cache_dir,
                base_dir=base_dir,
                course_start_dts=COURSE_START_DATES_MAP.get(args.course, []),
                embedding_method=embedding_method,
                course=args.course,
                raw_emb_map=raw_emb_map,
                svd_pipeline=daily_svd,
                k_policy=k,
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

                if budget_exhausted[b]:
                    write_empty_optimized_costs(opt_path, X_base)
                    continue

                # Recompute the daily budget from prior optimized spend if requested.
                daily_budget = b
                if args.dynamic_budget:
                    try:
                        daily_budget = calculate_dynamic_daily_budget(
                            campaign_budget=b,
                            start_dt=pd.to_datetime(start_dt),
                            end_dt=pd.to_datetime(end_dt),
                            opt_date=day,
                            bids_dir=bids_dir,
                        )
                        print(f"  Dynamic budget for {day.date()}: ${daily_budget:.2f}")
                    except ValueError as exc:
                        if "No more campaign budget remaining" in str(exc):
                            print(f"  {exc}")
                            write_empty_optimized_costs(opt_path, X_base)
                            budget_exhausted[b] = True
                            continue
                        raise

                if daily_budget <= 0:
                    print(f"  Budget exhausted for {day.date()} (budget={b}); writing empty placeholder.")
                    write_empty_optimized_costs(opt_path, X_base)
                    budget_exhausted[b] = True
                    continue

                # Optimize bids for day t
                m, cost_vars, pred_vars, X_opt = optimize_bids(
                    X_base.copy(),
                    str(model_path),
                    budget=daily_budget,
                    kw_df=kw_df_daily,
                    order_budget=order_budget,
                    max_purch=max_purch,
                    base_dir=base_dir,
                    min_spend=min_spend,
                )
                sol = extract_solution(m, cost_vars, pred_vars, str(model_path), X_opt)

                # Save optimized costs
                if sol is not None:
                    sol.to_csv(opt_path, index=False)

    print(f"\nBacktest complete.")

if __name__ == "__main__":
    main()