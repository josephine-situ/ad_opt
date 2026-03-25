from __future__ import annotations
"""Evaluates backtest results using a Gold-Standard Oracle model.

The Oracle is trained on full data (0:T) with CV over SVD dimensionality k.
The best k_eval is selected, and the Oracle model + its SVD transform are
frozen.  Each daily solution from the policy backtest is then scored by
transforming its features through the Oracle SVD and predicting with the
frozen Oracle model.

Example Usage:
    python scripts/backtest_eval.py --course gen_ai --exp-name exp1 --masked
    python scripts/backtest_eval.py --course gen_ai --exp-name svd_sweep --k-policy 10 20 50 100
"""

import json
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
from scripts.modeling import _to_float32_csr, train_best_model, train_oracle_model
from scripts.backtest_daily import feature_matrix_cached, select_keywords
from utils.data_pipeline import format_keyword_data, get_conversion_rates
from utils.date_features import COURSE_START_DATES_MAP
from utils.embeddings import (
    get_raw_bert_embeddings_cached,
    replace_embeddings,
)
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
    p.add_argument("--embedding-method", default="bert", choices=["bert", "llm"],
                   help="Embedding method: bert or llm (default: bert)")
    p.add_argument("--k-policy", type=int, nargs="+", default=[50],
                   help="Policy SVD dims that were used in the backtest (default: 50). "
                        "Use 0 for full BERT embeddings (no SVD). "
                        "Results will be read from the corresponding k<N> subdirectories.")
    p.add_argument("--k-candidates", type=int, nargs="+", default=[10, 20, 50, 100, 384],
                   help="SVD dim candidates for Oracle CV (default: 10 20 50 100 384).")

    args = p.parse_args()
    if args.budget is None:
        try:
            from scripts.run_pipeline import calculate_daily_budget
            args.budget = [calculate_daily_budget(args.course)]
        except ImportError as e:
            raise ImportError("Could not import calculate_daily_budget. Please check config.") from e
    # Map sentinel 0 → None (no SVD, full BERT embeddings)
    args.k_policy = [None if k == 0 else k for k in args.k_policy]
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

    # Load Base Data
    embedding_method = args.embedding_method
    data_file = base_dir / f"clean/ad_opt_data_{embedding_method}.csv"
    df = pd.read_csv(data_file)

    if args.course != "sys_eng":
        df = df[df["Region"] != "C"].copy()
    df["Day"] = pd.to_datetime(df["Day"])

    kw_df_all = pd.read_csv(base_dir / "gkp/keywords_classified.csv")

    # Merge Origin into Main Data (Actuals)
    df = df.merge(kw_df_all[["Keyword", "Origin"]], on="Keyword", how="left")

    # ── Base (non-embedding) features ─────────────────────────────────
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

    eval_models_dir = Path(f"opt_results/{args.course}/eval_models")
    eval_models_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-compute raw BERT embeddings ───────────────────────────────
    raw_emb_map: dict | None = None
    if embedding_method == "bert":
        all_keywords = list(
            set(df["Keyword"].unique().tolist())
            | set(kw_df_all["Keyword"].unique().tolist())
        )
        raw_emb_cache = base_dir / "cache" / "raw_bert_embeddings.pkl"
        raw_emb_map = get_raw_bert_embeddings_cached(
            all_keywords, cache_path=raw_emb_cache,
        )
        print(f"Raw BERT embeddings: {len(raw_emb_map)} keywords")

    # ── 1. Train / load Gold-Standard Oracle (f^{0:T}) ────────────────
    oracle_model_path = eval_models_dir / f"oracle_model_{embedding_method}.joblib"
    oracle_svd_path = eval_models_dir / f"oracle_svd_pipeline_{embedding_method}.joblib"
    oracle_features_path = eval_models_dir / f"oracle_features_{embedding_method}.json"
    oracle_cv_path = eval_models_dir / f"oracle_cv_results_{embedding_method}.json"

    if oracle_model_path.exists():
        print("Loading cached Oracle model ...")
        model = joblib.load(oracle_model_path)
        oracle_svd = joblib.load(oracle_svd_path) if oracle_svd_path.exists() else None
        oracle_features = json.loads(oracle_features_path.read_text()) if oracle_features_path.exists() else None
        oracle_cv = json.loads(oracle_cv_path.read_text()) if oracle_cv_path.exists() else {}

        # Derive e_metrics from CV results for summary rows
        e_metrics = {"Oracle_k": oracle_svd['n_components'] if oracle_svd else None}
        e_metrics.update({f"CV_MSE_k{k}": v for k, v in oracle_cv.items()})
    else:
        if embedding_method == "bert" and raw_emb_map is not None:
            print("Training Oracle model with CV over k ...")
            model, oracle_svd, best_k, oracle_cv, oracle_features = train_oracle_model(
                df, features_base, raw_emb_map,
                k_candidates=tuple(args.k_candidates),
            )
            joblib.dump(model, oracle_model_path)
            joblib.dump(oracle_svd, oracle_svd_path)
            oracle_features_path.write_text(json.dumps(oracle_features))
            oracle_cv_path.write_text(json.dumps({str(k): v for k, v in oracle_cv.items()}))

            e_metrics = {"Oracle_k": oracle_svd['n_components']}
            e_metrics.update({f"CV_MSE_k{k}": v for k, v in oracle_cv.items()})
        else:
            # LLM or fallback: train simple full-data model (no SVD sweep)
            print("Training full eval model (LLM / no SVD) ...")
            llm_cols = ["llm_relevance_score"] if "llm_relevance_score" in df.columns else []
            oracle_features = features_base + llm_cols
            oracle_svd = None
            model, params, cv_mse, train_mse, train_r2, _ = train_best_model(
                df, oracle_features, df["Day"].max().date(),
            )
            joblib.dump(model, oracle_model_path)
            joblib.dump(oracle_svd, oracle_svd_path)
            oracle_features_path.write_text(json.dumps(oracle_features))
            oracle_cv = {}
            oracle_cv_path.write_text(json.dumps(oracle_cv))

            e_metrics = {"CV_MSE": cv_mse, "Train_MSE": train_mse,
                         "Train_R2": train_r2, "Best_Params": str(params)}

    print(f"Oracle features ({len(oracle_features)}): {oracle_features[:5]} ...")

    # Get purchase rate by region
    rates_df = get_conversion_rates(base_dir=base_dir)

    k_policy_list = args.k_policy if embedding_method == "bert" else [None]

    # ── Outer loop: evaluate each k_policy backtest ───────────────────
    for k in k_policy_list:
        k_label = k if k is not None else "full"
        k_suffix = f"/k{k}" if k is not None else "/k_full"

        base_results_dir = Path(f"opt_results/{args.course}/backtests/{args.exp_name}{k_suffix}")
        models_dir = Path(f"models/{args.course}/backtests/{args.exp_name}{k_suffix}")
        cache_dir = base_results_dir / "cache"

        # Load Hist Metrics if available
        hist_metrics = {}
        hist_metrics_file = models_dir / "hist_model_metrics.csv"
        if hist_metrics_file.exists():
            hm_df = pd.read_csv(hist_metrics_file)
            hm_df["Day"] = pd.to_datetime(hm_df["Day"])
            for _, row in hm_df.iterrows():
                hist_metrics[row["Day"].date()] = row.to_dict()

        eval_summary_rows = []

        for day in opt_days:
            print(f"\nEvaluating Day: {day.date()} | k_policy={k_label}")
            obs = df[df["Day"] == day].copy()

            # Re-embed actuals with Oracle SVD
            if oracle_svd is not None and raw_emb_map is not None:
                obs, _ = replace_embeddings(obs, raw_emb_map, oracle_svd)

            # 2. Evaluate Actuals
            obs_zero = obs.copy()
            obs_zero["Cost"] = 0.0

            pred_act_clicks = model.predict(obs[oracle_features])
            pred_act_base = model.predict(obs_zero[oracle_features])
            print(f"  Actual Clicks Prediction: {pred_act_clicks.sum():.2f}, "
                  f"Base Prediction: {pred_act_base.sum():.2f}, "
                  f"Observed: {obs['Clicks'].sum():.2f}")
            val_act_diff = pred_act_clicks - pred_act_base
            val_act_clicks = val_act_diff.sum()

            # Prepare Actuals DF for country breakdown
            obs_breakdown = obs[["Keyword", "Region", "Origin", "Match type", "Cost"]].copy()
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
                course_start_dts=COURSE_START_DATES_MAP.get(args.course, []),
                embedding_method=embedding_method,
                course=args.course,
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

                # Re-embed with Oracle SVD for Gold-Standard evaluation
                if oracle_svd is not None and raw_emb_map is not None:
                    X_day, _ = replace_embeddings(X_day, raw_emb_map, oracle_svd)

                X_day["Cost"] = X_day["Optimal Cost"]

                # Predict Opt using Oracle
                pred_opt = model.predict(X_day[oracle_features])

                # Baseline (Cost = 0)
                X_day_zero = X_day.copy()
                X_day_zero["Cost"] = 0.0
                pred_opt_base = model.predict(X_day_zero[oracle_features])
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

                    curr_active = sol[sol["Optimal Cost"] > 1e-6].copy()
                    prev_active = prev_sol[prev_sol["Optimal Cost"] > 1e-6].copy()

                    key_cols = ["Keyword", "Region", "Match type"]

                    curr_keys = set(curr_active[key_cols].itertuples(index=False, name=None))
                    prev_keys = set(prev_active[key_cols].itertuples(index=False, name=None))

                    if len(curr_keys) > 0:
                        new_keys = curr_keys - prev_keys
                        pct_new_keywords = len(new_keys) / len(curr_keys)
                    else:
                        pct_new_keywords = 0.0

                    prev_active_sub = prev_active[key_cols + ["Optimal Cost"]].rename(columns={"Optimal Cost": "Prev_Cost"})
                    curr_sub = sol[key_cols + ["Optimal Cost"]].rename(columns={"Optimal Cost": "Curr_Cost"})

                    merged_cost = prev_active_sub.merge(curr_sub, on=key_cols, how="inner")

                    if len(merged_cost) > 0:
                        merged_cost["pct_change"] = abs((merged_cost["Curr_Cost"] - merged_cost["Prev_Cost"]) / merged_cost["Prev_Cost"])
                        avg_cost_change = merged_cost["pct_change"].mean()

                # --- Calculate Predicted Conversions ---
                X_day["t_Clicks_OptCost"] = pred_opt_lift

                opt_clicks_region = X_day.groupby('Region')['t_Clicks_OptCost'].sum().reset_index()
                act_clicks_region = obs_breakdown.groupby('Region')['t_Clicks_ActCost'].sum().reset_index()

                opt_cost_region = X_day.groupby('Region')['Optimal Cost'].sum().reset_index().rename(columns={'Optimal Cost': 'Opt_Cost_Reg'})
                act_cost_region = obs_breakdown.groupby('Region')['Cost'].sum().reset_index().rename(columns={'Cost': 'Act_Cost_Reg'})

                opt_clicks_reg_org = X_day.groupby(['Region', 'Origin'])['t_Clicks_OptCost'].sum().reset_index()
                act_clicks_reg_org = obs_breakdown.groupby(['Region', 'Origin'])['t_Clicks_ActCost'].sum().reset_index()

                df_region = rates_df.merge(opt_clicks_region, on='Region', how='left').rename(columns={'t_Clicks_OptCost': 'Opt_Clicks'})
                df_region = df_region.merge(act_clicks_region, on='Region', how='left').rename(columns={'t_Clicks_ActCost': 'Act_Clicks'})
                df_region = df_region.merge(opt_cost_region, on='Region', how='left').rename(columns={'Opt_Cost_Reg': 'Opt_Spend'})
                df_region = df_region.merge(act_cost_region, on='Region', how='left').rename(columns={'Act_Cost_Reg': 'Act_Spend'})
                df_region = df_region.fillna(0)

                df_region['Opt_Purchases'] = df_region['Opt_Clicks'] * df_region['Purch_rate']
                df_region['Act_Purchases'] = df_region['Act_Clicks'] * df_region['Purch_rate']

                val_opt_purch = df_region['Opt_Purchases'].sum()
                val_act_purch = df_region['Act_Purchases'].sum()

                def calc_group_purchases(clicks_df, click_col, group_col):
                     m = clicks_df.merge(rates_df, on='Region', how='left')
                     m['Purch_rate'] = m['Purch_rate'].fillna(0)
                     m['Predicted_Purch'] = m[click_col] * m['Purch_rate']
                     return m.groupby(group_col)['Predicted_Purch'].sum().to_dict()

                opt_purch_origin_map = calc_group_purchases(opt_clicks_reg_org, 't_Clicks_OptCost', 'Origin')
                act_purch_origin_map = calc_group_purchases(act_clicks_reg_org, 't_Clicks_ActCost', 'Origin')

                opt_clicks_reg_mt = X_day.groupby(['Region', 'Match type'])['t_Clicks_OptCost'].sum().reset_index()
                act_clicks_reg_mt = obs_breakdown.groupby(['Region', 'Match type'])['t_Clicks_ActCost'].sum().reset_index()

                opt_purch_mt_map = calc_group_purchases(opt_clicks_reg_mt, 't_Clicks_OptCost', 'Match type')
                act_purch_mt_map = calc_group_purchases(act_clicks_reg_mt, 't_Clicks_ActCost', 'Match type')

                breakdown_file = run_dir / f"region_breakdown_{day.strftime('%Y-%m-%d')}.csv"
                df_region[['Region', 'Opt_Purchases', 'Act_Purchases', 'Opt_Clicks', 'Act_Clicks', 'Opt_Spend', 'Act_Spend']].to_csv(breakdown_file, index=False)

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

                if "Gurobi Pred over Base" in sol.columns:
                    tm1_Clicks_OptCost = sol["Gurobi Pred over Base"].sum()
                elif "t_Clicks_OptCost" in sol.columns:
                    tm1_Clicks_OptCost = 0.0
                else:
                    tm1_Clicks_OptCost = 0.0

                if act_file.exists():
                    act_df_existing = pd.read_csv(act_file)
                    if "t_Clicks_ActCost" not in act_df_existing.columns:
                        obs_out = obs[["Keyword", "Region", "Match type", "Cost", "Clicks"]].copy()
                        obs_out["t_Clicks_ActCost"] = val_act_diff
                        obs_out.to_csv(act_file, index=False)

                hm = hist_metrics.get(day.date(), {})

                # --- Dynamic Compilation of Summary Row ---
                row_dict = {
                    "Day": day.date(),
                    "k_policy": k_label,
                    "Budget": int(b),
                    "t_Clicks_OptCost": val_opt_clicks,
                    "t_Clicks_ActCost": val_act_clicks,
                    "tm1_Clicks_OptCost": tm1_Clicks_OptCost,
                    "Actual_Clicks": real_act_clicks,
                    "Opt_Cost": val_opt_cost,
                    "Act_Cost": act_cost,
                    "Opt_Purch": val_opt_purch,
                    "Act_Purch": val_act_purch,
                    "N_Opt": len(X_day),
                    "N_Obs": len(obs),
                    "Avg_Cost_Change": avg_cost_change,
                    "Pct_New_Keywords": pct_new_keywords,

                    # Oracle Metrics
                    "Oracle_k": e_metrics.get("Oracle_k"),

                    # Hist Model Metrics
                    "Hist_MSE": hm.get("Hist_MSE"),
                    "Hist_R2": hm.get("Hist_R2"),
                    "Hist_Bias": hm.get("Hist_Bias"),
                }

                # Add Region/Origin Breakdowns Dynamically
                all_regions = set(X_day['Region'].unique()) | set(obs['Region'].unique())
                for reg in all_regions:
                    row_dict[f"Opt_Cost_Region_{reg}"] = X_day[X_day['Region'] == reg]['Optimal Cost'].sum()
                    row_dict[f"Act_Cost_Region_{reg}"] = obs[obs['Region'] == reg]['Cost'].sum()
                    row_dict[f"Opt_Clicks_Region_{reg}"] = X_day[X_day['Region'] == reg]['t_Clicks_OptCost'].sum()
                    row_dict[f"Act_Clicks_Region_{reg}"] = obs_breakdown[obs_breakdown['Region'] == reg]['t_Clicks_ActCost'].sum()
                    row_dict[f"Opt_Purch_Region_{reg}"] = df_region[df_region['Region'] == reg]['Opt_Purchases'].sum()
                    row_dict[f"Act_Purch_Region_{reg}"] = df_region[df_region['Region'] == reg]['Act_Purchases'].sum()

                all_origins = ['new', 'existing', 'existing searches']
                for org in all_origins:
                    row_dict[f"Opt_Cost_Origin_{org}"] = X_day[X_day['Origin'] == org]['Optimal Cost'].sum()
                    row_dict[f"Act_Cost_Origin_{org}"] = obs[obs['Origin'] == org]['Cost'].sum()
                    row_dict[f"Opt_Clicks_Origin_{org}"] = X_day[X_day['Origin'] == org]['t_Clicks_OptCost'].sum()
                    row_dict[f"Act_Clicks_Origin_{org}"] = obs_breakdown[obs_breakdown['Origin'] == org]['t_Clicks_ActCost'].sum()
                    row_dict[f"Opt_Purch_Origin_{org}"] = opt_purch_origin_map.get(org, 0.0)
                    row_dict[f"Act_Purch_Origin_{org}"] = act_purch_origin_map.get(org, 0.0)

                all_match_types = ['Exact match', 'Phrase match', 'Broad match']
                for mt in all_match_types:
                    row_dict[f"Opt_Cost_Match_{mt}"] = X_day[X_day['Match type'] == mt]['Optimal Cost'].sum()
                    row_dict[f"Act_Cost_Match_{mt}"] = obs[obs['Match type'] == mt]['Cost'].sum()
                    row_dict[f"Opt_Clicks_Match_{mt}"] = X_day[X_day['Match type'] == mt]['t_Clicks_OptCost'].sum()
                    row_dict[f"Act_Clicks_Match_{mt}"] = obs_breakdown[obs_breakdown['Match type'] == mt]['t_Clicks_ActCost'].sum()
                    row_dict[f"Opt_Purch_Match_{mt}"] = opt_purch_mt_map.get(mt, 0.0)
                    row_dict[f"Act_Purch_Match_{mt}"] = act_purch_mt_map.get(mt, 0.0)

                eval_summary_rows.append(row_dict)

        # Save results per k_policy
        if eval_summary_rows:
            res_df = pd.DataFrame(eval_summary_rows)
            out_path = base_results_dir / "evaluation_results.csv"
            res_df.to_csv(out_path, index=False)
            print(f"Saved evaluation results to {out_path}")

if __name__ == "__main__":
    main()