"""
Compare Gurobi vs. SCIP solver outputs for consistency.

Runs the full bid-optimization pipeline once with each solver on an
identical feature matrix, then reports:
  - wall-clock time for each solver
  - objective value comparison
  - per-keyword cost & prediction differences
  - summary statistics (MAE, max diff, correlation)

Example usage:
    python scripts/compare_solvers.py --course gen_ai
    python scripts/compare_solvers.py --course ml --budget 353.99
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COURSE_CONFIG
from utils.date_features import COURSE_START_DATES_MAP
from scripts.optimization import (
    create_feature_matrix,
    optimize_bids,
    extract_solution,
)
from tidy_get_data import load_or_cache
from scripts.modeling import _to_float32_csr  # noqa: F401 – needed to unpickle


def _run_solver(solver_name, X, model_path, budget, kw_df, order_budget, max_purch, base_dir, time_limit=600):
    """Run optimize_bids + extract_solution with a given solver and return results + timing."""
    print(f"\n{'='*60}")
    print(f"  SOLVER: {solver_name.upper()}  (time limit: {time_limit}s)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    model, cost_vars, pred_vars, X_opt = optimize_bids(
        X.copy(),
        model_path,
        budget=budget,
        kw_df=kw_df,
        order_budget=order_budget,
        max_purch=max_purch,
        base_dir=base_dir,
        solver=solver_name,
        time_limit=time_limit,
    )
    solve_time = time.perf_counter() - t0

    results_df = extract_solution(model, cost_vars, pred_vars, model_path, X_opt)
    total_time = time.perf_counter() - t0

    obj_val = model.get_obj_val() if model.is_optimal_or_limit() else None

    return {
        "results": results_df,
        "obj_val": obj_val,
        "solve_time": solve_time,
        "total_time": total_time,
        "status": model.raw_status,
    }


def compare_results(gurobi_out, scip_out, tol=1e-2):
    """Print a comparison report of the two solver runs."""
    print(f"\n{'='*60}")
    print("  COMPARISON REPORT")
    print(f"{'='*60}")

    # ---- Timing ----
    print("\n--- Timing ---")
    print(f"  Gurobi  solve: {gurobi_out['solve_time']:>8.2f}s   total: {gurobi_out['total_time']:>8.2f}s")
    print(f"  SCIP    solve: {scip_out['solve_time']:>8.2f}s   total: {scip_out['total_time']:>8.2f}s")
    speedup = scip_out['total_time'] / max(gurobi_out['total_time'], 1e-9)
    print(f"  SCIP / Gurobi time ratio: {speedup:.2f}x")

    # ---- Objective ----
    print("\n--- Objective Value ---")
    g_obj = gurobi_out["obj_val"]
    s_obj = scip_out["obj_val"]
    if g_obj is not None and s_obj is not None:
        print(f"  Gurobi: {g_obj:.6f}")
        print(f"  SCIP:   {s_obj:.6f}")
        obj_diff = abs(g_obj - s_obj)
        obj_rel = obj_diff / max(abs(g_obj), 1e-9)
        print(f"  Abs diff: {obj_diff:.6f}   Rel diff: {obj_rel:.6%}")
    else:
        print("  [Warning] One or both solvers did not find an optimal/feasible solution.")
        return

    # ---- Per-keyword comparison ----
    g_df = gurobi_out["results"]
    s_df = scip_out["results"]
    if g_df is None or s_df is None:
        print("  [Warning] Cannot compare per-keyword results (missing solution).")
        return

    # Merge on common key
    key_cols = ["Keyword", "Region", "Match type"]
    merged = g_df[key_cols + ["Optimal Cost", "Solver Pred"]].merge(
        s_df[key_cols + ["Optimal Cost", "Solver Pred"]],
        on=key_cols,
        how="outer",
        suffixes=("_grb", "_scip"),
    )
    merged = merged.fillna(0.0)

    print(f"\n--- Per-Keyword Comparison ({len(merged)} rows after outer join) ---")

    # Cost comparison
    cost_diff = (merged["Optimal Cost_grb"] - merged["Optimal Cost_scip"]).abs()
    print(f"  Cost  – MAE: {cost_diff.mean():.6f}   Max: {cost_diff.max():.6f}")
    if merged["Optimal Cost_grb"].std() > 0 and merged["Optimal Cost_scip"].std() > 0:
        cost_corr = merged["Optimal Cost_grb"].corr(merged["Optimal Cost_scip"])
        print(f"  Cost  – Pearson r: {cost_corr:.6f}")

    # Prediction comparison
    pred_diff = (merged["Solver Pred_grb"] - merged["Solver Pred_scip"]).abs()
    print(f"  Pred  – MAE: {pred_diff.mean():.6f}   Max: {pred_diff.max():.6f}")
    if merged["Solver Pred_grb"].std() > 0 and merged["Solver Pred_scip"].std() > 0:
        pred_corr = merged["Solver Pred_grb"].corr(merged["Solver Pred_scip"])
        print(f"  Pred  – Pearson r: {pred_corr:.6f}")

    # Rows unique to one solver
    only_grb = len(g_df.merge(s_df[key_cols], on=key_cols, how="left", indicator=True).query("_merge=='left_only'"))
    only_scip = len(s_df.merge(g_df[key_cols], on=key_cols, how="left", indicator=True).query("_merge=='left_only'"))
    print(f"\n  Keywords selected by Gurobi only: {only_grb}")
    print(f"  Keywords selected by SCIP only:   {only_scip}")

    # Overall verdict
    print("\n--- Verdict ---")
    if obj_rel < tol and cost_diff.mean() < tol * 10:
        print(f"  PASS: Solutions are consistent within tolerance ({tol}).")
    else:
        print(f"  WARN: Solutions differ beyond tolerance ({tol}). "
              "This may be expected for MIP problems with multiple optima.")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Compare Gurobi vs. SCIP on the bid-optimization MIP")
    parser.add_argument("--course", required=True, help="Course name (e.g. gen_ai, ml, sys_eng)")
    parser.add_argument("--embedding-method", default="bert", choices=["bert", "llm"])
    parser.add_argument("--budget", type=float, default=None, help="Budget (default: first from config)")
    parser.add_argument("--order-budget", action="store_true", default=True)
    parser.add_argument("--max-purch", action="store_true", default=True)
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance for PASS/WARN verdict (default 0.01)")
    parser.add_argument("--time-limit", type=int, default=3600, help="Solver time limit in seconds per solver (default: 3600)")
    parser.add_argument("--output", type=str, default="opt_results/comp_solvers.csv", help="Save merged comparison CSV to this path")
    args = parser.parse_args()

    if args.budget is None:
        args.budget = COURSE_CONFIG[args.course]["budgets"][0]

    base_dir = Path(f"data/{args.course}")
    model_path = f"models/{args.course}_xgb_clicks_model_{args.embedding_method}.joblib"

    # ---- Build feature matrix (shared) ----
    kw_df = pd.read_csv(base_dir / "gkp/keywords_classified.csv")
    keywords = kw_df["Keyword"].tolist()

    cache_dir = Path(f"opt_results/{args.course}/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    X = load_or_cache(
        create_feature_matrix,
        cache_dir / "feature_matrix.parquet",
        False,
        keywords,
        None,
        COURSE_START_DATES_MAP.get(args.course, []),
        base_dir,
    )
    X = X[X["Region"] != "C"]

    print(f"Course: {args.course}  |  Budget: {args.budget}  |  Rows: {len(X)}")
    print(f"Embedding: {args.embedding_method}  |  Order budget: {args.order_budget}  |  Max purch: {args.max_purch}")

    # ---- Run both solvers ----
    scip_out   = _run_solver("scip",   X, model_path, args.budget, kw_df, args.order_budget, args.max_purch, base_dir, time_limit=args.time_limit)
    gurobi_out = _run_solver("gurobi", X, model_path, args.budget, kw_df, args.order_budget, args.max_purch, base_dir, time_limit=args.time_limit)

    # ---- Compare ----
    merged = compare_results(gurobi_out, scip_out, tol=args.tol)

    if args.output and merged is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"\n[Info] Merged comparison saved to '{out_path}'.")


if __name__ == "__main__":
    main()
