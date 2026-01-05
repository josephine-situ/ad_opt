"""Analyze exploration-lambda sweep outputs.

Reads bid CSVs written by scripts/bid_optimization.py into:
    opt_results/bids/lambda_sweep/<cache_key>/

Computes per-lambda:
- total conv value / cost / profit
- objective decomposition: profit component vs exploration component
- % new keywords by unique keyword count among active keywords
- % cost spent on new keywords
- % profit from new keywords

Usage:
    python -u scripts/analyze_exploration_lambda.py --cache-key bert_glm_xgb_2026-01-04_full

If --cache-key is omitted, defaults to:
    bert_glm_xgb_<today YYYY-MM-DD>_full

Optional:
    --out-csv opt_results/analysis/lambda_sweep_summary.csv
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import json

import pandas as pd


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def _summarize_bids_file(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path)

    # Required numeric columns
    for c in ["bid", "conv_value", "cost", "profit"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {path}")

    bid = _to_float(df["bid"])
    conv = _to_float(df["conv_value"])
    cost = _to_float(df["cost"])
    profit = _to_float(df["profit"])

    lmbda = None
    if "lambda" in df.columns:
        lmbda = float(_to_float(df["lambda"]).iloc[0])
    else:
        # Fallback: parse from filename segment "_lambda_<tag>.csv"
        name = path.name
        if "_lambda_" in name:
            tag = name.split("_lambda_", 1)[1].rsplit(".csv", 1)[0]
            tag = tag.replace("p", ".").replace("m", "-")
            try:
                lmbda = float(tag)
            except ValueError:
                lmbda = 0.0
        else:
            lmbda = 0.0

    is_new = None
    if "is_new_keyword" in df.columns:
        is_new = df["is_new_keyword"].astype(bool)
    else:
        is_new = pd.Series([False] * len(df))

    active = cost.to_numpy() > 1e-9
    active_kws = set(df.loc[active, "keyword"].astype(str).tolist()) if "keyword" in df.columns else set()
    active_new_kws = set(df.loc[active & is_new.to_numpy(), "keyword"].astype(str).tolist()) if "keyword" in df.columns else set()

    total_conv = float(conv.sum())
    total_cost = float(cost.sum())
    total_profit = float(profit.sum())

    new_bid_sum = float(bid[is_new].sum())
    new_cost = float(cost[is_new].sum())
    new_profit = float(profit[is_new].sum())

    explore_component = float(lmbda) * new_bid_sum
    objective_total = total_profit + explore_component

    pct_new_unique = (len(active_new_kws) / len(active_kws)) if active_kws else 0.0
    pct_cost_new = (new_cost / total_cost) if total_cost > 0 else 0.0
    pct_profit_new = (new_profit / total_profit) if abs(total_profit) > 1e-12 else 0.0

    return {
        "lambda": float(lmbda),
        "objective_total": objective_total,
        "objective_profit_component": total_profit,
        "objective_exploration_component": explore_component,
        "total_conv_value": total_conv,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "new_cost": new_cost,
        "new_profit": new_profit,
        "pct_cost_new": float(pct_cost_new),
        "pct_profit_new": float(pct_profit_new),
        "active_unique_keywords": int(len(active_kws)),
        "active_unique_new_keywords": int(len(active_new_kws)),
        "pct_active_unique_new_keywords": float(pct_new_unique),
        "file": str(path),
    }


def _load_metrics_json(files: List[Path]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for p in files:
        with open(p, "r") as f:
            rows.append(json.load(f))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "lambda" in df.columns:
        df = df.sort_values("lambda").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze bid optimization lambda sweep outputs")
    ap.add_argument(
        "--cache-key",
        default=None,
        help="Cache key used during sweep (default: bert_glm_xgb_<today>_full)",
    )
    ap.add_argument(
        "--target-day",
        default=date.today().strftime("%Y-%m-%d"),
        help="Target day in YYYY-MM-DD (default: today's date)",
    )
    ap.add_argument(
        "--sweep-dir",
        default=None,
        help="Override sweep directory (default: opt_results/bids/lambda_sweep/<cache_key>)",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to write summary CSV (default: opt_results/analysis/lambda_sweep_<cache_key>.csv)",
    )

    args = ap.parse_args()

    if args.cache_key is None:
        # Keep in sync with submit_bid_optimization_job.sh defaults.
        td = str(args.target_day).strip()
        if td.lower() in {"today", ""}:
            td = date.today().strftime("%Y-%m-%d")
        args.cache_key = f"bert_glm_xgb_{td}_full"

    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else Path("opt_results/bids") / "lambda_sweep" / args.cache_key
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    bid_files = sorted(sweep_dir.glob("optimized_bids_*_lambda_*.csv"))
    if bid_files:
        rows = [_summarize_bids_file(p) for p in bid_files]
        df = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    else:
        # Backward compatible fallback
        metric_files = sorted(sweep_dir.glob("metrics_lambda_*.json"))
        if not metric_files:
            raise FileNotFoundError(f"No sweep bid CSVs or metrics JSON found in {sweep_dir}")
        df = _load_metrics_json(metric_files)

    # Friendly table
    keep_cols = [
        "lambda",
        "objective_total",
        "objective_profit_component",
        "objective_exploration_component",
        "total_conv_value",
        "total_cost",
        "total_profit",
        "new_profit",
        "active_unique_keywords",
        "active_unique_new_keywords",
        "pct_active_unique_new_keywords",
        "pct_cost_new",
        "pct_profit_new",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Format pct columns as percents for print
    df_print = df.copy()
    for c in ["pct_active_unique_new_keywords", "pct_cost_new", "pct_profit_new"]:
        if c in df_print.columns:
            df_print[c] = (100.0 * df_print[c].astype(float)).round(2)

    print("=" * 80)
    print(f"Lambda sweep summary: {args.cache_key}")
    print(f"Dir: {sweep_dir}")
    print("=" * 80)
    print(df_print[keep_cols].to_string(index=False))

    out_csv = Path(args.out_csv) if args.out_csv else Path("opt_results/analysis") / f"lambda_sweep_{args.cache_key}.csv"
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
