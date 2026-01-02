"""Sweep exploration lambda for bid optimization.

Runs scripts/bid_optimization.py for a set of lambda values and summarizes
key metrics (profit, spend, objective, and % new keywords selected).

Example:
  python scripts/analyze_exploration_lambda.py --lambdas 0,0.1,1,5 \
    --embedding-method bert --alg-conv xgb --alg-clicks xgb --budget 68096.51 --max-bid 100
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

import pandas as pd


def _parse_lambdas(s: str) -> List[float]:
    vals: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("--lambdas must contain at least one value")
    return vals


def _safe_lam_str(lam: float) -> str:
    return str(lam).replace(".", "p").replace("-", "m")


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep exploration lambda and summarize outcomes.")
    p.add_argument("--lambdas", type=str, required=True, help="Comma-separated lambdas, e.g. 0,0.1,1,5")

    p.add_argument(
        "--formulation-lp",
        type=str,
        default=None,
        help="Optional cached formulation path (.lp). If provided, build once (if missing) and reuse for each lambda.",
    )

    # Pass-through args to bid_optimization.py
    p.add_argument("--embedding-method", type=str, default="bert", choices=["tfidf", "bert"])
    p.add_argument("--alg-conv", type=str, default="glm", choices=["lr", "glm", "xgb", "ort", "mort"])
    p.add_argument("--alg-clicks", type=str, default="xgb", choices=["lr", "glm", "xgb", "ort", "mort"])
    p.add_argument("--budget", type=float, default=400.0)
    p.add_argument("--max-bid", type=float, default=50.0)
    p.add_argument("--target-day", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="data/clean")
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--trust-box", type=str, default="drop", choices=["drop", "off"])
    p.add_argument("--trust-box-training", type=str, default="data/clean/ad_opt_data_bert.csv")

    p.add_argument(
        "--bid-threshold",
        type=float,
        default=1e-6,
        help="Minimum bid to count a row as 'selected' (default: 1e-6)",
    )

    args = p.parse_args()

    lambdas = _parse_lambdas(args.lambdas)

    out_dir = Path("opt_results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    formulation_lp = Path(args.formulation_lp) if args.formulation_lp else None
    if formulation_lp is not None:
        if not formulation_lp.exists():
            # Build the base formulation once (profit-only objective) and exit.
            build_cmd = [
                "python",
                "-u",
                "scripts/bid_optimization.py",
                "--embedding-method",
                args.embedding_method,
                "--alg-conv",
                args.alg_conv,
                "--alg-clicks",
                args.alg_clicks,
                "--budget",
                str(args.budget),
                "--max-bid",
                str(args.max_bid),
                "--data-dir",
                args.data_dir,
                "--models-dir",
                args.models_dir,
                "--trust-box",
                args.trust_box,
                "--trust-box-training",
                args.trust_box_training,
                "--explore-lambda",
                "0.0",
                "--formulation-lp",
                str(formulation_lp),
                "--write-formulation-only",
            ]
            if args.target_day is not None:
                build_cmd += ["--target-day", args.target_day]
            if args.trial is not None:
                build_cmd += ["--trial", str(args.trial)]

            print(f"\n=== Building cached formulation: {formulation_lp} ===")
            subprocess.run(build_cmd, check=True)

        if not formulation_lp.exists():
            raise FileNotFoundError(f"Formulation cache not found after build attempt: {formulation_lp}")

    for lam in lambdas:
        if formulation_lp is not None:
            cmd = [
                "python",
                "-u",
                "scripts/bid_optimization.py",
                "--reuse-formulation",
                "--formulation-lp",
                str(formulation_lp),
                "--explore-lambda",
                str(lam),
            ]
        else:
            cmd = [
                "python",
                "-u",
                "scripts/bid_optimization.py",
                "--embedding-method",
                args.embedding_method,
                "--alg-conv",
                args.alg_conv,
                "--alg-clicks",
                args.alg_clicks,
                "--budget",
                str(args.budget),
                "--max-bid",
                str(args.max_bid),
                "--data-dir",
                args.data_dir,
                "--models-dir",
                args.models_dir,
                "--trust-box",
                args.trust_box,
                "--trust-box-training",
                args.trust_box_training,
                "--explore-lambda",
                str(lam),
            ]
            if args.target_day is not None:
                cmd += ["--target-day", args.target_day]
            if args.trial is not None:
                cmd += ["--trial", str(args.trial)]

        print(f"\n=== Running lambda={lam:g} ===")
        subprocess.run(cmd, check=True)

        model_suffix = f"{args.alg_conv}_{args.alg_clicks}"
        bids_dir = Path("opt_results/bids")
        if lam != 0.0:
            bids_file = bids_dir / f"optimized_bids_{args.embedding_method}_{model_suffix}_lambda{_safe_lam_str(lam)}.csv"
        else:
            bids_file = bids_dir / f"optimized_bids_{args.embedding_method}_{model_suffix}.csv"

        if not bids_file.exists():
            raise FileNotFoundError(f"Expected output not found: {bids_file}")

        df = pd.read_csv(bids_file)

        bid = _to_float(df.get("bid", pd.Series(dtype=float)))
        cost = _to_float(df.get("cost", pd.Series(dtype=float)))
        profit = _to_float(df.get("profit", pd.Series(dtype=float)))
        bonus = _to_float(df.get("exploration_bonus", pd.Series(dtype=float)))

        # is_new_keyword may come back as bool or string
        is_new = df.get("is_new_keyword", False)
        if isinstance(is_new, pd.Series):
            is_new = is_new.astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            is_new = pd.Series([False] * len(df))

        selected = bid.fillna(0.0) > float(args.bid_threshold)

        total_spend = float(cost.fillna(0.0).sum())
        total_profit = float(profit.fillna(0.0).sum())
        total_bonus = float(bonus.fillna(0.0).sum())
        total_objective = total_profit + total_bonus

        selected_count = int(selected.sum())
        selected_new_count = int((selected & is_new).sum())
        pct_new_selected = (selected_new_count / selected_count * 100.0) if selected_count > 0 else 0.0

        spend_new = float(cost.fillna(0.0)[selected & is_new].sum())
        pct_spend_new = (spend_new / total_spend * 100.0) if total_spend > 0 else 0.0

        rows.append(
            {
                "lambda": lam,
                "selected_rows": selected_count,
                "selected_new_rows": selected_new_count,
                "pct_new_selected": pct_new_selected,
                "total_spend": total_spend,
                "total_profit": total_profit,
                "exploration_bonus": total_bonus,
                "objective": total_objective,
                "pct_spend_new": pct_spend_new,
                "bids_file": str(bids_file).replace("\\\\", "/"),
            }
        )

    summary = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    out_file = out_dir / f"exploration_lambda_sweep_{args.embedding_method}_{args.alg_conv}_{args.alg_clicks}.csv"
    summary.to_csv(out_file, index=False)

    print(f"\nSaved sweep summary to {out_file}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
