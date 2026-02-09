"""
Sensitivity analysis: keyword concentration risk.

For each backtest day, randomly drops a fraction of keywords within each
(Region, Match type) group and rescales the remaining bids so that per-group
spend is preserved.  Repeats the random draw several times, evaluates
predicted clicks / purchases with the full-data evaluation
model, and produces a summary table (CSV + LaTeX) comparable to
analyze_backtest_results.py.

Example:
    python scripts/sensitivity_analysis.py --course gen_ai --exp-name exp103_fix_max_clicks
    python scripts/sensitivity_analysis.py --course ml --exp-name exp103_fix_max_clicks --n-reps 30
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_daily import feature_matrix_cached, select_keywords
from utils.data_pipeline import get_conversion_rates
from utils.date_features import COURSE_START_DATES_MAP
from config import COURSE_CONFIG


# ── Feature specification (must match backtest_eval.py) ──────────────── #

_FEATURE_COLS_BASE = [
    "Match type", "Region",
    "day_of_week", "is_weekend", "month",
    "is_public_holiday", "days_to_next_course_start",
    "last_month_searches", "three_month_avg", "six_month_avg",
    "mom_change", "search_trend",
    "Competition (indexed value)",
    "Top of page bid (low range)", "Top of page bid (high range)",
    "Cost",
]


def _get_feature_cols(X: pd.DataFrame, embedding_method: str) -> list[str]:
    """Return the ordered feature column list, matching backtest_eval.py."""
    if embedding_method == "llm":
        emb = [c for c in X.columns if c == "llm_relevance_score"]
    else:
        emb = [c for c in X.columns if c.startswith("bert_")]
    return _FEATURE_COLS_BASE + emb


# ── Keyword dropping & rescaling ─────────────────────────────────────── #

def drop_keywords_and_rescale(
    bids_df: pd.DataFrame, drop_frac: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Within each (Region, Match type) group, randomly drop *drop_frac* of
    keywords and scale survivors so that per-group spend is unchanged."""
    if drop_frac == 0:
        return bids_df.copy()

    parts: list[pd.DataFrame] = []
    for _, grp in bids_df.groupby(["Region", "Match type"]):
        n_drop = int(len(grp) * drop_frac)
        if n_drop >= len(grp):
            continue                              # all keywords dropped
        if n_drop == 0:
            parts.append(grp)
            continue
        drop_idx = rng.choice(grp.index, size=n_drop, replace=False)
        kept = grp.drop(drop_idx).copy()
        spent_before = grp["Optimal Cost"].sum()
        spent_after  = kept["Optimal Cost"].sum()
        if spent_after > 0:
            kept["Optimal Cost"] *= spent_before / spent_after
        parts.append(kept)

    if parts:
        return pd.concat(parts, ignore_index=True)
    return bids_df.iloc[:0].copy()


# ── Prediction helpers ───────────────────────────────────────────────── #

def _predict_lift(X_day: pd.DataFrame, model, features: list[str]) -> np.ndarray:
    """Predicted click lift = pred(cost) − pred(cost=0)."""
    pred = model.predict(X_day[features])
    X_zero = X_day.assign(Cost=0.0)
    pred_base = model.predict(X_zero[features])
    return pred - pred_base


def evaluate_bids(
    sol: pd.DataFrame,
    X_base: pd.DataFrame,
    model,
    features: list[str],
    loc_df: pd.DataFrame,
) -> dict:
    """Merge bids with features, predict clicks, derive purchases."""
    if sol.empty:
        return dict(clicks=0.0, cost=0.0, purch=0.0, n_kws=0)

    X = X_base.merge(
        sol[["Keyword", "Region", "Match type", "Optimal Cost"]],
        on=["Keyword", "Region", "Match type"],
        how="right",
    )
    X["Cost"] = X["Optimal Cost"]
    lift = _predict_lift(X, model, features)
    X["lift"] = lift

    # Purchases via location-level click proportions
    clicks_reg = X.groupby("Region")["lift"].sum().reset_index()
    merged = loc_df.merge(clicks_reg, on="Region", how="left").fillna(0)
    merged["purch"] = merged["lift"] * merged["Click_prop"] * merged["Purch_rate"]

    return dict(
        clicks=float(lift.sum()),
        cost=float(X["Optimal Cost"].sum()),
        purch=float(merged["purch"].sum()),
        n_kws=len(X),
    )


# ── LaTeX table ──────────────────────────────────────────────────────── #

def generate_sensitivity_table(summary_df: pd.DataFrame) -> str:
    """Produce a LaTeX table keyed on Drop % (including an Actual row)."""
    df = summary_df.copy()

    # Sort: Actual first, then ascending drop %
    df["_sk"] = df["Drop %"].apply(lambda x: -1 if x == "Actual" else x)
    df = df.sort_values("_sk").drop(columns=["_sk"])

    # ── format first column ──
    df["Drop %"] = df["Drop %"].apply(
        lambda x: "Actual" if x == "Actual" else f"{x:.0f}\\%"
    )

    # ── mean ± se columns ──
    df["Clicks"] = df.apply(
        lambda r: f"{r['avg clicks']:,.1f} $\\pm$ {r['se clicks']:,.1f}", axis=1)
    df["Purch"] = df.apply(
        lambda r: f"{r['avg purch']:,.2f} $\\pm$ {r['se purch']:,.2f}", axis=1)
    df["Cost"] = df.apply(
        lambda r: f"\\${r['avg cost']:,.2f} $\\pm$ {r['se cost']:,.2f}", axis=1)
    df["Kws"] = df.apply(
        lambda r: f"{r['avg kws']:,.0f} $\\pm$ {r['se kws']:,.0f}", axis=1)
    df[r"Clicks/\$"] = df["clicks/$"].map("{:,.3f}".format)

    # ── improvement columns ──
    imp_cols = ["imp clicks", "imp purch", r"imp clicks/$"]
    for col in imp_cols:
        df[col] = df[col].apply(
            lambda x: "---" if x == 0 and col in imp_cols else f"{x * 100:,.1f}\\%"
        )
    # Mark Actual improvements as dashes
    act_mask = df["Drop %"] == "Actual"
    for col in imp_cols:
        df.loc[act_mask, col] = "---"

    # ── bold best improvement in clicks (excl. Actual) ──
    non_act = df[~act_mask].copy()
    df = df.astype(object)
    if not non_act.empty:
        vals = non_act["imp clicks"].str.replace(r"[\\%,]", "", regex=True).astype(float)
        best_idx = vals.idxmax()
        for col in [c for c in df.columns if not c.startswith("_")]:
            df.at[best_idx, col] = f"\\textbf{{{df.at[best_idx, col]}}}"

    # ── MultiIndex columns ──
    col_map = [
        ("Drop %",            ("", r"Drop \%")),
        ("Clicks",            ("Metrics", "Clicks")),
        ("Purch",             ("Metrics", "Purch")),
        ("Cost",              ("Metrics", "Cost")),
        (r"Clicks/\$",        ("Metrics", r"Clicks/\$")),
        ("Kws",               ("Metrics", "Kws")),
        ("imp clicks",        ("Improvement", "Clicks")),
        ("imp purch",         ("Improvement", "Purch")),
        (r"imp clicks/$",     ("Improvement", r"Clicks/\$")),
    ]
    existing = [old for old, _ in col_map if old in df.columns]
    df = df[existing]
    df.columns = pd.MultiIndex.from_tuples(
        [new for old, new in col_map if old in existing]
    )

    n_metrics = sum(1 for c in df.columns if c[0] == "Metrics")
    n_imp     = sum(1 for c in df.columns if c[0] == "Improvement")
    total     = 1 + n_metrics + n_imp
    col_fmt   = "l" + "c" * (total - 1)

    latex = df.to_latex(
        index=False, escape=False, multicolumn_format="c", column_format=col_fmt
    )
    latex = latex.replace(r"\hline", r"\toprule", 1)
    if latex.strip().endswith(r"\hline"):
        latex = latex.strip()[:-6] + r"\bottomrule"

    # ── inject cmidrules ──
    lines = latex.split("\n")
    new_lines: list[str] = []
    header_replaced = False

    header_row = " "
    if n_metrics:
        header_row += fr"& \multicolumn{{{n_metrics}}}{{c}}{{Metrics}} "
    if n_imp:
        header_row += fr"& \multicolumn{{{n_imp}}}{{c}}{{Improvement}} "
    header_row += r"\\"

    cmid = ""
    col = 2
    if n_metrics:
        cmid += fr"\cmidrule(lr){{{col}-{col + n_metrics - 1}}} "
        col += n_metrics
    if n_imp:
        cmid += fr"\cmidrule(lr){{{col}-{col + n_imp - 1}}}"

    for line in lines:
        if "Metrics" in line and "Improvement" in line and not header_replaced:
            new_lines.append(header_row)
            new_lines.append(cmid)
            header_replaced = True
        elif "Drop" in line and "Clicks" in line:
            new_lines.append(line)
            new_lines.append(r"\midrule")
        elif r"\bottomrule" in line:
            new_lines.append(line)
        else:
            new_lines.append(line)

    final = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + "\n".join(new_lines)
        + "\n}\n\\end{table}"
    )
    return final


# ── CLI & main loop ──────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(
        description="Keyword-drop sensitivity analysis for backtest results"
    )
    p.add_argument("--course", default="gen_ai")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--budget", type=int, default=None,
                   help="Budget level (default: first budget in COURSE_CONFIG)")
    p.add_argument("--n-reps", type=int, default=20,
                   help="Monte-Carlo repetitions per day for stochastic drop levels")
    p.add_argument("--drop-pcts", type=float, nargs="+",
                   default=[0, 25, 50, 75],
                   help="Drop percentages to evaluate (default: 0 25 50 75 100)")
    p.add_argument("--embedding-method", default="bert",
                   choices=["bert", "llm"])
    p.add_argument("--keywords-n", type=int, default=None,
                   help="Number of keywords (must match backtest run)")
    p.add_argument("--masked", action="store_true",
                   help="Use masked keywords (must match backtest run)")
    return p.parse_args()


def main():
    args = parse_args()
    course = args.course

    if args.budget is None:
        args.budget = int(COURSE_CONFIG[course]["budgets"][0])

    base_dir        = Path(f"data/{course}")
    base_results_dir = Path(f"opt_results/{course}/backtests/{args.exp_name}")
    run_dir         = base_results_dir / f"budget_{args.budget}"
    bids_dir        = run_dir / "bids"
    cache_dir       = base_results_dir / "cache"
    eval_models_dir = Path(f"opt_results/{course}/eval_models")

    # ── load keyword list (same as backtest_eval.py) ──
    kw_df_all = pd.read_csv(base_dir / "gkp/keywords_classified.csv")

    # ── load evaluation model ──
    model_path = eval_models_dir / f"eval_model_full_{args.embedding_method}.joblib"
    if not model_path.exists():
        print(f"Eval model not found at {model_path}. Run backtest_eval.py first.")
        return
    model = joblib.load(model_path)
    print(f"Loaded eval model from {model_path}")

    # ── conversion / purchase rates by location ──
    loc_df = get_conversion_rates(base_dir=base_dir)

    # ── actual metrics from evaluation_results.csv ──
    eval_csv = base_results_dir / "evaluation_results.csv"
    if not eval_csv.exists():
        print(f"Evaluation results not found at {eval_csv}. Run backtest_eval.py first.")
        return
    eval_all = pd.read_csv(eval_csv)
    eval_df = eval_all[eval_all["Budget"] == args.budget].copy()
    if eval_df.empty:
        print(f"No evaluation results for budget {args.budget}")
        return

    # ── discover backtest days from bids files ──
    bids_files = sorted(bids_dir.glob("optimized_costs_*.csv"))
    if not bids_files:
        print(f"No bids files found in {bids_dir}")
        return
    days = [pd.to_datetime(f.stem.replace("optimized_costs_", "")) for f in bids_files]
    print(f"Found {len(days)} backtest days, {len(args.drop_pcts)} drop levels, "
          f"{args.n_reps} reps")

    course_start_dts = COURSE_START_DATES_MAP.get(course, [])
    features: list[str] | None = None

    # ── collect per-day, per-drop, per-rep results ──
    results: list[dict] = []               # one row per (day, drop_pct)
    actual_rows: list[dict] = []           # one row per day

    for day in days:
        print(f"  Day {day.date()}", end="", flush=True)

        sol = pd.read_csv(bids_dir / f"optimized_costs_{day.date()}.csv")

        # Reconstruct the same keyword list used in backtest_eval so the
        # cached feature matrix is reused (sol keywords are always a subset).
        seed = int(day.strftime("%Y%m%d"))
        _, keywords, _ = select_keywords(
            kw_df_all, args.keywords_n, args.masked, seed=seed
        )

        X_base = feature_matrix_cached(
            keywords=keywords,
            opt_date=day,
            cache_dir=cache_dir,
            base_dir=base_dir,
            course_start_dts=course_start_dts,
            embedding_method=args.embedding_method,
            course=course,
        )
        if features is None:
            features = _get_feature_cols(X_base, args.embedding_method)

        # Actual metrics (from evaluation_results.csv)
        day_eval = eval_df[eval_df["Day"] == str(day.date())]
        if not day_eval.empty:
            r = day_eval.iloc[0]
            actual_rows.append(dict(
                clicks=r["t_Clicks_ActCost"],
                cost=r["Act_Cost"],
                purch=r.get("Act_Purch", 0),
                n_kws=r["N_Obs"],
            ))

        # Sensitivity: each drop level
        for drop_pct in args.drop_pcts:
            frac = drop_pct / 100.0
            n_reps = 1 if drop_pct in (0, 100) else args.n_reps
            rep_metrics: list[dict] = []

            for rep in range(n_reps):
                seed = int(day.strftime("%Y%m%d")) * 1000 + int(drop_pct) * 10 + rep
                rng = np.random.default_rng(seed)
                sol_dropped = drop_keywords_and_rescale(sol, frac, rng)
                m = evaluate_bids(sol_dropped, X_base, model, features, loc_df)
                rep_metrics.append(m)

            # Average across reps → one observation per (day, drop_pct)
            results.append(dict(
                drop_pct=drop_pct,
                day=day,
                clicks=np.mean([m["clicks"] for m in rep_metrics]),
                cost=np.mean([m["cost"] for m in rep_metrics]),
                purch=np.mean([m["purch"] for m in rep_metrics]),
                n_kws=np.mean([m["n_kws"] for m in rep_metrics]),
            ))

        print(f"  ✓")

    # ── aggregate across days ────────────────────────────────────────── #
    res_df = pd.DataFrame(results)
    act_df = pd.DataFrame(actual_rows)

    avg_act_clicks = act_df["clicks"].mean()
    avg_act_purch  = act_df["purch"].mean()
    act_cpd = (act_df["clicks"].sum() / act_df["cost"].sum()
               if act_df["cost"].sum() > 0 else 0)

    summary_rows: list[dict] = []

    # Actual row
    summary_rows.append({
        "Drop %":     "Actual",
        "avg clicks": act_df["clicks"].mean(),
        "se clicks":  act_df["clicks"].sem(),
        "avg purch":  act_df["purch"].mean(),
        "se purch":   act_df["purch"].sem(),
        "avg cost":   act_df["cost"].mean(),
        "se cost":    act_df["cost"].sem(),
        "clicks/$":   act_cpd,
        "avg kws":    act_df["n_kws"].mean(),
        "se kws":     act_df["n_kws"].sem(),
        "imp clicks":    0,
        "imp purch":     0,
        "imp clicks/$":  0,
    })

    # One row per drop %
    for pct, grp in res_df.groupby("drop_pct"):
        avg_c     = grp["clicks"].mean()
        avg_purch = grp["purch"].mean()
        total_c   = grp["clicks"].sum()
        total_cost = grp["cost"].sum()
        cpd = total_c / total_cost if total_cost > 0 else 0

        summary_rows.append({
            "Drop %":     pct,
            "avg clicks": avg_c,
            "se clicks":  grp["clicks"].sem(),
            "avg purch":  avg_purch,
            "se purch":   grp["purch"].sem(),
            "avg cost":   grp["cost"].mean(),
            "se cost":    grp["cost"].sem(),
            "clicks/$":   cpd,
            "avg kws":    grp["n_kws"].mean(),
            "se kws":     grp["n_kws"].sem(),
            "imp clicks":   (avg_c - avg_act_clicks) / avg_act_clicks if avg_act_clicks else 0,
            "imp purch":    (avg_purch - avg_act_purch) / avg_act_purch if avg_act_purch else 0,
            "imp clicks/$": (cpd - act_cpd) / act_cpd if act_cpd else 0,
        })

    summary_df = pd.DataFrame(summary_rows).fillna(0)

    # ── output ───────────────────────────────────────────────────────── #
    out_csv = base_results_dir / "sensitivity_analysis.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved CSV  → {out_csv}")

    latex = generate_sensitivity_table(summary_df)
    out_tex = base_results_dir / "sensitivity_analysis.tex"
    with open(out_tex, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX → {out_tex}")
    print(latex)


if __name__ == "__main__":
    main()
