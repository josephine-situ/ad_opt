"""Rolling-window backtest with simulated cold-start masking.

Implements a chronological rolling backtest (no random splits):
- Iterate day-by-day from start_date to end_date.
- At each day t:
  - Train models on history <= t-1 after masking 20% of t-1 combo-keywords.
  - Optimize bids with an exploration bonus on (masked + real-new) combos.
  - Score profit via proportional replay; apply a discovery-rate tax for new combos.

Model choices (per user spec):
- EPC model: Ridge regression (scikit-learn)
- Clicks model: XGBoost regressor

Notes
-----
- Treat every (Keyword, Match type, Region) as a separate keyword (a "combo-keyword").
- The exploration term is a solver-side carrot only; it is NOT counted in backtest profit.

Outputs
-------
Writes a concise bundle under:
  opt_results/backtests/rolling_window_{start}_{end}/
    - config.json
    - daily_results.csv
    - lambda_summary.csv

Optionally saves per-day bids if --save-daily-bids is set.

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Add repo root for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_pipeline import get_gkp_data, impute_missing_data
from utils.keyword_matching import fuzzy_fill_from_gkp


try:
    import joblib  # type: ignore
except Exception as e:
    raise ImportError("joblib is required (installed with scikit-learn).") from e

try:
    import gurobipy as gp  # type: ignore
except Exception as e:
    raise ImportError(
        "gurobipy is required for optimization. Install with: pip install gurobipy"
    ) from e

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


from scripts.bid_optimization import optimize_bids_embedded
from scripts.bid_optimization import compute_clicks_bid_bounds_from_history


DISCOVERY_RATE_DEFAULT = 1.00


def _normalize_str(x: object) -> str:
    return str(x).strip()


def _combo_key(keyword: object, match_type: object, region: object) -> str:
    return f"{_normalize_str(keyword).lower()}||{_normalize_str(match_type)}||{_normalize_str(region)}"


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s).normalize()


def _date_range_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if end < start:
        return []
    days = int((end - start).days)
    return [start + pd.Timedelta(days=i) for i in range(days + 1)]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_float32_csr(X):
    # Avoid importing scipy; xgboost accepts numpy arrays and sparse matrices.
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(X):
            return X.tocsr().astype(np.float32)
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Match the repo's open-source modeling preprocessor."""

    categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor, numeric_cols, categorical_cols


def get_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "epc":
        excluded = {"Conv. value", "Clicks", "EPC", "Day", "Keyword"}
    elif target == "clicks":
        excluded = {"Clicks", "Conv. value", "EPC", "Day", "Keyword"}
    else:
        raise ValueError(f"Unsupported target: {target}")

    cols = [c for c in df.columns if c not in excluded]
    return df[cols].copy()


def get_target(df: pd.DataFrame, target: str) -> pd.Series:
    if target == "epc":
        return df["EPC"]
    if target == "clicks":
        return df["Clicks"]
    raise ValueError(f"Unsupported target: {target}")


def prepare_xyw(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    """Prepare X, y, and optional sample weights.

    For EPC:
      - Filter Clicks>0 (EPC undefined otherwise)
      - sample_weight = Clicks
    """

    if target == "epc":
        mask = df["Clicks"] > 0
        df_use = df.loc[mask].copy()
        sample_weight = df_use["Clicks"].astype(float)
    else:
        df_use = df.copy()
        sample_weight = None

    X = get_features(df_use, target)
    y = get_target(df_use, target)
    return X, y, sample_weight


@dataclass(frozen=True)
class TrainedModelArtifacts:
    epc_ridge_model_path: Path
    epc_ridge_preproc_path: Path
    clicks_xgb_model_path: Path
    clicks_xgb_preproc_path: Path


def train_models_no_cv(
    *,
    train_df: pd.DataFrame,
    embedding_method: str,
    out_dir: Path,
    seed: int,
) -> TrainedModelArtifacts:
    """Train Ridge(EPC) + XGB(clicks) without random CV splits."""

    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost (or `pip install -e .[ml_open]`)."
        )

    _ensure_dir(out_dir)

    # --- EPC: Ridge ---
    X_epc, y_epc, w_epc = prepare_xyw(train_df, "epc")
    preproc_epc, _, _ = build_preprocessor(X_epc)

    ridge = Ridge(alpha=5.0)
    ridge_pipe = Pipeline(steps=[("preprocess", preproc_epc), ("model", ridge)])

    fit_kwargs: Dict[str, object] = {}
    if w_epc is not None:
        fit_kwargs["model__sample_weight"] = w_epc

    ridge_pipe.fit(X_epc, y_epc, **fit_kwargs)

    epc_ridge_model_path = out_dir / f"ridge_{embedding_method}_epc.joblib"
    epc_ridge_preproc_path = out_dir / f"ridge_{embedding_method}_epc_preprocess.joblib"
    joblib.dump(ridge_pipe, epc_ridge_model_path)
    joblib.dump(ridge_pipe.named_steps["preprocess"], epc_ridge_preproc_path)

    # --- Clicks: XGB ---
    X_clk, y_clk, _w_clk = prepare_xyw(train_df, "clicks")
    preproc_clk, _, _ = build_preprocessor(X_clk)

    # Keep the ensemble small/shallow to remain embeddable.
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_estimators=20,
        learning_rate=0.3,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=int(os.cpu_count() or 1),
    )

    xgb_pipe = Pipeline(
        steps=[
            ("preprocess", preproc_clk),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", model),
        ]
    )

    xgb_pipe.fit(X_clk, y_clk)

    clicks_xgb_model_path = out_dir / f"xgb_mse_{embedding_method}_clicks.json"
    clicks_xgb_preproc_path = out_dir / f"xgb_mse_{embedding_method}_clicks_preprocess.joblib"
    xgb_pipe.named_steps["model"].save_model(clicks_xgb_model_path)
    joblib.dump(xgb_pipe.named_steps["preprocess"], clicks_xgb_preproc_path)

    return TrainedModelArtifacts(
        epc_ridge_model_path=epc_ridge_model_path,
        epc_ridge_preproc_path=epc_ridge_preproc_path,
        clicks_xgb_model_path=clicks_xgb_model_path,
        clicks_xgb_preproc_path=clicks_xgb_preproc_path,
    )


def load_keyword_embeddings(embedding_method: str) -> pd.DataFrame:
    """Load the repo's precomputed unique keyword embeddings."""

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "data" / "clean" / f"unique_keyword_embeddings_{embedding_method}.csv",
        repo_root / "data" / "embeddings" / f"unique_keyword_embeddings_{embedding_method}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "Keyword" not in df.columns:
                raise ValueError(f"Embeddings file {p} is missing 'Keyword' column")
            return df
    raise FileNotFoundError(
        f"Could not find unique keyword embeddings for '{embedding_method}'. Tried: {candidates}"
    )


def build_candidate_feature_matrix(
    *,
    keyword_df: pd.DataFrame,
    combos_df: pd.DataFrame,
    target_day: pd.Timestamp,
    embedding_method: str,
    gkp_dir: str,
) -> Tuple[pd.DataFrame, List[int], List[str], List[str]]:
    """Build a feature matrix for a specific list of (Keyword, Region, Match type) combos."""

    # Normalize day
    filter_day = pd.to_datetime(target_day).normalize()

    # Ensure required columns exist
    required_cols = {"Keyword", "Region", "Match type"}
    missing = required_cols - set(combos_df.columns)
    if missing:
        raise ValueError(f"combos_df missing columns: {sorted(missing)}")

    # Build base
    combo_df = combos_df[["Keyword", "Region", "Match type"]].copy()
    combo_df["Day"] = filter_day

    # Merge embeddings
    kw_emb = keyword_df.copy()
    kw_emb["Keyword_join"] = kw_emb["Keyword"].astype(str).str.lower().str.strip()
    combo_df["Keyword_join"] = combo_df["Keyword"].astype(str).str.lower().str.strip()

    result = combo_df.merge(
        kw_emb.drop(columns=["Keyword"], errors="ignore"),
        on="Keyword_join",
        how="left",
    )

    # Merge GKP
    gkp_df = get_gkp_data(gkp_dir=gkp_dir)
    if not gkp_df.empty:
        gkp_df = gkp_df.copy()
        gkp_df["Keyword"] = gkp_df["Keyword"].astype(str).str.lower().str.strip()
        gkp_df["Keyword_join"] = gkp_df["Keyword"]

        result = result.merge(
            gkp_df.drop(columns=["Keyword"], errors="ignore"),
            left_on="Keyword_join",
            right_on="Keyword_join",
            how="left",
        )

        fuzzy_fill_from_gkp(
            result,
            keyword_col="Keyword",
            gkp_df=gkp_df,
            gkp_keyword_col="Keyword",
            verbose=False,
            source_display_col="Keyword",
            print_all_mappings=False,
        )

    # Time-series stats
    search_cols = sorted([c for c in result.columns if c.startswith("searches_")])
    if search_cols:

        def _ts_stats_from_row(row: pd.Series) -> pd.Series:
            values = []
            for c in search_cols:
                try:
                    values.append(float(row[c]))
                except Exception:
                    values.append(np.nan)

            last_val = values[-1] if values else np.nan

            three = [v for v in values[-3:] if not np.isnan(v)]
            six = [v for v in values[-6:] if not np.isnan(v)]

            three_avg = (sum(three) / len(three)) if three else np.nan
            six_avg = (sum(six) / len(six)) if six else np.nan

            mom = np.nan
            if len(values) >= 2 and not np.isnan(values[-1]) and not np.isnan(values[-2]):
                prev_val = values[-2]
                curr_val = values[-1]
                if prev_val > 0:
                    mom = ((curr_val - prev_val) / prev_val) * 100.0
                else:
                    mom = 100.0 if curr_val > 0 else 0.0

            trend = np.nan
            if len(six) >= 2:
                x = np.arange(len(six), dtype=float)
                y = np.array(six, dtype=float)
                trend = np.polyfit(x, y, 1)[0]

            return pd.Series(
                {
                    "last_month_searches": last_val,
                    "three_month_avg": three_avg,
                    "six_month_avg": six_avg,
                    "mom_change": mom,
                    "search_trend": trend,
                }
            )

        ts_stats = result.apply(_ts_stats_from_row, axis=1)
        for col in ts_stats.columns:
            result[col] = ts_stats[col]

        result = result.drop(columns=search_cols, errors="ignore")

    # Provide Avg. CPC proxy if missing
    if "Avg. CPC" not in result.columns:
        low = result.get("Top of page bid (low range)")
        high = result.get("Top of page bid (high range)")
        if low is not None and high is not None:
            result["Avg. CPC"] = (
                pd.to_numeric(low, errors="coerce") + pd.to_numeric(high, errors="coerce")
            ) / 2.0
        else:
            result["Avg. CPC"] = 0.0

    # Impute
    result = impute_missing_data(result)

    # Date features
    from utils.date_features import calculate_date_features

    regions = sorted(result["Region"].astype(str).unique().tolist())
    date_features = calculate_date_features(filter_day, regions=regions)
    result["day_of_week"] = date_features["day_of_week"]
    result["is_weekend"] = date_features["is_weekend"]
    result["month"] = date_features["month"]
    result["is_public_holiday"] = date_features["is_public_holiday"]
    result["days_to_next_course_start"] = date_features["days_to_next_course_start"]

    # Mapping lists for extract_solution
    # Here we treat each combo as its own "keyword" by mapping rows back to keyword_df index.
    keyword_idx_list: List[int] = []
    region_list: List[str] = []
    match_list: List[str] = []

    kw_lookup = {str(k).lower().strip(): int(i) for i, k in enumerate(keyword_df["Keyword"].astype(str))}

    for _, row in result.iterrows():
        kw_key = str(row["Keyword"]).lower().strip()
        if kw_key not in kw_lookup:
            raise KeyError(f"Keyword '{row['Keyword']}' not found in keyword_df")
        keyword_idx_list.append(kw_lookup[kw_key])
        region_list.append(str(row["Region"]))
        match_list.append(str(row["Match type"]))

    # Drop helper columns
    result = result.drop(columns=["Day", "Keyword", "Keyword_join", "_kw_key"], errors="ignore")

    # Match repo conventions
    result.columns = result.columns.str.replace(".", "_", regex=False)

    # Keep raw categoricals as strings for Ridge/XGB preprocessors
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        result[col] = result[col].astype(float)

    result.reset_index(drop=True, inplace=True)
    return result, keyword_idx_list, region_list, match_list


def _safe_ratio(bid: float, cpc_obs: float, cap: float) -> float:
    if bid <= 0:
        return 0.0
    if cpc_obs is None or not np.isfinite(cpc_obs) or cpc_obs <= 0:
        return float(cap)
    return float(min(bid / cpc_obs, cap))


def _mvar_vals(v) -> np.ndarray:
    if hasattr(v, "X"):
        return np.asarray(v.X, dtype=float)
    return np.asarray([vv.X for vv in v], dtype=float)


def _aggregate_observed_by_combo(day_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed logs at combo level (robust to duplicates)."""

    d = day_df.copy()
    d["combo_key"] = d.apply(lambda r: _combo_key(r["Keyword"], r["Match type"], r["Region"]), axis=1)
    obs = (
        d.groupby("combo_key", as_index=False)
        .agg(
            {
                "Clicks": "sum",
                "Cost": "sum",
                "Conv. value": "sum",
            }
        )
        .copy()
    )
    obs = obs.rename(
        columns={
            "Clicks": "obs_clicks",
            "Cost": "obs_cost",
            "Conv. value": "obs_conv_value",
        }
    )
    obs["obs_cpc"] = np.where(obs["obs_clicks"] > 0, obs["obs_cost"] / obs["obs_clicks"], np.nan)
    obs["obs_epc"] = np.where(obs["obs_clicks"] > 0, obs["obs_conv_value"] / obs["obs_clicks"], 0.0)
    return obs


def score_day(
    *,
    day_df: pd.DataFrame,
    bids_df: pd.DataFrame,
    new_combo_keys: Set[str],
    discovery_rate: float,
    ratio_cap: float,
) -> Dict[str, float]:
    """Compute replay metrics for day t using proportional replay + new-keyword tax."""

    obs = _aggregate_observed_by_combo(day_df)

    bids_df = bids_df.copy()
    bids_df["combo_key"] = bids_df.apply(lambda r: _combo_key(r["keyword"], r["match"], r["region"]), axis=1)
    bid_map = bids_df.set_index("combo_key")["bid"].to_dict()

    profits_old = 0.0
    profits_new_taxed = 0.0
    profits_new_raw = 0.0

    replay_cost_total = 0.0
    replay_revenue_total = 0.0
    n_old = 0
    n_new = 0

    observed_clicks_total = 0.0
    replay_clicks_total = 0.0

    for _, row in obs.iterrows():
        ck = str(row["combo_key"])
        clicks_obs = float(row["obs_clicks"])
        cost_obs = float(row["obs_cost"])
        conv_obs = float(row["obs_conv_value"])

        cpc_obs = (cost_obs / clicks_obs) if clicks_obs > 0 else float("nan")
        epc_obs = (conv_obs / clicks_obs) if clicks_obs > 0 else 0.0

        bid = float(bid_map.get(ck, 0.0))
        ratio = _safe_ratio(bid, cpc_obs, ratio_cap)
        new_clicks = clicks_obs * ratio

        observed_clicks_total += float(clicks_obs)
        replay_clicks_total += float(new_clicks)

        revenue = new_clicks * epc_obs
        cost = new_clicks * bid
        raw_profit = revenue - cost

        replay_revenue_total += float(revenue)
        replay_cost_total += float(cost)

        if ck in new_combo_keys:
            profits_new_raw += float(raw_profit)
            profits_new_taxed += float(raw_profit) * float(discovery_rate)
            n_new += 1
        else:
            profits_old += raw_profit
            n_old += 1

    return {
        "profit_old": float(profits_old),
        "profit_new": float(profits_new_taxed),
        "profit_new_raw": float(profits_new_raw),
        "profit_total": float(profits_old + profits_new_taxed),
        "observed_clicks_total": float(observed_clicks_total),
        "replay_clicks_total": float(replay_clicks_total),
        "replay_revenue_total": float(replay_revenue_total),
        "replay_cost_total": float(replay_cost_total),
        "n_old": float(n_old),
        "n_new": float(n_new),
    }


def solve_from_base_lp(
    *,
    base_lp_path: Path,
    exploration_lambda: float,
    new_row_mask: np.ndarray,
    time_limit_s: int,
    solver_output_flag: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Reload cached base formulation and re-optimize for a new λ.

    This keeps the same constraints and base objective, and just adds
    + λ * sum_{i in new} b_i.

    Returns bids_df and (b,f,g) vectors extracted from the solved model.
    """

    model = gp.read(str(base_lp_path))
    model.setParam("TimeLimit", int(time_limit_s))
    model.setParam("OutputFlag", int(solver_output_flag))
    model.setParam("NonConvex", 2)

    base_obj = model.getObjective()

    new_idx = np.where(np.asarray(new_row_mask, dtype=bool))[0]
    if float(exploration_lambda) != 0.0 and len(new_idx) > 0:
        explore = gp.quicksum(model.getVarByName(f"b[{int(i)}]") for i in new_idx)
        model.setObjective(base_obj + float(exploration_lambda) * explore, gp.GRB.MAXIMIZE)
    else:
        model.setObjective(base_obj, gp.GRB.MAXIMIZE)

    model.optimize()

    # Extract arrays by name.
    K = len(new_row_mask)
    b = np.array([model.getVarByName(f"b[{i}]").X for i in range(K)], dtype=float)
    f = np.array([model.getVarByName(f"f[{i}]").X for i in range(K)], dtype=float)
    g = np.array([model.getVarByName(f"g[{i}]").X for i in range(K)], dtype=float)

    # For a reload-based solve, we don't reconstruct the full mapping here.
    # Caller should build bids_df from these arrays with their own mapping.
    return model, b, f, g


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()

    ap.add_argument("--embedding-method", default="bert", choices=["bert", "tfidf"])
    ap.add_argument("--start-date", default="2025-11-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument(
        "--step-days",
        type=int,
        default=1,
        help="Step size for rolling iteration (1=day-by-day, 7=week-by-week).",
    )

    ap.add_argument(
        "--lambdas",
        default="0,10,100",
        help="Comma-separated λ values for exploration bonus (e.g. '0,0.1,0.5').",
    )

    ap.add_argument("--mask-frac", type=float, default=0.20)
    ap.add_argument("--discovery-rate", type=float, default=DISCOVERY_RATE_DEFAULT)
    ap.add_argument("--ratio-cap", type=float, default=2.00)

    ap.add_argument("--gkp-dir", default="data/gkp")
    ap.add_argument("--data-path", default="")

    ap.add_argument("--max-bid", type=float, default=100.0)
    ap.add_argument(
        "--no-historical-bid-bounds",
        dest="historical_bid_bounds",
        action="store_false",
        help="Disable per-row bid bounds derived from historical Avg. CPC up to t-1 (default: enabled).",
    )
    ap.set_defaults(historical_bid_bounds=True)
    ap.add_argument(
        "--budget-mode",
        choices=["observed"],
        default="observed",
        help="Budget per day. Currently supports only 'observed' (sum of observed Cost for day t).",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=300)
    ap.add_argument("--solver-output", type=int, default=0)

    ap.add_argument(
        "--universe",
        choices=["train", "t-1"],
        default="train",
        help="Candidate universe. 'train' = all combos seen in training history (unmasked). 't-1' = only combos in day t-1.",
    )

    ap.add_argument(
        "--save-daily-bids",
        action="store_true",
        default=True,
        help="Save per-day bid tables to opt_results/backtests/.../bids (default: on).",
    )
    ap.add_argument(
        "--no-save-daily-bids",
        dest="save_daily_bids",
        action="store_false",
        help="Disable saving per-day bid tables.",
    )
    ap.add_argument(
        "--keep-day-cache",
        action="store_true",
        default=True,
        help="Keep per-day base formulation cache files (default: on).",
    )
    ap.add_argument(
        "--no-keep-day-cache",
        dest="keep_day_cache",
        action="store_false",
        help="Delete per-day cache files after each day (smaller output).",
    )
    ap.add_argument(
        "--keep-day-models",
        action="store_true",
        default=True,
        help="Keep per-day trained model artifacts (default: on).",
    )
    ap.add_argument(
        "--no-keep-day-models",
        dest="keep_day_models",
        action="store_false",
        help="Delete per-day trained model artifacts after each day (smaller output).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run by skipping already-computed (day, lambda) rows in daily_results.csv.",
    )

    args = ap.parse_args()

    start = _parse_date(args.start_date)
    end = _parse_date(args.end_date)
    lambdas = [float(x) for x in str(args.lambdas).split(",") if str(x).strip() != ""]
    if not lambdas:
        lambdas = [0.0]

    embedding_method = str(args.embedding_method)

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = repo_root / "data" / "clean" / f"ad_opt_data_{embedding_method}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df["Day"] = pd.to_datetime(df["Day"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Day"]).copy()

    # Ensure EPC exists (should already in cleaned data)
    if "EPC" not in df.columns:
        df["EPC"] = np.where(df["Clicks"] > 0, df["Conv. value"] / df["Clicks"], 0.0)

    # Keep all rows up to end date for training history; only score days in [start, end].
    df = df[df["Day"] <= end].copy()

    out_root = repo_root / "opt_results" / "backtests" / f"rolling_window_{start.date().isoformat()}_{end.date().isoformat()}"
    _ensure_dir(out_root)

    if args.save_daily_bids:
        _ensure_dir(out_root / "bids")

    cfg = {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "embedding_method": embedding_method,
        "mask_frac": float(args.mask_frac),
        "discovery_rate": float(args.discovery_rate),
        "ratio_cap": float(args.ratio_cap),
        "lambdas": lambdas,
        "universe": str(args.universe),
        "max_bid": float(args.max_bid),
        "historical_bid_bounds": bool(args.historical_bid_bounds),
        "budget_mode": str(args.budget_mode),
        "time_limit_s": int(args.time_limit),
        "seed": int(args.seed),
        "step_days": int(args.step_days),
    }
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    kw_emb = load_keyword_embeddings(embedding_method)
    kw_emb_key_to_idx: Dict[str, int] = {}
    for i, v in enumerate(kw_emb["Keyword"].astype(str).str.lower().str.strip().tolist()):
        if v not in kw_emb_key_to_idx:
            kw_emb_key_to_idx[v] = int(i)

    rng_global = np.random.default_rng(int(args.seed))

    daily_path = out_root / "daily_results.csv"
    actuals_path = out_root / "actuals_by_day.csv"

    # Default behavior is to start fresh unless --resume is set.
    if not args.resume:
        daily_path.unlink(missing_ok=True)
        actuals_path.unlink(missing_ok=True)
    done_pairs: Set[Tuple[str, float]] = set()
    if args.resume and daily_path.exists():
        try:
            existing = pd.read_csv(daily_path)
            if {"day", "lambda"}.issubset(existing.columns):
                for _, r in existing[["day", "lambda"]].dropna().iterrows():
                    done_pairs.add((str(r["day"]), float(r["lambda"])))
        except Exception:
            # If resume parsing fails, fall back to recomputing.
            done_pairs = set()

    # Only iterate over scoring days in the backtest window.
    days_present = sorted(df.loc[(df["Day"] >= start) & (df["Day"] <= end), "Day"].unique())
    step_days = int(args.step_days)
    if step_days < 1:
        raise ValueError("--step-days must be >= 1")
    if step_days > 1:
        days_present = days_present[::step_days]

    for t in days_present:
        t = pd.to_datetime(t).normalize()
        t_minus_1 = (t - pd.Timedelta(days=1)).normalize()

        # Day t logs (scoring day). Always write actual baseline metrics if present.
        day_t = df[df["Day"] == t].copy()
        if day_t.empty:
            continue
        day_t["combo_key"] = day_t.apply(lambda r: _combo_key(r["Keyword"], r["Match type"], r["Region"]), axis=1)

        actual_cost = float(day_t["Cost"].sum())
        actual_revenue = float(day_t["Conv. value"].sum())
        actual_clicks = float(day_t["Clicks"].sum())
        actual_profit = float(actual_revenue - actual_cost)
        actual_row = {
            "day": t.date().isoformat(),
            "actual_cost": actual_cost,
            "actual_revenue": actual_revenue,
            "actual_clicks": actual_clicks,
            "actual_profit": actual_profit,
        }
        if not actuals_path.exists():
            pd.DataFrame([actual_row]).to_csv(actuals_path, index=False)
        else:
            # Append only if this day isn't already present
            try:
                existing_days = set(pd.read_csv(actuals_path)["day"].astype(str).tolist())
            except Exception:
                existing_days = set()
            if actual_row["day"] not in existing_days:
                pd.DataFrame([actual_row]).to_csv(actuals_path, mode="a", header=False, index=False)

        # Training history: strictly before t
        hist = df[df["Day"] <= t_minus_1].copy()
        if hist.empty:
            # No history: skip optimization (nothing to train)
            continue

        # Build combo keys
        hist["combo_key"] = hist.apply(lambda r: _combo_key(r["Keyword"], r["Match type"], r["Region"]), axis=1)

        day_tm1 = df[df["Day"] == t_minus_1].copy()
        if not day_tm1.empty:
            day_tm1["combo_key"] = day_tm1.apply(
                lambda r: _combo_key(r["Keyword"], r["Match type"], r["Region"]), axis=1
            )
            combos_tm1 = sorted(day_tm1["combo_key"].unique().tolist())
        else:
            combos_tm1 = []

        # --- Masking: select 20% of combos that have history in t-1 ---
        mask_frac = float(args.mask_frac)
        n_mask = int(np.floor(mask_frac * len(combos_tm1)))
        # Deterministic per-day RNG for reproducibility across λ sweeps
        day_seed = int(args.seed) + int(t.strftime("%Y%m%d"))
        rng = np.random.default_rng(day_seed)
        masked: Set[str] = set(rng.choice(combos_tm1, size=n_mask, replace=False).tolist()) if n_mask > 0 else set()

        # Remove masked combos from ALL training history to simulate cold-start
        train_df = hist[~hist["combo_key"].isin(masked)].copy()

        # Real new combos = combos in day t that are NOT present in unmasked training
        train_combos = set(train_df["combo_key"].unique().tolist())
        combos_t = set(day_t["combo_key"].unique().tolist())
        real_new = {ck for ck in combos_t if ck not in train_combos}
        new_combo_keys = set(masked) | set(real_new)

        # Candidate universe
        if str(args.universe) == "train":
            candidate_combo_keys = set(train_combos) | combos_t | masked
        else:
            candidate_combo_keys = set(combos_tm1) | combos_t

        # Build candidate combos_df
        # We need Keyword/Region/Match type per combo key; pull from hist/day_t.
        ref = pd.concat([hist[["Keyword", "Match type", "Region", "combo_key"]], day_t[["Keyword", "Match type", "Region", "combo_key"]]], axis=0)
        ref = ref.drop_duplicates(subset=["combo_key"]).copy()
        ref = ref[ref["combo_key"].isin(candidate_combo_keys)].copy()
        combos_df = ref[["Keyword", "Region", "Match type", "combo_key"]].copy()

        # Build keyword_df subset for embeddings
        needed_keywords = sorted(combos_df["Keyword"].astype(str).str.lower().str.strip().unique().tolist())
        kw_emb_use = kw_emb.copy()
        kw_emb_use["kw_key"] = kw_emb_use["Keyword"].astype(str).str.lower().str.strip()
        kw_subset = kw_emb_use[kw_emb_use["kw_key"].isin(needed_keywords)].drop(columns=["kw_key"], errors="ignore")

        if kw_subset.empty:
            continue

        # --- Train models for the day ---
        day_dir = out_root / "_day_artifacts" / f"day={t.date().isoformat()}"
        _ensure_dir(day_dir)

        artifacts = train_models_no_cv(
            train_df=train_df.drop(columns=["combo_key"], errors="ignore"),
            embedding_method=embedding_method,
            out_dir=day_dir,
            seed=int(args.seed),
        )

        # --- Build candidate feature matrix for optimization ---
        gkp_dir = Path(str(args.gkp_dir))
        if not gkp_dir.is_absolute():
            gkp_dir = repo_root / gkp_dir

        X_raw, kw_idx_list, region_list, match_list = build_candidate_feature_matrix(
            keyword_df=kw_subset,
            combos_df=combos_df,
            target_day=t,
            embedding_method=embedding_method,
            gkp_dir=str(gkp_dir),
        )

        # New-row mask (aligned to candidate rows)
        row_combo_keys = combos_df["combo_key"].astype(str).tolist()
        new_row_mask = np.array([ck in new_combo_keys for ck in row_combo_keys], dtype=bool)

        # Budget
        if str(args.budget_mode) == "observed":
            budget = actual_cost
        else:
            raise ValueError(f"Unsupported budget_mode: {args.budget_mode}")

        # Per-day base formulation cache (used for λ re-solves)
        cache_dir = out_root / "_day_cache" / f"day={t.date().isoformat()}"
        _ensure_dir(cache_dir)
        base_lp_path = cache_dir / "base.lp"

        # Optional: per-row historical bid bounds from history up to t-1.
        # Compute once per day (reused for all λ re-solves).
        bid_lbs: Optional[np.ndarray] = None
        bid_ubs: Optional[np.ndarray] = None
        if bool(args.historical_bid_bounds):
            # IMPORTANT: use unmasked training history (train_df) so masked combos do not
            # receive their own exact historical bounds.
            training_csv = cache_dir / "training_history_for_bounds.csv"
            cpc_col = next((c for c in ("Avg. CPC", "Avg_ CPC") if c in train_df.columns), None)
            cols = ["Keyword", "Region", "Match type"] + ([cpc_col] if cpc_col is not None else [])
            try:
                train_df[cols].to_csv(training_csv, index=False)
            except Exception:
                # If we can't persist history for bounds, fall back to no bounds.
                training_csv = Path("__missing__")

            kw_idx_bounds: List[int] = []
            missing_kw = False
            for kw in combos_df["Keyword"].astype(str).str.lower().str.strip().tolist():
                idx = kw_emb_key_to_idx.get(kw)
                if idx is None:
                    missing_kw = True
                    idx = 0
                kw_idx_bounds.append(int(idx))

            if missing_kw:
                # Still proceed; bounds helper will fall back to exact-only or skip.
                pass

            bid_lbs, bid_ubs, bounds_debug = compute_clicks_bid_bounds_from_history(
                training_csv=training_csv,
                keyword_df=kw_emb,
                kw_idx_list=kw_idx_bounds,
                region_list=region_list,
                match_list=match_list,
                embedding_method=embedding_method,
                max_bid_cap=float(args.max_bid),
                min_active_bid=0.01,
            )
            try:
                bounds_debug.to_csv(cache_dir / "bid_bounds_debug.csv", index=False)
            except Exception:
                pass

        # Solve for each λ.
        # First λ builds the full model and writes base.lp; subsequent λ reload base.lp.
        bids_by_lambda: Dict[float, pd.DataFrame] = {}

        for j, lmbda in enumerate(lambdas):
            day_key = t.date().isoformat()
            pair = (day_key, float(lmbda))
            if pair in done_pairs:
                continue

            if j == 0:
                model, b, f, g = optimize_bids_embedded(
                    X_ort=X_raw,
                    X_lr=None,
                    weights_dict=None,
                    budget=budget,
                    max_bid=float(args.max_bid),
                    bid_lower_bounds=bid_lbs,
                    bid_upper_bounds=bid_ubs,
                    epc_model=None,
                    clicks_model=None,
                    epc_xgb_paths=None,
                    clicks_xgb_paths=(artifacts.clicks_xgb_model_path, artifacts.clicks_xgb_preproc_path),
                    epc_ridge_preproc_path=artifacts.epc_ridge_preproc_path,
                    clicks_ridge_preproc_path=None,
                    epc_ridge_model_path=artifacts.epc_ridge_model_path,
                    clicks_ridge_model_path=None,
                    alg_epc="ridge",
                    alg_clicks="xgb",
                    embedding_method=embedding_method,
                    exploration_lambda=float(lmbda),
                    new_row_mask=new_row_mask,
                    save_base_formulation_to=base_lp_path if len(lambdas) > 1 else None,
                    write_lp=False,
                    solver_output_flag=int(args.solver_output),
                )

                b_vals = _mvar_vals(b)
                f_vals = _mvar_vals(f)
                g_vals = _mvar_vals(g)

                K = len(new_row_mask)
                if len(b_vals) != K:
                    raise RuntimeError(f"Solver returned {len(b_vals)} bids but expected {K}")

                bids_df = pd.DataFrame(
                    {
                        "keyword": [kw_subset.iloc[kw_idx_list[i]]["Keyword"] for i in range(K)],
                        "region": region_list,
                        "match": match_list,
                        "bid": b_vals,
                        "bid_lb": (bid_lbs if bid_lbs is not None else np.full(K, np.nan)).astype(float),
                        "bid_ub": (bid_ubs if bid_ubs is not None else np.full(K, np.nan)).astype(float),
                        "predicted_clicks": g_vals,
                        "predicted_epc": f_vals,
                        "is_new_keyword": new_row_mask.astype(bool),
                    }
                )
                bids_df["conv_value"] = bids_df["predicted_clicks"] * bids_df["predicted_epc"]
                bids_df["cost"] = bids_df["bid"] * bids_df["predicted_clicks"]
                bids_df["profit"] = bids_df["conv_value"] - bids_df["cost"]

                # Add observed (day t) clicks/cost/value per combo for debugging.
                obs_combo = _aggregate_observed_by_combo(day_t)
                bids_df["combo_key"] = bids_df.apply(
                    lambda r: _combo_key(r["keyword"], r["match"], r["region"]), axis=1
                )
                bids_df = bids_df.merge(obs_combo, on="combo_key", how="left")
                for c in ("obs_clicks", "obs_cost", "obs_conv_value"):
                    if c in bids_df.columns:
                        bids_df[c] = bids_df[c].fillna(0.0)

                # Per-combo proportional replay diagnostics
                cap = float(args.ratio_cap)
                bids_df["replay_ratio"] = bids_df.apply(
                    lambda r: _safe_ratio(float(r["bid"]), float(r.get("obs_cpc", float("nan"))), cap), axis=1
                )
                bids_df["replay_clicks"] = bids_df["obs_clicks"].astype(float) * bids_df["replay_ratio"].astype(float)
                bids_df["replay_revenue"] = bids_df["replay_clicks"].astype(float) * bids_df.get("obs_epc", 0.0).astype(float)
                bids_df["replay_cost"] = bids_df["replay_clicks"].astype(float) * bids_df["bid"].astype(float)
                bids_df["replay_profit"] = bids_df["replay_revenue"].astype(float) - bids_df["replay_cost"].astype(float)

            else:
                # Reload and re-solve from cached base formulation.
                model_reload = gp.read(str(base_lp_path))
                model_reload.setParam("TimeLimit", int(args.time_limit))
                model_reload.setParam("OutputFlag", int(args.solver_output))
                model_reload.setParam("NonConvex", 2)

                base_obj = model_reload.getObjective()
                new_idx = np.where(new_row_mask)[0]
                if float(lmbda) != 0.0 and len(new_idx) > 0:
                    explore = gp.quicksum(model_reload.getVarByName(f"b[{int(i)}]") for i in new_idx)
                    model_reload.setObjective(base_obj + float(lmbda) * explore, gp.GRB.MAXIMIZE)
                else:
                    model_reload.setObjective(base_obj, gp.GRB.MAXIMIZE)

                model_reload.optimize()

                # Extract values and build bids_df consistent with extract_solution output.
                K = len(new_row_mask)
                b_vals = np.array([model_reload.getVarByName(f"b[{i}]").X for i in range(K)], dtype=float)
                f_vals = np.array([model_reload.getVarByName(f"f[{i}]").X for i in range(K)], dtype=float)
                g_vals = np.array([model_reload.getVarByName(f"g[{i}]").X for i in range(K)], dtype=float)

                bids_df = pd.DataFrame(
                    {
                        "keyword": [kw_subset.iloc[kw_idx_list[i]]["Keyword"] for i in range(K)],
                        "region": region_list,
                        "match": match_list,
                        "bid": b_vals,
                        "bid_lb": (bid_lbs if bid_lbs is not None else np.full(K, np.nan)).astype(float),
                        "bid_ub": (bid_ubs if bid_ubs is not None else np.full(K, np.nan)).astype(float),
                        "predicted_clicks": g_vals,
                        "predicted_epc": f_vals,
                        "is_new_keyword": new_row_mask.astype(bool),
                    }
                )
                bids_df["conv_value"] = bids_df["predicted_clicks"] * bids_df["predicted_epc"]
                bids_df["cost"] = bids_df["bid"] * bids_df["predicted_clicks"]
                bids_df["profit"] = bids_df["conv_value"] - bids_df["cost"]

                # Add observed (day t) clicks/cost/value per combo for debugging.
                obs_combo = _aggregate_observed_by_combo(day_t)
                bids_df["combo_key"] = bids_df.apply(
                    lambda r: _combo_key(r["keyword"], r["match"], r["region"]), axis=1
                )
                bids_df = bids_df.merge(obs_combo, on="combo_key", how="left")
                for c in ("obs_clicks", "obs_cost", "obs_conv_value"):
                    if c in bids_df.columns:
                        bids_df[c] = bids_df[c].fillna(0.0)

                # Per-combo proportional replay diagnostics
                cap = float(args.ratio_cap)
                bids_df["replay_ratio"] = bids_df.apply(
                    lambda r: _safe_ratio(float(r["bid"]), float(r.get("obs_cpc", float("nan"))), cap), axis=1
                )
                bids_df["replay_clicks"] = bids_df["obs_clicks"].astype(float) * bids_df["replay_ratio"].astype(float)
                bids_df["replay_revenue"] = bids_df["replay_clicks"].astype(float) * bids_df.get("obs_epc", 0.0).astype(float)
                bids_df["replay_cost"] = bids_df["replay_clicks"].astype(float) * bids_df["bid"].astype(float)
                bids_df["replay_profit"] = bids_df["replay_revenue"].astype(float) - bids_df["replay_cost"].astype(float)

            bids_by_lambda[float(lmbda)] = bids_df

            # Optional: save daily bids
            if args.save_daily_bids:
                lam_tag = str(float(lmbda)).replace(".", "p").replace("-", "m")
                bids_df.to_csv(out_root / "bids" / f"bids_{t.date().isoformat()}_lambda_{lam_tag}.csv", index=False)

            # Score
            scored = score_day(
                day_df=day_t,
                bids_df=bids_df,
                new_combo_keys=new_combo_keys,
                discovery_rate=float(args.discovery_rate),
                ratio_cap=float(args.ratio_cap),
            )

            row = {
                "day": day_key,
                "lambda": float(lmbda),
                "budget": float(budget),
                "n_candidates": int(len(new_row_mask)),
                "n_new_candidates": int(new_row_mask.sum()),
                "n_masked": int(len(masked)),
                "n_real_new": int(len(real_new)),
                "actual_cost": float(actual_cost),
                "actual_revenue": float(actual_revenue),
                "actual_clicks": float(actual_clicks),
                "actual_profit": float(actual_profit),
                **scored,
            }

            # Improvements vs actual observed baseline
            row["profit_improvement"] = float(row["profit_total"] - actual_profit)
            row["profit_improvement_pct"] = float(row["profit_improvement"] / actual_profit) if actual_profit != 0 else float("nan")
            row["cost_change"] = float(row.get("replay_cost_total", float("nan")) - actual_cost)
            row["cost_change_pct"] = float(row["cost_change"] / actual_cost) if actual_cost != 0 else float("nan")

            row_df = pd.DataFrame([row])

            if daily_path.exists() and args.resume:
                # Guard against schema mismatch when resuming.
                try:
                    with open(daily_path, "r", encoding="utf-8") as f:
                        header = f.readline().strip().split(",")
                    existing_cols = [h.strip() for h in header if h.strip()]
                    new_cols = [c for c in row_df.columns if c not in existing_cols]
                    if new_cols:
                        raise RuntimeError(
                            "Cannot resume because daily_results.csv has an older schema missing columns: "
                            f"{new_cols}. Delete the existing daily_results.csv (or rerun without --resume)."
                        )
                    # Align order and fill any missing existing columns
                    for c in existing_cols:
                        if c not in row_df.columns:
                            row_df[c] = np.nan
                    row_df = row_df[existing_cols]
                    row_df.to_csv(daily_path, mode="a", header=False, index=False)
                except Exception as e:
                    raise
            else:
                write_header = not daily_path.exists()
                row_df.to_csv(daily_path, mode="a", header=write_header, index=False)
            done_pairs.add(pair)

        # Cleanup per-day cache/model artifacts unless requested
        if not args.keep_day_cache:
            try:
                for p in cache_dir.glob("*"):
                    p.unlink(missing_ok=True)
                cache_dir.rmdir()
            except Exception:
                pass

        if not args.keep_day_models:
            try:
                for p in day_dir.glob("*"):
                    p.unlink(missing_ok=True)
                day_dir.rmdir()
            except Exception:
                pass

    # Write summary (based on whatever has been written so far)
    daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    if daily.empty:
        return

    summary = (
        daily.groupby("lambda", as_index=False)
        .agg(
            {
                "profit_old": "sum",
                "profit_new": "sum",
                "profit_total": "sum",
                "actual_profit": "sum",
                "actual_cost": "sum",
                "actual_revenue": "sum",
                "actual_clicks": "sum",
                "observed_clicks_total": "sum",
                "replay_clicks_total": "sum",
                "profit_improvement": "sum",
                "cost_change": "sum",
                "budget": "sum",
            }
        )
        .sort_values("lambda")
    )
    summary.to_csv(out_root / "lambda_summary.csv", index=False)


if __name__ == "__main__":
    main()
