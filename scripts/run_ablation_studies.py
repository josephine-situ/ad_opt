"""Run ablation studies for ad optimization modeling.

This script runs two experiment families:

1) Embedding-dimension ablation (BERT embeddings)
   - Generates datasets for multiple embedding dimensions (e.g., 10..50)
     - Fits and selects the best model among {glm, xgb, rf} by cross-validation,
         then reports test metrics for the selected model

2) Feature-combination ablation
   - Uses a fixed dataset (single embedding dimension)
     - Fits and selects the best model among {glm, xgb, rf} by cross-validation,
         then reports test metrics for the selected model

Design goals
------------
- Reuse the same preprocessing + metrics conventions as
  `scripts/prediction_modeling_tweedie.py`.
- Avoid overwriting production models by default.
- Make the ablation runner self-contained: it can optionally call
  `scripts/tidy_get_data.py` to materialize the needed train/test CSVs.

Examples
--------
# 1) Find best embedding dimensionality (10..50 step 10)
python scripts/run_ablation_studies.py embedding-dims --dims 10 20 30 40 50 --targets clicks epc

# 2) Compare feature sets at a fixed dimension
python scripts/run_ablation_studies.py feature-combos --n-components 50 --targets clicks epc

Outputs
-------
Writes CSV results under: opt_results/ablations/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from scripts.prediction_modeling_tweedie import (
    _as_numpy,
    _default_xgb_n_jobs,
    _grid_search_cv_mse,
    _to_float32_csr,
    build_preprocessor,
    compute_conditional_mae,
    compute_global_bias,
    compute_top_decile_lift,
    load_data,
    prepare_xyw,
)

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None


_BASE_FEATURE_COLS: List[str] = [
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
    "Avg. CPC",
]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cols_hash(cols: Sequence[str]) -> str:
    """Stable fingerprint for a selected raw feature set."""

    h = hashlib.md5()  # nosec - non-cryptographic fingerprint
    for c in cols:
        h.update(str(c).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _embedding_cols(df: pd.DataFrame, embedding_method: str) -> List[str]:
    prefix = f"{embedding_method.lower()}_"
    cols = [c for c in df.columns if c.startswith(prefix)]
    # fallback: tolerate older naming patterns
    if not cols:
        cols = [c for c in df.columns if embedding_method.lower() in c]
    return cols


def _ensure_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competition × match-type interactions if source cols exist.

    This mirrors the (commented-out) logic in `scripts/tidy_get_data.py` but keeps
    the ablation runner independent from the data pipeline.
    """

    out = df.copy()

    if "Competition (indexed value)" not in out.columns or "Match type" not in out.columns:
        return out

    comp = pd.to_numeric(out["Competition (indexed value)"], errors="coerce")
    is_exact = (out["Match type"] == "Exact match").astype(int)
    is_phrase = (out["Match type"] == "Phrase match").astype(int)
    is_broad = (out["Match type"] == "Broad match").astype(int)

    out["competition_x_is_exact"] = (comp * is_exact).fillna(0.0)
    out["competition_x_is_phrase"] = (comp * is_phrase).fillna(0.0)
    out["competition_x_is_broad"] = (comp * is_broad).fillna(0.0)

    return out


@dataclass(frozen=True)
class FitResult:
    cv_mse: float
    mse_train: float
    mse_test: float
    r2_test: float
    best_params: Dict[str, object]
    metrics: Dict[str, float]


def fit_glm_candidate_mse(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> FitResult:
    """Fit the same GLM candidate used by prediction_modeling_tweedie (Ridge, MSE) without saving."""

    preprocessor, _, _ = build_preprocessor(X_train)
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    param_grid = {
        "model__alpha": [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    }
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if sample_weight_train is None:
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=lambda est, X, y: -mean_squared_error(_as_numpy(y), _as_numpy(est.predict(X))),
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best: Pipeline = grid.best_estimator_
        best_params = dict(grid.best_params_)
        cv_mse = float(-grid.best_score_)
    else:
        best, best_params, cv_mse = _grid_search_cv_mse(
            base_estimator=pipe,
            param_grid={k: list(v) for k, v in param_grid.items()},
            X=X_train,
            y=y_train,
            cv=cv,
            sample_weight=sample_weight_train,
        )

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    mse_train = float(mean_squared_error(y_train, y_pred_train, sample_weight=sample_weight_train))
    mse_test = float(mean_squared_error(y_test, y_pred_test, sample_weight=sample_weight_test))

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test, sample_weight=sample_weight_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test, sample_weight=sample_weight_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test, sample_weight=sample_weight_test),
        "r2_score": float(r2_score(y_test, y_pred_test, sample_weight=sample_weight_test)),
    }

    return FitResult(
        cv_mse=float(cv_mse),
        mse_train=mse_train,
        mse_test=mse_test,
        r2_test=float(metrics["r2_score"]),
        best_params=best_params,
        metrics=metrics,
    )


def _fit_xgb_like_candidate_mse(
    *,
    kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> FitResult:
    if xgb is None:
        raise ImportError("xgboost is not installed")

    if kind not in {"xgb", "rf"}:
        raise ValueError("kind must be 'xgb' or 'rf'")

    preprocessor, _, _ = build_preprocessor(X_train)

    xgb_n_jobs = _default_xgb_n_jobs()

    if kind == "xgb":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_estimators=20,
            learning_rate=0.3,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=xgb_n_jobs,
        )
        param_grid = {
            "model__n_estimators": [5, 10, 20],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.1, 0.3],
            "model__subsample": [1.0],
            "model__colsample_bytree": [1.0],
        }
    else:
        # Mirror prediction_modeling_tweedie.py: XGBRFRegressor as an RF-like ensemble.
        model = xgb.XGBRFRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_estimators=20,
            learning_rate=1.0,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=xgb_n_jobs,
        )
        param_grid = {
            "model__n_estimators": [5, 10, 20],
            "model__max_depth": [2, 3, 4],
            # RF mode; keep learning_rate fixed at 1.0
            "model__learning_rate": [1.0],
            "model__subsample": [1.0],
            "model__colsample_bytree": [1.0],
        }

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", model),
        ]
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # IMPORTANT: mirror the stability precaution from prediction_modeling_tweedie.py.
    # Use n_jobs=1 in GridSearchCV and let XGBoost thread internally.
    if sample_weight_train is None:
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=lambda est, X, y: -mean_squared_error(_as_numpy(y), _as_numpy(est.predict(X))),
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best: Pipeline = grid.best_estimator_
        best_params = dict(grid.best_params_)
        cv_mse = float(-grid.best_score_)
    else:
        best, best_params, cv_mse = _grid_search_cv_mse(
            base_estimator=pipe,
            param_grid={k: list(v) for k, v in param_grid.items()},
            X=X_train,
            y=y_train,
            cv=cv,
            sample_weight=sample_weight_train,
        )

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    mse_train = float(mean_squared_error(y_train, y_pred_train, sample_weight=sample_weight_train))
    mse_test = float(mean_squared_error(y_test, y_pred_test, sample_weight=sample_weight_test))

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test, sample_weight=sample_weight_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test, sample_weight=sample_weight_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test, sample_weight=sample_weight_test),
        "r2_score": float(r2_score(y_test, y_pred_test, sample_weight=sample_weight_test)),
    }

    return FitResult(
        cv_mse=float(cv_mse),
        mse_train=mse_train,
        mse_test=mse_test,
        r2_test=float(metrics["r2_score"]),
        best_params=best_params,
        metrics=metrics,
    )


def fit_best_model_mse(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series],
    sample_weight_test: Optional[pd.Series],
) -> Tuple[str, FitResult, Dict[str, FitResult]]:
    """Fit glm/xgb/rf candidates and return the best by CV MSE.

    Returns:
      - best_model_name in {glm,xgb,rf}
      - best FitResult
      - dict of all fitted candidate results
    """

    candidates: Dict[str, FitResult] = {}
    candidates["glm"] = fit_glm_candidate_mse(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        cv_folds=cv_folds,
        sample_weight_train=sample_weight_train,
        sample_weight_test=sample_weight_test,
    )

    if xgb is not None:
        candidates["xgb"] = _fit_xgb_like_candidate_mse(
            kind="xgb",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            cv_folds=cv_folds,
            sample_weight_train=sample_weight_train,
            sample_weight_test=sample_weight_test,
        )
        # candidates["rf"] = _fit_xgb_like_candidate_mse(
        #     kind="rf",
        #     X_train=X_train,
        #     y_train=y_train,
        #     X_test=X_test,
        #     y_test=y_test,
        #     seed=seed,
        #     cv_folds=cv_folds,
        #     sample_weight_train=sample_weight_train,
        #     sample_weight_test=sample_weight_test,
        # )

    best_name = min(candidates.keys(), key=lambda k: float(candidates[k].cv_mse))
    return best_name, candidates[best_name], candidates


def ensure_dataset(
    *,
    embedding_method: str,
    n_components: int,
    data_dir: Path,
    reports_dir: Path,
    diversity_mode: str,
    force_rebuild: bool,
    log_file: Optional[str],
) -> None:
    """Ensure train/test CSVs exist in data_dir by running tidy_get_data if needed."""

    train_path = data_dir / f"train_{embedding_method}.csv"
    test_path = data_dir / f"test_{embedding_method}.csv"

    if train_path.exists() and test_path.exists() and not force_rebuild:
        # If diversity features are requested, ensure the cached dataset includes them.
        if diversity_mode in {"training", "prod"}:
            try:
                cols = pd.read_csv(train_path, nrows=1).columns.astype(str).tolist()
            except Exception:
                cols = []
            required = {"keyword_entropy_pred", "keyword_hhi_pred"}
            if required.issubset(set(cols)):
                return
        else:
            return

    data_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "tidy_get_data.py"),
        "--embedding-method",
        embedding_method,
        "--n-components",
        str(n_components),
        "--output-dir",
        str(data_dir.as_posix()),
        "--data-dir",
        str(reports_dir.as_posix()),
        "--diversity-mode",
        diversity_mode,
    ]

    if force_rebuild:
        cmd.append("--force-reload")

    # If log_file is None: tidy_get_data will write to logs/ by default.
    # If log_file == "": disable file logging.
    if log_file is not None:
        cmd.extend(["--log-file", log_file])

    print(f"[Data] Building dataset: method={embedding_method} n={n_components} dir={data_dir}")
    subprocess.run(cmd, check=True)


def select_feature_matrix(
    *,
    X: pd.DataFrame,
    embedding_method: str,
    use_hhi: bool,
    use_entropy: bool,
    use_interaction: bool,
) -> pd.DataFrame:
    emb_cols = _embedding_cols(X, embedding_method)

    cols: List[str] = []

    # Required base features (fail fast if missing)
    missing = [c for c in _BASE_FEATURE_COLS if c not in X.columns]
    if missing:
        raise KeyError(f"Missing base feature columns: {missing}")
    cols.extend(_BASE_FEATURE_COLS)

    if use_hhi and "keyword_hhi_pred" in X.columns:
        cols.append("keyword_hhi_pred")
    if use_entropy and "keyword_entropy_pred" in X.columns:
        cols.append("keyword_entropy_pred")

    if use_interaction:
        for c in ("competition_x_is_exact", "competition_x_is_phrase", "competition_x_is_broad"):
            if c in X.columns:
                cols.append(c)

    cols.extend(emb_cols)

    # Deduplicate while preserving order
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]

    return X[cols].copy()


def run_embedding_dim_ablation(
    *,
    embedding_method: str,
    dims: Sequence[int],
    targets: Sequence[str],
    data_root: Path,
    reports_dir: Path,
    results_dir: Path,
    diversity_mode: str,
    force_rebuild_data: bool,
    seed: int,
    cv_folds: int,
    log_file: Optional[str],
) -> Path:
    rows: List[Dict[str, object]] = []

    for n in dims:
        dim_dir = data_root / f"{embedding_method}_{n}d"
        ensure_dataset(
            embedding_method=embedding_method,
            n_components=n,
            data_dir=dim_dir,
            reports_dir=reports_dir,
            diversity_mode=diversity_mode,
            force_rebuild=force_rebuild_data,
            log_file=log_file,
        )

        df_train, df_test = load_data(data_dir=str(dim_dir), embedding_method=embedding_method)

        for target in targets:
            # Prepare X/y + sample weights using canonical logic
            Xtr_raw, ytr, wtr, _ = prepare_xyw(df_train, target)
            Xte_raw, yte, wte, _ = prepare_xyw(df_test, target)

            # For embedding-dimension ablation, default to *baseline* feature set
            # so the study isolates embedding dimensionality.
            Xtr = select_feature_matrix(
                X=Xtr_raw,
                embedding_method=embedding_method,
                use_hhi=False,
                use_entropy=False,
                use_interaction=False,
            )
            Xte = select_feature_matrix(
                X=Xte_raw,
                embedding_method=embedding_method,
                use_hhi=False,
                use_entropy=False,
                use_interaction=False,
            )

            best_model, best_fit, all_fits = fit_best_model_mse(
                X_train=Xtr,
                y_train=ytr,
                X_test=Xte,
                y_test=yte,
                seed=seed,
                cv_folds=cv_folds,
                sample_weight_train=wtr,
                sample_weight_test=wte,
            )

            rows.append(
                {
                    "study": "embedding_dims",
                    "embedding_method": embedding_method,
                    "n_components": int(n),
                    "target": target,
                    "feature_set": "baseline",
                    "n_features_raw": int(Xtr.shape[1]),
                    "feature_cols_hash": _cols_hash(list(Xtr.columns)),
                    "selected_model": best_model,
                    "cv_mse": best_fit.cv_mse,
                    "mse_train": best_fit.mse_train,
                    "mse_test": best_fit.mse_test,
                    "r2_test": best_fit.r2_test,
                    "best_params": json.dumps(best_fit.best_params, sort_keys=True),
                    "cv_mse_glm": float(all_fits["glm"].cv_mse),
                    "cv_mse_xgb": float(all_fits["xgb"].cv_mse) if "xgb" in all_fits else float("nan"),
                    "cv_mse_rf": float(all_fits["rf"].cv_mse) if "rf" in all_fits else float("nan"),
                    **{k: float(v) for k, v in best_fit.metrics.items()},
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"embedding_dim_ablation_{embedding_method}_{_timestamp()}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def run_feature_combo_ablation(
    *,
    embedding_method: str,
    n_components: int,
    targets: Sequence[str],
    data_root: Path,
    reports_dir: Path,
    results_dir: Path,
    diversity_mode: str,
    force_rebuild_data: bool,
    seed: int,
    cv_folds: int,
    log_file: Optional[str],
) -> Path:
    rows: List[Dict[str, object]] = []

    dim_dir = data_root / f"{embedding_method}_{n_components}d"
    ensure_dataset(
        embedding_method=embedding_method,
        n_components=n_components,
        data_dir=dim_dir,
        reports_dir=reports_dir,
        diversity_mode=diversity_mode,
        force_rebuild=force_rebuild_data,
        log_file=log_file,
    )

    df_train, df_test = load_data(data_dir=str(dim_dir), embedding_method=embedding_method)

    # Create interaction features at runtime (if possible)
    df_train = _ensure_interaction_features(df_train)
    df_test = _ensure_interaction_features(df_test)

    feature_sets: List[Tuple[str, Dict[str, bool]]] = [
        ("baseline", {"use_hhi": False, "use_entropy": False, "use_interaction": False}),
        ("baseline+hhi", {"use_hhi": True, "use_entropy": False, "use_interaction": False}),
        ("baseline+entropy", {"use_hhi": False, "use_entropy": True, "use_interaction": False}),
        ("baseline+interaction", {"use_hhi": False, "use_entropy": False, "use_interaction": True}),
        ("baseline+hhi+entropy", {"use_hhi": True, "use_entropy": True, "use_interaction": False}),
        ("baseline+hhi+interaction", {"use_hhi": True, "use_entropy": False, "use_interaction": True}),
        ("baseline+entropy+interaction", {"use_hhi": False, "use_entropy": True, "use_interaction": True}),
        ("baseline+hhi+entropy+interaction", {"use_hhi": True, "use_entropy": True, "use_interaction": True}),
    ]

    for target in targets:
        Xtr_raw, ytr, wtr, _ = prepare_xyw(df_train, target)
        Xte_raw, yte, wte, _ = prepare_xyw(df_test, target)

        for fs_name, flags in feature_sets:
            Xtr = select_feature_matrix(X=Xtr_raw, embedding_method=embedding_method, **flags)
            Xte = select_feature_matrix(X=Xte_raw, embedding_method=embedding_method, **flags)

            best_model, best_fit, all_fits = fit_best_model_mse(
                X_train=Xtr,
                y_train=ytr,
                X_test=Xte,
                y_test=yte,
                seed=seed,
                cv_folds=cv_folds,
                sample_weight_train=wtr,
                sample_weight_test=wte,
            )

            rows.append(
                {
                    "study": "feature_combos",
                    "embedding_method": embedding_method,
                    "n_components": int(n_components),
                    "target": target,
                    "feature_set": fs_name,
                    "n_features_raw": int(Xtr.shape[1]),
                    "feature_cols_hash": _cols_hash(list(Xtr.columns)),
                    "selected_model": best_model,
                    "cv_mse": best_fit.cv_mse,
                    "mse_train": best_fit.mse_train,
                    "mse_test": best_fit.mse_test,
                    "r2_test": best_fit.r2_test,
                    "best_params": json.dumps(best_fit.best_params, sort_keys=True),
                    "cv_mse_glm": float(all_fits["glm"].cv_mse),
                    "cv_mse_xgb": float(all_fits["xgb"].cv_mse) if "xgb" in all_fits else float("nan"),
                    "cv_mse_rf": float(all_fits["rf"].cv_mse) if "rf" in all_fits else float("nan"),
                    **{k: float(v) for k, v in best_fit.metrics.items()},
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"feature_combo_ablation_{embedding_method}_{n_components}d_{_timestamp()}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _parse_targets(vals: Sequence[str]) -> List[str]:
    allowed = {"clicks", "epc"}
    out = []
    for v in vals:
        v2 = v.strip().lower()
        if v2 not in allowed:
            raise ValueError(f"Unknown target '{v}'. Allowed: {sorted(allowed)}")
        out.append(v2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies (embedding dims + feature combos).")
    # Don't hard-fail when launched without a subcommand (e.g. from a debugger).
    # We'll print help and exit 0 instead.
    sub = parser.add_subparsers(dest="cmd")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--embedding-method", default="bert", choices=["bert", "tfidf"], help="Embedding method")
        p.add_argument(
            "--targets",
            nargs="+",
            default=["clicks", "epc"],
            help="Targets to evaluate: clicks epc",
        )
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--cv-folds", type=int, default=5)
        p.add_argument(
            "--reports-dir",
            type=str,
            default=str((_REPO_ROOT / "data" / "reports").as_posix()),
            help="Input reports directory (passed to tidy_get_data)",
        )
        p.add_argument(
            "--data-root",
            type=str,
            default=str((_REPO_ROOT / "data" / "clean" / "ablations").as_posix()),
            help="Root folder to store per-experiment datasets",
        )
        p.add_argument(
            "--results-dir",
            type=str,
            default=str((_REPO_ROOT / "ablations").as_posix()),
            help="Folder to write ablation result CSVs",
        )
        p.add_argument(
            "--diversity-mode",
            type=str,
            default=None,
            choices=["off", "training", "prod", None],
            help=(
                "Diversity mode for tidy_get_data. If omitted, defaults to 'off' for embedding-dims and 'training' for feature-combos."
            ),
        )
        p.add_argument(
            "--force-rebuild-data",
            action="store_true",
            help="Force tidy_get_data to recompute (skips caches)",
        )
        p.add_argument(
            "--log-file",
            type=str,
            default=None,
            help=(
                "Log file path forwarded to tidy_get_data. Use empty string to disable file logging. "
                "If omitted, tidy_get_data writes to logs/ by default."
            ),
        )

    p1 = sub.add_parser("embedding-dims", help="Ablate embedding dimensionality (baseline feature set).")
    add_common(p1)
    p1.add_argument("--dims", nargs="+", type=int, required=True, help="Embedding dimensions to evaluate")

    p2 = sub.add_parser("feature-combos", help="Ablate feature combinations at a fixed embedding dimension.")
    add_common(p2)
    p2.add_argument("--n-components", type=int, required=True, help="Embedding dimension to use")

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        return

    targets = _parse_targets(args.targets)
    reports_dir = Path(args.reports_dir)
    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)

    if args.cmd == "embedding-dims":
        diversity_mode = args.diversity_mode if args.diversity_mode is not None else "off"
        out_csv = run_embedding_dim_ablation(
            embedding_method=args.embedding_method,
            dims=args.dims,
            targets=targets,
            data_root=data_root,
            reports_dir=reports_dir,
            results_dir=results_dir,
            diversity_mode=diversity_mode,
            force_rebuild_data=bool(args.force_rebuild_data),
            seed=int(args.seed),
            cv_folds=int(args.cv_folds),
            log_file=args.log_file,
        )
        print(f"[Done] Wrote results: {out_csv}")
        return

    if args.cmd == "feature-combos":
        diversity_mode = args.diversity_mode if args.diversity_mode is not None else "training"
        out_csv = run_feature_combo_ablation(
            embedding_method=args.embedding_method,
            n_components=int(args.n_components),
            targets=targets,
            data_root=data_root,
            reports_dir=reports_dir,
            results_dir=results_dir,
            diversity_mode=diversity_mode,
            force_rebuild_data=bool(args.force_rebuild_data),
            seed=int(args.seed),
            cv_folds=int(args.cv_folds),
            log_file=args.log_file,
        )
        print(f"[Done] Wrote results: {out_csv}")
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
