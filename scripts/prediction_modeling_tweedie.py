"""Prediction Modeling (No IAI) for Ad Optimization
================================================

NOTE: The filename is historically misnamed.

The exported linear-model weights used by `scripts/bid_optimization.py` come
from a Ridge regression baseline trained to minimize (optionally weighted) MSE.

Trains prediction models to predict:
- conversion: "Conv. value"
- clicks: "Clicks"
- epc: "EPC"

This script is a drop-in *workflow* replacement for `scripts/prediction_modeling.py`
that does NOT require InterpretableAI (IAI) or a license.

Key outputs
-----------
1) A linear Ridge baseline trained with scikit-learn's `Ridge`.
    - Saves a fitted pipeline to `models/ridge_{embedding}_{target}.joblib`
   - Exports coefficients in the CSV format already consumed by
     `scripts/bid_optimization.py`:
       - `models/weights_{embedding}_{target}_numeric.csv`
       - `models/weights_{embedding}_{target}_categorical.csv` (if any)
       - `models/weights_{embedding}_{target}_constant.csv`

2) Optionally, an XGBoost model trained as an MSE baseline.
    - Saves to `models/xgb_mse_{embedding}_{target}.json`

Usage
-----
python scripts/prediction_modeling_tweedie.py --target conversion --embedding-method bert
python scripts/prediction_modeling_tweedie.py --target clicks --embedding-method bert --models ridge xgb

Notes
-----
- Targets are assumed to be non-negative.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tee_logging import setup_tee_logging

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise ImportError(
        "joblib is required (it is normally installed with scikit-learn)."
    ) from e

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import scipy.sparse as sp  # type: ignore
except Exception:
    sp = None

try:
    # Optional dependency. This is a cloud-based client; data will be sent to PriorLabs servers.
    import tabpfn_client  # type: ignore
    from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor  # type: ignore
except Exception:
    tabpfn_client = None
    TabPFNClientRegressor = None


def _configure_tabpfn_client_from_env() -> None:
    """Configure tabpfn-client without interactive prompts.

    `tabpfn_client.init()` triggers an interactive login flow. For scripts/cluster
    usage we instead rely on an access token provided via environment variable.
    """

    if tabpfn_client is None:  # pragma: no cover
        return

    token_env_candidates = (
        "TABPFN_ACCESS_TOKEN",
        "TABPFN_CLIENT_ACCESS_TOKEN",
        "PRIORLABS_ACCESS_TOKEN",
    )

    token = None
    for key in token_env_candidates:
        val = os.environ.get(key)
        if val:
            token = val
            break

    if token is None:
        raise RuntimeError(
            "TabPFN client is installed but no access token was found in the environment. "
            "Set TABPFN_ACCESS_TOKEN (recommended) and rerun. "
            "If you don't have a token yet, run an interactive login once via Python: "
            "`python -c \"import tabpfn_client; print(tabpfn_client.get_access_token())\"` "
            "and then export that token as TABPFN_ACCESS_TOKEN."
        )

    # Avoid printing the token. This persists credentials in the client's cache.
    tabpfn_client.set_access_token(token)


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor to enforce non-negative predictions.

    This is useful because downstream bid logic and Tweedie scoring assume
    non-negative predictions.
    """

    def __init__(self, base_estimator, eps: float = 1e-12):
        self.base_estimator = base_estimator
        self.eps = eps

    def fit(self, X, y, **fit_params):
        self.base_estimator.fit(X, y, **fit_params)
        # Let sklearn know this wrapper is fitted.
        self.is_fitted_ = True
        return self

    def predict(self, X):
        pred = self.base_estimator.predict(X)
        return np.maximum(np.asarray(pred, dtype=float), float(self.eps))


def _to_float32_dense(X):
    return np.asarray(X, dtype=np.float32)


def _default_xgb_n_jobs() -> int:
    """Choose a reasonable thread count for XGBoost on clusters.

    We prefer SLURM_CPUS_PER_TASK when available. This keeps training within the
    resources allocated by the scheduler.
    """

    for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        val = os.environ.get(key)
        if val:
            try:
                n = int(val)
                if n >= 1:
                    return n
            except Exception:
                pass
    return int(os.cpu_count() or 1)


def _to_float32_csr(X):
    """Cast matrices to float32; prefer CSR for sparse.

    This avoids a class of XGBoost crashes/slow paths on some HPC builds.
    """

    if sp is not None and sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def load_data(data_dir: str = "data/clean", embedding_method: str = "tfidf") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data."""
    print(f"Loading data from {data_dir}...")
    df_train = pd.read_csv(f"{data_dir}/train_{embedding_method}.csv")
    df_test = pd.read_csv(f"{data_dir}/test_{embedding_method}.csv")
    print(f"  Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    return df_train, df_test


def get_features(df: pd.DataFrame, target: str = "conversion") -> Tuple[pd.DataFrame, List[str]]:
    """Extract feature matrix.

    Mirrors `scripts/prediction_modeling.py`:
    - When target is 'conversion', Clicks is included as a predictor.
    - When target is 'epc', both Conv. value and Clicks are excluded.
    - When target is 'clicks', Conv. value is excluded.
    """

    if target == "conversion":
        excluded_cols = {"Conv. value", "EPC", "Day", "Keyword"}
    elif target == "epc":
        excluded_cols = {"Conv. value", "Clicks", "EPC", "Day", "Keyword"}
    else:  # clicks
        excluded_cols = {"Clicks", "Conv. value", "EPC", "Day", "Keyword"}

    feature_cols = [col for col in df.columns if col not in excluded_cols]
    return df[feature_cols].copy(), feature_cols


def get_target(df: pd.DataFrame, target: str = "conversion") -> pd.Series:
    if target == "conversion":
        return df["Conv. value"]
    if target == "epc":
        if "EPC" not in df.columns:
            raise KeyError("EPC target requires an 'EPC' column")
        return df["EPC"]
    if target == "clicks":
        return df["Clicks"]
    raise ValueError(f"Unknown target: {target}")




def _as_numpy(x: Iterable[float]) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()  # type: ignore[no-any-return]
    return np.asarray(list(x), dtype=float)


def _weighted_mean(values: Iterable[float], sample_weight: Optional[Iterable[float]]) -> float:
    v = _as_numpy(values)
    if sample_weight is None:
        return float(np.mean(v))
    w = _as_numpy(sample_weight)
    return float(np.average(v, weights=w))


def _quantiles(values: Iterable[float], qs: Iterable[float] = (0.0, 0.01, 0.5, 0.99, 1.0)) -> Dict[float, float]:
    v = _as_numpy(values)
    if v.size == 0:
        return {float(q): float("nan") for q in qs}
    out: Dict[float, float] = {}
    for q in qs:
        out[float(q)] = float(np.quantile(v, q))
    return out


def _print_diagnostics(
    *,
    label: str,
    y_true: Iterable[float],
    y_pred: Iterable[float],
    sample_weight: Optional[Iterable[float]] = None,
) -> None:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)

    mean_y = _weighted_mean(yt, None)
    mean_p = _weighted_mean(yp, None)
    bias = compute_global_bias(yt, yp)

    msg = f"  [{label}] mean(y)={mean_y:.4f} mean(pred)={mean_p:.4f} bias={bias:.4f}"
    if sample_weight is not None:
        wmean_y = _weighted_mean(yt, sample_weight)
        wmean_p = _weighted_mean(yp, sample_weight)
        wbias = compute_global_bias(yt, yp, sample_weight=sample_weight)
        msg += f" | wmean(y)={wmean_y:.4f} wmean(pred)={wmean_p:.4f} wbias={wbias:.4f}"
    print(msg)

    qy = _quantiles(yt)
    qp = _quantiles(yp)
    print(
        "  [{label}] quantiles y: "
        "min={y0:.4g} p1={y1:.4g} med={y50:.4g} p99={y99:.4g} max={y100:.4g}".format(
            label=label,
            y0=qy[0.0],
            y1=qy[0.01],
            y50=qy[0.5],
            y99=qy[0.99],
            y100=qy[1.0],
        )
    )
    print(
        "  [{label}] quantiles pred: "
        "min={p0:.4g} p1={p1:.4g} med={p50:.4g} p99={p99:.4g} max={p100:.4g}".format(
            label=label,
            p0=qp[0.0],
            p1=qp[0.01],
            p50=qp[0.5],
            p99=qp[0.99],
            p100=qp[1.0],
        )
    )


def _grid_search_cv_mse(
    *,
    base_estimator: Pipeline,
    param_grid: Dict[str, List[object]],
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
    sample_weight: Optional[pd.Series],
) -> Tuple[Pipeline, Dict[str, object], float]:
    """Simple, weight-aware CV grid search for MSE.

    We implement this because sklearn's GridSearchCV scoring callback doesn't
    receive per-fold sample_weight.

    Returns (best_estimator_fit_on_full_data, best_params, best_cv_mse).
    """

    best_params: Optional[Dict[str, object]] = None
    best_score: float = float("inf")

    for params in ParameterGrid(param_grid):
        fold_scores: List[float] = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_va = X.iloc[val_idx]
            y_va = y.iloc[val_idx]

            w_tr = None
            w_va = None
            if sample_weight is not None:
                w_tr = sample_weight.iloc[train_idx]
                w_va = sample_weight.iloc[val_idx]

            est = clone(base_estimator)
            est.set_params(**params)

            # Fit with sample weights if available
            if w_tr is not None:
                est.fit(X_tr, y_tr, model__sample_weight=_as_numpy(w_tr))
            else:
                est.fit(X_tr, y_tr)

            y_hat = _as_numpy(est.predict(X_va))
            fold_scores.append(
                float(
                    mean_squared_error(
                        _as_numpy(y_va),
                        y_hat,
                        sample_weight=_as_numpy(w_va) if w_va is not None else None,
                    )
                )
            )

        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("Grid search failed to evaluate any parameters.")

    best = clone(base_estimator)
    best.set_params(**best_params)
    
    # Fit final model with sample weights if available
    if sample_weight is not None:
        best.fit(X, y, model__sample_weight=_as_numpy(sample_weight))
    else:
        best.fit(X, y)

    return best, best_params, best_score


def compute_global_bias(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    sample_weight: Optional[Iterable[float]] = None,
) -> float:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    if sample_weight is None:
        return float(np.mean(yp) - np.mean(yt))
    sw = _as_numpy(sample_weight)
    return float(np.average(yp, weights=sw) - np.average(yt, weights=sw))


def compute_top_decile_lift(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    sample_weight: Optional[Iterable[float]] = None,
) -> float:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    if yt.size == 0:
        return float("nan")
    if sample_weight is None:
        overall_mean = float(np.mean(yt))
    else:
        sw = _as_numpy(sample_weight)
        overall_mean = float(np.average(yt, weights=sw))
    if overall_mean == 0:
        return float("nan")
    n_top = max(1, int(0.1 * yt.size))
    top_idx = np.argsort(yp)[-n_top:]
    if sample_weight is None:
        top_mean = float(np.mean(yt[top_idx]))
    else:
        sw = _as_numpy(sample_weight)
        top_mean = float(np.average(yt[top_idx], weights=sw[top_idx]))
    return float(top_mean / overall_mean)


def compute_conditional_mae(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    sample_weight: Optional[Iterable[float]] = None,
) -> float:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    mask = yt > 0
    if not np.any(mask):
        return float("nan")
    abs_err = np.abs(yt[mask] - yp[mask])
    if sample_weight is None:
        return float(np.mean(abs_err))
    sw = _as_numpy(sample_weight)
    return float(np.average(abs_err, weights=sw[mask]))


def prepare_xyw(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
    """Prepare features, target, and optional sample weights.

        For aggregated EPC modeling:
            - Filter rows where Clicks <= 0 (EPC undefined)
            - Target: EPC (existing column)
            - sample_weight: Clicks
    """

    df_use = df
    sample_weight: Optional[pd.Series] = None

    if target == "epc":
        if "Clicks" not in df.columns or "EPC" not in df.columns:
            raise KeyError("EPC target requires 'Clicks' and 'EPC' columns")
        mask = df["Clicks"] > 0
        n_before = int(len(df))
        n_after = int(mask.sum())
        if n_after < n_before:
            print(f"  [EPC] Filtered rows: {n_before} -> {n_after} (Clicks>0)")
        df_use = df.loc[mask].copy()
        sample_weight = df_use["Clicks"].astype(float)

    X, feature_cols = get_features(df_use, target)
    y = get_target(df_use, target)
    return X, y, sample_weight, feature_cols


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create a ColumnTransformer that one-hot encodes categoricals and scales numerics."""

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
            # Drop one level per categorical feature to avoid the dummy-variable trap:
            # with full one-hot + an intercept, the design matrix contains redundant
            # constant columns which can cause huge, unstable coefficients.
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


def build_preprocessor_tabpfn(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Preprocessor tuned for TabPFN.

    TabPFN typically expects a dense numeric matrix. One-hot encoding can
    explode dimensionality for high-cardinality categoricals (and sparse ->
    dense can be prohibitively large), so we use ordinal encoding instead.
    """

    categorical_cols = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # always dense
    )

    return preprocessor, numeric_cols, categorical_cols


def train_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    target: str,
    embedding_method: str,
    seed: int,
    out_dir: Path,
    tabpfn_max_train_rows: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> Tuple[Pipeline, float, Dict[str, float]]:
    """Train a TabPFN regressor.

    Notes:
    - TabPFN does not optimize Tweedie deviance; we still report Tweedie D^2
      for comparability.
    - TabPFN does not support sample weights in `fit`; weights are used only
      for evaluation metrics.
    """

    if TabPFNClientRegressor is None:
        raise ImportError(
            "tabpfn-client is not installed. Install with: pip install tabpfn-client "
            "(or `pip install -e .[tabpfn]` if using this repo's extras)."
        )

    _configure_tabpfn_client_from_env()

    print("\n--- TabPFN ---")

    if sample_weight_train is not None:
        print("  [TabPFN] Note: sample_weight is ignored during fit; used only for evaluation.")

    if tabpfn_max_train_rows > 0 and len(X_train) > tabpfn_max_train_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=tabpfn_max_train_rows, replace=False)
        X_fit = X_train.iloc[idx].copy()
        y_fit = y_train.iloc[idx].copy()
        print(f"  [TabPFN] Subsampled train rows: {len(X_train)} -> {len(X_fit)}")
    else:
        X_fit, y_fit = X_train, y_train

    preprocessor, _, _ = build_preprocessor_tabpfn(X_fit)

    # Cloud estimator; behaves like an sklearn regressor.
    model = TabPFNClientRegressor()
    model = NonNegativeRegressor(model)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cast", FunctionTransformer(_to_float32_dense)),
            ("model", model),
        ]
    )

    pipe.fit(X_fit, y_fit)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train, sample_weight=sample_weight_train)
    mse_test = mean_squared_error(y_test, y_pred_test, sample_weight=sample_weight_test)

    print(f"  Train MSE (lower is better): {mse_train:.4f}")
    print(f"  Test  MSE (lower is better): {mse_test:.4f}")

    _print_diagnostics(label="TabPFN train", y_true=y_train, y_pred=y_pred_train, sample_weight=sample_weight_train)
    _print_diagnostics(label="TabPFN test", y_true=y_test, y_pred=y_pred_test, sample_weight=sample_weight_test)

    model_path = out_dir / f"tabpfn_{embedding_method}_{target}.joblib"
    joblib.dump(pipe, model_path)
    print(f"  Saved pipeline to {model_path}")

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test, sample_weight=sample_weight_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test, sample_weight=sample_weight_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test, sample_weight=sample_weight_test),
        "r2_score": r2_score(y_test, y_pred_test, sample_weight=sample_weight_test),
    }

    print(f"  [TabPFN] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [TabPFN] Global bias: nan")
    print(f"  [TabPFN] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [TabPFN] Top decile lift: nan")
    print(f"  [TabPFN] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [TabPFN] Conditional MAE: nan")
    print(f"  [TabPFN] R^2: {metrics['r2_score']:.4f}" if not np.isnan(metrics["r2_score"]) else "  [TabPFN] R^2: nan")

    return pipe, mse_test, metrics


def _validate_non_negative_target(y: pd.Series, target: str) -> None:
    if (y < 0).any():
        n_bad = int((y < 0).sum())
        raise ValueError(
            f"Target '{target}' contains {n_bad} negative values. "
            "Tweedie models with log link assume non-negative targets."
        )


def train_glm_mse(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    target: str,
    embedding_method: str,
    seed: int,
    out_dir: Path,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> Tuple[Pipeline, float, Dict[str, float]]:
    raise RuntimeError(
        "train_glm_mse was renamed to train_ridge_mse. "
        "Update callers to use train_ridge_mse."
    )


def train_ridge_mse(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    target: str,
    embedding_method: str,
    seed: int,
    out_dir: Path,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> Tuple[Pipeline, float, Dict[str, float]]:
    print("\n--- Ridge Regression (MSE) ---")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )

    # Keep grid modest; this is a baseline that also produces coefficients.
    param_grid = {
        "model__alpha": [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if sample_weight_train is None:
        # GridSearchCV maximizes score; use negative MSE as the scoring objective.
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
        best_params = grid.best_params_
    else:
        # Weight-aware CV selection
        best, best_params, best_cv_mse = _grid_search_cv_mse(
            base_estimator=pipe,
            param_grid={k: list(v) for k, v in param_grid.items()},
            X=X_train,
            y=y_train,
            cv=cv,
            sample_weight=sample_weight_train,
        )
        print(f"  [GLM] Best CV MSE (weighted): {best_cv_mse:.6f}")

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train, sample_weight=sample_weight_train)
    mse_test = mean_squared_error(y_test, y_pred_test, sample_weight=sample_weight_test)

    print(f"  Best params: {best_params}")
    print(f"  Train MSE (lower is better): {mse_train:.4f}")
    print(f"  Test  MSE (lower is better): {mse_test:.4f}")

    _print_diagnostics(label="Ridge train", y_true=y_train, y_pred=y_pred_train, sample_weight=sample_weight_train)
    _print_diagnostics(label="Ridge test", y_true=y_test, y_pred=y_pred_test, sample_weight=sample_weight_test)

    # Save fitted pipeline.
    # Historical naming used "glm_*"; we now save as "ridge_*".
    model_path = out_dir / f"ridge_{embedding_method}_{target}.joblib"
    joblib.dump(best, model_path)
    print(f"  Saved pipeline to {model_path}")

    # Save the fitted preprocessing pipeline for use in bid_optimization.
    # This is required to reproduce identical transformations when embedding the linear model.
    preproc_path = out_dir / f"ridge_{embedding_method}_{target}_preprocess.joblib"
    joblib.dump(best.named_steps["preprocess"], preproc_path)
    print(f"  Saved preprocessor to {preproc_path}")

    # Export weights in existing CSV format
    export_ridge_weights(best, numeric_cols, categorical_cols, embedding_method, target, out_dir)

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test, sample_weight=sample_weight_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test, sample_weight=sample_weight_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test, sample_weight=sample_weight_test),
        "r2_score": r2_score(y_test, y_pred_test, sample_weight=sample_weight_test),
    }

    print(f"  [Ridge] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [Ridge] Global bias: nan")
    print(f"  [Ridge] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [Ridge] Top decile lift: nan")
    print(f"  [Ridge] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [Ridge] Conditional MAE: nan")
    print(f"  [Ridge] R^2: {metrics['r2_score']:.4f}" if not np.isnan(metrics["r2_score"]) else "  [Ridge] R^2: nan")

    return best, mse_test, metrics


def export_ridge_weights(
    fitted_ridge_pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
    embedding_method: str,
    target: str,
    out_dir: Path,
) -> None:
    """Export Ridge intercept + coefficients to the repo's existing weight CSV format."""

    preprocess: ColumnTransformer = fitted_ridge_pipeline.named_steps["preprocess"]
    model: Ridge = fitted_ridge_pipeline.named_steps["model"]

    feature_names = list(preprocess.get_feature_names_out())
    coefs = model.coef_.ravel()

    if len(feature_names) != len(coefs):
        raise RuntimeError(
            f"Feature name count ({len(feature_names)}) != coef count ({len(coefs)})."
        )

    # Constant
    const_path = out_dir / f"weights_{embedding_method}_{target}_constant.csv"
    pd.DataFrame({"constant": [float(model.intercept_)]}).to_csv(const_path, index=False)

    # Numeric weights
    numeric_rows: List[Dict[str, float]] = []

    # Categorical weights
    categorical_rows: List[Dict[str, object]] = []

    # Build mapping from OneHotEncoder feature name -> (feature, level)
    cat_map: Dict[str, Tuple[str, str]] = {}
    if categorical_cols:
        cat_pipe: Pipeline = preprocess.named_transformers_["cat"]
        ohe: OneHotEncoder = cat_pipe.named_steps["onehot"]
        ohe_names = list(ohe.get_feature_names_out(categorical_cols))

        # Create an unambiguous mapping using the known feature prefix
        for col, cats in zip(categorical_cols, ohe.categories_):
            for cat in cats:
                raw_name = f"{col}_{cat}"
                if raw_name in ohe_names:
                    cat_map[raw_name] = (col, str(cat))

    for name, weight in zip(feature_names, coefs):
        if name.startswith("num__"):
            numeric_rows.append({"feature": name[len("num__") :], "weight": float(weight)})
        elif name.startswith("cat__"):
            raw = name[len("cat__") :]
            if raw in cat_map:
                feature, level = cat_map[raw]
            else:
                # Fallback: keep raw name split if possible
                # (rare unless feature names include unusual separators)
                feature, level = raw.split("_", 1) if "_" in raw else (raw, "")
            categorical_rows.append({"feature": feature, "level": level, "weight": float(weight)})
        else:
            # Should not happen with our preprocessor
            numeric_rows.append({"feature": name, "weight": float(weight)})

    numeric_path = out_dir / f"weights_{embedding_method}_{target}_numeric.csv"
    pd.DataFrame(numeric_rows).to_csv(numeric_path, index=False)

    if categorical_rows:
        cat_path = out_dir / f"weights_{embedding_method}_{target}_categorical.csv"
        pd.DataFrame(categorical_rows).to_csv(cat_path, index=False)
    else:
        # If there are no categoricals, don't create an empty file.
        cat_path = out_dir / f"weights_{embedding_method}_{target}_categorical.csv"
        if cat_path.exists():
            cat_path.unlink()

    print(f"  Exported weights: {numeric_path.name}, {const_path.name}" + (f", {cat_path.name}" if categorical_rows else ""))


def train_xgb_mse(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    target: str,
    embedding_method: str,
    seed: int,
    out_dir: Path,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> Tuple[Pipeline, float, Dict[str, float]]:
    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost "
            "(or `pip install -e .[ml_open]` if using this repo's extras)."
        )

    print("\n--- XGBoost (MSE) ---")

    preprocessor, _, _ = build_preprocessor(X_train)

    # IMPORTANT: XGBoost has been observed to segfault under joblib/loky
    # multiprocessing on some clusters (esp. when xgboost is installed outside
    # the active conda env). We therefore run GridSearchCV serially (n_jobs=1)
    # and use XGBoost's internal threading (n_jobs) to utilize CPUs.
    xgb_n_jobs = _default_xgb_n_jobs()

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        # Keep the ensemble intentionally small/shallow so it can be embedded
        # into a mixed-integer optimization (MIO) model.
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

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", model),
        ]
    )

    # Keep the grid small: embedding complexity primarily scales with the
    # number of trees and the tree depth.
    param_grid = {
        "model__n_estimators": [5, 10, 20],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.1, 0.3],
        # Deterministic full-sample trees (helps reproducibility and is simpler
        # to reason about downstream).
        "model__subsample": [1.0],
        "model__colsample_bytree": [1.0],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    if sample_weight_train is None:
        # GridSearchCV maximizes score; use negative MSE as the scoring objective.
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
        best_params = grid.best_params_
    else:
        best, best_params, best_cv_mse = _grid_search_cv_mse(
            base_estimator=pipe,
            param_grid={k: list(v) for k, v in param_grid.items()},
            X=X_train,
            y=y_train,
            cv=cv,
            sample_weight=sample_weight_train,
        )
        print(f"  [XGB] Best CV MSE (weighted): {best_cv_mse:.6f}")

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train, sample_weight=sample_weight_train)
    mse_test = mean_squared_error(y_test, y_pred_test, sample_weight=sample_weight_test)

    print(f"  Best params: {best_params}")
    print(f"  Train MSE (lower is better): {mse_train:.4f}")
    print(f"  Test  MSE (lower is better): {mse_test:.4f}")

    _print_diagnostics(label="XGB train", y_true=y_train, y_pred=y_pred_train, sample_weight=sample_weight_train)
    _print_diagnostics(label="XGB test", y_true=y_test, y_pred=y_pred_test, sample_weight=sample_weight_test)

    # Save just the fitted booster in XGBoost's native format
    xgb_path = out_dir / f"xgb_mse_{embedding_method}_{target}.json"
    best.named_steps["model"].save_model(xgb_path)
    print(f"  Saved model to {xgb_path}")

    # Save the fitted preprocessing pipeline too.
    # The booster JSON does NOT include sklearn preprocessing state, so persisting
    # this is required to reproduce identical transformations later (e.g., for
    # distilling into an ORT).
    preproc_path = out_dir / f"xgb_mse_{embedding_method}_{target}_preprocess.joblib"
    joblib.dump(best.named_steps["preprocess"], preproc_path)
    print(f"  Saved preprocessor to {preproc_path}")

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test, sample_weight=sample_weight_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test, sample_weight=sample_weight_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test, sample_weight=sample_weight_test),
        "r2_score": r2_score(y_test, y_pred_test, sample_weight=sample_weight_test),
    }

    print(f"  [XGB] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [XGB] Global bias: nan")
    print(f"  [XGB] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [XGB] Top decile lift: nan")
    print(f"  [XGB] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [XGB] Conditional MAE: nan")
    print(f"  [XGB] R^2: {metrics['r2_score']:.4f}" if not np.isnan(metrics["r2_score"]) else "  [XGB] R^2: nan")

    return best, mse_test, metrics


def train_rf_mse(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    target: str,
    embedding_method: str,
    seed: int,
    out_dir: Path,
    cv_folds: int,
    sample_weight_train: Optional[pd.Series] = None,
    sample_weight_test: Optional[pd.Series] = None,
) -> Tuple[Pipeline, float, Dict[str, float]]:
    """Train an XGBoost random-forest style model with MSE objective.

    Note: scikit-learn's RandomForestRegressor does not natively optimize Tweedie deviance.
    XGBoost's XGBRFRegressor provides an RF-like ensemble while supporting MSE.
    """

    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost "
            "(or `pip install -e .[ml_open]` if using this repo's extras)."
        )

    print("\n--- XGBoost RF (MSE) ---")

    preprocessor, _, _ = build_preprocessor(X_train)

    # See note in train_xgb_mse re: avoiding joblib/loky multiprocessing.
    xgb_n_jobs = _default_xgb_n_jobs()

    model = xgb.XGBRFRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bynode=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        learning_rate=0.01,  # Low learning rate for RF-like behavior (not boosting)
        tree_method="hist",
        n_jobs=xgb_n_jobs,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", model),
        ]
    )

    param_grid = {
        "model__max_depth": [2, 3, 4],
        "model__n_estimators": [5, 10, 20],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bynode": [0.7, 0.9],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

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
        best_params = grid.best_params_
    else:
        best, best_params, best_cv_mse = _grid_search_cv_mse(
            base_estimator=pipe,
            param_grid={k: list(v) for k, v in param_grid.items()},
            X=X_train,
            y=y_train,
            cv=cv,
            sample_weight=sample_weight_train,
        )
        print(f"  [RF] Best CV MSE (weighted): {best_cv_mse:.6f}")

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train, sample_weight=sample_weight_train)
    mse_test = mean_squared_error(y_test, y_pred_test, sample_weight=sample_weight_test)

    print(f"  Best params: {best_params}")
    print(f"  Train MSE (lower is better): {mse_train:.4f}")
    print(f"  Test  MSE (lower is better): {mse_test:.4f}")

    _print_diagnostics(label="RF train", y_true=y_train, y_pred=y_pred_train, sample_weight=sample_weight_train)
    _print_diagnostics(label="RF test", y_true=y_test, y_pred=y_pred_test, sample_weight=sample_weight_test)

    rf_path = out_dir / f"rf_mse_{embedding_method}_{target}.json"
    best.named_steps["model"].save_model(rf_path)
    print(f"  Saved model to {rf_path}")

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test, sample_weight=sample_weight_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test, sample_weight=sample_weight_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test, sample_weight=sample_weight_test),
        "r2_score": r2_score(y_test, y_pred_test, sample_weight=sample_weight_test),
    }

    print(f"  [RF] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [RF] Global bias: nan")
    print(f"  [RF] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [RF] Top decile lift: nan")
    print(f"  [RF] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [RF] Conditional MAE: nan")
    print(f"  [RF] R^2: {metrics['r2_score']:.4f}" if not np.isnan(metrics["r2_score"]) else "  [RF] R^2: nan")

    return best, mse_test, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tweedie-loss prediction models (no IAI).")
    parser.add_argument(
        "--target",
        type=str,
        default="clicks",
        choices=["conversion", "epc", "clicks"],
        help="Target variable: conversion (Conv. value), epc (EPC), or clicks (default: epc)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/clean",
        help="Data directory (default: data/clean)",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        default="bert",
        choices=["tfidf", "bert"],
        help="Embedding method used in data (default: bert)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["ridge", "xgb", "rf"],
        choices=["ridge", "xgb", "rf", "tabpfn"],
        help="Which models to train (default: ridge xgb rf tabpfn)",
    )
    parser.add_argument(
        "--tabpfn-max-train-rows",
        type=int,
        default=10000,
        help="Max rows to fit TabPFN on (subsamples if larger; default: 10000; set 0 to disable)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for grid search")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Output directory for trained models/weights (default: models)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Log file path. Default: logs/model_performance_<target>_<embedding>.log. "
            "Set to empty string '' to disable file logging."
        ),
    )

    args = parser.parse_args()

    # Tee stdout/stderr to a log file (plus console).
    # Preserves previous default filename for easier comparisons across runs.
    default_log_file = Path("logs") / f"model_performance_{args.target}_{args.embedding_method}.log"
    log_file_arg = args.log_file if args.log_file is not None else str(default_log_file)
    log_path = setup_tee_logging(log_file=log_file_arg)
    if log_path is not None:
        print(f"[Logging] Tee output to {log_path}")

    print("=" * 70)
    print("Prediction Modeling for Ad Optimization (No IAI)")
    print("=" * 70)
    print(f"Target: {args.target}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Embedding method: {args.embedding_method}")
    print("=" * 70)

    out_dir = Path(args.models_dir)
    out_dir.mkdir(exist_ok=True)

    df_train, df_test = load_data(args.data_dir, args.embedding_method)

    X_train, y_train, w_train, features = prepare_xyw(df_train, args.target)
    X_test, y_test, w_test, _ = prepare_xyw(df_test, args.target)

    _validate_non_negative_target(y_train, args.target)
    _validate_non_negative_target(y_test, args.target)

    print(f"\nFeatures ({len(features)}): {', '.join(features[:5])}...")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results: Dict[str, Dict[str, object]] = {}

    if "ridge" in args.models:
        _, score, metrics = train_ridge_mse(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            cv_folds=args.cv_folds,
            sample_weight_train=w_train,
            sample_weight_test=w_test,
        )
        results["Ridge"] = {"score": score, "metrics": metrics}

    if "xgb" in args.models:
        _, score, metrics = train_xgb_mse(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            cv_folds=args.cv_folds,
            sample_weight_train=w_train,
            sample_weight_test=w_test,
        )
        results["XGB"] = {"score": score, "metrics": metrics}

    if "rf" in args.models:
        _, score, metrics = train_rf_mse(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            cv_folds=args.cv_folds,
            sample_weight_train=w_train,
            sample_weight_test=w_test,
        )
        results["RF"] = {"score": score, "metrics": metrics}

    if "tabpfn" in args.models:
        _, score, metrics = train_tabpfn(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            tabpfn_max_train_rows=args.tabpfn_max_train_rows,
            sample_weight_train=w_train,
            sample_weight_test=w_test,
        )
        results["TabPFN"] = {"score": score, "metrics": metrics}

    # Calculate target statistics
    y_test_mean = float(y_test.mean())
    y_test_std = float(y_test.std())
    y_test_var = float(y_test.var())
    y_test_min = float(y_test.min())
    y_test_max = float(y_test.max())
    y_test_median = float(y_test.median())
    
    # Baseline MSE: variance (predicting the mean for all samples)
    baseline_mse = y_test_var

    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("Model Performance Summary (Test MSE - lower is better)")
    summary_lines.append("=" * 70)
    
    # Target statistics
    summary_lines.append("\nTarget Statistics (for context):")
    summary_lines.append(f"  Mean: {y_test_mean:.4f}")
    summary_lines.append(f"  Std Dev: {y_test_std:.4f}")
    summary_lines.append(f"  Variance (baseline MSE): {baseline_mse:.4f}")
    summary_lines.append(f"  Median: {y_test_median:.4f}")
    summary_lines.append(f"  Min: {y_test_min:.4f}")
    summary_lines.append(f"  Max: {y_test_max:.4f}")
    summary_lines.append("")
    
    for model_name, info in sorted(results.items(), key=lambda x: float(x[1]["score"]), reverse=False):
        score = float(info["score"])
        metrics = info["metrics"]  # type: ignore[assignment]
        line = (
            f"  {model_name:6s}: MSE={score:.4f} | R^2={metrics['r2_score']:.4f} | "
            f"bias={metrics['global_bias']:.4f} | "
            f"top-decile lift={metrics['top_decile_lift']:.4f} | "
            f"cMAE={metrics['conditional_mae']:.4f}"
        )
        summary_lines.append(line)

    if results:
        best_model = min(results.items(), key=lambda x: float(x[1]["score"]))[0]
        summary_lines.append(f"\nBest model: {best_model}")
    
    summary_lines.append("=" * 70)
    
    # Print to console and log file
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    main()
