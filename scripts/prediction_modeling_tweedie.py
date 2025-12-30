"""Prediction Modeling (No IAI) for Ad Optimization
================================================

Trains Tweedie-loss regression models to predict:
- conversion: "Conv. value"
- clicks: "Clicks"
- epc: "EPC"

This script is a drop-in *workflow* replacement for `scripts/prediction_modeling.py`
that does NOT require InterpretableAI (IAI) or a license.

Key outputs
-----------
1) A Tweedie GLM (linear) trained with scikit-learn's `TweedieRegressor`.
   - Saves a fitted pipeline to `models/glm_{embedding}_{target}.joblib`
   - Exports coefficients in the CSV format already consumed by
     `scripts/bid_optimization.py`:
       - `models/weights_{embedding}_{target}_numeric.csv`
       - `models/weights_{embedding}_{target}_categorical.csv` (if any)
       - `models/weights_{embedding}_{target}_constant.csv`

2) Optionally, an XGBoost model trained with Tweedie objective.
   - Saves to `models/xgb_tweedie_{embedding}_{target}.json`

Usage
-----
python scripts/prediction_modeling_tweedie.py --target conversion --embedding-method tfidf
python scripts/prediction_modeling_tweedie.py --target clicks --embedding-method bert --models glm xgb

Notes
-----
- Tweedie variance power defaults:
  - clicks: 1.0 (Poisson)
  - conversion / epc: 1.5
- Targets are assumed to be non-negative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        return df["EPC"]
    if target == "clicks":
        return df["Clicks"]
    raise ValueError(f"Unknown target: {target}")


def get_tweedie_power(target: str) -> float:
    # clicks ~= counts -> Poisson
    return 1.0 if target == "clicks" else 1.5


def _as_numpy(x: Iterable[float]) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()  # type: ignore[no-any-return]
    return np.asarray(list(x), dtype=float)


def tweedie_d2_score(y_true: Iterable[float], y_pred: Iterable[float], power: float) -> float:
    """Compute Tweedie D^2 (deviance-based R^2).

    D^2 = 1 - D(y, y_pred) / D(y, y_mean)
    """

    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)

    # mean_tweedie_deviance requires strictly positive y_pred when power != 0
    yp = np.maximum(yp, 1e-12)

    baseline = np.full_like(yt, np.mean(yt), dtype=float)
    baseline = np.maximum(baseline, 1e-12)

    dev_model = mean_tweedie_deviance(yt, yp, power=power)
    dev_null = mean_tweedie_deviance(yt, baseline, power=power)

    if dev_null == 0:
        return float("nan")
    return float(1.0 - (dev_model / dev_null))


def compute_global_bias(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    return float(np.mean(yp) - np.mean(yt))


def compute_top_decile_lift(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    if yt.size == 0:
        return float("nan")
    overall_mean = float(np.mean(yt))
    if overall_mean == 0:
        return float("nan")
    n_top = max(1, int(0.1 * yt.size))
    top_idx = np.argsort(yp)[-n_top:]
    top_mean = float(np.mean(yt[top_idx]))
    return float(top_mean / overall_mean)


def compute_conditional_mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = _as_numpy(y_true)
    yp = _as_numpy(y_pred)
    mask = yt > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
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


def _validate_non_negative_target(y: pd.Series, target: str) -> None:
    if (y < 0).any():
        n_bad = int((y < 0).sum())
        raise ValueError(
            f"Target '{target}' contains {n_bad} negative values. "
            "Tweedie models with log link assume non-negative targets."
        )


def train_glm_tweedie(
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
) -> Tuple[Pipeline, float, Dict[str, float]]:
    print("\n--- Tweedie GLM (scikit-learn) ---")
    power = get_tweedie_power(target)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                TweedieRegressor(
                    power=power,
                    link="log",
                    alpha=1.0,
                    max_iter=5000,
                ),
            ),
        ]
    )

    # Keep grid modest; this is a baseline that also produces coefficients.
    param_grid = {
        "model__alpha": [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # GridSearchCV maximizes score; use negative deviance as the scoring objective.
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=lambda est, X, y: -mean_tweedie_deviance(
            _as_numpy(y), np.maximum(_as_numpy(est.predict(X)), 1e-12), power=power
        ),
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)
    best: Pipeline = grid.best_estimator_

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    d2_train = tweedie_d2_score(y_train, y_pred_train, power=power)
    d2_test = tweedie_d2_score(y_test, y_pred_test, power=power)

    print(f"  Tweedie variance power: {power}")
    print(f"  Best alpha: {grid.best_params_['model__alpha']}")
    print(f"  Train Tweedie D^2 (higher is better): {d2_train:.4f}")
    print(f"  Test  Tweedie D^2 (higher is better): {d2_test:.4f}")

    # Save fitted pipeline
    model_path = out_dir / f"glm_{embedding_method}_{target}.joblib"
    joblib.dump(best, model_path)
    print(f"  Saved pipeline to {model_path}")

    # Export weights in existing CSV format
    export_glm_weights(best, numeric_cols, categorical_cols, embedding_method, target, out_dir)

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test),
    }

    print(f"  [GLM] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [GLM] Global bias: nan")
    print(f"  [GLM] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [GLM] Top decile lift: nan")
    print(f"  [GLM] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [GLM] Conditional MAE: nan")

    return best, d2_test, metrics


def export_glm_weights(
    fitted_glm_pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
    embedding_method: str,
    target: str,
    out_dir: Path,
) -> None:
    """Export GLM intercept + coefficients to the repo's existing weight CSV format."""

    preprocess: ColumnTransformer = fitted_glm_pipeline.named_steps["preprocess"]
    model: TweedieRegressor = fitted_glm_pipeline.named_steps["model"]

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


def train_xgb_tweedie(
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
) -> Tuple[Pipeline, float, Dict[str, float]]:
    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost "
            "(or `pip install -e .[ml_open]` if using this repo's extras)."
        )

    print("\n--- XGBoost (Tweedie) ---")
    power = get_tweedie_power(target)

    preprocessor, _, _ = build_preprocessor(X_train)

    model = xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=power,
        random_state=seed,
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    param_grid = {
        "model__max_depth": [3, 5, 7],
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bytree": [0.7, 0.9],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=lambda est, X, y: -mean_tweedie_deviance(
            _as_numpy(y), np.maximum(_as_numpy(est.predict(X)), 1e-12), power=power
        ),
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)
    best: Pipeline = grid.best_estimator_

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    d2_train = tweedie_d2_score(y_train, y_pred_train, power=power)
    d2_test = tweedie_d2_score(y_test, y_pred_test, power=power)

    print(f"  Tweedie variance power: {power}")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Train Tweedie D^2 (higher is better): {d2_train:.4f}")
    print(f"  Test  Tweedie D^2 (higher is better): {d2_test:.4f}")

    # Save just the fitted booster in XGBoost's native format
    xgb_path = out_dir / f"xgb_tweedie_{embedding_method}_{target}.json"
    best.named_steps["model"].save_model(xgb_path)
    print(f"  Saved model to {xgb_path}")

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test),
    }

    print(f"  [XGB] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [XGB] Global bias: nan")
    print(f"  [XGB] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [XGB] Top decile lift: nan")
    print(f"  [XGB] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [XGB] Conditional MAE: nan")

    return best, d2_test, metrics


def train_rf_tweedie(
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
) -> Tuple[Pipeline, float, Dict[str, float]]:
    """Train an XGBoost random-forest style model with Tweedie objective.

    Note: scikit-learn's RandomForestRegressor does not natively optimize Tweedie deviance.
    XGBoost's XGBRFRegressor provides an RF-like ensemble while supporting reg:tweedie.
    """

    if xgb is None:
        raise ImportError(
            "xgboost is not installed. Install with: pip install xgboost "
            "(or `pip install -e .[ml_open]` if using this repo's extras)."
        )

    print("\n--- XGBoost RF (Tweedie) ---")
    power = get_tweedie_power(target)

    preprocessor, _, _ = build_preprocessor(X_train)

    model = xgb.XGBRFRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=power,
        random_state=seed,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bynode=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        learning_rate=1.0,
        tree_method="hist",
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    param_grid = {
        "model__max_depth": [3, 5, 7],
        "model__n_estimators": [300, 600],
        "model__subsample": [0.7, 0.9],
        "model__colsample_bynode": [0.7, 0.9],
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=lambda est, X, y: -mean_tweedie_deviance(
            _as_numpy(y), np.maximum(_as_numpy(est.predict(X)), 1e-12), power=power
        ),
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)
    best: Pipeline = grid.best_estimator_

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

    d2_train = tweedie_d2_score(y_train, y_pred_train, power=power)
    d2_test = tweedie_d2_score(y_test, y_pred_test, power=power)

    print(f"  Tweedie variance power: {power}")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Train Tweedie D^2 (higher is better): {d2_train:.4f}")
    print(f"  Test  Tweedie D^2 (higher is better): {d2_test:.4f}")

    rf_path = out_dir / f"rf_tweedie_{embedding_method}_{target}.json"
    best.named_steps["model"].save_model(rf_path)
    print(f"  Saved model to {rf_path}")

    metrics = {
        "global_bias": compute_global_bias(y_test, y_pred_test),
        "top_decile_lift": compute_top_decile_lift(y_test, y_pred_test),
        "conditional_mae": compute_conditional_mae(y_test, y_pred_test),
    }

    print(f"  [RF] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics["global_bias"]) else "  [RF] Global bias: nan")
    print(f"  [RF] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics["top_decile_lift"]) else "  [RF] Top decile lift: nan")
    print(f"  [RF] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics["conditional_mae"]) else "  [RF] Conditional MAE: nan")

    return best, d2_test, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tweedie-loss prediction models (no IAI).")
    parser.add_argument(
        "--target",
        type=str,
        default="clicks",
        choices=["conversion", "epc", "clicks"],
        help="Target variable: conversion (Conv. value), epc (EPC), or clicks (default: clicks)",
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
        default="tfidf",
        choices=["tfidf", "bert"],
        help="Embedding method used in data (default: tfidf)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["glm", "xgb", "rf"],
        choices=["glm", "xgb", "rf"],
        help="Which models to train (default: glm xgb rf)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for grid search")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Output directory for trained models/weights (default: models)",
    )

    args = parser.parse_args()

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

    X_train, features = get_features(df_train, args.target)
    X_test, _ = get_features(df_test, args.target)
    y_train = get_target(df_train, args.target)
    y_test = get_target(df_test, args.target)

    _validate_non_negative_target(y_train, args.target)
    _validate_non_negative_target(y_test, args.target)

    print(f"\nFeatures ({len(features)}): {', '.join(features[:5])}...")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    results: Dict[str, Dict[str, object]] = {}

    if "glm" in args.models:
        _, score, metrics = train_glm_tweedie(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            cv_folds=args.cv_folds,
        )
        results["GLM"] = {"score": score, "metrics": metrics}

    if "xgb" in args.models:
        _, score, metrics = train_xgb_tweedie(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            cv_folds=args.cv_folds,
        )
        results["XGB"] = {"score": score, "metrics": metrics}

    if "rf" in args.models:
        _, score, metrics = train_rf_tweedie(
            X_train,
            y_train,
            X_test,
            y_test,
            target=args.target,
            embedding_method=args.embedding_method,
            seed=args.seed,
            out_dir=out_dir,
            cv_folds=args.cv_folds,
        )
        results["RF"] = {"score": score, "metrics": metrics}

    print("\n" + "=" * 70)
    print("Model Performance Summary (Test Tweedie D^2)")
    print("=" * 70)
    for model_name, info in sorted(results.items(), key=lambda x: float(x[1]["score"]), reverse=True):
        score = float(info["score"])
        metrics = info["metrics"]  # type: ignore[assignment]
        print(
            f"  {model_name:6s}: {score:.4f} | "
            f"bias={metrics['global_bias']:.4f} | "
            f"top-decile lift={metrics['top_decile_lift']:.4f} | "
            f"cMAE={metrics['conditional_mae']:.4f}"
        )

    if results:
        best_model = max(results.items(), key=lambda x: float(x[1]["score"]))[0]
        print(f"\nBest model: {best_model}")
    print("=" * 70)


if __name__ == "__main__":
    main()
