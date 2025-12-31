"""mirrored_ORT

Train a "mirrored" Optimal Regression Tree (ORT-H) to approximate a trained XGBoost model.

Workflow:
1) Load train/test feature matrices (same format as scripts/prediction_modeling.py).
2) Load a trained XGBoost model (either IAI JSON via `write_json`, or an XGBoost-native
    booster JSON saved by `xgboost.XGBRegressor.save_model` as in
    scripts/prediction_modeling_tweedie.py).
3) Compute XGB predictions on train; use them as pseudo-labels.
4) Fit an OptimalTreeRegressor with hyperplane splits (ORT-H) via grid search.
5) Report test MSE (against the true target) and save the fitted ORT model.

This is intended for use by scripts/bid_optimization.py (algorithm key: "mort").

Example:
    python scripts/mirrored_ORT.py --target conversion --embedding-method bert \
        --xgb-model models/xgb_bert_conversion.json

Notes:
- Requires Interpretable AI (IAI) and a valid license.
- Hyperplane splits are enabled by setting `hyperplane_config`.

Docs reference:
- https://docs.interpretable.ai/stable/OptimalTrees/quickstart/regression/
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.iai_setup import iai
except ImportError:
    print("ERROR: Could not set up IAI. Install with: pip install iai")
    print("Note: IAI requires a valid license.")
    raise


def _load_xgb_predictor(
    *,
    xgb_path: Path,
    xgb_source: str,
    X_train_raw: pd.DataFrame,
) -> Any:
    """Load an XGB predictor.

    Supported sources:
    - 'iai': JSON created by `iai_learner.write_json(...)` and loaded via `iai.read_json`.
    - 'xgboost_booster': JSON created by `xgboost.XGBRegressor.save_model(...)`.
    - 'auto': try 'iai' first; if it fails, fall back to 'xgboost_booster'.

    For the xgboost booster case, we reconstruct the preprocessing pipeline using the
    same `build_preprocessor` and casting function used during training in
    scripts/prediction_modeling_tweedie.py.
    """

    if xgb_source not in {"auto", "iai", "xgboost_booster"}:
        raise ValueError("--xgb-source must be one of: auto, iai, xgboost_booster")

    if xgb_source in {"auto", "iai"}:
        try:
            return iai.read_json(str(xgb_path))
        except Exception as e:
            if xgb_source == "iai":
                raise RuntimeError(
                    "Failed to load XGB model via `iai.read_json`. "
                    "If this model was saved by xgboost/sklearn (prediction_modeling_tweedie.py), "
                    "rerun with `--xgb-source xgboost_booster`."
                ) from e

    # Fallback / explicit: XGBoost native booster JSON
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "xgboost is required to load booster JSONs. Install with: pip install xgboost"
        ) from e

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required to rebuild the preprocessing pipeline for booster JSONs. "
            "Install with: pip install scikit-learn"
        ) from e

    try:
        # Reuse the exact preprocessor used during training.
        from scripts import prediction_modeling_tweedie as pmt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import scripts.prediction_modeling_tweedie (needed to rebuild preprocessing for booster JSONs)."
        ) from e

    # Prefer using a persisted, fitted preprocessor if available.
    # This ensures exact category -> column mapping and scaling matching training.
    preproc_path = xgb_path.with_name(xgb_path.stem + "_preprocess.joblib")
    try:
        import joblib  # type: ignore
    except Exception:
        joblib = None

    if joblib is not None and preproc_path.exists():
        preprocessor = joblib.load(preproc_path)
    else:
        preprocessor, _, _ = pmt.build_preprocessor(X_train_raw)
        # The booster JSON saved by `XGBRegressor.save_model` does NOT include the fitted
        # sklearn preprocessing pipeline. If we don't have a persisted preprocessor,
        # we fit one on the distillation training matrix.
        preprocessor.fit(X_train_raw)

    model = xgb.XGBRegressor()
    model.load_model(xgb_path)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cast", FunctionTransformer(pmt._to_float32_csr, accept_sparse=True)),
            ("model", model),
        ]
    )
    return pipe


def load_data(data_dir: str = "data/clean", embedding_method: str = "bert") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data."""
    df_train = pd.read_csv(f"{data_dir}/train_{embedding_method}.csv")
    df_test = pd.read_csv(f"{data_dir}/test_{embedding_method}.csv")
    return df_train, df_test


def get_features(df: pd.DataFrame, target: str = "conversion") -> Tuple[pd.DataFrame, List[str]]:
    """Extract model features.

    Mirrors scripts/prediction_modeling.get_features.
    """
    df_features = df.copy()

    if target == "conversion":
        excluded_cols = {"Conv. value", "EPC", "Day", "Keyword"}
    elif target == "epc":
        excluded_cols = {"Conv. value", "Clicks", "EPC", "Day", "Keyword"}
    else:  # target == 'clicks'
        excluded_cols = {"Clicks", "Conv. value", "EPC", "Day", "Keyword"}

    feature_cols = [col for col in df_features.columns if col not in excluded_cols]

    categorical_cols = df_features[feature_cols].select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df_features[col] = df_features[col].astype("category")

    return df_features[feature_cols], feature_cols


def get_target(df: pd.DataFrame, target: str = "conversion") -> pd.Series:
    if target == "conversion":
        return df["Conv. value"]
    if target == "epc":
        return df["EPC"]
    if target == "clicks":
        return df["Clicks"]
    raise ValueError(f"Unknown target: {target}")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def _candidate_hyperplane_configs(sparsity: str) -> List[Any]:
    """Return candidate python shapes for `hyperplane_config`.

    IAI's Python bindings have historically accepted different container types
    depending on version. We try a few common representations.

    Per docs, a single config with `sparsity=:all` enables ORT-H.
    """
    config_dict = {"sparsity": sparsity}
    return [
        config_dict,
        [config_dict],
        (config_dict,),
        (("sparsity", sparsity),),
    ]


def build_grid(
    seed: int,
    max_depth_grid: List[int],
    minbucket_grid: List[float],
    hyperplane_sparsity: str,
    n_folds: int,
) -> Any:
    """Build an IAI GridSearch for ORT-H (hyperplane splits)."""

    base = iai.OptimalTreeRegressor(
        random_seed=seed,
        criterion="mse",
        normalize_y=False,
        show_progress=False,
    )

    # Prefer putting hyperplane_config into the grid params so it is explicit.
    # But since bindings differ, we fall back to setting it on the learner.
    grid_params: Dict[str, Any] = {
        "max_depth": max_depth_grid,
        "minbucket": minbucket_grid,
    }

    hyperplane_configs = _candidate_hyperplane_configs(hyperplane_sparsity)

    for hp_cfg in hyperplane_configs:
        try:
            grid = iai.GridSearch(base, **grid_params, hyperplane_config=[hp_cfg])
            # quick check: if GridSearch accepted, return it
            return grid
        except TypeError:
            continue
        except Exception:
            continue

    # Fall back: set hyperplane_config on the base learner.
    last_err: Optional[BaseException] = None
    for hp_cfg in hyperplane_configs:
        try:
            base_hp = iai.OptimalTreeRegressor(
                random_seed=seed,
                criterion="mse",
                normalize_y=False,
                show_progress=False,
                hyperplane_config=hp_cfg,
            )
            grid = iai.GridSearch(base_hp, **grid_params)
            return grid
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not configure `hyperplane_config` for ORT-H on this IAI version. "
        "Try updating IAI or adjust the hyperplane_config representation."
    ) from last_err


def _print_run_header(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("Mirrored ORT-H distillation")
    print("=" * 70)
    print(f"Host: {socket.gethostname()}")
    print(f"PWD: {os.getcwd()}")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id:
        print(f"SLURM_JOB_ID: {slurm_job_id}")
    if slurm_array_task_id:
        print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"Target: {args.target}")
    print(f"Embedding method: {args.embedding_method}")
    print(f"XGB model: {args.xgb_model} (source={args.xgb_source})")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mirrored ORT-H from a trained XGB model.")
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help=(
            "Optional working directory to chdir into before loading data/models. "
            "Useful on clusters; if omitted, uses current working directory."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default="clicks",
        choices=["conversion", "epc", "clicks"],
        help="Target variable.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/clean",
        help="Data directory containing train/test CSVs (default: data/clean).",
    )
    parser.add_argument(
        "--embedding-method",
        type=str,
        default="bert",
        choices=["tfidf", "bert"],
        help="Embedding method used in the data files.",
    )
    parser.add_argument(
        "--xgb-model",
        type=str,
        default="models/xgb_tweedie_bert_clicks.json",
        help=(
            "Path to trained XGB model JSON. Supports IAI JSON (write_json) and XGBoost booster JSON "
            "(xgboost.XGBRegressor.save_model, as in prediction_modeling_tweedie.py)."
        ),
    )
    parser.add_argument(
        "--xgb-source",
        type=str,
        default="auto",
        choices=["auto", "iai", "xgboost_booster"],
        help="How to load --xgb-model (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Cross-validation folds for grid search.",
    )
    parser.add_argument(
        "--max-depth-grid",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="Grid for ORT max_depth.",
    )
    parser.add_argument(
        "--minbucket-grid",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05],
        help="Grid for ORT minbucket.",
    )
    parser.add_argument(
        "--hyperplane-sparsity",
        type=str,
        default="all",
        help="Hyperplane sparsity; docs show 'all' to allow any number of features per split.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save the mirrored ORT model.",
    )

    args = parser.parse_args()

    if args.workdir:
        os.chdir(args.workdir)

    _print_run_header(args)

    df_train, df_test = load_data(args.data_dir, args.embedding_method)
    X_train, _ = get_features(df_train, target=args.target)
    X_test, _ = get_features(df_test, target=args.target)
    y_train = get_target(df_train, target=args.target)
    y_test = get_target(df_test, target=args.target)

    xgb_path = Path(args.xgb_model)
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGB model not found: {xgb_path}")

    print(f"Loading XGB model: {xgb_path} (source={args.xgb_source})")
    xgb_predictor = _load_xgb_predictor(xgb_path=xgb_path, xgb_source=args.xgb_source, X_train_raw=X_train)

    # Distillation targets
    y_train_hat = np.asarray(xgb_predictor.predict(X_train), dtype=float)
    y_test_hat = np.asarray(xgb_predictor.predict(X_test), dtype=float)

    print("Training mirrored ORT-H (hyperplane splits enabled) via grid search...")
    grid = build_grid(
        seed=args.seed,
        max_depth_grid=args.max_depth_grid,
        minbucket_grid=args.minbucket_grid,
        hyperplane_sparsity=args.hyperplane_sparsity,
        n_folds=args.n_folds,
    )

    grid.fit_cv(X_train, y_train_hat, validation_criterion="mse", n_folds=args.n_folds, verbose=True)

    # Evaluate on true target (what we ultimately care about)
    y_pred_test = np.asarray(grid.predict(X_test), dtype=float)
    test_mse_true = mse(np.asarray(y_test, dtype=float), y_pred_test)

    # Also compute mimic error vs XGB predictions (distillation fidelity)
    test_mse_mimic = mse(y_test_hat, y_pred_test)

    print(f"Test MSE vs true {args.target}: {test_mse_true:.6f}")
    print(f"Test MSE vs XGB predictions (mimic): {test_mse_mimic:.6f}")

    # Save model
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_model = models_dir / f"mort_{args.embedding_method}_{args.target}.json"
    learner = grid.get_learner()
    learner.write_json(str(out_model))
    print(f"Saved mirrored ORT model to {out_model}")

    # Save metrics
    metrics = {
        "target": args.target,
        "embedding_method": args.embedding_method,
        "xgb_model": str(xgb_path),
        "test_mse_true": test_mse_true,
        "test_mse_mimic": test_mse_mimic,
        "best_params": getattr(grid, "get_best_params", lambda: None)(),
    }
    metrics_path = models_dir / f"mort_{args.embedding_method}_{args.target}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
