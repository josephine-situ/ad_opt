"""
Prediction modeling for clicks. We only use XGB here as this was found to be the best model in prior experiments.
Handles training, evaluation, and saving of models. Used in backtests.

Example:
    python scripts/modeling.py --course ml

Input files:
    data/<course>/clean/train_<emb>.csv     Training split (from tidy_get_data.py)
    data/<course>/clean/test_<emb>.csv      Test split     (from tidy_get_data.py)

Output files:
    models/<course>_xgb_clicks_model_<emb>.joblib   Saved sklearn Pipeline (preprocessor + XGBoost)
    logs/modeling_<course>_<emb>_*.log               Run log

Estimated run time (HP Spectre x360, i7-1065G7 @ 1.30 GHz, 4C/8T, 16 GB RAM, no discrete GPU):
    ~1-3 min per course (GridSearchCV with 5-fold CV)
"""

from pathlib import Path
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.sparse as sp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import setup_tee_logging

def _to_float32_csr(X):
    """Cast matrices to float32; prefer CSR for sparse.

    This avoids a class of XGBoost crashes/slow paths on some HPC builds.
    """

    if sp is not None and sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def evaluate_model(model, X_test, y_test):
    '''Evaluate the model on test data.'''

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    bias = (y_pred - y_test).mean()

    # Actual avg of predicted top keywords vs avg of true values
    n_top = round(0.1 * len(y_test))
    top_idx = y_pred.argsort()[-n_top:]
    lift = y_test.iloc[top_idx].mean() / y_test.mean()

    metrics = {
        'MSE': mse,
        'R2': r2,
        'Bias': bias,
        'Lift': lift
    }

    return metrics

def train_best_model(df_day, features, day_date):
    """
    Train a single best model for the day using GridSearchCV.
    Returns: pipeline, best_params, cv_score (negative MSE), in_sample_score (MSE), r2, bias
    """
    X, y = df_day[features], df_day["Clicks"]
    cat = list(X.select_dtypes(include=["object", "category", "bool"]).columns)
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(with_mean=False), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat),
        ],
        remainder="drop",
    )
    
    # Base estimator (same hyperparams range as typically used, but can be searched)
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    
    pipeline = Pipeline(
        [
            ("preprocess", pre),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", xgb_reg),
        ]
    )

    # Keep the grid small to reduce overfitting risk on small daily data
    param_grid = {
        "model__n_estimators": [5, 10, 20],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.1, 0.3],
    }
    
    # Use deterministic seed based on day
    if hasattr(day_date, 'strftime'):
        seed_val = int(day_date.strftime('%Y%m%d'))
    else:
        seed_val = 42

    cv = KFold(n_splits=5, shuffle=True, random_state=seed_val)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = -grid_search.best_score_ # Convert back to positive MSE
    
    # In-sample metrics on full data (best_model is already refitted on X, y)
    y_pred = best_model.predict(X)
    in_sample_mse = mean_squared_error(y, y_pred)
    in_sample_r2 = r2_score(y, y_pred)
    in_sample_bias = (y_pred - y).mean()
    
    return best_model, best_params, best_cv_score, in_sample_mse, in_sample_r2, in_sample_bias


def train_oracle_model(
    df_full,
    features_base,
    raw_emb_map,
    k_candidates=(10, 20, 50, 100, 384),
    embedding_prefix='bert',
):
    """Train the Gold-Standard Oracle model via CV over SVD dimensionality *k*.

    For each candidate *k*:
      1. Fit SVD(*k*) on the **full** keyword set.
      2. Replace embedding columns with the SVD-transformed embeddings.
      3. Run ``train_best_model`` (GridSearchCV on XGBoost).
      4. Record CV MSE.

    The *k* with the lowest CV MSE is selected (``k_eval``), and a final
    model is trained on the full data with that *k*.

    Args:
        df_full: Full dataset (Day 0 … T) containing ``'Keyword'`` and
                 ``'Day'`` columns plus all base features.
        features_base: Feature column names **excluding** embedding columns.
        raw_emb_map: ``{keyword: raw_embedding_vector}`` dict (from
                     :func:`get_raw_bert_embeddings_cached`).
        k_candidates: Tuple of SVD component counts to try.  Values ≥ the
                      raw embedding dimensionality are treated as "no SVD".
        embedding_prefix: Column prefix for embedding columns (default
                          ``'bert'``).

    Returns:
        oracle_model: Fitted ``sklearn.pipeline.Pipeline``.
        oracle_svd_pipeline: Fitted SVD pipeline dict.
        best_k: Selected ``n_components`` (``int`` or ``None``).
        cv_results: ``{k_label: cv_mse}`` dict.
        oracle_features: List of feature names used by the Oracle.
    """
    from utils.embeddings import fit_svd_pipeline, replace_embeddings

    unique_keywords = df_full['Keyword'].unique()
    raw_matrix = np.array([raw_emb_map[kw] for kw in unique_keywords])
    embedding_dim = raw_matrix.shape[1]

    cv_results = {}

    for k in k_candidates:
        # Normalise k – values ≥ embedding_dim mean "no SVD"
        k_eff = None if (k is None or k >= embedding_dim) else k
        k_label = embedding_dim if k_eff is None else k_eff

        # Skip duplicate labels (e.g. 384 and 768 both map to 384)
        if k_label in cv_results:
            continue

        print(f"  Oracle CV: k={k_label} …")
        svd_pipe = fit_svd_pipeline(raw_matrix, n_components=k_eff)
        df_k, emb_cols = replace_embeddings(
            df_full.copy(), raw_emb_map, svd_pipe, prefix=embedding_prefix,
        )
        features_k = list(features_base) + emb_cols

        _, _, cv_mse, _, _, _ = train_best_model(
            df_k, features_k, df_full['Day'].max(),
        )
        cv_results[k_label] = cv_mse
        print(f"    k={k_label}: CV MSE = {cv_mse:.6f}")

    # ── Pick best k ─────────────────────────────────────────────────────
    best_k_label = min(cv_results, key=cv_results.get)
    best_k = None if best_k_label >= embedding_dim else best_k_label
    print(f"  Oracle best k={best_k_label} (CV MSE={cv_results[best_k_label]:.6f})")

    # ── Refit final model with best k on all data ───────────────────────
    svd_pipe_best = fit_svd_pipeline(raw_matrix, n_components=best_k)
    df_best, emb_cols_best = replace_embeddings(
        df_full.copy(), raw_emb_map, svd_pipe_best, prefix=embedding_prefix,
    )
    features_best = list(features_base) + emb_cols_best

    oracle_model, best_params, cv_mse, train_mse, train_r2, train_bias = (
        train_best_model(df_best, features_best, df_full['Day'].max())
    )

    print(
        f"  Oracle final: params={best_params}, "
        f"CV MSE={cv_mse:.4f}, Train R²={train_r2:.4f}"
    )

    return oracle_model, svd_pipe_best, best_k, cv_results, features_best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--course', default='gen_ai', help='Course name (default: gen_ai)')
    parser.add_argument('--embedding-method', default='bert', choices=['bert', 'llm'], help='Embedding method: bert or llm (default: bert)')
    parser.add_argument('--train-only', action='store_true', help='Train on train set only (default: train on full data)')
    args = parser.parse_args()

    log_path = setup_tee_logging(
        log_file=None,
        default_log_dir='logs',
        default_log_prefix=f'modeling_{args.course}_{args.embedding_method}',
    )

    print(f"[Logging] Tee output to {log_path}")
    print(f"Course: {args.course}")
    print(f"Embedding method: {args.embedding_method}")

    # Load processed training and test data
    base_dir = Path(f'data/{args.course}')
    embedding_method = args.embedding_method
    df_train = pd.read_csv(base_dir / f'clean/train_{embedding_method}.csv')
    df_test = pd.read_csv(base_dir / f'clean/test_{embedding_method}.csv')
    print(f"Loaded training data: {df_train.shape}, test data: {df_test.shape}")

    # Features
    features = [
        'Match type', 'Region', 'day_of_week', 'is_weekend', 'month',
        'is_public_holiday', 'days_to_next_course_start', 'last_month_searches',
        'three_month_avg', 'six_month_avg', 'mom_change', 'search_trend',
        'Competition (indexed value)', 'Top of page bid (low range)',
        'Top of page bid (high range)', 'Cost'
    ]
    
    # Add embedding-specific columns
    if embedding_method == 'llm':
        if 'llm_relevance_score' in df_train.columns:
            features.append('llm_relevance_score')
    else:
        bert_cols = [col for col in df_train.columns if col.startswith('bert_')]
        features.extend(bert_cols)
    
    target = 'Clicks'

    print(f"Using target: {target}")
    print(f"Using {len(features)} features for modeling.")

    # Use train_best_model for consistency with backtest_daily
    if args.train_only:
        df_fit = df_train
        train_label = "train set only"
    else:
        df_fit = pd.concat([df_train, df_test], ignore_index=True)
        train_label = "full data (train + test)"

    print(f"Training on: {train_label} ({len(df_fit)} rows)")

    best_model, best_params, best_cv_mse, in_sample_mse, in_sample_r2, in_sample_bias = train_best_model(
        df_fit, features=features, day_date=None
    )

    # Derive CV R² from CV MSE: R² = 1 - MSE / Var(y)
    y_var = df_fit[target].var()
    cv_r2 = 1 - best_cv_mse / y_var if y_var > 0 else float('nan')

    print(f"Best hyperparameters: {best_params}")
    print(f"CV MSE ({train_label}): {best_cv_mse:.4f}")
    print(f"CV R2  ({train_label}): {cv_r2:.4f}")
    print(f"In-sample metrics ({train_label}): {{'MSE': {in_sample_mse:.4f}, 'R2': {in_sample_r2:.4f}, 'Bias': {in_sample_bias:.4f}}}")

    # Held-out test evaluation (only meaningful when --train-only)
    if args.train_only:
        test_metrics = evaluate_model(best_model, df_test[features], df_test[target])
        print(f"Held-out test metrics: {test_metrics}")

    # Save the best model
    # Use course-specific and embedding-specific model name
    xgb_path = Path(f'models/{args.course}_xgb_clicks_model_{embedding_method}.joblib')
    joblib.dump(best_model, xgb_path)
    print(f"Saved best model to {xgb_path}")

if __name__ == '__main__':
    main()