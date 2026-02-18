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

def train_best_model(df_day, features, day_date, n_estimators=None):
    """
    Train a single best model for the day using GridSearchCV.

    Parameters
    ----------
    n_estimators : int or None
        If *None* (default), n_estimators is included in the grid search
        ([5, 10, 20]).  If an int is supplied, n_estimators is **fixed** at
        that value and only max_depth / learning_rate are searched.

    Returns: pipeline, best_params, cv_score (positive MSE), in_sample_mse, r2, bias
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
    
    # Base estimator
    xgb_kwargs = dict(
        objective="reg:squarederror",
        random_state=42,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    if n_estimators is not None:
        xgb_kwargs["n_estimators"] = n_estimators

    xgb_reg = xgb.XGBRegressor(**xgb_kwargs)
    
    pipeline = Pipeline(
        [
            ("preprocess", pre),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", xgb_reg),
        ]
    )

    # Keep the grid small to reduce overfitting risk on small daily data
    param_grid = {
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.1, 0.3],
    }
    if n_estimators is None:
        param_grid["model__n_estimators"] = [5, 10, 20]
    
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
    if n_estimators is not None:
        best_params["model__n_estimators"] = n_estimators
    best_cv_score = -grid_search.best_score_ # Convert back to positive MSE
    
    # In-sample metrics on full data (best_model is already refitted on X, y)
    y_pred = best_model.predict(X)
    in_sample_mse = mean_squared_error(y, y_pred)
    in_sample_r2 = r2_score(y, y_pred)
    in_sample_bias = (y_pred - y).mean()
    
    return best_model, best_params, best_cv_score, in_sample_mse, in_sample_r2, in_sample_bias


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