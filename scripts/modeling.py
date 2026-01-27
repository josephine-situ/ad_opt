"""
Prediction modeling for clicks. We only use XGB here as this was found to be the best model in prior experiments.
Handles training, evaluation, and saving of models. Used in backtests.
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

def train_xgb_mse(df_train, df_test, features, target, param_grid):
    '''Train and evaluate an XGBoost model for regression using MSE loss. Use CV for hyperparameter tuning.
    Returns the trained model and prints evaluation metrics.
    '''

    # Prepare X and y
    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    # Define preprocessor
    categorical_cols = list(X_train.select_dtypes(include=["object", "category", "bool"]).columns)
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False), numeric_cols),
            ('cat', OneHotEncoder(sparse_output=True), categorical_cols)
        ],
        remainder='drop' # or 'passthrough' if you want to keep other columns
    )

    # Define model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Pipeline
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cast", FunctionTransformer(_to_float32_csr, accept_sparse=True)),
            ("model", xgb_model),
        ]
    )

    # Grid search with CV
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )
    
    grid_search.fit(X_train, y_train)

    # Evaluate best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")
    train_metrics = evaluate_model(best_model, X_train, y_train)
    print(f"Training metrics: {train_metrics}")
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print(f"Test metrics: {test_metrics}")

    return best_model


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--course', default='gen_ai', help='Course name (default: gen_ai)')
    args = parser.parse_args()

    log_path = setup_tee_logging(
        log_file=None,
        default_log_dir='logs',
        default_log_prefix=f'modeling_{args.course}',
    )

    print(f"[Logging] Tee output to {log_path}")
    print(f"Course: {args.course}")

    # Load processed training and test data
    base_dir = Path(f'data/{args.course}')
    df_train = pd.read_csv(base_dir / 'clean/train_bert.csv')
    df_test = pd.read_csv(base_dir / 'clean/test_bert.csv')
    print(f"Loaded training data: {df_train.shape}, test data: {df_test.shape}")

    # Features
    features = [
        'Match type', 'Region', 'day_of_week', 'is_weekend', 'month',
        'is_public_holiday', 'days_to_next_course_start', 'last_month_searches',
        'three_month_avg', 'six_month_avg', 'mom_change', 'search_trend',
        'Competition (indexed value)', 'Top of page bid (low range)',
        'Top of page bid (high range)', 'Cost'
    ]
    bert_cols = [col for col in df_train.columns if col.startswith('bert_')]
    features.extend(bert_cols)
    target = 'Clicks'

    print(f"Using target: {target}")
    print(f"Using {len(features)} features for modeling.")

    # Keep the grid small: embedding complexity primarily scales with the
    # number of trees and the tree depth. 
    param_grid = {
        "model__n_estimators": [5, 10, 20],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.1, 0.3],
        "model__subsample": [1.0],
        "model__colsample_bytree": [1.0],
    }

    best_model = train_xgb_mse(df_train, df_test, features, target, param_grid)

    # Save the best model
    # Use course-specific model name
    xgb_path = Path(f'models/{args.course}_xgb_clicks_model.joblib')
    joblib.dump(best_model, xgb_path)
    print(f"Saved best model to {xgb_path}")

if __name__ == '__main__':
    main()