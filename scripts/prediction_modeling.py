"""
Prediction Modeling for Ad Optimization
========================================
Trains multiple regression models to predict conversion value and clicks.
Uses IAI (Interpretable AI) library for model training and cross-validation.

Models trained:
- Linear Regression (LR)
- Optimal Cart Trees (OCT)
- Random Forests (RF)
- XGBoost (XGB)

Usage:
    python prediction_modeling.py --target conversion
    python prediction_modeling.py --target clicks
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import IAI with automatic configuration
try:
    from utils.iai_setup import iai
except ImportError:
    print("ERROR: Could not set up IAI. Install with: pip install iai")
    print("Note: IAI requires a valid license.")
    sys.exit(1)


def load_data(data_dir='data/clean', embedding_method='tfidf'):
    """Load training and test data."""
    print(f"Loading data from {data_dir}...")
    df_train = pd.read_csv(f"{data_dir}/train_{embedding_method}.csv")
    df_test = pd.read_csv(f"{data_dir}/test_{embedding_method}.csv")
    print(f"  Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    return df_train, df_test


def get_features(df, target='conversion'):
    """Extract all feature columns except target variable and metadata.
    When target is 'conversion', Clicks is included as a predictor.
    When target is 'epc', both Conv. value and Clicks are excluded (they're used to calculate EPC).
    Convert categorical columns to category dtype for IAI compatibility."""
    
    # Make a copy to avoid modifying original
    df_features = df.copy()
    
    # Exclude target variable and metadata columns
    if target == 'conversion':
        excluded_cols = {'Conv. value', 'EPC', 'Day', 'Keyword'}
    elif target == 'epc':
        excluded_cols = {'Conv. value', 'Clicks', 'EPC', 'Day', 'Keyword'}
    else:  # target == 'clicks'
        excluded_cols = {'Clicks', 'Conv. value', 'EPC', 'Day', 'Keyword'}
    
    feature_cols = [col for col in df_features.columns if col not in excluded_cols]
    
    # Identify and convert categorical columns (string dtype) to category dtype
    categorical_cols = df_features[feature_cols].select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_features[col] = df_features[col].astype('category')
    
    return df_features[feature_cols], feature_cols


def get_target(df, target='conversion'):
    """Get target variable."""
    if target == 'conversion':
        return df['Conv. value']
    elif target == 'epc':
        return df['EPC']
    elif target == 'clicks':
        return df['Clicks']
    else:
        raise ValueError(f"Unknown target: {target}")


def get_tweedie_power(target: str) -> float:
    """Return Tweedie variance power based on the prediction target."""
    return 1.0 if target == 'clicks' else 1.5


def compute_global_bias(y_true, y_pred):
    """Global bias = mean(predictions) - mean(actuals)."""
    actual_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    if np.isnan(actual_mean) or np.isnan(pred_mean):
        return np.nan
    return float(pred_mean - actual_mean)


def compute_top_decile_lift(y_true, y_pred):
    """Top decile lift = avg actuals in top 10% predictions divided by global avg."""
    if len(y_true) == 0:
        return np.nan
    actuals = y_true.values if hasattr(y_true, 'values') else np.asarray(y_true)
    preds = np.asarray(y_pred)
    overall_mean = np.mean(actuals)
    if overall_mean == 0:
        return np.nan
    n_top = max(1, int(0.1 * len(actuals)))
    top_idx = np.argsort(preds)[-n_top:]
    top_mean = np.mean(actuals[top_idx])
    return float(top_mean / overall_mean) if overall_mean else np.nan


def compute_conditional_mae(y_true, y_pred):
    """Conditional MAE computed on rows where actual > 0."""
    if hasattr(y_true, 'values'):
        actuals = y_true.values
    else:
        actuals = np.asarray(y_true)
    preds = np.asarray(y_pred)
    mask = actuals > 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(actuals[mask] - preds[mask])))


def evaluate_additional_metrics(model, X_test, y_test, model_name):
    """Compute and print supplemental metrics for a fitted model."""
    preds = np.asarray(model.predict(X_test))
    metrics = {
        'global_bias': compute_global_bias(y_test, preds),
        'top_decile_lift': compute_top_decile_lift(y_test, preds),
        'conditional_mae': compute_conditional_mae(y_test, preds)
    }
    print(f"  [{model_name}] Global bias: {metrics['global_bias']:.4f}" if not np.isnan(metrics['global_bias']) else f"  [{model_name}] Global bias: nan")
    print(f"  [{model_name}] Top decile lift: {metrics['top_decile_lift']:.4f}" if not np.isnan(metrics['top_decile_lift']) else f"  [{model_name}] Top decile lift: nan")
    print(f"  [{model_name}] Conditional MAE: {metrics['conditional_mae']:.4f}" if not np.isnan(metrics['conditional_mae']) else f"  [{model_name}] Conditional MAE: nan")
    return metrics


def train_linear_regression(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train linear regression with feature selection."""
    print(f"\n--- Linear Regression ---")
    tweedie_power = get_tweedie_power(target)
    
    grid_lr = iai.GridSearch(
        iai.OptimalFeatureSelectionRegressor(
            random_seed=seed,
            criterion='tweedie',
            tweedie_variance_power=tweedie_power
        ),
        sparsity=[10, 15, 20, 25]
    )
    
    grid_lr.fit_cv(X_train, y_train, validation_criterion='tweedie', n_folds=5)
    
    # Evaluate on test set
    tweedie_train = grid_lr.score(X_train, y_train, criterion='tweedie')
    tweedie_test = grid_lr.score(X_test, y_test, criterion='tweedie')
    
    print(f"  Tweedie variance power: {tweedie_power}")
    print(f"  Train Tweedie score (0-1, higher is better): {tweedie_train:.4f}")
    print(f"  Test Tweedie score (0-1, higher is better):  {tweedie_test:.4f}")
    
    # Save model
    lnr = grid_lr.get_learner()
    model_path = f"models/lr_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    metrics = evaluate_additional_metrics(grid_lr, X_test, y_test, "LR")
    
    return grid_lr, tweedie_test, metrics


def train_ort(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train Optimal Regression Tree."""
    print(f"\n--- Optimal Regression Tree ---")
    tweedie_power = get_tweedie_power(target)
    
    grid_ort = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=seed,
            criterion='tweedie',
            tweedie_variance_power=tweedie_power,
            show_progress=False,  # <--- Disables the "inner" progress bar
            normalize_y=False,
        ),
        max_depth=[2, 4, 6, 8],
        minbucket=[0.02, 0.05, 0.1],
    )
    
    grid_ort.fit_cv(X_train, y_train, validation_criterion='tweedie', n_folds=3, verbose=True)
    
    # Evaluate on test set
    tweedie_train = grid_ort.score(X_train, y_train, criterion='tweedie')
    tweedie_test = grid_ort.score(X_test, y_test, criterion='tweedie')
    
    print(f"  Tweedie variance power: {tweedie_power}")
    print(f"  Train Tweedie score (0-1, higher is better): {tweedie_train:.4f}")
    print(f"  Test Tweedie score (0-1, higher is better):  {tweedie_test:.4f}")
    
    # Save model
    lnr = grid_ort.get_learner()
    model_path = f"models/ort_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    metrics = evaluate_additional_metrics(grid_ort, X_test, y_test, "ORT")
    return grid_ort, tweedie_test, metrics


def train_random_forest(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train Random Forest."""
    print(f"\n--- Random Forest ---")
    tweedie_power = get_tweedie_power(target)
    
    grid_rf = iai.GridSearch(
        iai.RandomForestRegressor(
            random_seed=seed,
            criterion='tweedie',
            tweedie_variance_power=tweedie_power,
            normalize_y=False,
        ),
        max_depth=[2, 4, 6, 8],
        minbucket=[0.01, 0.02, 0.05],
        num_trees=[20, 25, 50, 100]
    )
    
    grid_rf.fit_cv(X_train, y_train, validation_criterion='tweedie', n_folds=5)
    
    # Evaluate on test set
    tweedie_train = grid_rf.score(X_train, y_train, criterion='tweedie')
    tweedie_test = grid_rf.score(X_test, y_test, criterion='tweedie')
    
    print(f"  Tweedie variance power: {tweedie_power}")
    print(f"  Train Tweedie score (0-1, higher is better): {tweedie_train:.4f}")
    print(f"  Test Tweedie score (0-1, higher is better):  {tweedie_test:.4f}")
    
    # Save model
    lnr = grid_rf.get_learner()
    model_path = f"models/rf_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    metrics = evaluate_additional_metrics(grid_rf, X_test, y_test, "RF")
    return grid_rf, tweedie_test, metrics


def train_xgboost(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train XGBoost."""
    print(f"\n--- XGBoost ---")
    tweedie_power = get_tweedie_power(target)
    
    grid_xgb = iai.GridSearch(
        iai.XGBoostRegressor(
            random_seed=seed,
            criterion='tweedie',
            tweedie_variance_power=tweedie_power,
            normalize_y=False,
        ),
        max_depth=[2, 4, 6, 8],
        minbucket=[0.01, 0.02, 0.05],
        num_estimators=[20, 25, 50, 100]
    )
    
    grid_xgb.fit_cv(X_train, y_train, validation_criterion='tweedie', n_folds=5)
    
    # Evaluate on test set
    tweedie_train = grid_xgb.score(X_train, y_train, criterion='tweedie')
    tweedie_test = grid_xgb.score(X_test, y_test, criterion='tweedie')
    
    print(f"  Tweedie variance power: {tweedie_power}")
    print(f"  Train Tweedie score (0-1, higher is better): {tweedie_train:.4f}")
    print(f"  Test Tweedie score (0-1, higher is better):  {tweedie_test:.4f}")
    
    # Save model
    lnr = grid_xgb.get_learner()
    model_path = f"models/xgb_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    metrics = evaluate_additional_metrics(grid_xgb, X_test, y_test, "XGB")
    return grid_xgb, tweedie_test, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train prediction models for ad optimization."
    )
    parser.add_argument(
        '--target',
        type=str,
        default='conversion',
        choices=['conversion', 'epc', 'clicks'],
        help='Target variable: conversion (Conv. value), epc (EPC), or clicks (default: conversion)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/clean',
        help='Data directory (default: data/clean)'
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='tfidf',
        choices=['tfidf', 'bert'],
        help='Embedding method used in data (default: tfidf)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['ort', 'rf', 'xgb'], # removed 'lr' as using tweedie loss
        choices=['ort', 'rf', 'xgb'],
        help='Models to train (default: all)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Prediction Modeling for Ad Optimization")
    print("=" * 70)
    print(f"Target: {args.target}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Embedding method: {args.embedding_method}")
    print("=" * 70)
    
    try:
        # Load data
        df_train, df_test = load_data(args.data_dir, args.embedding_method)
        
        X_train, features = get_features(df_train, args.target)
        X_test, _ = get_features(df_test, args.target)
        y_train = get_target(df_train, args.target)
        y_test = get_target(df_test, args.target)
        
        print(f"\nFeatures ({len(features)}): {', '.join(features[:5])}...")
        print(f"Target: {args.target}")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Ensure models directory exists
        Path('models').mkdir(exist_ok=True)
        
        # Train models
        results = {}
        
        if 'lr' in args.models:
            _, score, metrics = train_linear_regression(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['LR'] = {'score': score, 'metrics': metrics}
        
        if 'ort' in args.models:
            _, score, metrics = train_ort(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['ORT'] = {'score': score, 'metrics': metrics}
        
        if 'rf' in args.models:
            _, score, metrics = train_random_forest(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['RF'] = {'score': score, 'metrics': metrics}
        
        if 'xgb' in args.models:
            _, score, metrics = train_xgboost(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['XGB'] = {'score': score, 'metrics': metrics}
        
        # Print summary
        print("\n" + "=" * 70)
        print("Model Performance Summary (Test Tweedie Score)")
        print("=" * 70)
        for model_name, info in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
            score = info['score']
            metrics = info['metrics']
            print(f"  {model_name:6s}: {score:.4f} | bias={metrics['global_bias']:.4f} | top-decile lift={metrics['top_decile_lift']:.4f} | cMAE={metrics['conditional_mae']:.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1]['score'])[0]
        print(f"\nBest model: {best_model}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
