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
    Convert categorical columns to category dtype for IAI compatibility."""
    
    # Make a copy to avoid modifying original
    df_features = df.copy()
    
    # Exclude target variable and metadata columns
    if target == 'conversion':
        excluded_cols = {'Conv. value', 'Day', 'Keyword'}
    else:  # target == 'clicks'
        excluded_cols = {'Clicks', 'Conv. value', 'Day', 'Keyword'}
    
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
    elif target == 'clicks':
        return df['Clicks']
    else:
        raise ValueError(f"Unknown target: {target}")


def train_linear_regression(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train linear regression with feature selection."""
    print(f"\n--- Linear Regression ---")
    
    grid_lr = iai.GridSearch(
        iai.OptimalFeatureSelectionRegressor(random_seed=seed),
        sparsity=[10, 15, 20, 25]
    )
    
    grid_lr.fit_cv(X_train, y_train, validation_criterion='mse', n_folds=5)
    
    # Evaluate on test set
    r2_train = grid_lr.score(X_train, y_train, criterion='mse')
    r2_test = grid_lr.score(X_test, y_test, criterion='mse')
    
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²:  {r2_test:.4f}")
    
    # Save model
    lnr = grid_lr.get_learner()
    model_path = f"models/lr_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    return grid_lr, r2_test


def train_ort(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train Optimal Regression Tree."""
    print(f"\n--- Optimal Regression Tree ---")
    
    grid_ort = iai.GridSearch(
        iai.OptimalTreeRegressor(
            random_seed=seed,
            show_progress=False,  # <--- Disables the "inner" progress bar
        ),
        max_depth=[2, 4, 6, 8],
        minbucket=[0.01, 0.02, 0.05],
        
    )
    
    grid_ort.fit_cv(X_train, y_train, validation_criterion='mse', n_folds=5, verbose=True)
    
    # Evaluate on test set
    r2_train = grid_ort.score(X_train, y_train, criterion='mse')
    r2_test = grid_ort.score(X_test, y_test, criterion='mse')
    
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²:  {r2_test:.4f}")
    
    # Save model
    lnr = grid_ort.get_learner()
    model_path = f"models/ort_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    return grid_ort, r2_test


def train_random_forest(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train Random Forest."""
    print(f"\n--- Random Forest ---")
    
    grid_rf = iai.GridSearch(
        iai.RandomForestRegressor(random_seed=seed),
        max_depth=[2, 4, 6, 8],
        minbucket=[0.01, 0.02, 0.05],
        num_trees=[20, 25, 50, 100]
    )
    
    grid_rf.fit_cv(X_train, y_train, validation_criterion='mse', n_folds=5)
    
    # Evaluate on test set
    r2_train = grid_rf.score(X_train, y_train, criterion='mse')
    r2_test = grid_rf.score(X_test, y_test, criterion='mse')
    
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²:  {r2_test:.4f}")
    
    # Save model
    lnr = grid_rf.get_learner()
    model_path = f"models/rf_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    return grid_rf, r2_test


def train_xgboost(X_train, y_train, X_test, y_test, target='conversion', embedding_method='tfidf', seed=42):
    """Train XGBoost."""
    print(f"\n--- XGBoost ---")
    
    grid_xgb = iai.GridSearch(
        iai.XGBoostRegressor(random_seed=seed),
        max_depth=[2, 4, 6, 8],
        minbucket=[0.01, 0.02, 0.05],
        num_estimators=[20, 25, 50, 100]
    )
    
    grid_xgb.fit_cv(X_train, y_train, validation_criterion='mse', n_folds=5)
    
    # Evaluate on test set
    r2_train = grid_xgb.score(X_train, y_train, criterion='mse')
    r2_test = grid_xgb.score(X_test, y_test, criterion='mse')
    
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²:  {r2_test:.4f}")
    
    # Save model
    lnr = grid_xgb.get_learner()
    model_path = f"models/xgb_{embedding_method}_{target}.json"
    lnr.write_json(model_path)
    print(f"Model saved to {model_path}")
    
    return grid_xgb, r2_test


def main():
    parser = argparse.ArgumentParser(
        description="Train prediction models for ad optimization."
    )
    parser.add_argument(
        '--target',
        type=str,
        default='conversion',
        choices=['conversion', 'clicks'],
        help='Target variable: conversion or clicks (default: conversion)'
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
        default=['lr', 'ort', 'rf', 'xgb'],
        choices=['lr', 'ort', 'rf', 'xgb'],
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
            _, r2 = train_linear_regression(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['LR'] = r2
        
        if 'ort' in args.models:
            _, r2 = train_ort(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['ORT'] = r2
        
        if 'rf' in args.models:
            _, r2 = train_random_forest(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['RF'] = r2
        
        if 'xgb' in args.models:
            _, r2 = train_xgboost(X_train, y_train, X_test, y_test, args.target, args.embedding_method)
            results['XGB'] = r2
        
        # Print summary
        print("\n" + "=" * 70)
        print("Model Performance Summary (Test R²)")
        print("=" * 70)
        for model_name, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name:6s}: {r2:.4f}")
        
        best_model = max(results, key=results.get)
        print(f"\nBest model: {best_model}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
