"""
Bid Optimization with Linear Programming
==========================================
Optimizes keyword bids using Gurobi linear programming solver.
Maximizes profit by setting optimal bids for keywords across regions and match types.

Requires:
- Gurobi solver and license: https://www.gurobi.com/
- gurobipy: pip install gurobipy
- Pre-trained models (lr_{embedding}_conversion.json, lr_{embedding}_clicks.json)
- Embeddings file will be auto-generated if missing

Usage:
    python bid_optimization.py --embedding-method bert --budget 68096.51 --max-bid 100
    python bid_optimization.py --embedding-method tfidf --budget 68096.51
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_embeddings

# Check for required libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not found. Install with: pip install gurobipy")
    print("Note: Gurobi requires a valid license.")
    sys.exit(1)

try:
    from utils.iai_setup import iai
except ImportError:
    print("WARNING: Could not set up IAI. Will load weights from CSV files instead.")
    iai = None


def load_embeddings_data(keywords, embedding_method='bert', output_dir='data/embeddings'):
    """Load embeddings from file."""
    embeddings_file = Path(output_dir) / f'unique_keyword_embeddings_{embedding_method}.csv'
    
    # Use the utility function to load embeddings
    embedding_df = load_embeddings(
        embeddings_file,
        embedding_method=embedding_method,
        keywords=keywords
    )
    
    return embedding_df


def create_feature_matrix_with_embeddings(keywords, embedding_method='bert', n_components=50, models_dir='models'):
    """Generate embeddings for keywords using saved embedding pipeline.
    
    Loads the saved vectorizer/transformer, SVD, and normalizer from models_dir
    and applies them to the provided keywords to create embeddings.
    
    Args:
        keywords: list or Series of keyword strings
        embedding_method: 'bert' or 'tfidf'
        n_components: number of SVD components used (should match training)
        models_dir: directory containing saved pipeline pickle files
    
    Returns:
        DataFrame with columns ['{method}_0' ... '{method}_N', 'Keyword']
    """
    import pickle
    
    # Ensure keywords is a list
    if not isinstance(keywords, list):
        keywords = list(keywords)
    
    print(f"Generating {embedding_method.upper()} embeddings for {len(keywords)} keywords...")
    
    # Load the saved pipeline
    pipeline_file = Path(models_dir) / f'{embedding_method}_pipeline_{n_components}d.pkl'
    if not pipeline_file.exists():
        raise FileNotFoundError(
            f"Embedding pipeline not found: {pipeline_file}. "
            f"Please ensure tidy_get_data.py has been run with add_embeddings(save_models=True)."
        )
    
    with open(pipeline_file, 'rb') as f:
        pipeline = pickle.load(f)
    
    vectorizer = pipeline['vectorizer'] if 'vectorizer' in pipeline else pipeline.get('transformer')
    svd = pipeline.get('svd')
    normalizer = pipeline.get('normalizer')
    
    if vectorizer is None or svd is None or normalizer is None:
        raise ValueError(
            f"Invalid pipeline format. Expected keys: 'vectorizer'/'transformer', 'svd', 'normalizer'. "
            f"Got keys: {list(pipeline.keys())}"
        )
    
    # Apply the pipeline
    if embedding_method == 'tfidf':
        # TF-IDF pipeline
        embeddings = vectorizer.transform(keywords)  # sparse matrix
        embeddings = svd.transform(embeddings)  # dense array
    else:  # bert
        # BERT pipeline
        embeddings = vectorizer.encode(keywords)  # array of shape (n, 384)
        embeddings = svd.transform(embeddings)  # reduce to n_components
    
    # Apply L2 normalization
    embeddings = normalizer.transform(embeddings)
    
    # Create DataFrame with embedding columns
    embedding_cols = [f'{embedding_method}_{i}' for i in range(n_components)]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    embedding_df['Keyword'] = keywords
    
    print(f"  Generated embeddings with shape {embeddings.shape}")
    
    return embedding_df


def load_weights_from_csv(embedding_method='bert', models_dir='models'):
    """Load weights and constants from CSV files (no IAI required).
    
    Falls back to IAI if CSV files don't exist.
    
    Args:
        embedding_method: 'bert' or 'tfidf'
        models_dir: directory containing weight CSV files
    
    Returns:
        dict with keys: 'conv_const', 'conv_weights', 'clicks_const', 'clicks_weights'
    """
    print(f"Loading weights for embedding method '{embedding_method}'...")
    
    models_dir = Path(models_dir)
    
    # Check if CSV files exist
    conv_numeric_file = models_dir / f'weights_{embedding_method}_conversion_numeric.csv'
    conv_const_file = models_dir / f'weights_{embedding_method}_conversion_constant.csv'
    clicks_numeric_file = models_dir / f'weights_{embedding_method}_clicks_numeric.csv'
    clicks_const_file = models_dir / f'weights_{embedding_method}_clicks_constant.csv'
    
    # If CSV files exist, load from them
    if conv_numeric_file.exists() and conv_const_file.exists() and \
       clicks_numeric_file.exists() and clicks_const_file.exists():
        print(f"  Loading from CSV files...")
        
        conv_cat_file = models_dir / f'weights_{embedding_method}_conversion_categorical.csv'
        clicks_cat_file = models_dir / f'weights_{embedding_method}_clicks_categorical.csv'
        
        # Load conversion numeric weights
        conv_numeric_df = pd.read_csv(conv_numeric_file)
        conv_weights = dict(zip(conv_numeric_df['feature'], conv_numeric_df['weight']))
        
        # Load conversion categorical weights if available
        if conv_cat_file.exists():
            conv_cat_df = pd.read_csv(conv_cat_file)
            for _, row in conv_cat_df.iterrows():
                feature = row['feature']
                level = row['level']
                weight = row['weight']
                if feature not in conv_weights:
                    conv_weights[feature] = {}
                if not isinstance(conv_weights[feature], dict):
                    # If already have numeric, convert to dict
                    numeric_val = conv_weights[feature]
                    conv_weights[feature] = {feature: numeric_val}
                conv_weights[feature][level] = weight
        
        # Load conversion constant
        conv_const = pd.read_csv(conv_const_file)['constant'].iloc[0]
        
        # Load clicks numeric weights
        clicks_numeric_df = pd.read_csv(clicks_numeric_file)
        clicks_weights = dict(zip(clicks_numeric_df['feature'], clicks_numeric_df['weight']))
        
        # Load clicks categorical weights if available
        if clicks_cat_file.exists():
            clicks_cat_df = pd.read_csv(clicks_cat_file)
            for _, row in clicks_cat_df.iterrows():
                feature = row['feature']
                level = row['level']
                weight = row['weight']
                if feature not in clicks_weights:
                    clicks_weights[feature] = {}
                if not isinstance(clicks_weights[feature], dict):
                    # If already have numeric, convert to dict
                    numeric_val = clicks_weights[feature]
                    clicks_weights[feature] = {feature: numeric_val}
                clicks_weights[feature][level] = weight
        
        # Load clicks constant
        clicks_const = pd.read_csv(clicks_const_file)['constant'].iloc[0]
        
        print(f"  Loaded from CSV files ✓")
        return {
            'conv_const': conv_const,
            'conv_weights': conv_weights,
            'clicks_const': clicks_const,
            'clicks_weights': clicks_weights
        }
    
    # If CSV files don't exist, fall back to IAI
    else:
        print(f"  CSV files not found, falling back to IAI...")
        if iai is None:
            raise FileNotFoundError(
                f"CSV weight files not found and IAI not available. "
                f"Please run 'python scratch.py' first to extract weights."
            )
        
        # Load models using IAI
        conversion_model = models_dir / f'lr_{embedding_method}_conversion.json'
        clicks_model = models_dir / f'lr_{embedding_method}_clicks.json'
        
        if not conversion_model.exists():
            raise FileNotFoundError(f"Conversion model not found: {conversion_model}")
        if not clicks_model.exists():
            raise FileNotFoundError(f"Clicks model not found: {clicks_model}")
        
        lnr_conv = iai.read_json(str(conversion_model))
        lnr_clicks = iai.read_json(str(clicks_model))
        
        # Extract weights and constants using IAI
        weights_conv_tuple = lnr_conv.get_prediction_weights()
        weights_clicks_tuple = lnr_clicks.get_prediction_weights()
        
        if isinstance(weights_conv_tuple, tuple):
            conv_weights = weights_conv_tuple[0]
        else:
            conv_weights = weights_conv_tuple
        
        if isinstance(weights_clicks_tuple, tuple):
            clicks_weights = weights_clicks_tuple[0]
        else:
            clicks_weights = weights_clicks_tuple
        
        conv_const = lnr_conv.get_prediction_constant()
        clicks_const = lnr_clicks.get_prediction_constant()
        
        print(f"  Loaded from IAI models ✓")
        return {
            'conv_const': conv_const,
            'conv_weights': conv_weights,
            'clicks_const': clicks_const,
            'clicks_weights': clicks_weights
        }


def load_models(embedding_method='bert', alg_conv='lr', alg_clicks='lr', models_dir='models'):
    """Load pre-trained prediction models based on embedding method and algorithm type.
    
    Args:
        embedding_method: 'bert' or 'tfidf'
        alg_conv: algorithm type for conversion model - 'lr' (linear regression), 'ort' (optimal regression tree), 
                  'rf' (random forest), 'xgb' (xgboost)
        alg_clicks: algorithm type for clicks model - same options as alg_conv
        models_dir: directory containing model files
    
    Returns:
        tuple of (conversion_model, clicks_model)
    """
    print(f"Loading models for embedding method '{embedding_method}'...")
    print(f"  Conversion model: {alg_conv}")
    print(f"  Clicks model: {alg_clicks}")
    
    conversion_model = Path(models_dir) / f'{alg_conv}_{embedding_method}_conversion.json'
    clicks_model = Path(models_dir) / f'{alg_clicks}_{embedding_method}_clicks.json'
    
    if not conversion_model.exists():
        raise FileNotFoundError(f"Conversion model not found: {conversion_model}")
    if not clicks_model.exists():
        raise FileNotFoundError(f"Clicks model not found: {clicks_model}")
    
    lnr_conv = iai.read_json(str(conversion_model))
    lnr_clicks = iai.read_json(str(clicks_model))
    
    print(f"  Loaded conversion model from {conversion_model}")
    print(f"  Loaded clicks model from {clicks_model}")
    
    return lnr_conv, lnr_clicks


def extract_weights(lnr_conv, lnr_clicks, embedding_method='bert', n_embeddings=50):
    """Extract ALL weights from trained models (not just embeddings)."""
    print(f"Extracting model weights...")
    
    # Get weights using IAI's method
    weights_conv_tuple = lnr_conv.get_prediction_weights()
    weights_clicks_tuple = lnr_clicks.get_prediction_weights()
    
    # Handle tuple format (continuous, categorical)
    if isinstance(weights_conv_tuple, tuple):
        weights_conv = weights_conv_tuple[0]
    else:
        weights_conv = weights_conv_tuple
    
    if isinstance(weights_clicks_tuple, tuple):
        weights_clicks = weights_clicks_tuple[0]
    else:
        weights_clicks = weights_clicks_tuple
    
    conv_const = lnr_conv.get_prediction_constant()
    clicks_const = lnr_clicks.get_prediction_constant()
    
    print(f"\n  Conversion model weights:")
    for key, val in sorted(weights_conv.items(), key=lambda x: str(x[0])):
        print(f"    {key}: {val:.6f}")
    
    print(f"\n  Clicks model weights:")
    for key, val in sorted(weights_clicks.items(), key=lambda x: str(x[0])):
        print(f"    {key}: {val:.6f}")
    
    return {
        'conv_const': conv_const,
        'conv_weights': weights_conv,
        'clicks_const': clicks_const,
        'clicks_weights': weights_clicks
    }


def create_feature_matrix(keyword_df, embedding_method='bert', target_day=None, regions=None, match_types=None, training_data_path='data/clean/ad_opt_data_bert.csv', weights_dict=None, alg_conv='lr', alg_clicks='lr'):
    """Create feature matrix/matrices for all keyword-region-match combinations for a specific day.
    
    Merges on exact keyword-match type-region combinations from historical data.
    If target_day is None, uses the latest date in data and adjusts date features for today.
    
    For mixed models (ORT + LR), returns two versions:
    - X_ort: categorical features as strings (for ORT model access)
    - X_lr: categorical features one-hot encoded (for LR model access)
    
    For single model type, returns the appropriate version as X.
    
    Includes:
    - Embeddings from keyword_df
    - Region
    - Match type
    - All historical features (Avg. CPC, Competition, etc.)
    - Date-adjusted features (day_of_week, month, days_to_next_course_start, is_public_holiday, is_weekend)
    
    Args:
        keyword_df: DataFrame with keywords and embeddings
        embedding_method: 'bert' or 'tfidf'
        target_day: str, date in format 'YYYY-MM-DD' (e.g., '2024-11-04'). If None, uses latest date.
        regions: list of regions (default: ["USA", "Region_A", "Region_B", "Region_C"])
        match_types: list of match types (default: ["broad match", "exact match", "phrase match"])
        training_data_path: path to training data file
        weights_dict: dict with 'conv_weights' and 'clicks_weights' to filter features. If None, keeps all.
        alg_conv: algorithm type for conversion model ('lr', 'ort', etc.).
        alg_clicks: algorithm type for clicks model ('lr', 'ort', etc.).
    
    Returns:
        For ORT-only: (X_ort, keyword_idx_list, region_list, match_list)
        For LR-only: (X_lr, keyword_idx_list, region_list, match_list)
        For mixed: (X_ort, X_lr, keyword_idx_list, region_list, match_list)
    """
    if regions is None:
        regions = ["USA", "A", "B", "C"]
    if match_types is None:
        match_types = ["Broad match", "Exact match", "Phrase match"]
    
    # Extract model weights if provided
    if weights_dict is None:
        weights_dict = {}
    conv_weights = weights_dict.get('conv_weights', {})
    clicks_weights = weights_dict.get('clicks_weights', {})
    
    num_keywords = len(keyword_df)
    n_combos = num_keywords * len(regions) * len(match_types)
    
    print(f"Creating feature matrix...")
    print(f"  Target day: {target_day}")
    print(f"  Keywords: {num_keywords}, Regions: {len(regions)}, Matches: {len(match_types)}")
    print(f"  Total combinations: {n_combos}")
    
    # Load training data
    print(f"  Loading training data from {training_data_path}...")
    training_data = pd.read_csv(training_data_path)
    training_data['Day'] = pd.to_datetime(training_data['Day'])
    
    # Determine which date to use
    if target_day is not None:
        if isinstance(target_day, str):
            target_day = pd.to_datetime(target_day)
        filter_day = target_day
        print(f"  Using target day: {target_day.date()}")
    else:
        # Use latest date in data
        filter_day = training_data['Day'].max()
        print(f"  Using latest date in data: {filter_day.date()}")
    
    # Build all combinations we need
    combinations = []
    for kw in keyword_df['Keyword']:
        for region in regions:
            for match in match_types:
                combinations.append({
                    'Keyword': kw,
                    'Region': region,
                    'Match type': match,
                })
    
    combo_df = pd.DataFrame(combinations)
    combo_df['Day'] = filter_day
    print(f"  Created {len(combo_df)} keyword-region-match combinations")
    
    # Use merge_asof to match on (Keyword, Match type, Region) with nearest date
    print(f"  Merging on exact (Keyword, Match type, Region) with nearest date...")
    
    # Merge using asof on date, exact match on categorical columns
    result = pd.merge_asof(
        combo_df.sort_values('Day'),
        training_data.sort_values('Day'),
        on='Day',
        by=['Keyword', 'Match type', 'Region'],
        direction='nearest'
    )
    
    # Drop combinations not found in training data (will have NaN in feature columns)
    initial_rows = len(combo_df)
    result = result.dropna(subset=['Avg. CPC'])
    final_rows = len(result)
    dropped_rows = initial_rows - final_rows
    
    print(f"  Matched {final_rows} combinations with features (nearest date), {dropped_rows} combinations not found in data and dropped")
    
    # If target_day != latest date in data, we need to adjust date features
    if target_day is None:
        from utils.date_features import calculate_date_features
        
        target_date = pd.to_datetime(datetime.today())
        date_features = calculate_date_features(target_date, regions=regions)
        
        print(f"  Adjusting date features to today ({target_date.date()})...")
        result['day_of_week'] = date_features['day_of_week']
        result['is_weekend'] = date_features['is_weekend']
        result['month'] = date_features['month']
        result['is_public_holiday'] = date_features['is_public_holiday']
        result['days_to_next_course_start'] = date_features['days_to_next_course_start']
        
        print(f"  Adjusted temporal features for {target_date.date()}")
    else:
        print(f"  Using date features from {filter_day.date()}")
    
    # Extract the order information for rows that actually made it into the feature matrix
    # (BEFORE dropping Day/Keyword columns so we still have access to them)
    keyword_idx_list = []
    region_list = []
    match_list = []
    
    for _, row in result.iterrows():
        kw_idx = keyword_df[keyword_df['Keyword'] == row['Keyword']].index[0]
        keyword_idx_list.append(kw_idx)
        region_list.append(row['Region'])
        match_list.append(row['Match type'])
    
    # Drop day column
    result = result.drop(columns=['Day', 'Keyword'])

    # Reset index
    result.reset_index(drop=True, inplace=True)
    
    # Convert column names: replace dots with underscores to match model feature names
    result.columns = result.columns.str.replace('.', '_', regex=False)
    
    # Determine if we need both versions (mixed models) or just one
    use_ort_conv = (alg_conv == 'ort')
    use_ort_clicks = (alg_clicks == 'ort')
    use_both = use_ort_conv and use_ort_clicks
    use_both_lr = (not use_ort_conv) and (not use_ort_clicks)
    is_mixed = (use_ort_conv and not use_ort_clicks) or (not use_ort_conv and use_ort_clicks)
    
    if use_both:
        # Both ORT: keep categorical as strings only
        print(f"  Keeping categorical features as strings (both models are ORT)")
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            result[col] = result[col].astype(float)
        X_ort = result
        X_lr = None
        
    elif use_both_lr:
        # Both LR: one-hot encode only
        print(f"  One-hot encoding categorical columns (both models are LR)")
        categorical_cols = result.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            result = pd.get_dummies(result, columns=categorical_cols, drop_first=False)
        result = result.astype(float)
        X_ort = None
        X_lr = result
        
    else:
        # Mixed models: create both versions
        print(f"  Creating both versions for mixed models ({alg_conv.upper()} + {alg_clicks.upper()})")
        
        # Version 1: ORT (categorical as strings)
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        X_ort = result.copy()
        for col in numeric_cols:
            X_ort[col] = X_ort[col].astype(float)
        
        # Version 2: LR (one-hot encoded)
        X_lr = result.copy()
        categorical_cols = X_lr.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            X_lr = pd.get_dummies(X_lr, columns=categorical_cols, drop_first=False)
        X_lr = X_lr.astype(float)
    
    print(f"  Final feature matrix shape: {(X_ort if X_ort is not None else X_lr).shape}")
    print(f"  Columns: {(X_ort if X_ort is not None else X_lr).columns.tolist()[:15]}...")
    
    # Return appropriate format
    if is_mixed:
        return X_ort, X_lr, keyword_idx_list, region_list, match_list
    elif use_both:
        return X_ort, keyword_idx_list, region_list, match_list
    else:  # use_both_lr
        return X_lr, keyword_idx_list, region_list, match_list


def optimize_bids(X, weights_dict, budget=68096.51, max_bid=100, n_active=14229):
    """Solve bid optimization problem using Gurobi.
    
    Uses all numeric features from the trained model.
    Feature matrix X should already be numeric (one-hot encoded if needed).
    """
    print(f"\nSolving bid optimization problem...")
    print(f"  Budget: ${budget:,.2f}")
    print(f"  Max bid: ${max_bid:.2f}")
    print(f"  Max active keywords: {n_active}")
    
    n = len(X)
    
    # Extract constants and weights
    conv_const = weights_dict['conv_const']
    conv_weights = weights_dict['conv_weights']
    clicks_const = weights_dict['clicks_const']
    clicks_weights = weights_dict['clicks_weights']
    
    # Feature matrix as numpy array
    X_arr = X.values
    
    # Build prediction vectors by applying all model weights
    conv_predictions = np.full(n, conv_const, dtype=float)
    clicks_predictions = np.full(n, clicks_const, dtype=float)
    
    print(f"\n  Applying conversion model weights...")
    # Apply feature weights for conversion
    for feature_name, weight in conv_weights.items():
        feature_str = str(feature_name).strip()
        
        # Try to find the column in X
        if feature_str in X.columns:
            col_idx = X.columns.get_loc(feature_str)
            col_data = X_arr[:, col_idx].astype(float)
            conv_predictions += weight * col_data
            print(f"    Applied weight for '{feature_str}': {weight:.4f}")
        elif feature_str == 'Avg. CPC' or feature_str == 'Avg_ CPC':
            # CPC is handled separately in objective
            pass
        else:
            # Try to find similar name (handle naming variations)
            matching_cols = [col for col in X.columns if feature_str.lower() in col.lower()]
            if matching_cols:
                print(f"    Note: Feature '{feature_str}' not found exactly, but found similar: {matching_cols}")
    
    print(f"  Applying clicks model weights...")
    for feature_name, weight in clicks_weights.items():
        feature_str = str(feature_name).strip()
        
        if feature_str in X.columns:
            col_idx = X.columns.get_loc(feature_str)
            col_data = X_arr[:, col_idx].astype(float)
            clicks_predictions += weight * col_data
            print(f"    Applied weight for '{feature_str}': {weight:.4f}")
        elif feature_str == 'Avg. CPC' or feature_str == 'Avg_ CPC':
            pass
        else:
            matching_cols = [col for col in X.columns if feature_str.lower() in col.lower()]
            if matching_cols:
                print(f"    Note: Feature '{feature_str}' not found exactly, but found similar: {matching_cols}")
    
    # Get CPC weights if they exist
    conv_cpc_weight = conv_weights.get('Avg. CPC', conv_weights.get('Avg_ CPC', 0.0))
    clicks_cpc_weight = clicks_weights.get('Avg. CPC', clicks_weights.get('Avg_ CPC', 0.0))
    
    print(f"\n  CPC weights: conv={conv_cpc_weight:.4f}, clicks={clicks_cpc_weight:.4f}")
    
    # Create model
    model = gp.Model('bid_optimization')
    model.setParam('OutputFlag', 1)  # Show solver output
    
    # Decision variables
    b = model.addMVar(shape=n, lb=0, name='bid')  # bid amounts
    z = model.addMVar(shape=n, vtype=GRB.BINARY, name='active')  # binary active indicator
    
    # Objective: Maximize profit
    # profit = (conv_predictions + conv_cpc_weight * bid) - (clicks_predictions + clicks_cpc_weight * bid) * bid
    # profit = conv_predictions + conv_cpc_weight * bid - clicks_predictions * bid - clicks_cpc_weight * bid^2
    profit = gp.quicksum(
        conv_predictions[i] + conv_cpc_weight * b[i] - (clicks_predictions[i] + clicks_cpc_weight * b[i]) * b[i]
        for i in range(n)
    )
    
    model.setObjective(profit, GRB.MAXIMIZE)
    
    # Constraints
    # Budget constraint
    model.addConstr(gp.quicksum(b) <= budget, name='budget')
    
    # Bid linking constraints (b[i] is only positive if z[i] = 1)
    model.addConstrs((b[i] <= max_bid * z[i] for i in range(n)), name='bid_ub')
    
    # Activity constraint
    model.addConstr(gp.quicksum(z) <= n_active, name='max_active')
    
    # Optimize
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print(f"  Status: OPTIMAL")
        print(f"  Optimal profit: ${model.objVal:,.2f}")
        return model, b, z
    else:
        print(f"  Status: {model.status}")
        return model, b, z

def embed_lr(model, weights, const, X, b, target):
    """
    Embeds linear regression constraints directly into Gurobi model.
    Creates prediction variables and adds constraints linking them to features and bids.
    
    Args:
        model: Gurobi model object
        weights: Dictionary of feature weights (can include dict values for categorical features)
        const: Constant term from the model
        X: Feature matrix (pandas DataFrame)
        b: Bid decision variables (Gurobi MVar)
        target: Target name ('conversion' or 'clicks') for variable naming
    
    Returns:
        tuple of (model, pred_vars) where pred_vars is list of prediction variables
    """
    K = len(X)
    pred_vars = []
    
    print(f"  Embedding {target} model constraints...")
    
    # Create prediction variables for this target
    for i in range(K):
        # Create prediction variable (can be negative)
        pred_var = model.addVar(lb=-GRB.INFINITY, name=f'{target}_pred_{i}')
        pred_vars.append(pred_var)
        
        # Build constraint expression: pred = const + weights·features + cpc_weight·bid
        expr = const
        
        # Add feature weights
        for feature, weight in weights.items():
            # Handle CPC weight specially - use decision variable b instead of feature matrix
            if feature in ['Avg. CPC', 'Avg_ CPC']:
                expr += weight * b[i]
                continue
            
            # Check if weight is a dict (categorical feature with one-hot encoding)
            if isinstance(weight, dict):
                # This is a categorical feature with multiple levels
                for level_name, level_weight in weight.items():
                    # Construct one-hot encoded column name
                    ohe_col_name = f"{feature}_{level_name}"
                    
                    if ohe_col_name not in X.columns:
                        raise ValueError(f"Error: One-hot encoded column '{ohe_col_name}' is missing from X dataframe for {target} model.")
                    
                    expr += level_weight * X.iloc[i][ohe_col_name]
            else:
                # This is a numeric feature
                if feature not in X.columns:
                    raise ValueError(f"Error: Feature '{feature}' is missing from X dataframe for {target} model.")
                
                expr += weight * X.iloc[i][feature]
        
        # Add constraint: pred_var == expr
        model.addConstr(pred_var == expr, name=f'{target}_constr_{i}')
    
    return model, pred_vars

def embed_ort(model, ort_model, X, b, target, max_bid=50.0, M=None, save_dir=None):
    """
    Embeds Optimal Regression Tree (ORT) constraints directly into Gurobi model using path-based formulation.
    Path-based formulation enforces all split conditions along the path to each leaf (OptiCL-style).
    For each leaf, creates multiple constraints (one per split node on path).
    
    Args:
        model: Gurobi model object
        ort_model: IAI ORT model object (from iai.read_json)
        X: Feature matrix (pandas DataFrame)
        b: Bid decision variables (Gurobi MVar)
        target: Target name ('conversion' or 'clicks') for variable naming
        max_bid: Maximum individual bid (used for M calculation) (default: 50.0)
        M: Big-M parameter for indicator constraints. If None, automatically calculated from data.
        save_dir: (Optional) Directory to save the tree visualization HTML. If None, no save.

    Returns:
        tuple of (model, pred_vars) where pred_vars is list of prediction variables (one per row)
    """
    K = len(X)
    pred_vars = []
    
    print(f"  Embedding {target} ORT model constraints (path-based formulation)...")
    
    # Save tree visualization if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        tree_file = save_dir / f'{target}_tree.html'
        ort_model.write_html(str(tree_file))
        print(f"    Saved {target} tree visualization to {tree_file}")
    
    # Extract tree structure from IAI model
    num_nodes = ort_model.get_num_nodes()
    leaf_nodes = []
    
    # Find all leaf nodes
    for node_idx in range(1, num_nodes + 1):
        if ort_model.is_leaf(node_index=node_idx):
            leaf_nodes.append(node_idx)
    
    print(f"    Found {len(leaf_nodes)} leaf nodes")
    
    # Get leaf predictions (constants)
    leaf_predictions = {}
    for leaf_idx in leaf_nodes:
        leaf_predictions[leaf_idx] = ort_model.get_regression_constant(node_index=leaf_idx)
    
    # Build path map: for each leaf, get all nodes on path from root
    def get_path_to_leaf(leaf_idx, ort_model):
        """Get list of (node_idx, direction) tuples on path from root to leaf.
        direction: 'lower' if node is on lower branch, 'upper' if on upper branch."""
        path = []
        current = leaf_idx
        
        while current != 1:  # 1 is root
            parent = ort_model.get_parent(node_index=current)
            lower_child = ort_model.get_lower_child(node_index=parent)
            
            if lower_child == current:
                path.append((parent, 'lower'))
            else:
                path.append((parent, 'upper'))
            
            current = parent
        
        return list(reversed(path))  # Return path from root to leaf
    
    # Precompute paths for all leaves
    leaf_paths = {leaf_id: get_path_to_leaf(leaf_id, ort_model) for leaf_id in leaf_nodes}
    
    # --- Calculate Big-M automatically if not provided ---
    if M is None:
        print(f"    Calculating principled Big-M (data-driven + bid contribution)...")
        max_feature_diff = 0.0
        max_bid_coeff = 0.0
        
        # Iterate through all split nodes to compute max deviations
        for node_idx in range(1, num_nodes + 1):
            if ort_model.is_leaf(node_index=node_idx):
                continue  # Skip leaf nodes
            
            # First, try to get split feature
            try:
                split_feature = ort_model.get_split_feature(node_index=node_idx)
            except Exception as e:
                print(f"      Warning: Skipping node {node_idx} (could not get split feature)")
                continue
            
            # Check if this is a hyperplane split
            try:
                is_hyperplane = ort_model.is_hyperplane_split(node_index=node_idx)
            except Exception as e:
                print(f"      Warning: Skipping node {node_idx} (could not determine split type)")
                continue
            
            # Try to get threshold first (for numeric/axis-aligned splits)
            try:
                split_threshold = ort_model.get_split_threshold(node_index=node_idx)
            except Exception as e:
                # No threshold - might be categorical, try to get categories
                try:
                    split_cats = ort_model.get_split_categories(node_index=node_idx)
                    # It's categorical, skip M calculation
                    continue
                except Exception as e2:
                    print(f"      Warning: Skipping node {node_idx} (could not get threshold or categories)")
                    continue
            
            if is_hyperplane:
                # Hyperplane split: weighted combination of features
                try:
                    weights_dict = ort_model.get_split_weights(node_index=node_idx)[0]
                except Exception as e:
                    print(f"      Warning: Could not extract weights for node {node_idx}")
                    continue
                
                # Compute max feature-based deviation (excluding bid terms)
                for i in range(K):
                    try:
                        split_expr = 0.0
                        bid_weight = 0.0
                        
                        for feat_name, weight in weights_dict.items():
                            if feat_name in ['Avg. CPC', 'Avg_ CPC']:
                                bid_weight = abs(weight)
                            else:
                                if feat_name in X.columns:
                                    split_expr += weight * X.iloc[i][feat_name]
                        
                        # Update max feature difference and max bid coefficient
                        feature_diff = abs(split_expr - split_threshold)
                        max_feature_diff = max(max_feature_diff, feature_diff)
                        max_bid_coeff = max(max_bid_coeff, bid_weight)
                    except Exception as e:
                        print(f"      Warning: Error computing split expr for node {node_idx}, row {i}")
                        continue
            else:
                # Axis-aligned split: single feature vs threshold
                for i in range(K):
                    try:
                        if split_feature in ['Avg. CPC', 'Avg_ CPC']:
                            # Pure bid term: coefficient is implicitly 1.0 when feature is b[i]
                            max_bid_coeff = max(max_bid_coeff, 1.0)
                        else:
                            if split_feature in X.columns:
                                split_expr = X.iloc[i][split_feature]
                                diff = abs(split_expr - split_threshold)
                                max_feature_diff = max(max_feature_diff, diff)
                    except Exception as e:
                        print(f"      Warning: Error processing feature {split_feature} for node {node_idx}, row {i}")
                        continue
        
        # Principled M: account for both feature deviations and maximum bid contribution
        # M = max_feature_diff + max_bid_coeff * max_bid
        M = max_feature_diff + max_bid_coeff * max_bid
        print(f"    Calculated Big-M = {M:.4f}")
        print(f"      (max feature diff: {max_feature_diff:.4f}, max bid coeff: {max_bid_coeff:.4f}, max_bid: {max_bid:.2f})")
    else:
        print(f"    Using provided Big-M = {M:.4f}")
    
    # Create prediction variables for each row
    # Also create leaf indicator variables: l[i, leaf_id] = 1 if row i reaches leaf_id
    for i in range(K):
        pred_var = model.addVar(lb=-GRB.INFINITY, name=f'{target}_pred_{i}')
        pred_vars.append(pred_var)
        
        # Create binary indicators for which leaf this row reaches
        leaf_indicators = {}
        for leaf_id in leaf_nodes:
            leaf_indicators[leaf_id] = model.addVar(vtype=GRB.BINARY, name=f'{target}_leaf_{i}_{leaf_id}')
        
        # Constraint: exactly one leaf must be active for each row
        model.addConstr(
            gp.quicksum(leaf_indicators[leaf_id] for leaf_id in leaf_nodes) == 1,
            name=f'{target}_one_leaf_{i}'
        )
        
        # Path-based constraints: for each leaf, add constraints for all splits on its path
        for leaf_id in leaf_nodes:
            path_to_leaf = leaf_paths[leaf_id]
            
            # For each split node on the path to this leaf
            for node_idx, direction in path_to_leaf:
                # Get split feature first
                try:
                    split_feature = ort_model.get_split_feature(node_index=node_idx)
                except Exception as e:
                    print(f"    Warning: Skipping path constraint for node {node_idx} (could not get split feature)")
                    continue
                
                # Check if this is a hyperplane split
                try:
                    is_hyperplane = ort_model.is_hyperplane_split(node_index=node_idx)
                except Exception as e:
                    print(f"    Warning: Skipping path constraint for node {node_idx} (could not determine split type)")
                    continue
                
                # Try to get threshold first (for numeric/axis-aligned and hyperplane splits)
                try:
                    split_threshold = ort_model.get_split_threshold(node_index=node_idx)
                except Exception as e:
                    # No threshold - try categorical
                    try:
                        split_cats = ort_model.get_split_categories(node_index=node_idx)
                    except Exception as e2:
                        print(f"    Warning: Skipping path constraint for node {node_idx} (could not get threshold or categories)")
                        continue
                    
                    # Categorical split: check if feature value is in the right category set
                    # split_cats is like {'USA': True, 'B': False, 'A': True, 'C': False}
                    # direction='lower' means feature should be in True categories, 'upper' means False
                    if direction == 'lower':
                        true_cats = {cat for cat, goes_lower in split_cats.items() if goes_lower}
                    else:
                        true_cats = {cat for cat, goes_lower in split_cats.items() if not goes_lower}
                    
                    if split_feature not in X.columns:
                        raise ValueError(f"Feature '{split_feature}' not found in X for {target} ORT model")
                    
                    # Create binary indicator: 1 if feature value is in true_cats, 0 otherwise
                    cat_indicator = model.addVar(vtype=GRB.BINARY, name=f'{target}_cat_{i}_{leaf_id}_{node_idx}')
                    
                    # Constraint: cat_indicator = 1 iff X[i, split_feature] in true_cats
                    feature_val = str(X.iloc[i][split_feature])
                    is_in_true_cats = 1 if feature_val in true_cats else 0
                    model.addConstr(cat_indicator == is_in_true_cats, name=f'{target}_cat_constr_{i}_{leaf_id}_{node_idx}')
                    
                    # Path constraint: if leaf is active, cat_indicator must be 1
                    model.addConstr(cat_indicator >= 1 - M * (1 - leaf_indicators[leaf_id]), 
                                   name=f'{target}_path_cat_{i}_{leaf_id}_{node_idx}')
                    continue
                
                # Build list of expression terms (will be combined with gp.quicksum for proper Gurobi expression handling)
                expr_terms = []
                
                if is_hyperplane:
                    # Hyperplane split: weighted combination of features
                    try:
                        weights_dict = ort_model.get_split_weights(node_index=node_idx)[0]
                    except Exception as e:
                        print(f"    Warning: Could not extract weights for node {node_idx}")
                        continue
                    
                    # Build expression: sum(weights[feat] * X[i, feat])
                    for feat_name, weight in weights_dict.items():
                        # Handle CPC weight specially - use decision variable b instead of feature matrix
                        if feat_name in ['Avg. CPC', 'Avg_ CPC']:
                            expr_terms.append(weight * b[i])
                        else:
                            if feat_name not in X.columns:
                                raise ValueError(f"Feature '{feat_name}' not found in X for {target} ORT model")
                            expr_terms.append(weight * X.iloc[i][feat_name])
                else:
                    # Axis-aligned split: single feature vs threshold
                    if split_feature in ['Avg. CPC', 'Avg_ CPC']:
                        expr_terms.append(b[i])
                    else:
                        if split_feature not in X.columns:
                            raise ValueError(f"Feature '{split_feature}' not found in X for {target} ORT model")
                        expr_terms.append(X.iloc[i][split_feature])
                
                # Use gp.quicksum to build proper Gurobi expression (handles both constants and variables)
                split_expr = gp.quicksum(expr_terms) if expr_terms else 0.0
                
                # Add constraint based on direction of split on path
                if direction == 'lower':
                    # If this leaf is activated, split_expr must be <= threshold
                    model.addConstr(
                        split_expr <= split_threshold + M * (1 - leaf_indicators[leaf_id]),
                        name=f'{target}_path_lower_{i}_{leaf_id}_{node_idx}'
                    )
                else:  # direction == 'upper'
                    # If this leaf is activated, split_expr must be > threshold
                    model.addConstr(
                        split_expr >= split_threshold + 1e-6 - M * (1 - leaf_indicators[leaf_id]),
                        name=f'{target}_path_upper_{i}_{leaf_id}_{node_idx}'
                    )
        
        # Link prediction variable to leaf predictions
        # pred_var = sum(leaf_predictions[leaf] * leaf_indicator[leaf])
        model.addConstr(
            pred_var == gp.quicksum(leaf_predictions[leaf_id] * leaf_indicators[leaf_id] for leaf_id in leaf_nodes),
            name=f'{target}_prediction_{i}'
        )
    
    return model, pred_vars


def _get_descendant_leaves(node_idx, ort_model, all_leaves):
    """
    Get all leaf descendants of a given node.
    
    Args:
        node_idx: Node index to start from
        ort_model: IAI ORT model
        all_leaves: List of all leaf node indices
    
    Returns:
        List of leaf indices that are descendants of node_idx
    """
    if node_idx in all_leaves:
        return [node_idx]
    
    if ort_model.is_leaf(node_index=node_idx):
        return [node_idx]
    
    descendants = []
    lower_child = ort_model.get_lower_child(node_index=node_idx)
    upper_child = ort_model.get_upper_child(node_index=node_idx)
    
    if lower_child is not None:
        descendants.extend(_get_descendant_leaves(lower_child, ort_model, all_leaves))
    if upper_child is not None:
        descendants.extend(_get_descendant_leaves(upper_child, ort_model, all_leaves))
    
    return descendants

def optimize_bids_embedded(X_ort=None, X_lr=None, weights_dict=None, budget=400, max_bid=50.0, conv_model=None, clicks_model=None):
    """Solve bid optimization problem using Gurobi with embedded ML model constraints.
    
    Supports embedding of:
    - Linear Regression (LR): via weights_dict with constants and weights
    - Optimal Regression Trees (ORT): via conv_model and clicks_model IAI objects
    
    Includes ReLU logic for both Conversion and Clicks to handle negative predictions
    while preserving the ability for y=0 to force results to 0.
    
    Args:
        X_ort: Feature matrix with categorical features as strings (for ORT models). If None, will use X_lr.
        X_lr: Feature matrix with categorical features one-hot encoded (for LR models). If None, will use X_ort.
        weights_dict: Dictionary with 'conv_const', 'conv_weights', 'clicks_const', 'clicks_weights'
        budget: Total budget for bids
        max_bid: Maximum individual bid
        conv_model: (Optional) IAI ORT model for conversion. If provided, uses ORT instead of LR
        clicks_model: (Optional) IAI ORT model for clicks. If provided, uses ORT instead of LR
    
    Returns:
        tuple of (model, b, z, y, f_eff, g_eff)
    """
    
    # --- Parameters ---
    # Big-M parameters must be upper bounds on the maximum possible values
    M_d = 400     # Max potential clicks (Big-M for g)
    M_c = 40000   # Max potential conversion value (Big-M for f)
    
    # Determine which models are being used
    use_ort_conv = conv_model is not None
    use_ort_clicks = clicks_model is not None
    
    # Select appropriate X for each model
    X_conv = X_ort if use_ort_conv else X_lr
    X_clicks = X_ort if use_ort_clicks else X_lr
    
    # Use whichever X is not None for size reference
    K = len(X_ort) if X_ort is not None else len(X_lr)
    
    model_type_str = ""
    if use_ort_conv or use_ort_clicks:
        if use_ort_conv and use_ort_clicks:
            model_type_str = "ORT"
        else:
            model_type_str = f"{'ORT' if use_ort_conv else 'LR'} (conversion) + {'ORT' if use_ort_clicks else 'LR'} (clicks)"
    else:
        model_type_str = "LR"
    
    print(f"\nSolving bid optimization with embedded {model_type_str} constraints...")
    print(f"  Budget: ${budget:,.2f}")
    print(f"  Keywords: {K}")

    # Create model
    model = gp.Model('bid_optimization')
    model.setParam('OutputFlag', 1) 
    model.setParam('TimeLimit', 60)

    # --- 0. Decision Variables ---
    
    # b_i: Bid amount
    b = model.addMVar(shape=K, lb=0, ub=max_bid, name='b')
    
    # y_i: Bidding Participation (1 if bidding, 0 otherwise)
    y = model.addMVar(shape=K, vtype=GRB.BINARY, name='y')
    
    # z_i: Click Sensor (1 if clicks > 0, 0 otherwise)
    z = model.addMVar(shape=K, vtype=GRB.BINARY, name='z')
    
    # Effective Values (The final results used in Objective)
    f_eff = model.addMVar(shape=K, lb=0, ub=M_c, name='f_eff')
    g_eff = model.addMVar(shape=K, lb=0, ub=M_d, name='g_eff')

    # Rectified Prediction Variables (Intermediate variables for ReLU logic)
    # These will hold max(0, prediction) independent of y
    f_rect = model.addMVar(shape=K, lb=0, ub=M_c, name='f_rect')
    g_rect = model.addMVar(shape=K, lb=0, ub=M_d, name='g_rect')

    # --- 1. Raw ML Predictions (Embedded) ---
    # Create tupledict to hold prediction variables
    f_hat_vars = []
    g_hat_vars = []
    
    # Create trees directory for ORT visualization
    trees_dir = Path('opt_results/trees') if (use_ort_conv or use_ort_clicks) else None
    
    # Embed conversion model
    if use_ort_conv:
        # Use ORT model
        model, f_hat_vars = embed_ort(model, conv_model, X_conv, b, target='conversion', max_bid=max_bid, save_dir=trees_dir)
    else:
        # Use LR model
        conv_const = weights_dict['conv_const']
        conv_weights = weights_dict['conv_weights']
        model, f_hat_vars = embed_lr(model, conv_weights, conv_const, X_conv, b, target='conversion')
    
    # Embed clicks model
    if use_ort_clicks:
        # Use ORT model
        model, g_hat_vars = embed_ort(model, clicks_model, X_clicks, b, target='clicks', max_bid=max_bid, save_dir=trees_dir)
    else:
        # Use LR model
        clicks_const = weights_dict['clicks_const']
        clicks_weights = weights_dict['clicks_weights']
        model, g_hat_vars = embed_lr(model, clicks_weights, clicks_const, X_clicks, b, target='clicks')
    
    model.update()
    
    # --- Total Budget Constraint ---
    model.addConstr(gp.quicksum(b) <= budget, name='TotalBudget')

    # --- 2. Bidding Participation ---
    # If y=0, b=0. If y=1, b >= 0.01
    model.addConstr(b <= max_bid * y, name='BidMaxBound')
    model.addConstr(b >= 0.01 * y, name='BidMinBound')

    # --- 3. Effective Value Overrides (The "Gate") ---
    # If y=0 (inactive), effective results must be 0
    model.addConstr(f_eff <= M_c * y, name='EffConvBound')
    model.addConstr(g_eff <= M_d * y, name='EffClickBound')

    # --- 4. MODEL RECOVERY (ReLU Logic) ---
    
    # Step A: Calculate Rectified Values [ f_rect = max(0, f_hat) ]
    # This prevents infeasibility when predictions are negative.
    for i in range(K):
        # Conversion ReLU
        model.addGenConstrMax(f_rect[i], [f_hat_vars[i]], constant=0.0, name=f"ReLU_Conv_{i}")
        # Clicks ReLU
        model.addGenConstrMax(g_rect[i], [g_hat_vars[i]], constant=0.0, name=f"ReLU_Click_{i}")

    # Step B: Link Effective to Rectified
    # f_eff <= f_rect.
    # Combined with Section 3, this means: f_eff <= min(f_rect, M*y)
    model.addConstr(f_eff <= f_rect, name='ConvRecovery')
    
    # For clicks (g), we want g_eff >= g_rect when active, but allow 0 when inactive.
    # The original constraint was: g_i >= g_hat - M(1-y)
    # The new robust constraint is: g_eff >= g_rect - M_d * (1 - y)
    #   If y=1: g_eff >= g_rect (since we want to pay at least the predicted cost)
    #   If y=0: g_eff >= g_rect - BigM (Becomes trivial, allows g_eff=0 via Section 3)
    model.addConstr(g_eff >= g_rect - M_d * (1 - y), name='ClickRecovery')

    # --- 5. Logical Dependency ---
    # Click Sensor: If g < 1, force z=0. If g >= 1, allow z=1.
    model.addConstr(g_eff <= M_d * z, name='ClickSensorBigM')
    model.addConstr(z <= g_eff, name='ClickSensorActivator')
    
    # Conversion Gate: No clicks (z=0) means no conversion (f=0)
    model.addConstr(f_eff <= M_c * z, name='ConversionGate')

    # --- Objective ---
    # Maximize Profit = Revenue (f) - Cost (b * g)
    total_cost = b @ g_eff
    total_revenue = gp.quicksum(f_eff)

    model.setObjective(total_revenue - total_cost, GRB.MAXIMIZE)

    # --- Save Model Formulation ---
    model_dir = Path('opt_results/formulations')
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model.update()
    
    # Determine model type string for filename
    if use_ort_conv or use_ort_clicks:
        model_type_str = f"{('ort' if use_ort_conv else 'lr')}_{'ort' if use_ort_clicks else 'lr'}"
    else:
        model_type_str = "lr_lr"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lp_file = model_dir / f'bid_optimization_{model_type_str}_{timestamp}.lp'
    model.write(str(lp_file))
    print(f"\n  Model formulation saved to {lp_file}")
    
    # --- Optimize ---
    model.setParam("NonConvex", 2) # Allow quadratic objective
    model.optimize()

    return model, b, z, y, f_eff, g_eff


def extract_solution(model, b, z, y, f_eff, g_eff, keyword_df, keyword_idx_list, region_list, match_list, X=None, weights_dict=None):
    """Extract non-zero bids from solution with predictions.
    
    Args:
        model: Gurobi model object (solved)
        b: Bid decision variables
        z: Click sensor binary variables (1 if clicks > 0)
        y: Bidding participation binary variables (1 if bidding)
        f_eff: Effective conversion values (from solver)
        g_eff: Effective clicks values (from solver)
        keyword_df: DataFrame with keywords
        keyword_idx_list: List mapping rows to keyword indices
        region_list: List of regions for each row
        match_list: List of match types for each row
        X: Feature matrix (optional)
        weights_dict: Dictionary with model weights (optional)
    """
    b_vals = b.X
    z_vals = z.X
    y_vals = y.X
    f_vals = f_eff.X
    g_vals = g_eff.X
    
    # Find active bids (where y=1)
    active_idx = np.where(y_vals >= 0.5)[0]
    
    print(f"\nSolution Summary:")
    print(f"  Active keywords (y=1): {len(active_idx)}")
    print(f"  Total spend: ${np.sum(b_vals):,.2f}")
    print(f"  Predicted profit: ${model.objVal:,.2f}")
    
    # Build result DataFrame with all active bids
    bids_df = pd.DataFrame({
        'bid': b_vals[active_idx],
        'keyword': [keyword_df.iloc[keyword_idx_list[i]]['Keyword'] for i in active_idx],
        'region': [region_list[i] for i in active_idx],
        'match': [match_list[i] for i in active_idx],
        'active': y_vals[active_idx],
        'predicted_conv_value': f_vals[active_idx],
        'predicted_clicks': g_vals[active_idx],
    })
    
    # Calculate profit: f_eff - b * g_eff
    bids_df['predicted_profit'] = bids_df['predicted_conv_value'] - bids_df['bid'] * bids_df['predicted_clicks']
    
    # Sort by bid (descending)
    bids_df = bids_df.sort_values('bid', ascending=False).reset_index(drop=True)
    
    return bids_df


def main():
    parser = argparse.ArgumentParser(
        description="Optimize keyword bids using linear programming."
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='bert',
        choices=['tfidf', 'bert'],
        help='Embedding method used for models (default: bert)'
    )
    parser.add_argument(
        '--alg-conv',
        type=str,
        default='lr',# 'ort',#
        choices=['lr', 'glm', 'ort'],
        help='Algorithm type for conversion model: lr (linear weights), glm (Tweedie GLM weights), ort (optimal tree; requires IAI) (default: lr)'
    )
    parser.add_argument(
        '--alg-clicks',
        type=str,
        default='ort', # 'lr', # 
        choices=['lr', 'glm', 'ort'],
        help='Algorithm type for clicks model: lr (linear weights), glm (Tweedie GLM weights), ort (optimal tree; requires IAI) (default: lr)'
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=400,
        help='Total budget for bids (default: 400)'
    )
    parser.add_argument(
        '--max-bid',
        type=float,
        default=50.0,
        help='Maximum individual bid (default: 50.0)'
    )
    parser.add_argument(
        '--target-day',
        type=str,
        default=None,
        help='Target day for optimization in format YYYY-MM-DD (e.g., 2024-11-04). If not provided, uses all data.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/clean',
        help='Directory containing embeddings and models (default: data/clean)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )

    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Bid Optimization with Linear Programming")
    print("=" * 70)
    print(f"Embedding method: {args.embedding_method}")
    print("=" * 70)
    
    try:
        # Load new keywords from classified keywords file
        classified_keywords_file = Path('data/gkp') / 'keywords_classified.csv'
        if classified_keywords_file.exists():
            classified_df = pd.read_csv(str(classified_keywords_file))
            new_keywords = classified_df['Keyword'].tolist()
            print(f"Loaded {len(new_keywords)} new keywords from {classified_keywords_file}")
            keywords_to_embed = new_keywords
        else:
            # Fall back to loading from training data
            training_data_file = Path(args.data_dir) / f'ad_opt_data_{args.embedding_method}.csv'
            training_df = pd.read_csv(str(training_data_file))
            keywords_to_embed = training_df['Keyword'].unique().tolist()
            print(f"Loaded {len(keywords_to_embed)} unique keywords from training data")
        
        # Generate embeddings using saved pipeline
        keyword_df = create_feature_matrix_with_embeddings(
            keywords_to_embed,
            embedding_method=args.embedding_method,
            n_components=50,
            models_dir=args.models_dir
        )
        
        # Load weights from CSV files (no IAI required)
        weights_dict = load_weights_from_csv(
            embedding_method=args.embedding_method,
            models_dir=args.models_dir
        )
        
        # Create feature matrix (includes all features and keyword values for target day)
        training_data_file = Path(args.data_dir) / f'ad_opt_data_{args.embedding_method}.csv'
        feature_matrix_result = create_feature_matrix(
            keyword_df, 
            embedding_method=args.embedding_method,
            target_day=args.target_day,
            training_data_path=str(training_data_file),
            weights_dict=weights_dict,
            alg_conv=args.alg_conv,
            alg_clicks=args.alg_clicks
        )
        
        # Unpack results based on model types
        is_mixed = (args.alg_conv == 'ort' and args.alg_clicks != 'ort') or (args.alg_conv != 'ort' and args.alg_clicks == 'ort')
        if is_mixed:
            X_ort, X_lr, kw_idx_list, region_list, match_list = feature_matrix_result
        else:
            X_result, kw_idx_list, region_list, match_list = feature_matrix_result
            X_ort = X_result if args.alg_conv == 'ort' and args.alg_clicks == 'ort' else None
            X_lr = X_result if args.alg_conv != 'ort' and args.alg_clicks != 'ort' else None
        
        X = X_ort if X_ort is not None else X_lr
        print(f"Feature matrix has {X.shape[1]} total features")
        
        # Save feature matrices
        data_dir = Path('opt_results/feature_matrices')
        data_dir.mkdir(exist_ok=True, parents=True)
        
        if X_ort is not None:
            ort_file = data_dir / f'X_ort_{args.embedding_method}_{args.alg_conv}_{args.alg_clicks}.csv'
            X_ort.to_csv(ort_file, index=False)
            print(f"Saved X_ort to {ort_file}")
        
        if X_lr is not None:
            lr_file = data_dir / f'X_lr_{args.embedding_method}_{args.alg_conv}_{args.alg_clicks}.csv'
            X_lr.to_csv(lr_file, index=False)
            print(f"Saved X_lr to {lr_file}")
        
        # Also save the mapping information (keyword indices, regions, match types)
        mapping_file = data_dir / f'mapping_{args.embedding_method}_{args.alg_conv}_{args.alg_clicks}.csv'
        mapping_df = pd.DataFrame({
            'keyword_idx': kw_idx_list,
            'region': region_list,
            'match_type': match_list
        })
        mapping_df.to_csv(mapping_file, index=False)
        print(f"Saved mapping to {mapping_file}")

        # Embedded mode: supports linear weights (LR/GLM) and ORT (IAI)
        conv_model = None
        clicks_model = None

        if args.alg_conv == 'ort':
            if iai is None:
                raise RuntimeError("IAI is required for ORT models. Please install IAI or use LR/GLM weights.")

            conversion_model_path = Path(args.models_dir) / f'ort_{args.embedding_method}_conversion.json'
            if not conversion_model_path.exists():
                raise FileNotFoundError(f"ORT conversion model not found: {conversion_model_path}")
            conv_model = iai.read_json(str(conversion_model_path))
            print(f"  Loaded ORT conversion model from {conversion_model_path}")

        if args.alg_clicks == 'ort':
            if iai is None:
                raise RuntimeError("IAI is required for ORT models. Please install IAI or use LR/GLM weights.")

            clicks_model_path = Path(args.models_dir) / f'ort_{args.embedding_method}_clicks.json'
            if not clicks_model_path.exists():
                raise FileNotFoundError(f"ORT clicks model not found: {clicks_model_path}")
            clicks_model = iai.read_json(str(clicks_model_path))
            print(f"  Loaded ORT clicks model from {clicks_model_path}")

        model, b, z, y, f_eff, g_eff = optimize_bids_embedded(
            X_ort=X_ort,
            X_lr=X_lr,
            weights_dict=weights_dict,
            budget=args.budget,
            max_bid=args.max_bid,
            conv_model=conv_model,
            clicks_model=clicks_model
        )

        if model.status == 2 or model.status == 9:
            output_dir = Path('opt_results/bids')
            output_dir.mkdir(exist_ok=True, parents=True)

            model_suffix = f"{args.alg_conv}_{args.alg_clicks}"
            output_file = output_dir / f'optimized_bids_{args.embedding_method}_{model_suffix}.csv'

            bids_df = extract_solution(model, b, z, y, f_eff, g_eff, keyword_df, kw_idx_list, region_list, match_list, X=X, weights_dict=weights_dict)

            bids_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            print(f"\nTop 10 bids:")
            print(bids_df.head(10).to_string(index=False))
        else:
            print(f"Optimization failed with status {model.status}")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("✓ Bid optimization completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
