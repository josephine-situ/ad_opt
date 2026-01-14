"""
Bid Optimization with Linear Programming
==========================================
Optimizes keyword bids using Gurobi linear programming solver.
Maximizes profit by setting optimal bids for keywords across regions and match types.

    Requires:
    - Gurobi solver and license: https://www.gurobi.com/
    - gurobipy: pip install gurobipy
    - Pre-trained models / weights for EPC + clicks (e.g., weights_{embedding}_epc_*.csv, weights_{embedding}_clicks_*.csv)
    - Embeddings file will be auto-generated if missing

Usage:
    python bid_optimization.py --embedding-method bert --budget 68096.51 --max-bid 100
    python bid_optimization.py --embedding-method tfidf --budget 68096.51
"""

import argparse
import sys
from pathlib import Path
import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_embeddings
from utils.data_pipeline import get_gkp_data, impute_missing_data
from utils.keyword_matching import fuzzy_fill_from_gkp

# Check for required libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: Gurobi not found. Install with: pip install gurobipy")
    print("Note: Gurobi requires a valid license.")
    sys.exit(1)

# Optional dependency: IAI (only needed for ORT/MORT workflows or when falling back to IAI for weights)
iai = None
_iai_import_attempted = False


def _get_iai(*, required: bool) -> object:
    """Lazily import IAI only when needed.

    This keeps non-IAI workflows (LR/GLM/XGB) from importing (or failing on) IAI.
    """

    global iai, _iai_import_attempted
    if iai is not None:
        return iai

    if _iai_import_attempted:
        if required:
            raise ImportError(
                "IAI is required for this operation (ORT/MORT or IAI weight fallback), but it could not be imported."
            )
        return None

    _iai_import_attempted = True
    try:
        from utils.iai_setup import iai as _iai  # type: ignore

        iai = _iai
        return iai
    except Exception as e:
        if required:
            raise ImportError(
                "IAI is required for this operation (ORT/MORT or IAI weight fallback), but it could not be imported."
            ) from e
        return None


try:
    import joblib  # type: ignore
except Exception:
    joblib = None

try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

try:
    import scipy.sparse as sp  # type: ignore
except Exception:
    sp = None


def generate_keyword_embeddings_df(keywords, embedding_method='bert', n_components=50, pipeline_dir='data/clean'):
    """Generate embeddings for keywords using a saved embedding pipeline.

    Loads the saved vectorizer/transformer, SVD, and normalizer from pipeline_dir
    and applies them to the provided keywords to create embeddings.
    
    Args:
        keywords: list or Series of keyword strings
        embedding_method: 'bert' or 'tfidf'
        n_components: number of SVD components used (should match training)
        pipeline_dir: directory containing saved pipeline pickle files (default: data/clean)
    
    Returns:
        DataFrame with columns ['{method}_0' ... '{method}_N', 'Keyword']
    """
    import pickle
    
    # Ensure keywords is a list
    if not isinstance(keywords, list):
        keywords = list(keywords)
    
    print(f"Generating {embedding_method.upper()} embeddings for {len(keywords)} keywords...")
    
    # Load the saved pipeline
    pipeline_file = Path(pipeline_dir) / f'{embedding_method}_pipeline_{n_components}d.pkl'
    if not pipeline_file.exists():
        raise FileNotFoundError(
            f"Embedding pipeline not found: {pipeline_file}. "
            f"Please ensure the data pipeline has been run with embedding pipeline persistence enabled."
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
        # BERT pipeline (SentenceTransformer)
        # Reload from fresh to avoid pickle/tokenizer issues while maintaining consistency
        from sentence_transformers import SentenceTransformer
        
        # Get the model name from the stored pipeline (ensures consistency with training)
        model_name = pipeline.get('model_name', 'all-MiniLM-L6-v2')
        
        try:
            # Reload the model fresh to ensure tokenizer is properly initialized
            # (pickled tokenizers can have state issues across environments)
            transformer = SentenceTransformer(model_name)
            embeddings = transformer.encode(keywords, convert_to_numpy=True)
            print(f"  Using SentenceTransformer model: {model_name}")
        except Exception as e:
            # Fallback: try to use the pickled vectorizer if reload fails
            print(f"  Warning: Could not reload SentenceTransformer, using pickled version: {e}")
            embeddings = vectorizer.encode(keywords)  # array of shape (n, 384)
        
        embeddings = svd.transform(embeddings)  # reduce to n_components (fitted on training data)
    
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
        dict with keys: 'epc_const', 'epc_weights', 'clicks_const', 'clicks_weights'
    """
    print(f"Loading weights for embedding method '{embedding_method}'...")
    
    models_dir = Path(models_dir)
    
    # Check if CSV files exist
    epc_numeric_file = models_dir / f'weights_{embedding_method}_epc_numeric.csv'
    epc_const_file = models_dir / f'weights_{embedding_method}_epc_constant.csv'
    clicks_numeric_file = models_dir / f'weights_{embedding_method}_clicks_numeric.csv'
    clicks_const_file = models_dir / f'weights_{embedding_method}_clicks_constant.csv'
    
    # If CSV files exist, load from them
    has_epc = epc_numeric_file.exists() and epc_const_file.exists()
    has_clicks = clicks_numeric_file.exists() and clicks_const_file.exists()

    if has_clicks and has_epc:
        print(f"  Loading from CSV files...")

        epc_cat_file = models_dir / f'weights_{embedding_method}_epc_categorical.csv'
        clicks_cat_file = models_dir / f'weights_{embedding_method}_clicks_categorical.csv'

        # Load EPC
        epc_numeric_df = pd.read_csv(epc_numeric_file)
        epc_weights = dict(zip(epc_numeric_df['feature'], epc_numeric_df['weight']))

        if epc_cat_file.exists():
            epc_cat_df = pd.read_csv(epc_cat_file)
            for _, row in epc_cat_df.iterrows():
                feature = row['feature']
                level = row['level']
                weight = row['weight']
                if feature not in epc_weights:
                    epc_weights[feature] = {}
                if not isinstance(epc_weights[feature], dict):
                    numeric_val = epc_weights[feature]
                    epc_weights[feature] = {feature: numeric_val}
                epc_weights[feature][level] = weight

        epc_const = pd.read_csv(epc_const_file)['constant'].iloc[0]

        # Load clicks
        clicks_numeric_df = pd.read_csv(clicks_numeric_file)
        clicks_weights = dict(zip(clicks_numeric_df['feature'], clicks_numeric_df['weight']))
        if clicks_cat_file.exists():
            clicks_cat_df = pd.read_csv(clicks_cat_file)
            for _, row in clicks_cat_df.iterrows():
                feature = row['feature']
                level = row['level']
                weight = row['weight']
                if feature not in clicks_weights:
                    clicks_weights[feature] = {}
                if not isinstance(clicks_weights[feature], dict):
                    numeric_val = clicks_weights[feature]
                    clicks_weights[feature] = {feature: numeric_val}
                clicks_weights[feature][level] = weight
        clicks_const = pd.read_csv(clicks_const_file)['constant'].iloc[0]

        print(f"  Loaded from CSV files ✓")
        return {
            'epc_const': epc_const,
            'epc_weights': epc_weights,
            'clicks_const': clicks_const,
            'clicks_weights': clicks_weights,
        }
    
    # If CSV files don't exist, fall back to IAI
    else:
        print(f"  CSV files not found, falling back to IAI...")

        iai_local = _get_iai(required=True)
        
        # Load models using IAI
        epc_model_path = models_dir / f'lr_{embedding_method}_epc.json'
        clicks_model = models_dir / f'lr_{embedding_method}_clicks.json'
        
        if not epc_model_path.exists():
            raise FileNotFoundError(f"EPC model not found: {epc_model_path}")
        if not clicks_model.exists():
            raise FileNotFoundError(f"Clicks model not found: {clicks_model}")

        lnr_epc = iai_local.read_json(str(epc_model_path))
        lnr_clicks = iai_local.read_json(str(clicks_model))
        
        # Extract weights and constants using IAI
        weights_epc_tuple = lnr_epc.get_prediction_weights()
        weights_clicks_tuple = lnr_clicks.get_prediction_weights()
        
        if isinstance(weights_epc_tuple, tuple):
            epc_weights = weights_epc_tuple[0]
        else:
            epc_weights = weights_epc_tuple
        
        if isinstance(weights_clicks_tuple, tuple):
            clicks_weights = weights_clicks_tuple[0]
        else:
            clicks_weights = weights_clicks_tuple
        
        epc_const = lnr_epc.get_prediction_constant()
        clicks_const = lnr_clicks.get_prediction_constant()
        
        print(f"  Loaded from IAI models ✓")
        return {
            'epc_const': epc_const,
            'epc_weights': epc_weights,
            'clicks_const': clicks_const,
            'clicks_weights': clicks_weights,
        }


def create_feature_matrix(
    keyword_df,
    embedding_method='bert',
    target_day=None,
    regions=None,
    match_types=None,
    alg_epc='lr',
    alg_clicks='lr',
    gkp_dir: str = 'data/gkp',
):
    """Create feature matrix/matrices for all keyword-region-match combinations for a specific day.
    
    This version is intentionally *independent* of historical ads training data.
    It builds features using:
    - Keyword embeddings from keyword_df
    - Region + match type
    - Google Keyword Planner (GKP) stats from data/gkp (competition, top-of-page bids, monthly searches)
    - Derived search time-series stats: last_month_searches, three_month_avg, six_month_avg, mom_change, search_trend
    - Date features for the requested target day (or today)
    
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
        alg_epc: algorithm type for EPC model ('lr', 'ort', etc.).
        alg_clicks: algorithm type for clicks model ('lr', 'ort', etc.).
        gkp_dir: directory containing GKP exports (default: data/gkp)
    
    Returns:
        For ORT-only: (X_ort, keyword_idx_list, region_list, match_list)
        For LR-only: (X_lr, keyword_idx_list, region_list, match_list)
        For mixed: (X_ort, X_lr, keyword_idx_list, region_list, match_list)
    """
    if regions is None:
        regions = ["USA", "A", "B", "C"]
    if match_types is None:
        match_types = ["Broad match", "Exact match", "Phrase match"]
    
    num_keywords = len(keyword_df)
    n_combos = num_keywords * len(regions) * len(match_types)
    
    print(f"Creating feature matrix...")
    print(f"  Target day: {target_day}")
    print(f"  Keywords: {num_keywords}, Regions: {len(regions)}, Matches: {len(match_types)}")
    print(f"  Total combinations: {n_combos}")
    
    # Determine which date to use for temporal features
    if target_day is not None:
        if isinstance(target_day, str):
            target_day = pd.to_datetime(target_day)
        filter_day = target_day
        print(f"  Using target day for temporal features: {filter_day.date()}")
    else:
        filter_day = pd.to_datetime(datetime.today())
        print(f"  Using today for temporal features: {filter_day.date()}")
    
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

    # --- Merge keyword embeddings onto each row ---
    # `keyword_df` is expected to contain columns like bert_0..bert_49 (or tfidf_*) plus 'Keyword'.
    # We join on a normalized keyword key to avoid case/whitespace mismatches.
    kw_emb = keyword_df.copy()
    kw_emb['Keyword_join'] = kw_emb['Keyword'].astype(str).str.lower().str.strip()
    combo_df = combo_df.copy()
    combo_df['Keyword_join'] = combo_df['Keyword'].astype(str).str.lower().str.strip()

    # Add embeddings (and any other keyword-level columns) to each combination.
    # Keep the original 'Keyword' from combo_df.
    combo_df = combo_df.merge(
        kw_emb.drop(columns=['Keyword'], errors='ignore'),
        on='Keyword_join',
        how='left'
    )

    # --- Feature sourcing (GKP-only) ---
    print(f"  Using GKP keyword stats from {gkp_dir} (no historical training merge)")
    gkp_df = get_gkp_data(gkp_dir=gkp_dir)

    # Normalize join keys for GKP merge
    gkp_df = gkp_df.copy()
    gkp_df['Keyword'] = gkp_df['Keyword'].astype(str).str.lower().str.strip()
    gkp_df['Keyword_join'] = gkp_df['Keyword']

    # Merge keyword stats onto each keyword-region-match row
    result = combo_df.merge(
        gkp_df.drop(columns=['Keyword'], errors='ignore'),
        left_on='Keyword_join',
        right_on='Keyword_join',
        how='left'
    )

    # Fill remaining unmatched rows via similar-word normalization.
    # This keeps the existing exact normalized merge behavior, and only fills leftover NaNs.
    fuzzy_fill_from_gkp(
        result,
        keyword_col='Keyword',
        gkp_df=gkp_df,
        gkp_keyword_col='Keyword',
        verbose=True,
        source_display_col='Keyword',
        print_all_mappings=False,
    )

    # Compute time-series stats from searches_YYYY_MM columns
    search_cols = sorted([c for c in result.columns if c.startswith('searches_')])
    if search_cols:
        def _ts_stats_from_row(row):
            values = [row[c] for c in search_cols]
            clean_vals = []
            for v in values:
                try:
                    clean_vals.append(float(v))
                except Exception:
                    clean_vals.append(np.nan)

            last_val = clean_vals[-1] if clean_vals else np.nan

            three = [v for v in clean_vals[-3:] if not np.isnan(v)]
            six = [v for v in clean_vals[-6:] if not np.isnan(v)]

            three_avg = (sum(three) / len(three)) if three else np.nan
            six_avg = (sum(six) / len(six)) if six else np.nan

            mom = np.nan
            if len(clean_vals) >= 2 and not np.isnan(clean_vals[-1]) and not np.isnan(clean_vals[-2]):
                prev_val = clean_vals[-2]
                curr_val = clean_vals[-1]
                if prev_val > 0:
                    mom = ((curr_val - prev_val) / prev_val) * 100.0
                else:
                    mom = 100.0 if curr_val > 0 else 0.0

            trend = np.nan
            if len(six) >= 2:
                x = np.arange(len(six), dtype=float)
                y = np.array(six, dtype=float)
                trend = np.polyfit(x, y, 1)[0]

            return pd.Series(
                {
                    'last_month_searches': last_val,
                    'three_month_avg': three_avg,
                    'six_month_avg': six_avg,
                    'mom_change': mom,
                    'search_trend': trend,
                }
            )

        ts_stats = result.apply(_ts_stats_from_row, axis=1)
        for col in ts_stats.columns:
            result[col] = ts_stats[col]

    # Drop raw monthly columns to keep feature matrix consistent
    if search_cols:
        result = result.drop(columns=search_cols, errors='ignore')

    # Provide Avg. CPC proxy if missing (optimizer replaces it with bid variable anyway)
    if 'Avg. CPC' not in result.columns:
        low = result.get('Top of page bid (low range)')
        high = result.get('Top of page bid (high range)')
        if low is not None and high is not None:
            result['Avg. CPC'] = (pd.to_numeric(low, errors='coerce') + pd.to_numeric(high, errors='coerce')) / 2.0
        else:
            result['Avg. CPC'] = 0.0

    # Impute missing values (uses repo's current imputation strategy)
    result = impute_missing_data(result)

    # Drop helper join key but keep Keyword/Region/Match type/Day
    result = result.drop(columns=['Keyword_join', '_kw_key'], errors='ignore')
    
    # Date features always come from the requested target day (or today)
    from utils.date_features import calculate_date_features
    date_features = calculate_date_features(filter_day, regions=regions)
    result['day_of_week'] = date_features['day_of_week']
    result['is_weekend'] = date_features['is_weekend']
    result['month'] = date_features['month']
    result['is_public_holiday'] = date_features['is_public_holiday']
    result['days_to_next_course_start'] = date_features['days_to_next_course_start']
    
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
    
    # Determine if we need both versions (mixed models) or just one.
    # - ORT/MORT need raw categoricals (strings).
    # - XGB models in this repo are trained with a saved sklearn ColumnTransformer
    #   (one-hot inside the preprocessor), so they need raw categoricals as input.
    # - Ridge models also use a saved sklearn ColumnTransformer preprocessor,
    #   so they also need raw categoricals as input.
    raw_cat_algs = {'ort', 'mort', 'xgb', 'ridge'}
    use_raw_epc = (alg_epc in raw_cat_algs)
    use_raw_clicks = (alg_clicks in raw_cat_algs)
    use_both_raw = use_raw_epc and use_raw_clicks
    use_both_lr = (not use_raw_epc) and (not use_raw_clicks)
    is_mixed = (use_raw_epc and not use_raw_clicks) or (not use_raw_epc and use_raw_clicks)
    
    if use_both_raw:
        # Both need raw categoricals: keep categorical as strings only
        print(f"  Keeping categorical features as strings (both models need raw categoricals)")
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            result[col] = result[col].astype(float)
        X_ort = result
        X_lr = None
        
    elif use_both_lr:
        # Both are linear-weight models: one-hot encode only
        print(f"  One-hot encoding categorical columns (both models use linear weights)")
        categorical_cols = result.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            result = pd.get_dummies(result, columns=categorical_cols, drop_first=False)
        result = result.astype(float)
        X_ort = None
        X_lr = result
        
    else:
        # Mixed models: create both versions
        print(f"  Creating both versions for mixed models ({alg_epc.upper()} + {alg_clicks.upper()})")
        
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
    elif use_both_raw:
        return X_ort, keyword_idx_list, region_list, match_list
    else:  # use_both_lr
        return X_lr, keyword_idx_list, region_list, match_list

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
        target: Target name ('epc' or 'clicks') for variable naming
    
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


def embed_glm(model, weights, const, X, b, target, *, link: str = 'log'):
    """Embed a (Tweedie) GLM into Gurobi.

    This repo's GLM models are trained via sklearn's TweedieRegressor, which
    uses a log link by default for most positive-mean Tweedie families.

    For link='log', we embed:
        eta_i = const + sum_j w_j * x_{ij} + w_cpc * b_i
        pred_i = exp(eta_i)

    Args:
        model: Gurobi model
        weights: weights dict in the same format as embed_lr
        const: intercept
        X: feature matrix (DataFrame)
        b: bid decision variables (MVar)
        target: 'epc' or 'clicks'
        link: 'log' or 'identity'
    """

    K = len(X)
    pred_vars = []

    print(f"  Embedding {target} GLM constraints (link={link})...")

    for i in range(K):
        # Linear predictor eta
        eta = model.addVar(lb=-GRB.INFINITY, name=f'{target}_eta_{i}')

        expr = const

        for feature, weight in weights.items():
            if feature in ['Avg. CPC', 'Avg_ CPC']:
                expr += weight * b[i]
                continue

            if isinstance(weight, dict):
                for level_name, level_weight in weight.items():
                    ohe_col_name = f"{feature}_{level_name}"
                    if ohe_col_name not in X.columns:
                        raise ValueError(
                            f"Error: One-hot encoded column '{ohe_col_name}' is missing from X dataframe for {target} GLM."
                        )
                    expr += level_weight * X.iloc[i][ohe_col_name]
            else:
                if feature not in X.columns:
                    raise ValueError(f"Error: Feature '{feature}' is missing from X dataframe for {target} GLM.")
                expr += weight * X.iloc[i][feature]

        model.addConstr(eta == expr, name=f'{target}_glm_eta_{i}')

        if link == 'identity':
            pred = model.addVar(lb=-GRB.INFINITY, name=f'{target}_pred_{i}')
            model.addConstr(pred == eta, name=f'{target}_glm_identity_{i}')
        elif link == 'log':
            pred = model.addVar(lb=0.0, name=f'{target}_pred_{i}')
            # y = exp(x) general constraint
            model.addGenConstrExp(eta, pred, name=f'{target}_glm_exp_{i}')
        else:
            raise ValueError(f"Unsupported GLM link: {link}")

        pred_vars.append(pred)

    return model, pred_vars


def embed_affine_in_bid(
    model,
    *,
    base: np.ndarray,
    slope: np.ndarray,
    b,
    target: str,
    link: str = "identity",
) -> Tuple[object, List[object]]:
    """Embed predictions of the form y_i = base_i + slope_i * b_i (or exp for log link).

    This is used when a trained sklearn model is linear in the *preprocessed* feature
    space and the preprocessing makes the bid feature appear as an affine function of b.
    """

    K = int(len(base))
    if len(slope) != K:
        raise ValueError(f"base length ({K}) != slope length ({len(slope)})")

    pred_vars: List[object] = []
    for i in range(K):
        eta = float(base[i]) + float(slope[i]) * b[i]
        if link == "identity":
            pred = model.addVar(lb=-GRB.INFINITY, name=f"{target}_pred_{i}")
            model.addConstr(pred == eta, name=f"{target}_affine_{i}")
        elif link == "log":
            pred = model.addVar(lb=0.0, name=f"{target}_pred_{i}")
            eta_var = model.addVar(lb=-GRB.INFINITY, name=f"{target}_eta_{i}")
            model.addConstr(eta_var == eta, name=f"{target}_affine_eta_{i}")
            model.addGenConstrExp(eta_var, pred, name=f"{target}_affine_exp_{i}")
        else:
            raise ValueError(f"Unsupported link: {link}")

        pred_vars.append(pred)

    return model, pred_vars


def _extract_ridge_coefficients(ridge_model_path: Path, preprocessor_path: Path) -> Tuple[float, np.ndarray, List[str]]:
    """Extract intercept and coefficients from a trained Ridge model.
    
    Args:
        ridge_model_path: Path to saved Ridge model (.joblib)
        preprocessor_path: Path to saved preprocessor (.joblib)
    
    Returns:
        Tuple of (intercept, coefficients_array, feature_names)
    """
    if joblib is None:
        raise ImportError("joblib is required to load Ridge models")
    
    # Load the Ridge model and preprocessor
    ridge_pipeline = joblib.load(ridge_model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Extract from pipeline
    ridge_model = ridge_pipeline.named_steps['model']
    
    intercept = float(ridge_model.intercept_)
    coefficients = ridge_model.coef_.ravel()
    feature_names = list(preprocessor.get_feature_names_out())
    # Strip the "num__" and "cat__" prefixes
    feature_names = [name.split('__', 1)[1] if '__' in name else name for name in feature_names]
    
    return intercept, coefficients, feature_names

def _extract_xgb_base_score(model_path: Path) -> float:
    """Extract base_score from an XGBoost model JSON saved via save_model()."""

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        learner = obj.get("learner", {})
        lmp = learner.get("learner_model_param", {})
        bs = lmp.get("base_score")
        if bs is None:
            # Some versions nest params differently.
            bs = learner.get("attributes", {}).get("base_score")
        if bs is None:
            return 0.0
        return float(bs)
    except Exception:
        return 0.0


def _align_X_for_preprocessor(preprocessor, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Align X to the fitted preprocessor's expected columns.

    Ensures missing columns exist (filled with NaN) and columns are ordered.
    Also handles the common 'Avg. CPC' vs 'Avg_ CPC' name mismatch.
    """

    def _expected_from_column_transformer(ct) -> Optional[List[str]]:
        """Best-effort: infer expected input columns from a fitted ColumnTransformer.

        Works for ColumnTransformers that were fit on DataFrames (column specs as names).
        If the ColumnTransformer was fit on numpy arrays (column specs as indices/slices),
        we cannot reliably reconstruct names here.
        """

        transformers = getattr(ct, "transformers_", None)
        if not transformers:
            return None

        expected_cols: List[str] = []
        used: set = set()

        for _name, _trans, colspec in transformers:
            if colspec is None or colspec == "drop":
                continue

            # Most common: list/tuple/np.ndarray of column names
            if isinstance(colspec, (list, tuple, np.ndarray, pd.Index)):
                if len(colspec) == 0:
                    continue
                first = colspec[0]
                if isinstance(first, str):
                    for c in list(colspec):
                        if c not in used:
                            expected_cols.append(c)
                            used.add(c)
                    continue

            # Anything else (slice, int indices, boolean masks, callables) -> give up.
            return None

        if getattr(ct, "remainder", "drop") == "passthrough":
            for c in X.columns:
                if c not in used:
                    expected_cols.append(c)
                    used.add(c)

        return expected_cols or None

    # 1) Preferred: sklearn exposes exact training-time input columns.
    expected: Optional[List[str]] = None
    if hasattr(preprocessor, "feature_names_in_"):
        expected = list(preprocessor.feature_names_in_)  # type: ignore[attr-defined]
    else:
        expected = _expected_from_column_transformer(preprocessor)

    X_use = X.copy()

    # 2) If we inferred expected by name, align by name.
    if expected is not None:
        if "Avg. CPC" in expected and "Avg. CPC" not in X_use.columns and "Avg_ CPC" in X_use.columns:
            X_use = X_use.rename(columns={"Avg_ CPC": "Avg. CPC"})
        elif "Avg_ CPC" in expected and "Avg_ CPC" not in X_use.columns and "Avg. CPC" in X_use.columns:
            X_use = X_use.rename(columns={"Avg. CPC": "Avg_ CPC"})

        for col in expected:
            if col not in X_use.columns:
                X_use[col] = np.nan

        X_use = X_use[expected]
        return X_use, expected

    # 3) Fallback: align by expected feature count (positional). This happens when the
    # preprocessor was fit on numpy arrays (no column names preserved).
    if hasattr(preprocessor, "n_features_in_"):
        n_in = int(getattr(preprocessor, "n_features_in_"))
        if X_use.shape[1] < n_in:
            # Pad with NaN columns to satisfy expected width.
            for j in range(n_in - X_use.shape[1]):
                X_use[f"__pad_{j}__"] = np.nan
        if X_use.shape[1] != n_in:
            X_use = X_use.iloc[:, :n_in]
        return X_use, list(X_use.columns)

    # 4) Last resort: do nothing.
    return X_use, list(X_use.columns)


def _parse_xgb_tree_paths(tree_obj: dict) -> List[Tuple[List[Tuple[int, str, float]], float]]:
    """Return list of (conditions, leaf_value) for one tree.

    condition tuple: (feature_index, op, threshold) with op in {'lt','ge'}.
    """

    paths: List[Tuple[List[Tuple[int, str, float]], float]] = []

    def _recurse(node: dict, conds: List[Tuple[int, str, float]]) -> None:
        if "leaf" in node:
            paths.append((list(conds), float(node["leaf"])))
            return

        split = node.get("split")
        if not (isinstance(split, str) and split.startswith("f")):
            raise ValueError(f"Unsupported split key: {split}")

        feat_idx = int(split[1:])
        thr = float(node.get("split_condition"))

        children = node.get("children")
        if not isinstance(children, list) or len(children) < 2:
            raise ValueError("Malformed XGBoost tree: missing children")

        child_by_id = {int(ch.get("nodeid")): ch for ch in children if "nodeid" in ch}
        yes_id = int(node.get("yes"))
        no_id = int(node.get("no"))
        yes_child = child_by_id.get(yes_id)
        no_child = child_by_id.get(no_id)
        if yes_child is None or no_child is None:
            raise ValueError("Malformed XGBoost tree: could not map yes/no children")

        # XGBoost convention: yes branch corresponds to feature < threshold.
        _recurse(yes_child, conds + [(feat_idx, "lt", thr)])
        _recurse(no_child, conds + [(feat_idx, "ge", thr)])

    _recurse(tree_obj, [])
    return paths


def embed_xgb(
    model,
    *,
    xgb_model_path: Path,
    preprocessor_path: Path,
    X: pd.DataFrame,
    b,
    target: str,
    max_bid: float,
) -> Tuple[object, List[object]]:
    """Embed an XGBoost regressor into Gurobi (Corrected Formulation).

    Assumptions:
    - The XGB model was trained on `preprocessor.transform(X_raw)`.
    - `X` is the raw feature DataFrame.
    - The bid decision variable `b[i]` modifies the 'Avg. CPC' feature.
    """

    if xgb is None or joblib is None:
        raise ImportError("xgboost and joblib are required. Pip install them first.")

    print(f"  Embedding {target} XGB constraints (path-based formulation)...")

    # 1. Load Preprocessor
    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {e}")

    # 2. Align Data
    # Note: Ensure _align_X_for_preprocessor is available in your scope
    X_use, expected_cols = _align_X_for_preprocessor(preprocessor, X)

    # 3. Identify CPC Column
    cpc_candidates = [c for c in ("Avg. CPC", "Avg_ CPC") if c in expected_cols]
    if not cpc_candidates:
        raise KeyError(f"Could not find Avg. CPC in expected columns: {expected_cols[:5]}...")
    cpc_col = cpc_candidates[0]

    # 4. Precompute Transformations
    # We calculate Z0 (value at bid=0) and the slope (dZ).
    X0 = X_use.copy()
    X0[cpc_col] = 0.0
    X1 = X_use.copy()
    X1[cpc_col] = 1.0

    try:
        Z0 = preprocessor.transform(X0)
        Z1 = preprocessor.transform(X1)
        dZ = Z1 - Z0  # The slope per unit of bid
    except Exception as e:
        print(f"Preprocessing crashed: {e}")
        raise

    # Handle Sparse/Dense conversion
    if sp is not None and sp.issparse(Z0):
        Z0 = Z0.tocsr()
        dZ = dZ.tocsr()
    else:
        Z0 = np.asarray(Z0)
        dZ = np.asarray(dZ)

    # 5. Load XGBoost Model
    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    
    # Note: Ensure _extract_xgb_base_score is available in your scope
    base_score = _extract_xgb_base_score(xgb_model_path)

    # 6. Parse Trees
    tree_dumps = booster.get_dump(dump_format="json")
    trees = [json.loads(s) for s in tree_dumps]

    # Note: Ensure _parse_xgb_tree_paths is available in your scope
    tree_paths = []
    used_features = set()

    for t in trees:
        paths = _parse_xgb_tree_paths(t)
        tree_paths.append(paths)
        for conds, _ in paths:
            for feat_idx, _, _ in conds:
                used_features.add(int(feat_idx))

    if not used_features:
        raise ValueError("XGB model appears to have no split features")

    # 7. Identify the CPC Feature Index in Transformed Space
    # Bid can affect multiple transformed columns (e.g., scaling + passthrough).
    # We embed each split feature as an affine function of b: v_i(b) = base_i + slope_i * b.
    tol = 1e-9

    # 8. Materialize base (Z0) and slope (dZ) for used features
    used_features_list = sorted(list(used_features))
    base_by_feat: Dict[int, np.ndarray] = {}
    slope_by_feat: Dict[int, np.ndarray] = {}
    
    if sp is not None and sp.issparse(Z0):
        for j in used_features_list:
            base_by_feat[j] = np.asarray(Z0.getcol(j).toarray()).ravel()
            slope_by_feat[j] = np.asarray(dZ.getcol(j).toarray()).ravel()
    else:
        for j in used_features_list:
            base_by_feat[j] = np.asarray(Z0[:, j]).ravel()
            slope_by_feat[j] = np.asarray(dZ[:, j]).ravel()

    # 9. Build Gurobi Constraints
    K = len(X_use)
    pred_vars = []

    for i in range(K):
        pred = model.addVar(lb=-GRB.INFINITY, name=f"{target}_pred_{i}")
        pred_vars.append(pred)

        tree_outputs = []

        for t_idx, paths in enumerate(tree_paths):
            leaf_inds = []
            leaf_vals = []

            for leaf_idx, (conds, leaf_val) in enumerate(paths):
                # Decision variable for "Is this leaf active?"
                z = model.addVar(vtype=GRB.BINARY, name=f"{target}_t{t_idx}_l{leaf_idx}_i{i}")
                leaf_inds.append(z)
                leaf_vals.append(float(leaf_val))

                infeasible = False
                split_counter = 0
                
                for feat_idx, op, thr in conds:
                    feat_idx = int(feat_idx)
                    thr = float(thr)

                    base_val = float(base_by_feat[feat_idx][i])
                    slope_val = float(slope_by_feat[feat_idx][i])

                    # Treat as constant if bid has (near) zero effect on this transformed feature.
                    if abs(slope_val) <= tol:
                        v = base_val
                        if np.isnan(v):
                            infeasible = True
                            break
                        if op == "lt":
                            # Exact XGBoost semantics: go left iff v < thr
                            if not (v < thr):
                                infeasible = True
                                break
                        elif op == "ge":
                            # Exact XGBoost semantics: go right iff v >= thr
                            if not (v >= thr):
                                infeasible = True
                                break
                        else:
                            raise ValueError(f"Unknown op: {op}")
                    else:
                        # Variable feature: v(b) = base + slope*b
                        v0 = base_val
                        v1 = base_val + slope_val * float(max_bid)
                        val_min = min(v0, v1)
                        val_max = max(v0, v1)

                        local_M = max(abs(val_max - thr), abs(val_min - thr)) + 10.0
                        expr = base_val + (slope_val * b[i])

                        if op == "lt":
                            model.addConstr(
                                expr <= thr - 1e-6 + local_M * (1 - z),
                                name=f"{target}_lt_t{t_idx}_l{leaf_idx}_i{i}_s{split_counter}"
                            )
                        elif op == "ge":
                            model.addConstr(
                                expr >= thr - local_M * (1 - z),
                                name=f"{target}_ge_t{t_idx}_l{leaf_idx}_i{i}_s{split_counter}"
                            )
                        else:
                            raise ValueError(f"Unknown op: {op}")
                    
                    split_counter += 1

                if infeasible:
                    # If constant features make this path impossible, force z=0
                    model.addConstr(z == 0, name=f"{target}_infeasible_t{t_idx}_l{leaf_idx}_i{i}")

            # One leaf per tree
            model.addConstr(gp.quicksum(leaf_inds) == 1, name=f"{target}_oneleaf_t{t_idx}_i{i}")

            # Accumulate output
            tree_out = model.addVar(lb=-GRB.INFINITY, name=f"{target}_out_t{t_idx}_i{i}")
            model.addConstr(
                tree_out == gp.quicksum(leaf_vals[k] * leaf_inds[k] for k in range(len(leaf_inds))),
                name=f"{target}_sum_t{t_idx}_i{i}"
            )
            tree_outputs.append(tree_out)

        # Final Prediction Link
        model.addConstr(
            pred == float(base_score) + gp.quicksum(tree_outputs),
            name=f"{target}_final_link_{i}"
        )

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
        target: Target name ('epc' or 'clicks') for variable naming
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

def optimize_bids_embedded(
    X_ort=None,
    X_lr=None,
    weights_dict=None,
    budget=400,
    max_bid=50.0,
    epc_model=None,
    clicks_model=None,
    epc_xgb_paths: Optional[Tuple[Path, Path]] = None,
    clicks_xgb_paths: Optional[Tuple[Path, Path]] = None,
    epc_ridge_preproc_path: Optional[Path] = None,
    clicks_ridge_preproc_path: Optional[Path] = None,
    epc_ridge_model_path: Optional[Path] = None,
    clicks_ridge_model_path: Optional[Path] = None,
    alg_epc: str = 'lr',
    alg_clicks: str = 'lr',
    embedding_method: str = 'bert',
    bid_lower_bounds: Optional[np.ndarray] = None,
    bid_upper_bounds: Optional[np.ndarray] = None,
    exploration_lambda: float = 0.0,
    new_row_mask: Optional[np.ndarray] = None,
    save_base_formulation_to: Optional[Path] = None,
    write_lp: bool = True,
    lp_dir: Optional[Path] = None,
    solver_output_flag: int = 1,
):
    """Solve bid optimization with embedded ML models (OptiCL-style).

    This implements the simpler formulation:

        max_{b,g,f}  sum_i g_i * (f_i - b_i)
        s.t.         sum_i g_i * b_i <= B
                     g_i = Model_clicks(b_i, w_i)
                     f_i = Model_epc(b_i, w_i)
                     b_i >= 0, g_i >= 0, f_i >= 0

    Notes:
    - The budget constraint and objective are bilinear/quadratic; Gurobi is run with NonConvex=2.

    Args:
        X_ort: Feature matrix with categorical features as strings (for ORT models). If None, will use X_lr.
        X_lr: Feature matrix with categorical features one-hot encoded (for LR/Ridge weights). If None, will use X_ort.
        weights_dict: Model weights dict with keys: 'epc_const', 'epc_weights', 'clicks_const', 'clicks_weights'.
        budget: Total budget constraint B.
        max_bid: Upper bound on each bid b_i.
        epc_model: Optional IAI ORT model for EPC.
        clicks_model: Optional IAI ORT model for clicks.
    Returns:
        tuple of (model, b, f, g) where f is EPC and g is clicks.
    """
    
    # --- Parameters ---
    # Big-M parameters must be upper bounds on the maximum possible values
    M_g = 400     # Max potential clicks (Big-M for g)
    M_f = 40000   # Max potential EPC (Big-M for f). Kept large for safety.

    raw_cat_algs = {'ort', 'mort', 'xgb', 'ridge'}

    use_ort_epc = (alg_epc in {'ort', 'mort'})
    use_ort_clicks = (alg_clicks in {'ort', 'mort'})
    use_xgb_epc = (alg_epc == 'xgb')
    use_xgb_clicks = (alg_clicks == 'xgb')

    # Select appropriate X for each model
    X_epc = X_ort if (alg_epc in raw_cat_algs) else X_lr
    X_clicks = X_ort if (alg_clicks in raw_cat_algs) else X_lr

    # Use whichever X is not None for size reference
    if X_ort is None and X_lr is None:
        raise ValueError("Must provide at least one of X_ort or X_lr")
    K = len(X_ort) if X_ort is not None else len(X_lr)

    model_type_str = ""
    if use_ort_epc or use_ort_clicks or use_xgb_epc or use_xgb_clicks:
        if (use_ort_epc and use_ort_clicks) or (use_xgb_epc and use_xgb_clicks):
            model_type_str = "TREE"
        else:
            model_type_str = f"{alg_epc.upper()} (epc) + {alg_clicks.upper()} (clicks)"
    else:
        model_type_str = "LINEAR"
    
    print(f"\nSolving bid optimization with embedded {model_type_str} constraints...")
    print(f"  Budget: ${budget:,.2f}")
    print(f"  Keywords: {K}")

    # Create model
    model = gp.Model('bid_optimization')
    model.setParam('OutputFlag', int(solver_output_flag)) 
    model.setParam('TimeLimit', 300)

    # --- 0. Decision Variables ---

    # z_i: Binary "Active" switch. 1 if bidding, 0 if turned off.
    z = model.addMVar(shape=K, vtype=GRB.BINARY, name='z')

    # b_i: Bid amount
    b = model.addMVar(shape=K, lb=0, ub=max_bid, name='b')

    # f_i: predicted EPC (non-negative)
    f = model.addMVar(shape=K, lb=0, ub=M_f, name='f')

    # g_i: predicted clicks (non-negative)
    g = model.addMVar(shape=K, lb=0, ub=M_g, name='g')

    # --- 1. Raw ML Predictions (Embedded) ---
    # Create tupledict to hold prediction variables
    f_hat_vars = []
    g_hat_vars = []
    
    # Create trees directory for ORT visualization
    trees_dir = Path('opt_results/trees') if (use_ort_epc or use_ort_clicks) else None
    
    # Embed EPC model
    if use_ort_epc:
        model, f_hat_vars = embed_ort(model, epc_model, X_epc, b, target='epc', max_bid=max_bid, save_dir=trees_dir)
    elif use_xgb_epc:
        if epc_xgb_paths is None:
            raise ValueError("epc_xgb_paths is required when alg_epc == 'xgb'")
        xgb_path, preproc_path = epc_xgb_paths
        model, f_hat_vars = embed_xgb(
            model,
            xgb_model_path=xgb_path,
            preprocessor_path=preproc_path,
            X=X_epc,
            b=b,
            target='epc',
            max_bid=max_bid,
        )
    else:
        if alg_epc == 'ridge':
            if epc_ridge_preproc_path is None:
                raise ValueError("epc_ridge_preproc_path is required when alg_epc == 'ridge'")
            try:
                preprocessor_epc = joblib.load(epc_ridge_preproc_path)
            except (AttributeError, ImportError) as e:
                raise RuntimeError(
                    f"Failed to load Ridge preprocessor from {epc_ridge_preproc_path}.\n"
                    f"Error: {e}"
                ) from e

            if epc_ridge_model_path is None:
                epc_ridge_model_path = Path('models') / f'ridge_{embedding_method}_epc.joblib'
            epc_intercept, epc_coeffs, _ = _extract_ridge_coefficients(
                epc_ridge_model_path,
                epc_ridge_preproc_path,
            )

            X_epc_aligned, expected_cols = _align_X_for_preprocessor(preprocessor_epc, X_epc)
            cpc_candidates = [c for c in ("Avg. CPC", "Avg_ CPC") if c in expected_cols]
            if not cpc_candidates:
                raise KeyError(
                    f"Could not find Avg. CPC in expected columns for Ridge EPC: {expected_cols[:5]}..."
                )
            cpc_col = cpc_candidates[0]

            X0 = X_epc_aligned.copy()
            X0[cpc_col] = 0.0
            X1 = X_epc_aligned.copy()
            X1[cpc_col] = 1.0

            Z0 = preprocessor_epc.transform(X0)
            Z1 = preprocessor_epc.transform(X1)
            dZ = Z1 - Z0

            base = epc_intercept + (Z0 @ epc_coeffs)
            slope = dZ @ epc_coeffs
            base = np.asarray(base).ravel()
            slope = np.asarray(slope).ravel()

            model, f_hat_vars = embed_affine_in_bid(
                model,
                base=base,
                slope=slope,
                b=b,
                target='epc',
                link='identity',
            )
        else:
            if weights_dict is None:
                raise ValueError("weights_dict is required when epc_model is not provided")

            if 'epc_const' not in weights_dict or 'epc_weights' not in weights_dict:
                raise KeyError("weights_dict must contain 'epc_const' and 'epc_weights'")

            epc_const = weights_dict['epc_const']
            epc_weights = weights_dict['epc_weights']

            model, f_hat_vars = embed_lr(model, epc_weights, epc_const, X_epc, b, target='epc')
    
    # Embed clicks model
    if use_ort_clicks:
        # Use ORT model
        model, g_hat_vars = embed_ort(model, clicks_model, X_clicks, b, target='clicks', max_bid=max_bid, save_dir=trees_dir)
    elif use_xgb_clicks:
        if clicks_xgb_paths is None:
            raise ValueError("clicks_xgb_paths is required when alg_clicks == 'xgb'")
        xgb_path, preproc_path = clicks_xgb_paths
        model, g_hat_vars = embed_xgb(
            model,
            xgb_model_path=xgb_path,
            preprocessor_path=preproc_path,
            X=X_clicks,
            b=b,
            target='clicks',
            max_bid=max_bid,
        )
    else:
        # Use LR model
        clicks_const = weights_dict['clicks_const']
        clicks_weights = weights_dict['clicks_weights']
        if alg_clicks == 'ridge':
            if clicks_ridge_preproc_path is None:
                raise ValueError("clicks_ridge_preproc_path is required when alg_clicks == 'ridge'")
            try:
                preprocessor_clicks = joblib.load(clicks_ridge_preproc_path)
            except (AttributeError, ImportError) as e:
                raise RuntimeError(
                    f"Failed to load Ridge preprocessor from {clicks_ridge_preproc_path}.\n"
                    f"Error: {e}"
                ) from e
            
            # Extract coefficients from actual Ridge model
            if clicks_ridge_model_path is None:
                clicks_ridge_model_path = Path('models') / f'ridge_{embedding_method}_clicks.joblib'
            clicks_intercept, clicks_coeffs, _ = _extract_ridge_coefficients(clicks_ridge_model_path, clicks_ridge_preproc_path)

            X_clicks_aligned, expected_cols = _align_X_for_preprocessor(preprocessor_clicks, X_clicks)
            cpc_candidates = [c for c in ("Avg. CPC", "Avg_ CPC") if c in expected_cols]
            if not cpc_candidates:
                raise KeyError(
                    f"Could not find Avg. CPC in expected columns for Ridge clicks: {expected_cols[:5]}..."
                )
            cpc_col = cpc_candidates[0]

            X0 = X_clicks_aligned.copy()
            X0[cpc_col] = 0.0
            X1 = X_clicks_aligned.copy()
            X1[cpc_col] = 1.0

            Z0 = preprocessor_clicks.transform(X0)
            Z1 = preprocessor_clicks.transform(X1)
            dZ = Z1 - Z0

            base = clicks_intercept + (Z0 @ clicks_coeffs)
            slope = dZ @ clicks_coeffs
            base = np.asarray(base).ravel()
            slope = np.asarray(slope).ravel()

            model, g_hat_vars = embed_affine_in_bid(
                model,
                base=base,
                slope=slope,
                b=b,
                target='clicks',
                link='identity',
            )
        else:
            model, g_hat_vars = embed_lr(model, clicks_weights, clicks_const, X_clicks, b, target='clicks')
    
    model.update()
    
    # --- 2. Link prediction vars with Semi-Continuous Logic ---
    
    # Step A: Enforce Bid Limits based on Z
    # If z=0 -> b=0. If z=1 -> 0.01 <= b <= max_bid
    model.addConstr(b <= max_bid * z, name="Bid_UpperBound")
    model.addConstr(b >= 0.01 * z, name="Bid_LowerBound")

    # Optional: tighter per-row bid bounds derived from historical ranges.
    if bid_lower_bounds is not None or bid_upper_bounds is not None:
        if bid_lower_bounds is None or bid_upper_bounds is None:
            raise ValueError("bid_lower_bounds and bid_upper_bounds must both be provided")
        if len(bid_lower_bounds) != K or len(bid_upper_bounds) != K:
            raise ValueError("bid_lower_bounds/bid_upper_bounds length must match number of rows")

        for i in range(K):
            lb_i = float(bid_lower_bounds[i])
            ub_i = float(bid_upper_bounds[i])
            model.addConstr(b[i] <= ub_i * z[i], name=f"Bid_UpperBound_Hist_{i}")
            model.addConstr(b[i] >= lb_i * z[i], name=f"Bid_LowerBound_Hist_{i}")

    # Step B: Link Predictions (f, g) using Indicator Constraints
    for i in range(K):
        # Case z=0: Force final outputs (EPC/Clicks) to 0
        model.addGenConstrIndicator(z[i], 0, f[i] == 0, name=f"ForceZeroEPC_{i}")
        model.addGenConstrIndicator(z[i], 0, g[i] == 0, name=f"ForceZeroClicks_{i}")

        # Case z=1: Final outputs match the Embedded Model Variables
        model.addGenConstrIndicator(z[i], 1, f[i] == f_hat_vars[i], name=f"ActiveEPC_{i}")
        model.addGenConstrIndicator(z[i], 1, g[i] == g_hat_vars[i], name=f"ActiveClicks_{i}")

    # --- Total Budget Constraint ---
    # Create auxiliary spend variables: spend[i] = b[i] * g[i]
    spend = model.addMVar(shape=K, lb=0, ub=max_bid*M_g, name='spend')
    for i in range(K):
        model.addConstr(spend[i] == b[i] * g[i], name=f'Spend_{i}')
    
    # sum_i spend_i <= B
    model.addConstr(gp.quicksum(spend) <= budget, name='TotalBudget')

    # --- Objective ---
    # Maximize Net Profit = sum_i g_i * (f_i - b_i)
    base_obj = (g @ f) - (b @ g)
    model.setObjective(base_obj, GRB.MAXIMIZE)

    # Optionally persist a base formulation (lambda-independent) for fast re-solves.
    if save_base_formulation_to is not None:
        save_base_formulation_to.parent.mkdir(exist_ok=True, parents=True)
        model.update()
        model.write(str(save_base_formulation_to))
        print(f"\n  Base formulation saved to {save_base_formulation_to}")

    # Add exploration term: + lambda * sum_{new keywords} CPC (bid)
    if float(exploration_lambda) != 0.0:
        if new_row_mask is None:
            raise ValueError("new_row_mask must be provided when exploration_lambda != 0")
        if len(new_row_mask) != K:
            raise ValueError("new_row_mask length must match number of rows")
        new_idx = np.where(np.asarray(new_row_mask, dtype=bool))[0]
        if len(new_idx) > 0:
            explore_expr = gp.quicksum(b[int(i)] for i in new_idx)
            model.setObjective(base_obj + float(exploration_lambda) * explore_expr, GRB.MAXIMIZE)
        else:
            print("Note: exploration_lambda != 0 but no rows are marked as new; exploration term is 0")

    # --- Save Model Formulation (optional) ---
    if bool(write_lp):
        model_dir = Path('opt_results/formulations') if lp_dir is None else Path(lp_dir)
        model_dir.mkdir(exist_ok=True, parents=True)

        model.update()

        # Determine model type string for filename
        model_type_str = f"{alg_epc}_{alg_clicks}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lp_file = model_dir / f'bid_optimization_{model_type_str}_{timestamp}.lp'
        model.write(str(lp_file))
        print(f"\n  Model formulation saved to {lp_file}")
    
    # --- Optimize ---
    model.setParam("NonConvex", 2)  # Allow bilinear terms in objective/constraints
    model.optimize()

    return model, b, f, g


def extract_solution(model, b, f, g, keyword_df, keyword_idx_list, region_list, match_list, X=None):
    """Extract non-zero bids from solution with predictions.

    Args:
        model: Gurobi model object (solved)
        b: Bid decision variables
        f: Predicted EPC variables (from solver)
        g: Predicted clicks variables (from solver)
        keyword_df: DataFrame with keywords
        keyword_idx_list: List mapping rows to keyword indices
        region_list: List of regions for each row
        match_list: List of match types for each row
        X: Feature matrix (optional)
    """
    def _vals(v):
        # Supports both gurobipy.MVar and list/array of Vars
        if hasattr(v, "X"):
            return np.asarray(v.X, dtype=float)
        return np.asarray([vv.X for vv in v], dtype=float)

    b_vals = _vals(b)
    f_vals = _vals(f)
    g_vals = _vals(g)

    print(f"\nSolution Summary:")
    print(f"  Total keywords: {len(b_vals)}")
    print(f"  Total spend: ${float(np.sum(b_vals * g_vals)):,.2f}")
    print(f"  Predicted profit (+ exploration bonus): ${model.objVal:,.2f}")
    
    # Build result DataFrame with all bids
    bids_df = pd.DataFrame({
        'keyword': [keyword_df.iloc[keyword_idx_list[i]]['Keyword'] for i in range(len(b_vals))],
        'region': region_list,
        'match': match_list,
        'bid': b_vals,
        'predicted_clicks': g_vals,
        'predicted_epc': f_vals,
    })

    if 'is_new_keyword' in keyword_df.columns:
        is_new_lookup = keyword_df['is_new_keyword'].to_numpy(dtype=bool)
        bids_df['is_new_keyword'] = [bool(is_new_lookup[int(keyword_idx_list[i])]) for i in range(len(b_vals))]
    else:
        # Fallback if is_new_keyword is missing from keyword_df (e.g. older cache)
        # We can't easily reconstruct it here without the original set, so default to False
        # or try to infer if we have access to the set.
        # Ideally, keyword_df should always have it if generated by this script.
        bids_df['is_new_keyword'] = False

    # Calculate derived columns
    bids_df['conv_value'] = bids_df['predicted_clicks'] * bids_df['predicted_epc']
    bids_df['cost'] = bids_df['bid'] * bids_df['predicted_clicks']
    bids_df['profit'] = bids_df['conv_value'] - bids_df['cost']
    
    # Load historical CPC data for existing keywords
    ad_opt_data_file = Path('data/clean/ad_opt_data_bert.csv')
    ad_opt_df = pd.read_csv(str(ad_opt_data_file))
    
    # Group by keyword/region/match type and calculate stats
    hist_stats = ad_opt_df.groupby(['Keyword', 'Region', 'Match type'])['Avg. CPC'].agg(
        hist_min_cpc='min',
        hist_avg_cpc='mean',
        hist_max_cpc='max'
    ).reset_index()
    
    # Merge historical stats
    bids_df = bids_df.merge(
        hist_stats,
        left_on=['keyword', 'region', 'match'],
        right_on=['Keyword', 'Region', 'Match type'],
        how='left'
    )
    
    # Drop source columns from hist_stats
    bids_df = bids_df.drop(columns=['Keyword', 'Region', 'Match type'], errors='ignore')
    
    # Sort by bid (desc), then keyword, region, match
    sort_cols = ['profit', 'bid', 'keyword', 'region', 'match']
    sort_asc = [False, False, True, True, True]
    bids_df = bids_df.sort_values(by=sort_cols, ascending=sort_asc).reset_index(drop=True)
    
    # Reorder columns: keyword, region, match, bid, clicks, epc, conv_value, cost, profit, then any historical columns
    base_cols = ['keyword', 'region', 'match', 'bid', 'predicted_clicks', 'predicted_epc', 'conv_value', 'cost', 'profit', 'is_new_keyword']
    hist_cols = [c for c in bids_df.columns if c.startswith('hist_')]
    final_cols = base_cols + hist_cols
    # Only keep columns that exist
    final_cols = [c for c in final_cols if c in bids_df.columns]
    bids_df = bids_df[final_cols]
    
    return bids_df


def _normalize_kw(s: str) -> str:
    return str(s).strip().lower()


def _cache_dir_for_key(cache_key: str) -> Path:
    return Path('opt_results/cache') / cache_key


def _try_acquire_lock(lock_file: Path, timeout_s: int = 1800, poll_s: float = 2.0) -> bool:
    """Cross-process lock using atomic file creation.

    Returns True if lock acquired, False if timed out.
    """

    start = time.time()
    lock_file.parent.mkdir(exist_ok=True, parents=True)
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f:
                f.write(f"pid={os.getpid()} time={datetime.now().isoformat()}\n")
            return True
        except FileExistsError:
            if time.time() - start > timeout_s:
                return False
            time.sleep(poll_s)


def _release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def _load_cached_artifacts(cache_dir: Path, *, alg_epc: str, alg_clicks: str, embedding_method: str) -> dict:
    kw_file = cache_dir / f"keyword_df_{embedding_method}.csv"
    mapping_file = cache_dir / f"mapping_{embedding_method}_{alg_epc}_{alg_clicks}.csv"
    X_ort_file = cache_dir / f"X_ort_{embedding_method}_{alg_epc}_{alg_clicks}.csv"
    X_lr_file = cache_dir / f"X_lr_{embedding_method}_{alg_epc}_{alg_clicks}.csv"
    meta_file = cache_dir / "meta.json"

    if not kw_file.exists() or not mapping_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Cache artifacts missing in {cache_dir}")

    keyword_df = pd.read_csv(kw_file)
    mapping_df = pd.read_csv(mapping_file)
    kw_idx_list = mapping_df['keyword_idx'].astype(int).tolist()
    region_list = mapping_df['region'].astype(str).tolist()
    match_list = mapping_df['match_type'].astype(str).tolist()

    X_ort = pd.read_csv(X_ort_file) if X_ort_file.exists() else None
    X_lr = pd.read_csv(X_lr_file) if X_lr_file.exists() else None

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    return {
        'keyword_df': keyword_df,
        'X_ort': X_ort,
        'X_lr': X_lr,
        'kw_idx_list': kw_idx_list,
        'region_list': region_list,
        'match_list': match_list,
        'meta': meta,
    }


def _write_cached_artifacts(
    cache_dir: Path,
    *,
    keyword_df: pd.DataFrame,
    X_ort: Optional[pd.DataFrame],
    X_lr: Optional[pd.DataFrame],
    kw_idx_list: List[int],
    region_list: List[str],
    match_list: List[str],
    alg_epc: str,
    alg_clicks: str,
    embedding_method: str,
    meta: dict,
) -> None:
    cache_dir.mkdir(exist_ok=True, parents=True)
    keyword_df.to_csv(cache_dir / f"keyword_df_{embedding_method}.csv", index=False)

    mapping_df = pd.DataFrame({'keyword_idx': kw_idx_list, 'region': region_list, 'match_type': match_list})
    mapping_df.to_csv(cache_dir / f"mapping_{embedding_method}_{alg_epc}_{alg_clicks}.csv", index=False)

    if X_ort is not None:
        X_ort.to_csv(cache_dir / f"X_ort_{embedding_method}_{alg_epc}_{alg_clicks}.csv", index=False)
    if X_lr is not None:
        X_lr.to_csv(cache_dir / f"X_lr_{embedding_method}_{alg_epc}_{alg_clicks}.csv", index=False)

    with open(cache_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


def _lambda_tag(lmbda: float) -> str:
    return str(float(lmbda)).replace('.', 'p').replace('-', 'm')


def _write_bids_outputs(
    *,
    bids_df: pd.DataFrame,
    embedding_method: str,
    model_suffix: str,
    cache_key: str,
    exploration_lambda: float,
    write_legacy: bool,
) -> Tuple[Path, Path]:
    """Write bids outputs.

    Returns (sweep_file, legacy_file). legacy_file may not exist if write_legacy=False.
    """

    lam_tag = _lambda_tag(exploration_lambda)

    sweep_dir = Path('opt_results/bids') / 'lambda_sweep' / cache_key
    sweep_dir.mkdir(exist_ok=True, parents=True)
    sweep_file = sweep_dir / f"optimized_bids_{embedding_method}_{model_suffix}_lambda_{lam_tag}.csv"
    bids_df.to_csv(sweep_file, index=False)

    legacy_file = Path('opt_results/bids') / f"optimized_bids_{embedding_method}_{model_suffix}.csv"
    if write_legacy:
        legacy_file.parent.mkdir(exist_ok=True, parents=True)
        bids_df.to_csv(legacy_file, index=False)

    return sweep_file, legacy_file


def validate_solution(optimized_bids, solver_epc_preds, solver_clicks_preds, X_epc, X_clicks, epc_model, clicks_model, 
                      epc_xgb_paths, clicks_xgb_paths, epc_ridge_preproc_path=None, clicks_ridge_preproc_path=None,
                      alg_epc='lr', alg_clicks='lr', embedding_method='bert',
                      preprocessor_epc=None, preprocessor_clicks=None):
    """Validate that optimized solution predictions match original models.
    
    Re-predicts EPC and clicks using the original trained models and compares
    against the solver's predictions to detect any alignment issues.
    
    Args:
        optimized_bids: Array of optimized bid values
        solver_epc_preds: EPC predictions from the solver/embedding
        solver_clicks_preds: Clicks predictions from the solver/embedding
        X_epc: Raw features for EPC model (with Avg. CPC column)
        X_clicks: Raw features for clicks model (with Avg. CPC column)
        epc_model: Trained EPC model (IAI if ORT, None for others)
        clicks_model: Trained clicks model (IAI if ORT, None for others)
        epc_xgb_paths: Tuple of (model_path, preprocessor_path) for XGB EPC
        clicks_xgb_paths: Tuple of (model_path, preprocessor_path) for XGB clicks
        epc_ridge_preproc_path: Path to Ridge EPC preprocessor
        clicks_ridge_preproc_path: Path to Ridge clicks preprocessor
        alg_epc: Algorithm for EPC ('lr', 'ridge', 'xgb', 'ort', 'mort')
        alg_clicks: Algorithm for clicks ('lr', 'ridge', 'xgb', 'ort', 'mort')
        embedding_method: 'bert' or 'tfidf'
        preprocessor_epc: Preprocessor for XGB EPC (loaded if not provided)
        preprocessor_clicks: Preprocessor for XGB clicks (loaded if not provided)
    """
    print("\n=== Solution Validation: Comparing Solver Predictions vs. Original Models ===")
    
    # Create a copy of X with optimized bids
    X_validate_epc = X_epc.copy()
    X_validate_clicks = X_clicks.copy()
    
    # Find CPC column names
    cpc_col_epc = next((c for c in ("Avg. CPC", "Avg_ CPC") if c in X_validate_epc.columns), None)
    cpc_col_clicks = next((c for c in ("Avg. CPC", "Avg_ CPC") if c in X_validate_clicks.columns), None)
    
    if cpc_col_epc is None or cpc_col_clicks is None:
        print("WARNING: Could not find Avg. CPC column for validation")
        return
    
    # Set CPC values to optimized bids
    X_validate_epc[cpc_col_epc] = optimized_bids
    X_validate_clicks[cpc_col_clicks] = optimized_bids
    
    try:
        # Predict EPC using original model
        if alg_epc == 'xgb':
            if epc_xgb_paths is None:
                print("WARNING: XGB EPC paths not available for validation")
                epc_preds = None
            else:
                xgb_path, preproc_path = epc_xgb_paths
                if preprocessor_epc is None:
                    preprocessor_epc = joblib.load(preproc_path)
                X_epc_aligned, _ = _align_X_for_preprocessor(preprocessor_epc, X_validate_epc)
                Z_epc = preprocessor_epc.transform(X_epc_aligned)
                if sp is not None and sp.issparse(Z_epc):
                    Z_epc = Z_epc.tocsr()
                else:
                    Z_epc = np.asarray(Z_epc)
                booster_epc = xgb.Booster()
                booster_epc.load_model(str(xgb_path))
                Z_epc_dmatrix = xgb.DMatrix(data=Z_epc)
                epc_preds = booster_epc.predict(Z_epc_dmatrix)
        elif alg_epc == 'ort' or alg_epc == 'mort':
            epc_preds = epc_model.predict(X_validate_epc.values)
        elif alg_epc == 'ridge':
            # Ridge EPC validation
            if epc_ridge_preproc_path is None:
                print("WARNING: EPC Ridge preprocessor path not available for validation")
                epc_preds = None
            else:
                try:
                    preprocessor_ridge_epc = joblib.load(epc_ridge_preproc_path)
                    ridge_epc_model_path = Path('models') / f'ridge_{embedding_method}_epc.joblib'
                    epc_intercept, epc_coeffs, _ = _extract_ridge_coefficients(ridge_epc_model_path, Path(epc_ridge_preproc_path))

                    X_epc_aligned, _ = _align_X_for_preprocessor(preprocessor_ridge_epc, X_validate_epc)
                    Z_epc = preprocessor_ridge_epc.transform(X_epc_aligned)
                    # Works for both dense and sparse matrices
                    epc_preds = epc_intercept + (Z_epc @ epc_coeffs)
                    epc_preds = np.asarray(epc_preds).ravel()
                except Exception as e:
                    print(f"Note: Ridge EPC validation failed: {e}")
                    epc_preds = None
        else:
            # LR or other - not implemented, skip validation
            print(f"Note: Validation not implemented for {alg_epc} EPC model")
            epc_preds = None
        
        # Predict clicks using original model
        if alg_clicks == 'xgb':
            if clicks_xgb_paths is None:
                print("WARNING: XGB clicks paths not available for validation")
                clicks_preds = None
            else:
                xgb_path, preproc_path = clicks_xgb_paths
                if preprocessor_clicks is None:
                    preprocessor_clicks = joblib.load(preproc_path)
                X_clicks_aligned, _ = _align_X_for_preprocessor(preprocessor_clicks, X_validate_clicks)
                Z_clicks = preprocessor_clicks.transform(X_clicks_aligned)
                if sp is not None and sp.issparse(Z_clicks):
                    Z_clicks = Z_clicks.tocsr()
                else:
                    Z_clicks = np.asarray(Z_clicks)
                booster_clicks = xgb.Booster()
                booster_clicks.load_model(str(xgb_path))
                Z_clicks_dmatrix = xgb.DMatrix(data=Z_clicks)
                clicks_preds = booster_clicks.predict(Z_clicks_dmatrix)
        elif alg_clicks == 'ort' or alg_clicks == 'mort':
            clicks_preds = clicks_model.predict(X_validate_clicks.values)
        elif alg_clicks == 'ridge':
            # Ridge clicks validation
            if clicks_ridge_preproc_path is None:
                print("WARNING: Clicks Ridge preprocessor path not available for validation")
                clicks_preds = None
            else:
                try:
                    preprocessor_ridge_clicks = joblib.load(clicks_ridge_preproc_path)
                    ridge_clicks_model_path = Path('models') / f'ridge_{embedding_method}_clicks.joblib'
                    clicks_intercept, clicks_coeffs, _ = _extract_ridge_coefficients(
                        ridge_clicks_model_path, Path(clicks_ridge_preproc_path)
                    )

                    X_clicks_aligned, _ = _align_X_for_preprocessor(preprocessor_ridge_clicks, X_validate_clicks)
                    Z_clicks = preprocessor_ridge_clicks.transform(X_clicks_aligned)
                    clicks_preds = clicks_intercept + (Z_clicks @ clicks_coeffs)
                    clicks_preds = np.asarray(clicks_preds).ravel()
                except Exception as e:
                    print(f"Note: Ridge clicks validation failed: {e}")
                    clicks_preds = None
        else:
            print(f"Note: Validation not implemented for {alg_clicks} clicks model")
            clicks_preds = None
        
        # Report validation results
        if epc_preds is not None:
            print(f"\nEPC Model ({alg_epc}):")
            print(f"  Model predictions range: [{epc_preds.min():.6f}, {epc_preds.max():.6f}]")
            print(f"  Mean: {epc_preds.mean():.6f}, Std: {epc_preds.std():.6f}")
            if np.isnan(epc_preds).any() or np.isinf(epc_preds).any():
                print(f"  ⚠️  WARNING: Found NaN/Inf in EPC model predictions!")
            else:
                print(f"  ✓ EPC model predictions valid")
            
            # Compare with solver predictions
            print(f"  Solver predictions range: [{solver_epc_preds.min():.6f}, {solver_epc_preds.max():.6f}]")
            print(f"  Mean: {solver_epc_preds.mean():.6f}, Std: {solver_epc_preds.std():.6f}")
            
            # Calculate differences
            epc_diff = epc_preds - solver_epc_preds
            epc_mae = np.abs(epc_diff).mean()
            epc_rmse = np.sqrt((epc_diff ** 2).mean())
            epc_max_abs = np.abs(epc_diff).max()
            print(f"  Difference (model - solver):")
            print(f"    MAE: {epc_mae:.6f}, RMSE: {epc_rmse:.6f}")
            print(f"    Max abs: {epc_max_abs:.12f}")

            if solver_epc_preds.min() >= -1e-9 and epc_preds.min() < -1e-9:
                epc_clipped = np.maximum(epc_preds, 0.0)
                epc_diff_clip = epc_clipped - solver_epc_preds
                epc_mae_clip = np.abs(epc_diff_clip).mean()
                epc_rmse_clip = np.sqrt((epc_diff_clip ** 2).mean())
                print(f"  (Also vs clipped model preds, since optimization enforces f>=0)")
                print(f"    MAE: {epc_mae_clip:.6f}, RMSE: {epc_rmse_clip:.6f}")

            # Use robust tolerance-based comparison to avoid false positives when std == 0.
            # Note: solver predictions can differ slightly due to numerical tolerances.
            if not np.allclose(epc_preds, solver_epc_preds, rtol=1e-6, atol=1e-6):
                print(f"  ⚠️  WARNING: Large difference between model and solver predictions!")
            else:
                print(f"  ✓ EPC predictions align well")
        
        if clicks_preds is not None:
            print(f"\nClicks Model ({alg_clicks}):")
            print(f"  Model predictions range: [{clicks_preds.min():.6f}, {clicks_preds.max():.6f}]")
            print(f"  Mean: {clicks_preds.mean():.6f}, Std: {clicks_preds.std():.6f}")
            if np.isnan(clicks_preds).any() or np.isinf(clicks_preds).any():
                print(f"  ⚠️  WARNING: Found NaN/Inf in clicks model predictions!")
            else:
                print(f"  ✓ Clicks model predictions valid")
            
            # Compare with solver predictions
            print(f"  Solver predictions range: [{solver_clicks_preds.min():.6f}, {solver_clicks_preds.max():.6f}]")
            print(f"  Mean: {solver_clicks_preds.mean():.6f}, Std: {solver_clicks_preds.std():.6f}")
            
            # Calculate differences
            clicks_diff = clicks_preds - solver_clicks_preds
            clicks_mae = np.abs(clicks_diff).mean()
            clicks_rmse = np.sqrt((clicks_diff ** 2).mean())
            clicks_max_abs = np.abs(clicks_diff).max()
            print(f"  Difference (model - solver):")
            print(f"    MAE: {clicks_mae:.6f}, RMSE: {clicks_rmse:.6f}")
            print(f"    Max abs: {clicks_max_abs:.12f}")

            if solver_clicks_preds.min() >= -1e-9 and clicks_preds.min() < -1e-9:
                clicks_clipped = np.maximum(clicks_preds, 0.0)
                clicks_diff_clip = clicks_clipped - solver_clicks_preds
                clicks_mae_clip = np.abs(clicks_diff_clip).mean()
                clicks_rmse_clip = np.sqrt((clicks_diff_clip ** 2).mean())
                print(f"  (Also vs clipped model preds, since optimization enforces g>=0)")
                print(f"    MAE: {clicks_mae_clip:.6f}, RMSE: {clicks_rmse_clip:.6f}")

            if not np.allclose(clicks_preds, solver_clicks_preds, rtol=1e-6, atol=1e-6):
                print(f"  ⚠️  WARNING: Large difference between model and solver predictions!")
            else:
                print(f"  ✓ Clicks predictions align well")
        
        # Save validation predictions
        if epc_preds is not None or clicks_preds is not None:
            validation_df = pd.DataFrame({'bid': optimized_bids})
            if epc_preds is not None:
                validation_df['epc_model_pred'] = epc_preds
                validation_df['epc_solver_pred'] = solver_epc_preds
            if clicks_preds is not None:
                validation_df['clicks_model_pred'] = clicks_preds
                validation_df['clicks_solver_pred'] = solver_clicks_preds
            
            val_dir = Path('opt_results/validation')
            val_dir.mkdir(exist_ok=True, parents=True)
            val_file = val_dir / f'solution_validation_{embedding_method}_{alg_epc}_{alg_clicks}.csv'
            validation_df.to_csv(val_file, index=False)
            print(f"\n✓ Validation predictions saved to {val_file}")
        
    except Exception as e:
        print(f"ERROR during solution validation: {e}")
        import traceback
        traceback.print_exc()


def _predict_linear_weights_at_bid0(*, X: pd.DataFrame, const: float, weights: dict) -> np.ndarray:
    """Predict using the repo's exported linear-weight dict format at bid=0.

    Assumes X is already in the same feature space as the weights (i.e. one-hot
    already applied if needed). The bid feature ('Avg. CPC'/'Avg_ CPC') is treated
    as 0 here.
    """

    preds = np.full(shape=(len(X),), fill_value=float(const), dtype=float)

    # Find CPC column in X, but treat bid as zero regardless of X value.
    cpc_cols = [c for c in ("Avg. CPC", "Avg_ CPC") if c in X.columns]
    cpc_col = cpc_cols[0] if cpc_cols else None

    for feature, weight in weights.items():
        if feature in ["Avg. CPC", "Avg_ CPC"]:
            # bid=0 => contributes nothing
            continue

        if isinstance(weight, dict):
            for level_name, level_weight in weight.items():
                ohe_col_name = f"{feature}_{level_name}"
                if ohe_col_name in X.columns:
                    preds += float(level_weight) * X[ohe_col_name].to_numpy(dtype=float)
        else:
            if feature == cpc_col:
                continue
            if feature in X.columns:
                preds += float(weight) * X[feature].to_numpy(dtype=float)

    return preds


def _predict_ridge_at_bid0(*, X_raw: pd.DataFrame, ridge_preproc_path: Path, ridge_model_path: Path) -> np.ndarray:
    """Predict Ridge outputs at bid=0 using saved preprocessor + model coefficients."""

    preprocessor = joblib.load(ridge_preproc_path)
    intercept, coeffs, _ = _extract_ridge_coefficients(ridge_model_path, ridge_preproc_path)

    X_aligned, expected_cols = _align_X_for_preprocessor(preprocessor, X_raw)
    cpc_candidates = [c for c in ("Avg. CPC", "Avg_ CPC") if c in expected_cols]
    if cpc_candidates:
        X_aligned = X_aligned.copy()
        X_aligned[cpc_candidates[0]] = 0.0

    Z = preprocessor.transform(X_aligned)
    preds = intercept + (Z @ coeffs)
    return np.asarray(preds).ravel()


def _predict_xgb_at_bid0(*, X_raw: pd.DataFrame, xgb_model_path: Path, preproc_path: Path) -> np.ndarray:
    """Predict XGB outputs at bid=0 using saved preprocessor + booster."""

    preprocessor = joblib.load(preproc_path)
    X_aligned, expected_cols = _align_X_for_preprocessor(preprocessor, X_raw)
    cpc_candidates = [c for c in ("Avg. CPC", "Avg_ CPC") if c in expected_cols]
    if cpc_candidates:
        X_aligned = X_aligned.copy()
        X_aligned[cpc_candidates[0]] = 0.0

    Z = preprocessor.transform(X_aligned)
    if sp is not None and sp.issparse(Z):
        Z = Z.tocsr()
    else:
        Z = np.asarray(Z)

    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    dm = xgb.DMatrix(data=Z)
    return np.asarray(booster.predict(dm)).ravel()


def _predict_iai_tree_at_bid0(*, X_raw: pd.DataFrame, ort_model) -> np.ndarray:
    """Predict ORT/MORT outputs at bid=0 by setting CPC feature to zero."""

    X0 = X_raw.copy()
    cpc_cols = [c for c in ("Avg. CPC", "Avg_ CPC") if c in X0.columns]
    if cpc_cols:
        X0[cpc_cols[0]] = 0.0
    return np.asarray(ort_model.predict(X0.values)).ravel()


def filter_rows_feasible_at_bid0(
    *,
    X_ort: Optional[pd.DataFrame],
    X_lr: Optional[pd.DataFrame],
    kw_idx_list: List[int],
    region_list: List[str],
    match_list: List[str],
    alg_epc: str,
    alg_clicks: str,
    embedding_method: str,
    weights_dict: Optional[dict],
    epc_model,
    clicks_model,
    epc_xgb_paths: Optional[Tuple[Path, Path]],
    clicks_xgb_paths: Optional[Tuple[Path, Path]],
    epc_ridge_preproc_path: Optional[Path],
    clicks_ridge_preproc_path: Optional[Path],
    tol: float = 1e-9,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[int], List[str], List[str]]:
    """Drop rows where either EPC or clicks at bid=0 is negative.

    This guarantees the optimization has a feasible solution with b=0.
    """

    raw_cat_algs = {"ort", "mort", "xgb", "ridge"}

    X_epc = X_ort if (alg_epc in raw_cat_algs) else X_lr
    X_clicks = X_ort if (alg_clicks in raw_cat_algs) else X_lr

    if X_epc is None or X_clicks is None:
        raise ValueError("Could not determine X for EPC/clicks filtering (X_ort/X_lr missing)")

    if len(X_epc) != len(kw_idx_list) or len(X_clicks) != len(kw_idx_list):
        raise ValueError("Row count mismatch between X and mapping lists")

    # --- Predict EPC at bid=0 ---
    if alg_epc in {"ort", "mort"}:
        if epc_model is None:
            raise ValueError("epc_model required to filter ORT/MORT EPC")
        epc0 = _predict_iai_tree_at_bid0(X_raw=X_epc, ort_model=epc_model)
    elif alg_epc == "xgb":
        if epc_xgb_paths is None:
            raise ValueError("epc_xgb_paths required to filter XGB EPC")
        xgb_path, preproc_path = epc_xgb_paths
        epc0 = _predict_xgb_at_bid0(X_raw=X_epc, xgb_model_path=xgb_path, preproc_path=preproc_path)
    elif alg_epc == "ridge":
        if epc_ridge_preproc_path is None:
            raise ValueError("epc_ridge_preproc_path required to filter Ridge EPC")
        ridge_path = Path("models") / f"ridge_{embedding_method}_epc.joblib"
        epc0 = _predict_ridge_at_bid0(X_raw=X_epc, ridge_preproc_path=Path(epc_ridge_preproc_path), ridge_model_path=ridge_path)
    else:
        if weights_dict is None:
            raise ValueError("weights_dict required to filter LR EPC")
        epc0 = _predict_linear_weights_at_bid0(
            X=X_epc,
            const=float(weights_dict["epc_const"]),
            weights=weights_dict["epc_weights"],
        )

    # --- Predict clicks at bid=0 ---
    if alg_clicks in {"ort", "mort"}:
        if clicks_model is None:
            raise ValueError("clicks_model required to filter ORT/MORT clicks")
        clicks0 = _predict_iai_tree_at_bid0(X_raw=X_clicks, ort_model=clicks_model)
    elif alg_clicks == "xgb":
        if clicks_xgb_paths is None:
            raise ValueError("clicks_xgb_paths required to filter XGB clicks")
        xgb_path, preproc_path = clicks_xgb_paths
        clicks0 = _predict_xgb_at_bid0(X_raw=X_clicks, xgb_model_path=xgb_path, preproc_path=preproc_path)
    elif alg_clicks == "ridge":
        if clicks_ridge_preproc_path is None:
            raise ValueError("clicks_ridge_preproc_path required to filter Ridge clicks")
        ridge_path = Path("models") / f"ridge_{embedding_method}_clicks.joblib"
        clicks0 = _predict_ridge_at_bid0(
            X_raw=X_clicks,
            ridge_preproc_path=Path(clicks_ridge_preproc_path),
            ridge_model_path=ridge_path,
        )
    else:
        if weights_dict is None:
            raise ValueError("weights_dict required to filter LR clicks")
        clicks0 = _predict_linear_weights_at_bid0(
            X=X_clicks,
            const=float(weights_dict["clicks_const"]),
            weights=weights_dict["clicks_weights"],
        )

    mask = (epc0 >= -tol) & (clicks0 >= -tol)
    keep = int(mask.sum())
    drop = int(len(mask) - keep)
    print(f"\nBid=0 feasibility filter: keeping {keep}/{len(mask)} rows (dropping {drop} with EPC<0 or clicks<0 at bid=0)")

    if keep == 0:
        raise RuntimeError("All rows were filtered out as infeasible at bid=0 (no feasible combos remain)")

    def _filt_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        return df.loc[mask].reset_index(drop=True)

    X_ort_f = _filt_df(X_ort)
    X_lr_f = _filt_df(X_lr)
    kw_idx_f = [kw_idx_list[i] for i in range(len(mask)) if bool(mask[i])]
    region_f = [region_list[i] for i in range(len(mask)) if bool(mask[i])]
    match_f = [match_list[i] for i in range(len(mask)) if bool(mask[i])]

    return X_ort_f, X_lr_f, kw_idx_f, region_f, match_f

def _compute_training_stats_and_bounds(
    csv_path: Path,
    exclude_cols: Optional[Set[str]] = None,
    allowed_cols: Optional[Set[str]] = None,
) -> Tuple[dict, float]:
    """Computes Mean/Covariance for Mahalanobis filter and finds Max CPC."""
    if exclude_cols is None:
        # Exclude bid itself and common *outcome/target* columns from historical ads data.
        # The trust ellipsoid should be based on predictor features only.
        exclude_cols = {
            "Avg_ CPC",
            "Avg. CPC",
            "Clicks",
            "Cost",
            "EPC",
            "Conv_ value",
        }
        
    print(f"Loading training data from {csv_path} for Trust Ellipsoid...")
    df = pd.read_csv(csv_path)
    # Keep feature naming consistent with the optimizer feature matrix.
    # (create_feature_matrix replaces '.' -> '_')
    df.columns = df.columns.str.replace('.', '_', regex=False)
    
    # 1. Find Max CPC for capping the optimizer (Safety Rail)
    cpc_col = next((c for c in ["Avg_ CPC", "Avg. CPC"] if c in df.columns), None)
    max_cpc = float('inf')
    if cpc_col:
        # robust max (ignoring infs)
        valid_cpcs = pd.to_numeric(df[cpc_col], errors='coerce')
        valid_cpcs = valid_cpcs[np.isfinite(valid_cpcs)]
        if not valid_cpcs.empty:
            max_cpc = valid_cpcs.max()
    
    # 2. Prepare Numeric Data for Covariance
    # Convert everything to numeric where possible; non-numeric becomes NaN.
    # This makes the covariance stats robust to currency/object columns.
    df_num = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    numeric_cols = [c for c in df_num.columns if df_num[c].notna().any()]
    # Drop columns we don't want in the distance calculation (like the bid itself)
    cols_to_use = [c for c in numeric_cols if c not in exclude_cols]

    # If the optimizer feature matrix doesn't have a training column, we cannot include it
    # (otherwise the Mahalanobis calc will KeyError on X_check[stat_cols]).
    if allowed_cols is not None:
        cols_to_use = [c for c in cols_to_use if c in allowed_cols]

    # Drop rows with NaNs/Infs to ensure clean covariance calculation
    data = df_num[cols_to_use].replace([np.inf, -np.inf], np.nan).dropna()
    
    if data.empty:
        raise ValueError("Training data has no valid numeric rows for covariance calculation.")

    # 3. Fit Covariance
    # Support_fraction=1.0 uses all data (Empirical). 
    # If your training data has many outliers, consider MinCovDet() instead.
    cov_estimator = EmpiricalCovariance().fit(data)
    
    stats = {
        'mean': cov_estimator.location_,
        'inv_cov': cov_estimator.precision_,
        'columns': cols_to_use
    }
    
    return stats, max_cpc

def _compute_training_numeric_ranges(training_csv: Path) -> Dict[str, Tuple[float, float]]:
    """Compute per-column (min,max) ranges from the training dataset.

    Uses numeric columns only. Column names are normalized to match the optimizer's
    feature naming ('.' -> '_').
    """

    df = pd.read_csv(str(training_csv))
    df.columns = df.columns.str.replace('.', '_', regex=False)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ranges: Dict[str, Tuple[float, float]] = {}
    for c in num_cols:
        s = pd.to_numeric(df[c], errors='coerce')
        lo = float(np.nanmin(s.to_numpy(dtype=float)))
        hi = float(np.nanmax(s.to_numpy(dtype=float)))
        if np.isfinite(lo) and np.isfinite(hi):
            ranges[c] = (lo, hi)
    return ranges

def calculate_training_stats(X_train: pd.DataFrame, exclude_cols={"Avg. CPC"}) -> dict:
    # 1. Select numeric features
    Xn = X_train.copy()
    Xn.columns = Xn.columns.str.replace('.', '_', regex=False)
    # Convert to numeric where possible
    Xn = Xn.apply(lambda s: pd.to_numeric(s, errors='coerce'))
    numeric_cols = [c for c in Xn.columns if Xn[c].notna().any()]
    cols_to_use = [c for c in numeric_cols if c not in exclude_cols]

    data = Xn[cols_to_use].replace([np.inf, -np.inf], np.nan).dropna()
    
    # 2. Fit Covariance (Robust to outliers is better, e.g., MinCovDet, but Empirical is faster)
    emp_cov = EmpiricalCovariance().fit(data)
    
    return {
        'mean': emp_cov.location_,
        'inv_cov': emp_cov.precision_, # Precision is the inverse covariance
        'columns': cols_to_use
    }

def apply_trust_ellipsoid_filter(
    *,
    X_ort: Optional[pd.DataFrame],
    X_lr: Optional[pd.DataFrame],
    kw_idx_list: List[int],
    region_list: List[str],
    match_list: List[str],
    training_stats: Dict[str, np.ndarray],
    p_value_threshold: float = 0.99,
    exclude_cols: Optional[Set[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[int], List[str], List[str]]:
    """Enforce a "Trust Ellipsoid" (Mahalanobis Distance) filter.
    
    This is tighter than a min/max box. It filters out points that are statistically 
    far from the training distribution center, accounting for feature correlations.

    Args:
        training_stats: Dict containing calculated artifacts from training data:
            - 'mean': numpy array of shape (n_features,)
            - 'inv_cov': numpy array of shape (n_features, n_features) (Inverse Covariance)
            - 'columns': list of column names used to calculate these stats
        p_value_threshold: Cutoff for the Chi-Square distribution (e.g., 0.99 keeps 99% of normal data).
    """

    if exclude_cols is None:
        exclude_cols = {"Avg_ CPC", "Avg. CPC", "Clicks", "EPC", "Conv_ Value", "Cost"}

    X_check = X_ort if X_ort is not None else X_lr
    if X_check is None:
        raise ValueError("Trust filter requires at least one of X_ort or X_lr")
        
    # 1. Identify valid numeric columns that align with training stats
    stat_cols = training_stats.get('columns')
    if stat_cols is None:
        raise ValueError("training_stats must contain a 'columns' key listing the feature order.")
        
    # Filter down to only the columns we want to check (removing IDs, decision vars, etc)
    # We must ensure the order matches the covariance matrix exactly.
    # Keep naming consistent with the optimizer matrix.
    X_check = X_check.copy()
    X_check.columns = X_check.columns.str.replace('.', '_', regex=False)

    try:
        X_numeric = X_check[stat_cols].apply(pd.to_numeric, errors='coerce')
    except KeyError:
        missing = set(stat_cols) - set(X_check.columns)
        print(f"Warning: Missing columns for Mahalanobis calculation: {missing}")
        return X_ort, X_lr, kw_idx_list, region_list, match_list

    # 2. Prepare Math Artifacts
    mu = training_stats['mean']       # Shape: (D,)
    inv_cov = training_stats['inv_cov'] # Shape: (D, D)
    
    # 3. Calculate Mahalanobis Distance for every row
    # D^2 = (x - mu)^T * Sigma^-1 * (x - mu)
    # We implement a vectorized version for speed:
    
    Xv = X_numeric.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    finite_rows = np.isfinite(Xv).all(axis=1)

    # Treat any non-finite row as out-of-distribution (drop it).
    mahal_dist_sq = np.full(shape=(Xv.shape[0],), fill_value=np.inf, dtype=float)
    if finite_rows.any():
        diff = Xv[finite_rows] - mu
        # (N, D) dot (D, D) -> (N, D)
        left_term = np.dot(diff, inv_cov)
        # Row-wise dot product: sum( (N, D) * (N, D), axis=1 ) -> (N,)
        mahal_dist_sq[finite_rows] = np.sum(left_term * diff, axis=1)

    # 4. Determine Threshold (Chi-Squared cut-off)
    # Degrees of freedom = number of features used in distance calc
    df = len(stat_cols)
    # The critical value such that p_value_threshold % of data falls inside
    critical_value = chi2.ppf(p_value_threshold, df)

    # 5. Create Mask
    # Keep if distance^2 <= critical_value
    mask = mahal_dist_sq <= critical_value

    keep = int(mask.sum())
    drop = int(len(mask) - keep)
    print(f"\nTrust-Ellipsoid: keeping {keep}/{len(mask)} rows (dropping {drop} extrapolations)")
    print(f"  (Threshold Chi2: {critical_value:.2f}, Max Obs Dist: {mahal_dist_sq.max():.2f})")

    if keep == 0:
        raise RuntimeError("All rows were dropped by the trust filter!")

    # 6. Apply Filter
    def _filt_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None: return None
        return df.loc[mask].reset_index(drop=True)

    X_ort_f = _filt_df(X_ort)
    X_lr_f = _filt_df(X_lr)
    kw_idx_f = [kw_idx_list[i] for i in range(len(mask)) if bool(mask[i])]
    region_f = [region_list[i] for i in range(len(mask)) if bool(mask[i])]
    match_f = [match_list[i] for i in range(len(mask)) if bool(mask[i])]

    return X_ort_f, X_lr_f, kw_idx_f, region_f, match_f

def apply_trust_box_filter(
    *,
    X_ort: Optional[pd.DataFrame],
    X_lr: Optional[pd.DataFrame],
    kw_idx_list: List[int],
    region_list: List[str],
    match_list: List[str],
    training_ranges: Dict[str, Tuple[float, float]],
    mode: str = "drop",
    exclude_cols: Optional[set] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[int], List[str], List[str]]:
    """Enforce "trust box" feature ranges based on training min/max.

    mode:
      - 'drop': drop any rows where any in-scope numeric feature is outside [min,max]

    Note: We exclude the bid feature ('Avg. CPC') here because bid is a decision variable.
    Bid range is handled separately by capping max_bid.
    """

    if exclude_cols is None:
        exclude_cols = {"Avg_ CPC", "Avg. CPC"}

    X_check = X_ort if X_ort is not None else X_lr
    if X_check is None:
        raise ValueError("Trust-box filter requires at least one of X_ort or X_lr")
    if len(X_check) != len(kw_idx_list):
        raise ValueError("Row count mismatch between X and mapping lists")

    if mode != "drop":
        raise ValueError(f"Unsupported trust-box mode: {mode} (supported: drop)")

    numeric_cols = X_check.select_dtypes(include=[np.number]).columns.tolist()
    in_scope = [c for c in numeric_cols if c in training_ranges and c not in exclude_cols]

    if not in_scope:
        print("\nTrust-box: no overlapping numeric columns found; skipping range filter")
        return X_ort, X_lr, kw_idx_list, region_list, match_list

    mask = np.ones(len(X_check), dtype=bool)
    for c in in_scope:
        lo, hi = training_ranges[c]
        v = pd.to_numeric(X_check[c], errors='coerce').to_numpy(dtype=float)
        # NaN counts as out-of-distribution.
        ok = np.isfinite(v) & (v >= lo) & (v <= hi)
        mask &= ok

    keep = int(mask.sum())
    drop = int(len(mask) - keep)
    print(f"\nTrust-box (drop): keeping {keep}/{len(mask)} rows (dropping {drop} out-of-range)")

    if keep == 0:
        raise RuntimeError("All rows were dropped by the trust-box range filter")

    def _filt_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        return df.loc[mask].reset_index(drop=True)

    X_ort_f = _filt_df(X_ort)
    X_lr_f = _filt_df(X_lr)
    kw_idx_f = [kw_idx_list[i] for i in range(len(mask)) if bool(mask[i])]
    region_f = [region_list[i] for i in range(len(mask)) if bool(mask[i])]
    match_f = [match_list[i] for i in range(len(mask)) if bool(mask[i])]

    return X_ort_f, X_lr_f, kw_idx_f, region_f, match_f


def _compute_clicks_bid_bounds_from_history(
    *,
    training_csv: Path,
    keyword_df: pd.DataFrame,
    kw_idx_list: List[int],
    region_list: List[str],
    match_list: List[str],
    embedding_method: str,
    max_bid_cap: float,
    min_active_bid: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Compute per-row bid (lb, ub) from historical Avg CPC, with nearest fallback.

    - If exact (keyword, match, region) exists: use its historical min/max.
    - Else: use nearest available (keyword, match, region) where nearest is by
      cosine similarity in embedding space, preferring:
        1) same region+match
        2) same region
        3) any
    """

    K = len(kw_idx_list)
    lbs = np.full(K, float(min_active_bid), dtype=float)
    # Default to tightest bounds if no history exists
    ubs = np.full(K, float(min_active_bid), dtype=float)

    debug_rows: List[Dict[str, Any]] = []

    def _kw_reg_mt(i: int) -> Tuple[str, str, str]:
        kw = str(keyword_df.iloc[kw_idx_list[i]]["Keyword"]).lower().strip()
        reg = str(region_list[i]).lower().strip()
        mt = str(match_list[i]).lower().strip()
        return kw, reg, mt

    def _add_debug(
        *,
        i: int,
        source: str,
        lb: float,
        ub: float,
        nn_keyword: Optional[str] = None,
        nn_region: Optional[str] = None,
        nn_match: Optional[str] = None,
        nn_hist_min: Optional[float] = None,
        nn_hist_max: Optional[float] = None,
        cosine_sim: Optional[float] = None,
        tier: Optional[str] = None,
    ) -> None:
        kw, reg, mt = _kw_reg_mt(i)
        debug_rows.append(
            {
                "keyword": kw,
                "region": reg,
                "match_type": mt,
                "lb": float(lb),
                "ub": float(ub),
                "source": source,
                "nn_keyword": nn_keyword,
                "nn_region": nn_region,
                "nn_match_type": nn_match,
                "nn_hist_min": nn_hist_min,
                "nn_hist_max": nn_hist_max,
                "cosine_sim": cosine_sim,
                "tier": tier,
            }
        )

    if not training_csv.exists():
        print(f"Warning: historical training CSV not found: {training_csv}; skipping historical bid bounds")
        for i in range(K):
            _add_debug(i=i, source="no_training_csv", lb=lbs[i], ub=ubs[i])
        return lbs, ubs, pd.DataFrame(debug_rows)

    hist = pd.read_csv(training_csv)
    hist.columns = hist.columns.str.replace('.', '_', regex=False)

    required_cols = {"Keyword", "Region", "Match type"}
    if not required_cols.issubset(set(hist.columns)):
        print(
            f"Warning: training CSV missing required columns {required_cols - set(hist.columns)}; skipping historical bid bounds"
        )
        for i in range(K):
            _add_debug(i=i, source="missing_required_cols", lb=lbs[i], ub=ubs[i])
        return lbs, ubs, pd.DataFrame(debug_rows)

    cpc_col = next((c for c in ("Avg_ CPC", "Avg. CPC") if c in hist.columns), None)
    if cpc_col is None:
        print("Warning: training CSV has no Avg CPC column; skipping historical bid bounds")
        for i in range(K):
            _add_debug(i=i, source="missing_avg_cpc", lb=lbs[i], ub=ubs[i])
        return lbs, ubs, pd.DataFrame(debug_rows)

    hist = hist.copy()
    hist["Keyword_join"] = hist["Keyword"].astype(str).str.lower().str.strip()
    hist["Region_join"] = hist["Region"].astype(str).str.lower().str.strip()
    hist["Match_join"] = hist["Match type"].astype(str).str.lower().str.strip()
    hist["Bid"] = pd.to_numeric(hist[cpc_col], errors="coerce")
    hist = hist[np.isfinite(hist["Bid"])].copy()

    # Positive-only to avoid lb=0 due to placeholders/missing.
    hist = hist[hist["Bid"] > 0].copy()
    if hist.empty:
        print("Warning: training CSV has no positive Avg CPC values; skipping historical bid bounds")
        for i in range(K):
            _add_debug(i=i, source="no_positive_cpc", lb=lbs[i], ub=ubs[i])
        return lbs, ubs, pd.DataFrame(debug_rows)

    bounds_df = (
        hist.groupby(["Keyword_join", "Region_join", "Match_join"], as_index=False)["Bid"]
        .agg(hist_min="min", hist_max="max")
        .reset_index(drop=True)
    )

    exact_map = {
        (r.Keyword_join, r.Region_join, r.Match_join): (float(r.hist_min), float(r.hist_max))
        for r in bounds_df.itertuples(index=False)
    }

    # Embeddings source for nearest-neighbor fallback.
    # IMPORTANT: nearest-neighbor requires vectors for *historical* keywords too.
    # The optimizer's keyword_df only contains the current run's keywords, so it
    # is usually insufficient as a candidate embedding store.
    emb_cols = [c for c in keyword_df.columns if c.startswith(f"{embedding_method}_")]
    if not emb_cols:
        # Exact-only fallback
        for i in range(K):
            kw, reg, mt = _kw_reg_mt(i)
            exact = exact_map.get((kw, reg, mt))
            if exact is None:
                _add_debug(i=i, source="no_embedding_cols_no_exact", lb=lbs[i], ub=ubs[i])
                continue
            lo, hi = exact
            lbs[i] = max(float(min_active_bid), lo)
            ubs[i] = min(float(max_bid_cap), hi)
            _add_debug(
                i=i,
                source="exact",
                lb=lbs[i],
                ub=ubs[i],
                nn_keyword=kw,
                nn_region=reg,
                nn_match=mt,
                nn_hist_min=float(lo),
                nn_hist_max=float(hi),
            )
        ubs = np.maximum(ubs, lbs)
        return lbs, ubs, pd.DataFrame(debug_rows)

    # Build an embedding lookup from a persisted "unique_keyword_embeddings" table
    # (covers the historical training universe) + the current keyword_df (covers
    # brand-new keywords).
    emb_lookup: Dict[str, np.ndarray] = {}

    def _add_embeddings_from_df(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if "Keyword" not in df.columns:
            return
        df_use = df.copy()
        df_use["Keyword_join"] = df_use["Keyword"].astype(str).str.lower().str.strip()
        cols = [c for c in emb_cols if c in df_use.columns]
        if not cols:
            return
        for r in df_use[["Keyword_join"] + cols].itertuples(index=False):
            v = np.asarray([getattr(r, c) for c in cols], dtype=float)
            if not np.isfinite(v).all():
                continue
            emb_lookup[str(getattr(r, "Keyword_join"))] = v

    # Prefer a persisted embedding table that matches the training CSV's pipeline.
    # IMPORTANT: Embeddings from different pipelines (e.g., different SVD fits) are
    # not comparable with cosine similarity; even identical keywords can end up with
    # strongly negative cosine similarity.
    emb_file_candidates = [
        training_csv.parent / f"unique_keyword_embeddings_{embedding_method}.csv",
        Path("data") / "clean" / f"unique_keyword_embeddings_{embedding_method}.csv",
        Path("data") / "embeddings" / f"unique_keyword_embeddings_{embedding_method}.csv",
    ]
    for f in emb_file_candidates:
        if f.exists():
            try:
                emb_df = load_embeddings(f, embedding_method=embedding_method)
                _add_embeddings_from_df(emb_df)
                break
            except Exception as e:
                print(f"Warning: failed to load embeddings from {f}: {type(e).__name__}: {e}")

    # Also include the current run's keyword_df so brand-new keywords can be queried.
    _add_embeddings_from_df(keyword_df)

    # Candidate rows w/ vectors
    cand_rows = []
    cand_vecs = []
    for r in bounds_df.itertuples(index=False):
        v = emb_lookup.get(str(r.Keyword_join))
        if v is None:
            continue
        cand_rows.append(r)
        cand_vecs.append(v)

    if not cand_rows:
        # Exact-only
        for i in range(K):
            kw, reg, mt = _kw_reg_mt(i)
            exact = exact_map.get((kw, reg, mt))
            if exact is None:
                _add_debug(i=i, source="no_candidate_vectors_no_exact", lb=lbs[i], ub=ubs[i])
                continue
            lo, hi = exact
            lbs[i] = max(float(min_active_bid), lo)
            ubs[i] = min(float(max_bid_cap), hi)
            _add_debug(
                i=i,
                source="exact",
                lb=lbs[i],
                ub=ubs[i],
                nn_keyword=kw,
                nn_region=reg,
                nn_match=mt,
                nn_hist_min=float(lo),
                nn_hist_max=float(hi),
            )
        ubs = np.maximum(ubs, lbs)
        return lbs, ubs, pd.DataFrame(debug_rows)

    cand_df = pd.DataFrame(
        {
            "Keyword_join": [r.Keyword_join for r in cand_rows],
            "Region_join": [r.Region_join for r in cand_rows],
            "Match_join": [r.Match_join for r in cand_rows],
            "hist_min": [float(r.hist_min) for r in cand_rows],
            "hist_max": [float(r.hist_max) for r in cand_rows],
        }
    )
    V = np.vstack(cand_vecs)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    reg_arr = cand_df["Region_join"].to_numpy(dtype=str)
    mt_arr = cand_df["Match_join"].to_numpy(dtype=str)

    for i in range(K):
        kw, reg, mt = _kw_reg_mt(i)

        exact = exact_map.get((kw, reg, mt))
        if exact is not None:
            lo, hi = exact
            lbs[i] = max(float(min_active_bid), lo)
            ubs[i] = min(float(max_bid_cap), hi)
            _add_debug(
                i=i,
                source="exact",
                lb=lbs[i],
                ub=ubs[i],
                nn_keyword=kw,
                nn_region=reg,
                nn_match=mt,
                nn_hist_min=float(lo),
                nn_hist_max=float(hi),
            )
            continue

        x = emb_lookup.get(kw)
        if x is None:
            _add_debug(i=i, source="no_embedding_for_keyword", lb=lbs[i], ub=ubs[i])
            continue
        x = np.asarray(x, dtype=float)
        x = x / (np.linalg.norm(x) + 1e-12)

        tier1 = (reg_arr == reg) & (mt_arr == mt)
        tier2 = (reg_arr == reg)
        if tier1.any():
            idxs = np.where(tier1)[0]
            tier = "same_region_and_match"
        elif tier2.any():
            idxs = np.where(tier2)[0]
            tier = "same_region"
        else:
            # Fall back to any region/match.
            idxs = np.arange(len(cand_df), dtype=int)
            tier = "any"

        sims = Vn[idxs] @ x
        best_local = int(np.argmax(sims))
        best = int(idxs[best_local])
        best_sim = float(sims[best_local])
        
        raw_lo = float(cand_df.loc[best, "hist_min"])
        raw_hi = float(cand_df.loc[best, "hist_max"])

        # Scale bounds by similarity (confidence)
        confidence = max(0.0, best_sim)
        lo = raw_lo * confidence
        hi = raw_hi * confidence

        lbs[i] = max(float(min_active_bid), lo)
        ubs[i] = min(float(max_bid_cap), hi)

        _add_debug(
            i=i,
            source="nearest_neighbor",
            lb=lbs[i],
            ub=ubs[i],
            nn_keyword=str(cand_df.loc[best, "Keyword_join"]),
            nn_region=str(cand_df.loc[best, "Region_join"]),
            nn_match=str(cand_df.loc[best, "Match_join"]),
            nn_hist_min=raw_lo,
            nn_hist_max=raw_hi,
            cosine_sim=best_sim,
            tier=tier,
        )

    lbs = np.clip(lbs, 0.0, float(max_bid_cap))
    ubs = np.clip(ubs, 0.0, float(max_bid_cap))
    ubs = np.maximum(ubs, lbs)
    debug_df = pd.DataFrame(debug_rows)
    return lbs, ubs, debug_df


def compute_clicks_bid_bounds_from_history(
    *,
    training_csv: Path,
    keyword_df: pd.DataFrame,
    kw_idx_list: List[int],
    region_list: List[str],
    match_list: List[str],
    embedding_method: str,
    max_bid_cap: float,
    min_active_bid: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Public wrapper for per-row historical bid bounds.

    This delegates to the internal implementation used by the main optimizer.
    """

    return _compute_clicks_bid_bounds_from_history(
        training_csv=training_csv,
        keyword_df=keyword_df,
        kw_idx_list=kw_idx_list,
        region_list=region_list,
        match_list=match_list,
        embedding_method=embedding_method,
        max_bid_cap=max_bid_cap,
        min_active_bid=min_active_bid,
    )


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
        default='ridge',
        choices=['lr', 'ridge', 'xgb', 'ort', 'mort'],
        help='Algorithm type for EPC model: lr (linear weights), ridge (Ridge baseline weights), xgb (XGBoost MSE; no IAI), ort (optimal tree; requires IAI), mort (mirrored ORT with hyperplanes; requires IAI) (default: lr)'
    )
    parser.add_argument(
        '--alg-clicks',
        type=str,
        default='xgb', # 'lr', # 
        choices=['lr', 'ridge', 'xgb', 'ort', 'mort'],
        help='Algorithm type for clicks model: lr (linear weights), ridge (Ridge baseline weights), xgb (XGBoost MSE; no IAI), ort (optimal tree; requires IAI), mort (mirrored ORT with hyperplanes; requires IAI) (default: lr)'
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
        help='Target day for optimization in format YYYY-MM-DD (e.g., 2024-11-04). If not provided, uses today\'s data.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/clean',
        help='Directory containing cleaned data artifacts and embedding pipelines (default: data/clean)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=None,
        help='Trial mode: limit to N keywords for quick testing (default: None, use all)'
    )

    parser.add_argument(
        '--trust-box',
        type=str,
        default='drop',
        choices=['drop', 'off'],
        help="Enforce training min/max ranges on numeric features: 'drop' removes out-of-range rows; 'off' disables (default: drop)"
    )

    parser.add_argument(
        '--trust-box-training',
        type=str,
        default='data/clean/ad_opt_data_bert.csv',
        help='Training CSV used to compute trust-box min/max (default: data/clean/ad_opt_data_bert.csv)'
    )

    parser.add_argument(
        '--exploration-lambda',
        type=float,
        default=0.0,
        help='Exploration term weight λ in objective: + λ * sum_{new keywords} CPC (bid). (default: 0)'
    )

    parser.add_argument(
        '--cache-key',
        type=str,
        default=None,
        help='Cache key for reusing embeddings/feature matrices and base formulation across lambda runs.'
    )

    parser.add_argument(
        '--cache-mode',
        type=str,
        default='auto',
        choices=['auto', 'build', 'reuse'],
        help="Cache behavior: 'build' forces rebuild; 'reuse' requires existing cache; 'auto' loads if present else builds. (default: auto)"
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validate_solution (useful for fast lambda sweeps).'
    )

    parser.add_argument(
        '--treat-existing-searches-as-new',
        action='store_true',
        help=(
            "Treat keywords with Origin='existing searches' as new keywords for the purposes of "
            "the exploration objective and output flags (does not change bidding constraints)."
        ),
    )

    
    args = parser.parse_args()

    # Resolve target-day early so downstream cache keys, file naming, and
    # directory layout consistently include a concrete date.
    if args.target_day is None:
        args.target_day = pd.Timestamp.today().strftime("%Y-%m-%d")
    elif isinstance(args.target_day, str):
        td = args.target_day.strip().lower()
        if td in {"today", ""}:
            args.target_day = pd.Timestamp.today().strftime("%Y-%m-%d")
        elif td in {"all", "none"}:
            args.target_day = None
    
    print("=" * 70)
    print("Bid Optimization with Linear Programming")
    print("=" * 70)
    print(f"Embedding method: {args.embedding_method}")
    if args.trial is not None:
        print(f"⚠️  TRIAL MODE: Limited to {args.trial} keywords")
    print("=" * 70)
    
    got_lock = False

    try:
        # Cache key default
        if args.cache_key is None:
            td = args.target_day if args.target_day is not None else 'all'
            tr = str(args.trial) if args.trial is not None else 'full'
            args.cache_key = f"{args.embedding_method}_{args.alg_conv}_{args.alg_clicks}_{td}_{tr}"  # noqa: B950

        cache_dir = _cache_dir_for_key(args.cache_key)
        lock_file = cache_dir / ".build.lock"
        base_lp = cache_dir / "base_formulation.lp"

        # Load new keywords from classified keywords file
        classified_keywords_file = Path('data/gkp') / 'keywords_classified.csv'
        if classified_keywords_file.exists():
            classified_df = pd.read_csv(str(classified_keywords_file))
            if 'Keyword' not in classified_df.columns:
                raise KeyError(f"Expected 'Keyword' column in {classified_keywords_file}")

            keywords_to_embed = classified_df['Keyword'].dropna().astype(str).tolist()

            new_kw_set: Set[str] = set()
            if 'Origin' in classified_df.columns:
                origin_lc = classified_df['Origin'].astype(str).str.lower()
                new_kw_set = set(
                    _normalize_kw(s)
                    for s in classified_df.loc[
                        origin_lc.eq('new'),
                        'Keyword',
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                if args.treat_existing_searches_as_new:
                    extra = classified_df.loc[origin_lc.eq('existing searches'), 'Keyword'].dropna().astype(str).tolist()
                    new_kw_set |= set(_normalize_kw(s) for s in extra)
            print(f"Loaded {len(keywords_to_embed)} keywords from {classified_keywords_file} (new={len(new_kw_set)})")
        else:
            # Fall back to loading from latest GKP export
            gkp_df = get_gkp_data(gkp_dir='data/gkp')
            if gkp_df.empty:
                raise FileNotFoundError(
                    "No keywords_classified.csv found and no GKP 'Saved Keywords Stats' file found in data/gkp."
                )
            keywords_to_embed = gkp_df['Keyword'].dropna().astype(str).tolist()
            print(f"Loaded {len(keywords_to_embed)} keywords from latest GKP export")
            new_kw_set = set(_normalize_kw(s) for s in keywords_to_embed)
        
        # Apply trial mode if requested
        if args.trial is not None:
            # Mix new keywords with existing ones (identified by Origin column)
            classified_df = pd.read_csv(str(classified_keywords_file))
            origin_lc = classified_df.get('Origin', '').astype(str).str.lower()
            new_keywords_df = classified_df[origin_lc.eq('new')]
            existing_keywords_df = classified_df[origin_lc.isin({'existing', 'existing searches'})]
            existing_searches_df = classified_df[origin_lc.eq('existing searches')]
            
            new_kws = new_keywords_df['Keyword'].dropna().astype(str).tolist()
            existing_kws = existing_keywords_df['Keyword'].dropna().astype(str).tolist()
            existing_searches_kws = existing_searches_df['Keyword'].dropna().astype(str).tolist()
            
            # Take half from new, half from existing (or adjust if not enough of one type)
            new_count = min(args.trial // 2, len(new_kws))
            existing_count = min(args.trial - new_count, len(existing_kws))
            
            trial_keywords = new_kws[:new_count] + existing_kws[:existing_count]
            keywords_to_embed = trial_keywords[:args.trial]
            print(f"⚠️  Trial mode: using {new_count} new + {existing_count} existing keywords = {len(keywords_to_embed)} total")

            new_kw_set = set(_normalize_kw(s) for s in new_kws[:new_count])
            if args.treat_existing_searches_as_new:
                # Only mark the *existing searches* subset as new.
                new_kw_set |= set(_normalize_kw(s) for s in existing_searches_kws[:existing_count])

        # Fast path: reuse cached base formulation + mapping + keyword_df
        if args.cache_mode in {'auto', 'reuse'} and base_lp.exists():
            try:
                cached = _load_cached_artifacts(
                    cache_dir,
                    alg_epc=args.alg_conv,
                    alg_clicks=args.alg_clicks,
                    embedding_method=args.embedding_method,
                )
                keyword_df = cached['keyword_df']
                kw_idx_list = cached['kw_idx_list']
                region_list = cached['region_list']
                match_list = cached['match_list']

                # Build row-level new mask from cached keyword_df
                if 'is_new_keyword' in keyword_df.columns:
                    is_new_lookup = keyword_df['is_new_keyword'].to_numpy(dtype=bool)
                    new_row_mask = np.asarray([bool(is_new_lookup[int(k)]) for k in kw_idx_list], dtype=bool)
                else:
                    new_row_mask = np.asarray(
                        [
                            _normalize_kw(keyword_df.iloc[int(k)]['Keyword']) in new_kw_set
                            for k in kw_idx_list
                        ],
                        dtype=bool,
                    )

                print(f"\nReusing cached base formulation: {base_lp}")
                model = gp.read(str(base_lp))
                model.setParam('OutputFlag', 1)
                model.setParam('TimeLimit', 300)
                model.setParam('NonConvex', 2)

                # Apply exploration objective
                if float(args.exploration_lambda) != 0.0 and new_row_mask.any():
                    obj = model.getObjective()
                    idxs = np.where(new_row_mask)[0]
                    explore = gp.quicksum(model.getVarByName(f"b[{int(i)}]") for i in idxs)
                    model.setObjective(obj + float(args.exploration_lambda) * explore, GRB.MAXIMIZE)

                model.optimize()

                # Extract vars (lists) for solution export
                K = len(kw_idx_list)
                b_vars = [model.getVarByName(f"b[{i}]") for i in range(K)]
                f_vars = [model.getVarByName(f"f[{i}]") for i in range(K)]
                g_vars = [model.getVarByName(f"g[{i}]") for i in range(K)]

                if model.status == 2 or model.status == 9:
                    model_suffix = f"{args.alg_conv}_{args.alg_clicks}"

                    bids_df = extract_solution(
                        model,
                        b_vars,
                        f_vars,
                        g_vars,
                        keyword_df,
                        kw_idx_list,
                        region_list,
                        match_list,
                        X=None,
                    )
                    bids_df['lambda'] = float(args.exploration_lambda)

                    sweep_file, _legacy = _write_bids_outputs(
                        bids_df=bids_df,
                        embedding_method=args.embedding_method,
                        model_suffix=model_suffix,
                        cache_key=args.cache_key,
                        exploration_lambda=float(args.exploration_lambda),
                        write_legacy=False,
                    )

                    print(f"\nResults saved to {sweep_file}")
                    print("\n" + "=" * 70)
                    print("✓ Bid optimization completed successfully! (cache reuse)")
                    print("=" * 70)
                    return

                print(f"Optimization failed with status {model.status}")
                sys.exit(1)
            except Exception as e:
                if args.cache_mode == 'reuse':
                    raise
                print(f"Cache reuse failed; falling back to rebuild. Reason: {type(e).__name__}: {e}")

        # Build path: acquire lock so parallel SLURM array tasks don't all rebuild at once.
        if args.cache_mode in {'auto', 'build'}:
            got_lock = _try_acquire_lock(lock_file)
            if not got_lock:
                raise TimeoutError(f"Timed out waiting for cache lock: {lock_file}")

        # Generate embeddings using saved pipeline
        keyword_df = generate_keyword_embeddings_df(
            keywords_to_embed,
            embedding_method=args.embedding_method,
            n_components=50,
            pipeline_dir=args.data_dir,
        )
        # Ensure is_new_keyword is correctly populated in the keyword_df
        keyword_df['is_new_keyword'] = keyword_df['Keyword'].astype(str).map(lambda s: _normalize_kw(s) in new_kw_set)
        
        # Only load weights for non-XGB models (LR/Ridge use weights_dict)
        if args.alg_conv != 'xgb' or args.alg_clicks != 'xgb':
            weights_dict = load_weights_from_csv(
                embedding_method=args.embedding_method,
                models_dir=args.models_dir
            )
        else:
            weights_dict = None
        
        # Create feature matrix (GKP-only; no historical training data)
        feature_matrix_result = create_feature_matrix(
            keyword_df,
            embedding_method=args.embedding_method,
            target_day=args.target_day,
            alg_epc=args.alg_conv,
            alg_clicks=args.alg_clicks,
            gkp_dir='data/gkp',
        )
        
        # Unpack results based on model types
        raw_cat_algs = {'ort', 'mort', 'xgb', 'ridge'}
        is_mixed = ((args.alg_conv in raw_cat_algs) and (args.alg_clicks not in raw_cat_algs)) or ((args.alg_conv not in raw_cat_algs) and (args.alg_clicks in raw_cat_algs))
        if is_mixed:
            X_ort, X_lr, kw_idx_list, region_list, match_list = feature_matrix_result
        else:
            X_result, kw_idx_list, region_list, match_list = feature_matrix_result
            X_ort = X_result if (args.alg_conv in raw_cat_algs) and (args.alg_clicks in raw_cat_algs) else None
            X_lr = X_result if (args.alg_conv not in raw_cat_algs) and (args.alg_clicks not in raw_cat_algs) else None
        
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

        # Embedded mode: supports linear weights (LR/Ridge), ORT/MORT (IAI), and XGB (no IAI)
        epc_model = None
        clicks_model = None
        epc_xgb_paths = None
        clicks_xgb_paths = None

        if args.alg_conv in {'ort', 'mort'}:
            iai_local = _get_iai(required=True)

            epc_model_path = Path(args.models_dir) / f'{args.alg_conv}_{args.embedding_method}_epc.json'
            if not epc_model_path.exists():
                raise FileNotFoundError(f"{args.alg_conv.upper()} EPC model not found: {epc_model_path}")
            epc_model = iai_local.read_json(str(epc_model_path))
            print(f"  Loaded {args.alg_conv.upper()} EPC model from {epc_model_path}")

        if args.alg_conv == 'xgb':
            epc_xgb_model_path = Path(args.models_dir) / f'xgb_mse_{args.embedding_method}_epc.json'
            epc_xgb_preproc_path = Path(args.models_dir) / f'xgb_mse_{args.embedding_method}_epc_preprocess.joblib'
            if not epc_xgb_model_path.exists():
                raise FileNotFoundError(f"XGB EPC model not found: {epc_xgb_model_path}")
            if not epc_xgb_preproc_path.exists():
                raise FileNotFoundError(f"XGB EPC preprocessor not found: {epc_xgb_preproc_path}")
            epc_xgb_paths = (epc_xgb_model_path, epc_xgb_preproc_path)
            print(f"  Using XGB EPC model from {epc_xgb_model_path}")

        if args.alg_clicks in {'ort', 'mort'}:
            iai_local = _get_iai(required=True)

            clicks_model_path = Path(args.models_dir) / f'{args.alg_clicks}_{args.embedding_method}_clicks.json'
            if not clicks_model_path.exists():
                raise FileNotFoundError(f"{args.alg_clicks.upper()} clicks model not found: {clicks_model_path}")
            clicks_model = iai_local.read_json(str(clicks_model_path))
            print(f"  Loaded {args.alg_clicks.upper()} clicks model from {clicks_model_path}")

        if args.alg_clicks == 'xgb':
            clicks_xgb_model_path = Path(args.models_dir) / f'xgb_mse_{args.embedding_method}_clicks.json'
            clicks_xgb_preproc_path = Path(args.models_dir) / f'xgb_mse_{args.embedding_method}_clicks_preprocess.joblib'
            if not clicks_xgb_model_path.exists():
                raise FileNotFoundError(f"XGB clicks model not found: {clicks_xgb_model_path}")
            if not clicks_xgb_preproc_path.exists():
                raise FileNotFoundError(f"XGB clicks preprocessor not found: {clicks_xgb_preproc_path}")
            clicks_xgb_paths = (clicks_xgb_model_path, clicks_xgb_preproc_path)
            print(f"  Using XGB clicks model from {clicks_xgb_model_path}")

        # Load Ridge preprocessors if needed
        epc_ridge_preproc_path = None
        clicks_ridge_preproc_path = None
        
        if args.alg_conv == 'ridge':
            epc_ridge_preproc_path = Path(args.models_dir) / f'ridge_{args.embedding_method}_epc_preprocess.joblib'
            if not epc_ridge_preproc_path.exists():
                raise FileNotFoundError(f"Ridge EPC preprocessor not found: {epc_ridge_preproc_path}")
            print(f"  Using Ridge EPC preprocessor from {epc_ridge_preproc_path}")
        
        if args.alg_clicks == 'ridge':
            clicks_ridge_preproc_path = Path(args.models_dir) / f'ridge_{args.embedding_method}_clicks_preprocess.joblib'
            if not clicks_ridge_preproc_path.exists():
                raise FileNotFoundError(f"Ridge clicks preprocessor not found: {clicks_ridge_preproc_path}")
            print(f"  Using Ridge clicks preprocessor from {clicks_ridge_preproc_path}")

        # --- Feasibility guard: ensure b=0 is feasible ---
        # Drop any keyword/region/match combos where either EPC or clicks is negative at bid=0.
        # This prevents infeasibility caused by f,g having lb=0 while model outputs can be negative.
        X_ort, X_lr, kw_idx_list, region_list, match_list = filter_rows_feasible_at_bid0(
            X_ort=X_ort,
            X_lr=X_lr,
            kw_idx_list=kw_idx_list,
            region_list=region_list,
            match_list=match_list,
            alg_epc=args.alg_conv,
            alg_clicks=args.alg_clicks,
            embedding_method=args.embedding_method,
            weights_dict=weights_dict,
            epc_model=epc_model,
            clicks_model=clicks_model,
            epc_xgb_paths=epc_xgb_paths,
            clicks_xgb_paths=clicks_xgb_paths,
            epc_ridge_preproc_path=epc_ridge_preproc_path,
            clicks_ridge_preproc_path=clicks_ridge_preproc_path,
        )

        # --- Trust box: enforce Trust Ellipsoid (Mahalanobis) ---
        effective_max_bid = float(args.max_bid)
        
        if args.trust_box != 'off':
            training_csv = Path(args.trust_box_training)
            if not training_csv.exists():
                raise FileNotFoundError(f"Trust-box training CSV not found: {training_csv}")
            
            # [CHANGED] Compute stats (Mean/Cov) instead of just ranges.
            # Restrict to columns that actually exist in the current optimizer feature matrix.
            X_for_stats = X_ort if X_ort is not None else X_lr
            allowed_cols = None
            if X_for_stats is not None:
                allowed_cols = set(X_for_stats.select_dtypes(include=[np.number]).columns.tolist())
            training_stats, train_cpc_max = _compute_training_stats_and_bounds(
                training_csv,
                allowed_cols=allowed_cols,
            )

            # [UNCHANGED] Cap bid to be within the training CPC range 
            # We still do this to prevent the solver from exploring bid spaces 
            # completely unseen in training (e.g. bidding $100 when max train CPC was $5).
            if np.isfinite(train_cpc_max):
                original_max = effective_max_bid
                effective_max_bid = min(effective_max_bid, train_cpc_max)
                if effective_max_bid < original_max:
                    print(f"Trust-box: capping max bid from {original_max} to {effective_max_bid:.4f} based on training max CPC")

            # [CHANGED] Apply the Ellipsoid Filter
            # This filters the *rows* (keywords) that have feature combinations 
            # statistically unlikely based on the training data.
            X_ort, X_lr, kw_idx_list, region_list, match_list = apply_trust_ellipsoid_filter(
                X_ort=X_ort,
                X_lr=X_lr,
                kw_idx_list=kw_idx_list,
                region_list=region_list,
                match_list=match_list,
                training_stats=training_stats,
                p_value_threshold=0.99, # Configurable: 0.99 = keep 99% of "normal" data
            )

        # --- Historical bid bounds for clicks model ---
        # Enforce bids within historical min/max for (keyword, match, region) when available;
        # otherwise use nearest available combo as a safety rail against extrapolation.
        bid_lbs, bid_ubs, bid_bounds_debug_df = _compute_clicks_bid_bounds_from_history(
            training_csv=Path(args.trust_box_training),
            keyword_df=keyword_df,
            kw_idx_list=kw_idx_list,
            region_list=region_list,
            match_list=match_list,
            embedding_method=args.embedding_method,
            max_bid_cap=effective_max_bid,
            min_active_bid=0.01,
        )

        # Save bounds + nearest-neighbor info for debugging
        try:
            cache_dir.mkdir(exist_ok=True, parents=True)
            bounds_debug_file = cache_dir / "clicks_bid_bounds_debug.csv"
            bid_bounds_debug_df.to_csv(bounds_debug_file, index=False)
            print(f"Saved bid bounds debug CSV: {bounds_debug_file}")
        except Exception as e:
            print(f"Warning: failed to write bid bounds debug CSV: {type(e).__name__}: {e}")

        model, b, f, g = optimize_bids_embedded(
            X_ort=X_ort,
            X_lr=X_lr,
            weights_dict=weights_dict,
            budget=args.budget,
            max_bid=effective_max_bid,
            epc_model=epc_model,
            clicks_model=clicks_model,
            epc_xgb_paths=epc_xgb_paths,
            clicks_xgb_paths=clicks_xgb_paths,
            epc_ridge_preproc_path=epc_ridge_preproc_path,
            clicks_ridge_preproc_path=clicks_ridge_preproc_path,
            alg_epc=args.alg_conv,
            alg_clicks=args.alg_clicks,
            embedding_method=args.embedding_method,
            bid_lower_bounds=bid_lbs,
            bid_upper_bounds=bid_ubs,
            exploration_lambda=float(args.exploration_lambda),
            new_row_mask=np.asarray([
                bool(keyword_df.iloc[int(kw_idx_list[i])].get('is_new_keyword', False))
                for i in range(len(kw_idx_list))
            ], dtype=bool),
            save_base_formulation_to=(base_lp if args.cache_mode in {'auto', 'build'} else None),
        )

        if model.status == 2 or model.status == 9:
            model_suffix = f"{args.alg_conv}_{args.alg_clicks}"
            bids_df = extract_solution(model, b, f, g, keyword_df, kw_idx_list, region_list, match_list, X=X)
            bids_df['lambda'] = float(args.exploration_lambda)

            # Validate solution predictions
            raw_cat_algs = {'ort', 'mort', 'xgb', 'ridge'}
            X_for_epc = X_ort if args.alg_conv in raw_cat_algs else X_lr
            X_for_clicks = X_ort if args.alg_clicks in raw_cat_algs else X_lr

            if not args.skip_validation:
                validate_solution(
                    optimized_bids=b.X,
                    solver_epc_preds=f.X,
                    solver_clicks_preds=g.X,
                    X_epc=X_for_epc,
                    X_clicks=X_for_clicks,
                    epc_model=epc_model,
                    clicks_model=clicks_model,
                    epc_xgb_paths=epc_xgb_paths if args.alg_conv == 'xgb' else None,
                    clicks_xgb_paths=clicks_xgb_paths if args.alg_clicks == 'xgb' else None,
                    epc_ridge_preproc_path=epc_ridge_preproc_path if args.alg_conv == 'ridge' else None,
                    clicks_ridge_preproc_path=clicks_ridge_preproc_path if args.alg_clicks == 'ridge' else None,
                    alg_epc=args.alg_conv,
                    alg_clicks=args.alg_clicks,
                    embedding_method=args.embedding_method
                )

            sweep_file, legacy_file = _write_bids_outputs(
                bids_df=bids_df,
                embedding_method=args.embedding_method,
                model_suffix=model_suffix,
                cache_key=args.cache_key,
                exploration_lambda=float(args.exploration_lambda),
                write_legacy=True,
            )

            # Cache artifacts for reuse
            if args.cache_mode in {'auto', 'build'}:
                meta = {
                    'created_at': datetime.now().isoformat(),
                    'cache_key': args.cache_key,
                    'embedding_method': args.embedding_method,
                    'alg_epc': args.alg_conv,
                    'alg_clicks': args.alg_clicks,
                    'target_day': args.target_day,
                    'trial': args.trial,
                }
                _write_cached_artifacts(
                    cache_dir,
                    keyword_df=keyword_df,
                    X_ort=X_ort,
                    X_lr=X_lr,
                    kw_idx_list=kw_idx_list,
                    region_list=region_list,
                    match_list=match_list,
                    alg_epc=args.alg_conv,
                    alg_clicks=args.alg_clicks,
                    embedding_method=args.embedding_method,
                    meta=meta,
                )

            if got_lock:
                _release_lock(lock_file)
                got_lock = False

            print(f"\nResults saved to {legacy_file}")
            print(f"Sweep copy saved to {sweep_file}")
            print(f"\nTop 10 bids:")
            print(bids_df.head(10).to_string(index=False))
        elif model.status == GRB.INFEASIBLE:
            print(f"Optimization failed with status {model.status} (Infeasible).")
            print("Computing IIS to locate the conflict...")
            
            # Compute the Irreducible Inconsistent Subsystem
            model.computeIIS()
            
            # Write the report to a file you can open in a text editor
            output_file = "infeasibility_report.ilp"
            model.write(output_file)
            print(f"Infeasibility report written to {output_file}")
            
            # Optional: Print the specific constraints causing the issue to the console
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"Constraint in IIS: {c.ConstrName}")

            if got_lock:
                _release_lock(lock_file)
                got_lock = False

            sys.exit(1)
        else:
            print(f"Optimization failed with status {model.status}")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("✓ Bid optimization completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        if got_lock:
            _release_lock(lock_file)
            got_lock = False
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
