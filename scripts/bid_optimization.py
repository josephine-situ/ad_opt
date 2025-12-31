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
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_embeddings
from utils.data_pipeline import get_gkp_data, impute_missing_data

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


def load_models(embedding_method='bert', alg_epc='lr', alg_clicks='lr', models_dir='models'):
    """Load pre-trained prediction models based on embedding method and algorithm type.
    
    Args:
        embedding_method: 'bert' or 'tfidf'
        alg_epc: algorithm type for EPC model - 'lr' (linear regression), 'ort' (optimal regression tree),
                  'mort' (mirrored optimal regression tree with hyperplanes), 'rf' (random forest), 'xgb' (xgboost)
        alg_clicks: algorithm type for clicks model - same options as alg_epc
        models_dir: directory containing model files
    
    Returns:
        tuple of (epc_model, clicks_model)
    """
    print(f"Loading models for embedding method '{embedding_method}'...")
    print(f"  EPC model: {alg_epc}")
    print(f"  Clicks model: {alg_clicks}")
    
    epc_model = Path(models_dir) / f'{alg_epc}_{embedding_method}_epc.json'
    clicks_model = Path(models_dir) / f'{alg_clicks}_{embedding_method}_clicks.json'
    
    if not epc_model.exists():
        raise FileNotFoundError(f"EPC model not found: {epc_model}")
    if not clicks_model.exists():
        raise FileNotFoundError(f"Clicks model not found: {clicks_model}")
    
    iai_local = _get_iai(required=True)
    lnr_epc = iai_local.read_json(str(epc_model))
    lnr_clicks = iai_local.read_json(str(clicks_model))
    
    print(f"  Loaded EPC model from {epc_model}")
    print(f"  Loaded clicks model from {clicks_model}")
    
    return lnr_epc, lnr_clicks


def extract_weights(lnr_epc, lnr_clicks, embedding_method='bert', n_embeddings=50):
    """Extract ALL weights from trained models (not just embeddings)."""
    print(f"Extracting model weights...")
    
    # Get weights using IAI's method
    weights_epc_tuple = lnr_epc.get_prediction_weights()
    weights_clicks_tuple = lnr_clicks.get_prediction_weights()
    
    # Handle tuple format (continuous, categorical)
    if isinstance(weights_epc_tuple, tuple):
        weights_epc = weights_epc_tuple[0]
    else:
        weights_epc = weights_epc_tuple
    
    if isinstance(weights_clicks_tuple, tuple):
        weights_clicks = weights_clicks_tuple[0]
    else:
        weights_clicks = weights_clicks_tuple
    
    epc_const = lnr_epc.get_prediction_constant()
    clicks_const = lnr_clicks.get_prediction_constant()
    
    print(f"\n  EPC model weights:")
    for key, val in sorted(weights_epc.items(), key=lambda x: str(x[0])):
        print(f"    {key}: {val:.6f}")
    
    print(f"\n  Clicks model weights:")
    for key, val in sorted(weights_clicks.items(), key=lambda x: str(x[0])):
        print(f"    {key}: {val:.6f}")
    
    return {
        'epc_const': epc_const,
        'epc_weights': weights_epc,
        'clicks_const': clicks_const,
        'clicks_weights': weights_clicks
    }


def create_feature_matrix(
    keyword_df,
    embedding_method='bert',
    target_day=None,
    regions=None,
    match_types=None,
    weights_dict=None,
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
        weights_dict: dict with 'epc_weights' and 'clicks_weights' to filter features. If None, keeps all.
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
    
    # Extract model weights if provided
    if weights_dict is None:
        weights_dict = {}
    epc_weights = weights_dict.get('epc_weights', {})
    clicks_weights = weights_dict.get('clicks_weights', {})
    
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
    gkp_df['Keyword_join'] = gkp_df['Keyword'].astype(str).str.lower().str.strip()

    # Merge keyword stats onto each keyword-region-match row
    result = combo_df.merge(
        gkp_df.drop(columns=['Keyword'], errors='ignore'),
        left_on='Keyword_join',
        right_on='Keyword_join',
        how='left'
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
    result = result.drop(columns=['Keyword_join'], errors='ignore')
    
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
    #   (one-hot inside the preprocessor), so they ALSO need raw categoricals.
    raw_cat_algs = {'ort', 'mort', 'xgb'}
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

    if hasattr(preprocessor, "feature_names_in_"):
        expected = list(preprocessor.feature_names_in_)  # type: ignore[attr-defined]
    else:
        expected = list(X.columns)

    X_use = X.copy()

    if "Avg. CPC" in expected and "Avg. CPC" not in X_use.columns and "Avg_ CPC" in X_use.columns:
        X_use = X_use.rename(columns={"Avg_ CPC": "Avg. CPC"})
    elif "Avg_ CPC" in expected and "Avg_ CPC" not in X_use.columns and "Avg. CPC" in X_use.columns:
        X_use = X_use.rename(columns={"Avg. CPC": "Avg_ CPC"})

    for col in expected:
        if col not in X_use.columns:
            X_use[col] = np.nan

    X_use = X_use[expected]
    return X_use, expected


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
    """Embed an XGBoost regressor (saved JSON) into Gurobi without IAI.

    Assumptions:
    - The XGB model was trained on `preprocessor.transform(X_raw)`.
    - `X` is the raw feature DataFrame (categoricals as strings).
    - The bid decision variable `b[i]` replaces the raw 'Avg. CPC' feature.

    This uses a leaf-indicator (path-based) formulation per tree.
    """

    if xgb is None:
        raise ImportError("xgboost is required for embed_xgb. Install with: pip install xgboost")
    if joblib is None:
        raise ImportError("joblib is required for embed_xgb. Install with: pip install joblib")

    print(f"  Embedding {target} XGB constraints (path-based formulation)...")

    preprocessor = joblib.load(preprocessor_path)
    X_use, expected_cols = _align_X_for_preprocessor(preprocessor, X)

    # Identify CPC column name as expected by the preprocessor.
    cpc_candidates = [c for c in ("Avg. CPC", "Avg_ CPC") if c in expected_cols]
    if not cpc_candidates:
        raise KeyError(
            "Could not find an Avg. CPC column expected by the saved preprocessor. "
            f"Expected one of ['Avg. CPC', 'Avg_ CPC'] in: {expected_cols[:20]}..."
        )
    cpc_col = cpc_candidates[0]

    # Precompute preprocessed feature constants (CPC set to 0) and CPC coefficient
    # (difference between CPC=1 and CPC=0). This is robust to scaling pipelines.
    X0 = X_use.copy()
    X0[cpc_col] = 0.0
    X1 = X_use.copy()
    X1[cpc_col] = 1.0

    Z0 = preprocessor.transform(X0)
    Z1 = preprocessor.transform(X1)
    dZ = Z1 - Z0

    if sp is not None and sp.issparse(Z0):
        Z0 = Z0.tocsr()
        dZ = dZ.tocsr()
    else:
        Z0 = np.asarray(Z0)
        dZ = np.asarray(dZ)

    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    base_score = _extract_xgb_base_score(xgb_model_path)

    tree_dumps = booster.get_dump(dump_format="json")
    trees = [json.loads(s) for s in tree_dumps]

    # Parse all leaf paths and collect used feature indices.
    tree_paths: List[List[Tuple[List[Tuple[int, str, float]], float]]] = []
    used_features: set[int] = set()
    cpc_split_thresholds: List[float] = []

    for t in trees:
        paths = _parse_xgb_tree_paths(t)
        tree_paths.append(paths)
        for conds, _leaf in paths:
            for feat_idx, _op, thr in conds:
                used_features.add(int(feat_idx))
                cpc_split_thresholds.append(float(thr))

    if not used_features:
        raise ValueError("XGB model appears to have no split features")

    # Determine which preprocessed feature index corresponds to CPC.
    if sp is not None and sp.issparse(dZ):
        nonzero_cols = set(map(int, dZ.nonzero()[1]))
    else:
        nonzero_cols = set(int(j) for j in range(dZ.shape[1]) if np.any(dZ[:, j] != 0))

    cpc_feature_candidates = sorted(list(nonzero_cols.intersection(used_features)))
    if len(cpc_feature_candidates) > 1:
        raise ValueError(
            f"Multiple variable-dependent preprocessed columns detected for CPC: {cpc_feature_candidates}. "
            "This embedding assumes only the CPC column varies with b."
        )
    cpc_feature_idx = cpc_feature_candidates[0] if cpc_feature_candidates else None

    cpc_coeff = 0.0
    if cpc_feature_idx is not None:
        if sp is not None and sp.issparse(dZ):
            # coefficient per row should be constant; take first row.
            cpc_coeff = float(dZ[0, cpc_feature_idx])
        else:
            cpc_coeff = float(dZ[0, cpc_feature_idx])

    # Materialize constants only for used features.
    used_features_list = sorted(list(used_features))
    const_by_feat: Dict[int, np.ndarray] = {}
    if sp is not None and sp.issparse(Z0):
        for j in used_features_list:
            const_by_feat[j] = np.asarray(Z0.getcol(j).toarray()).ravel()
    else:
        for j in used_features_list:
            const_by_feat[j] = np.asarray(Z0[:, j]).ravel()

    # Big-M for CPC splits (only needed if CPC is used in the trees).
    M_cpc = 0.0
    if cpc_feature_idx is not None:
        expr0 = 0.0
        expr1 = cpc_coeff * float(max_bid)
        expr_min = float(min(expr0, expr1))
        expr_max = float(max(expr0, expr1))
        for paths in tree_paths:
            for conds, _leaf in paths:
                for feat_idx, _op, thr in conds:
                    if int(feat_idx) != int(cpc_feature_idx):
                        continue
                    M_cpc = max(M_cpc, expr_max - float(thr), float(thr) - expr_min)
        M_cpc = float(M_cpc + 1e-6)

    K = len(X_use)
    pred_vars: List[object] = []

    # Build constraints row-by-row.
    for i in range(K):
        pred = model.addVar(lb=-GRB.INFINITY, name=f"{target}_pred_{i}")
        pred_vars.append(pred)

        tree_outputs = []

        for t_idx, paths in enumerate(tree_paths):
            leaf_inds = []
            leaf_vals = []

            for leaf_idx, (conds, leaf_val) in enumerate(paths):
                z = model.addVar(vtype=GRB.BINARY, name=f"{target}_xgb_t{t_idx}_l{leaf_idx}_i{i}")
                leaf_inds.append(z)
                leaf_vals.append(float(leaf_val))

                # Enforce split conditions along the path.
                infeasible = False
                for feat_idx, op, thr in conds:
                    feat_idx = int(feat_idx)
                    thr = float(thr)

                    if cpc_feature_idx is None or feat_idx != int(cpc_feature_idx):
                        # Constant feature. If the split is violated, this leaf is impossible.
                        v = float(const_by_feat[feat_idx][i])
                        if op == "lt":
                            if not (v < thr):
                                infeasible = True
                                break
                        elif op == "ge":
                            if not (v >= thr):
                                infeasible = True
                                break
                        else:
                            raise ValueError(f"Unknown op: {op}")
                    else:
                        # CPC-dependent feature.
                        expr = (cpc_coeff * b[i])
                        if op == "lt":
                            model.addConstr(expr <= thr + M_cpc * (1 - z), name=f"{target}_xgb_lt_t{t_idx}_l{leaf_idx}_i{i}_f{feat_idx}")
                        elif op == "ge":
                            model.addConstr(expr >= thr + 1e-6 - M_cpc * (1 - z), name=f"{target}_xgb_ge_t{t_idx}_l{leaf_idx}_i{i}_f{feat_idx}")
                        else:
                            raise ValueError(f"Unknown op: {op}")

                if infeasible:
                    model.addConstr(z == 0, name=f"{target}_xgb_infeasible_t{t_idx}_l{leaf_idx}_i{i}")

            # Exactly one leaf active per tree.
            model.addConstr(gp.quicksum(leaf_inds) == 1, name=f"{target}_xgb_oneleaf_t{t_idx}_i{i}")

            tree_out = model.addVar(lb=-GRB.INFINITY, name=f"{target}_xgb_tree_{t_idx}_i{i}")
            model.addConstr(
                tree_out == gp.quicksum(leaf_vals[k] * leaf_inds[k] for k in range(len(leaf_inds))),
                name=f"{target}_xgb_treeout_t{t_idx}_i{i}",
            )
            tree_outputs.append(tree_out)

        # Prediction is base_score + sum(tree outputs)
        model.addConstr(pred == float(base_score) + gp.quicksum(tree_outputs), name=f"{target}_xgb_predlink_{i}")

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
    alg_epc: str = 'lr',
    alg_clicks: str = 'lr',
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
        X_lr: Feature matrix with categorical features one-hot encoded (for LR/GLM weights). If None, will use X_ort.
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

    raw_cat_algs = {'ort', 'mort', 'xgb'}

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
    model.setParam('OutputFlag', 1) 
    model.setParam('TimeLimit', 60)

    # --- 0. Decision Variables ---

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
        if weights_dict is None:
            raise ValueError("weights_dict is required when epc_model is not provided")

        if 'epc_const' not in weights_dict or 'epc_weights' not in weights_dict:
            raise KeyError("weights_dict must contain 'epc_const' and 'epc_weights'")

        epc_const = weights_dict['epc_const']
        epc_weights = weights_dict['epc_weights']

        if alg_epc == 'glm':
            model, f_hat_vars = embed_glm(model, epc_weights, epc_const, X_epc, b, target='epc', link='log')
        else:
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
        if alg_clicks == 'glm':
            model, g_hat_vars = embed_glm(model, clicks_weights, clicks_const, X_clicks, b, target='clicks', link='log')
        else:
            model, g_hat_vars = embed_lr(model, clicks_weights, clicks_const, X_clicks, b, target='clicks')
    
    model.update()
    
    # --- Total Budget Constraint ---
    # --- 2. Link prediction vars and enforce non-negativity via bounds ---
    # f and g already have lb=0. These equalities force embedded model outputs to be non-negative.
    for i in range(K):
        model.addConstr(f[i] == f_hat_vars[i], name=f"EPCLink_{i}")
        model.addConstr(g[i] == g_hat_vars[i], name=f"ClicksLink_{i}")

    # --- Total Budget Constraint ---
    # sum_i g_i * b_i <= B
    model.addQConstr(b @ g <= budget, name='TotalBudget')

    # --- Objective ---
    # Maximize Net Profit = sum_i g_i * (f_i - b_i)
    model.setObjective((g @ f) - (b @ g), GRB.MAXIMIZE)

    # --- Save Model Formulation ---
    model_dir = Path('opt_results/formulations')
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


def extract_solution(model, b, f, g, keyword_df, keyword_idx_list, region_list, match_list, X=None, weights_dict=None):
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
        weights_dict: Dictionary with model weights (optional)
    """
    b_vals = b.X
    f_vals = f.X
    g_vals = g.X

    # Active rows: bid > 0 (tolerate numerical noise)
    active_idx = np.where(b_vals > 1e-9)[0]
    
    print(f"\nSolution Summary:")
    print(f"  Active rows (bid>0): {len(active_idx)}")
    print(f"  Total spend: ${float(np.sum(b_vals * g_vals)):,.2f}")
    print(f"  Predicted profit: ${model.objVal:,.2f}")
    
    # Build result DataFrame with all active bids
    bids_df = pd.DataFrame({
        'bid': b_vals[active_idx],
        'keyword': [keyword_df.iloc[keyword_idx_list[i]]['Keyword'] for i in active_idx],
        'region': [region_list[i] for i in active_idx],
        'match': [match_list[i] for i in active_idx],
        'predicted_epc': f_vals[active_idx],
        'predicted_clicks': g_vals[active_idx],
    })

    bids_df['predicted_spend'] = bids_df['bid'] * bids_df['predicted_clicks']
    bids_df['predicted_profit'] = bids_df['predicted_clicks'] * (bids_df['predicted_epc'] - bids_df['bid'])
    
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
        default='xgb',
        choices=['lr', 'glm', 'xgb', 'ort', 'mort'],
        help='Algorithm type for EPC model: lr (linear weights), glm (Tweedie GLM weights), xgb (XGBoost Tweedie; no IAI), ort (optimal tree; requires IAI), mort (mirrored ORT with hyperplanes; requires IAI) (default: lr)'
    )
    parser.add_argument(
        '--alg-clicks',
        type=str,
        default='xgb', # 'lr', # 
        choices=['lr', 'glm', 'xgb', 'ort', 'mort'],
        help='Algorithm type for clicks model: lr (linear weights), glm (Tweedie GLM weights), xgb (XGBoost Tweedie; no IAI), ort (optimal tree; requires IAI), mort (mirrored ORT with hyperplanes; requires IAI) (default: lr)'
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
        help='Directory containing cleaned data artifacts and embedding pipelines (default: data/clean)'
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
            # Fall back to loading from latest GKP export
            gkp_df = get_gkp_data(gkp_dir='data/gkp')
            if gkp_df.empty:
                raise FileNotFoundError(
                    "No keywords_classified.csv found and no GKP 'Saved Keywords Stats' file found in data/gkp."
                )
            keywords_to_embed = gkp_df['Keyword'].dropna().astype(str).tolist()
            print(f"Loaded {len(keywords_to_embed)} keywords from latest GKP export")
        
        # Generate embeddings using saved pipeline
        keyword_df = generate_keyword_embeddings_df(
            keywords_to_embed,
            embedding_method=args.embedding_method,
            n_components=50,
            pipeline_dir=args.data_dir,
        )
        
        # Load weights from CSV files (no IAI required)
        weights_dict = load_weights_from_csv(
            embedding_method=args.embedding_method,
            models_dir=args.models_dir
        )
        
        # Create feature matrix (GKP-only; no historical training data)
        feature_matrix_result = create_feature_matrix(
            keyword_df,
            embedding_method=args.embedding_method,
            target_day=args.target_day,
            weights_dict=weights_dict,
            alg_epc=args.alg_conv,
            alg_clicks=args.alg_clicks,
            gkp_dir='data/gkp',
        )
        
        # Unpack results based on model types
        raw_cat_algs = {'ort', 'mort', 'xgb'}
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

        # Embedded mode: supports linear weights (LR/GLM), ORT/MORT (IAI), and XGB (no IAI)
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
            epc_xgb_model_path = Path(args.models_dir) / f'xgb_tweedie_{args.embedding_method}_epc.json'
            epc_xgb_preproc_path = Path(args.models_dir) / f'xgb_tweedie_{args.embedding_method}_epc_preprocess.joblib'
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
            clicks_xgb_model_path = Path(args.models_dir) / f'xgb_tweedie_{args.embedding_method}_clicks.json'
            clicks_xgb_preproc_path = Path(args.models_dir) / f'xgb_tweedie_{args.embedding_method}_clicks_preprocess.joblib'
            if not clicks_xgb_model_path.exists():
                raise FileNotFoundError(f"XGB clicks model not found: {clicks_xgb_model_path}")
            if not clicks_xgb_preproc_path.exists():
                raise FileNotFoundError(f"XGB clicks preprocessor not found: {clicks_xgb_preproc_path}")
            clicks_xgb_paths = (clicks_xgb_model_path, clicks_xgb_preproc_path)
            print(f"  Using XGB clicks model from {clicks_xgb_model_path}")

        model, b, f, g = optimize_bids_embedded(
            X_ort=X_ort,
            X_lr=X_lr,
            weights_dict=weights_dict,
            budget=args.budget,
            max_bid=args.max_bid,
            epc_model=epc_model,
            clicks_model=clicks_model,
            epc_xgb_paths=epc_xgb_paths,
            clicks_xgb_paths=clicks_xgb_paths,
            alg_epc=args.alg_conv,
            alg_clicks=args.alg_clicks
        )

        if model.status == 2 or model.status == 9:
            output_dir = Path('opt_results/bids')
            output_dir.mkdir(exist_ok=True, parents=True)

            model_suffix = f"{args.alg_conv}_{args.alg_clicks}"
            output_file = output_dir / f'optimized_bids_{args.embedding_method}_{model_suffix}.csv'

            bids_df = extract_solution(model, b, f, g, keyword_df, kw_idx_list, region_list, match_list, X=X, weights_dict=weights_dict)

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
