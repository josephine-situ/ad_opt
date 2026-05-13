"""
Maximize clicks for a single day of data, based on a pre-trained XGB model.
1. Create feature matrix from raw data (and all keyword combos).
2. Load and embed pre-trained model.
3. Use Gurobi to maximize clicks under budget constraint.

Example usage:
    python scripts/optimization.py --course ml

Input files:
    data/<course>/gkp/keywords_classified.csv                       [FROM GKP / compare_keywords.py]
    data/<course>/gkp/Saved Keywords Stats *.csv                    [FROM GOOGLE KEYWORD PLANNER]
    data/<course>/clean/unique_keyword_embeddings_bert.csv           (from tidy_get_data.py, BERT only)
    models/<course>_xgb_clicks_model_<emb>.joblib                    (from modeling.py)
    data/<course>/reports/Purchase report.csv                        [FROM GOOGLE ADS REPORTS] (max-purch mode)
    config.py  (course start dates, budgets)

Output files:
    opt_results/<course>/bids/optimized_costs.csv     Optimal cost allocations per keyword-region-match type
    opt_results/<course>/cache/feature_matrix.parquet  Cached feature matrix

Estimated run time (HP Spectre x360, i7-1065G7 @ 1.30 GHz, 4C/8T, 16 GB RAM, no discrete GPU):
    ~5-15 min per course per budget (Gurobi MIP solve; depends on keyword count)
"""

from datetime import datetime
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_pipeline import get_date_features, get_gkp_data, impute_missing_data, merge_with_ads_data, get_conversion_rates
from utils.date_features import COURSE_START_DATES, COURSE_START_DATES_MAP
from config import COURSE_CONFIG
from utils.llm_scoring import get_llm_scores_cached
from utils.embeddings import fit_svd_pipeline, get_raw_bert_embeddings_cached
from tidy_get_data import load_or_cache
from scripts.modeling import _to_float32_csr # necessary to read model correctly

def check_embeddings(embedding_df, base_dir=Path('data/gen_ai')):
    '''Test consistency of embeddings'''

    # Load saved embeddings
    saved_emb = pd.read_csv(base_dir / 'clean/unique_keyword_embeddings_bert.csv')

    # 1. Ensure both are indexed by Keyword for easy alignment
    df1 = embedding_df.set_index('Keyword').sort_index()
    df2 = saved_emb.set_index('Keyword').sort_index()

    # 2. Find common keywords to avoid "Key Not Found" errors
    common_keywords = df1.index.intersection(df2.index)
    df1_shared = df1.loc[common_keywords]
    df2_shared = df2.loc[common_keywords]

    # This checks if the values are the same within a tiny tolerance (default 1e-08)
    is_consistent = np.allclose(df1_shared.values, df2_shared.values, atol=1e-5)

    if is_consistent:
        print("✅ Consistency Check Passed: Embeddings are identical.")
    else:
        # Calculate the average difference to see how far off they are
        diff = np.abs(df1_shared.values - df2_shared.values).mean()
        print(f"❌ Consistency Check Failed: Mean Absolute Difference is {diff}")


def get_emb_from_pipeline(keywords, base_dir=Path('data/gen_ai')):

    # Read possible keywords and create full BERT embeddings, matching run_pipeline.
    raw_emb_cache = base_dir / 'cache' / 'raw_bert_embeddings.pkl'
    raw_emb_map = get_raw_bert_embeddings_cached(keywords, cache_path=raw_emb_cache)

    raw_matrix = np.array([raw_emb_map[kw] for kw in keywords])
    svd_pipeline = fit_svd_pipeline(raw_matrix, n_components=None)
    embeddings = svd_pipeline['normalizer'].transform(raw_matrix)

    embedding_cols = [f'bert_{i}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    embedding_df['Keyword'] = keywords

    check_embeddings(embedding_df, base_dir)

    return embedding_df


def load_impression_multiplier_lookup(base_dir=None):
    """Load region/match-type impression multipliers from impr_multi.csv.

    The file is expected to contain a Region column plus one column each for
    Exact, Phrase, and Broad multipliers.
    """

    if base_dir is None:
        return {}

    mult_path = Path(base_dir) / 'gkp' / 'impr_multi.csv'
    if not mult_path.exists():
        print(f"[Info] Impression multiplier file not found: {mult_path}. Using 1.0 for all caps.")
        return {}

    mult_df = pd.read_csv(mult_path)
    required_cols = {'Region', 'Exact', 'Phrase', 'Broad'}
    if not required_cols.issubset(mult_df.columns):
        missing = sorted(required_cols.difference(mult_df.columns))
        raise ValueError(f"impr_multi.csv is missing expected columns: {missing}")

    match_type_to_col = {
        'Exact match': 'Exact',
        'Phrase match': 'Phrase',
        'Broad match': 'Broad',
    }

    lookup = {}
    for _, row in mult_df.iterrows():
        region = str(row['Region']).strip()
        for match_type, col_name in match_type_to_col.items():
            value = pd.to_numeric(row[col_name], errors='coerce')
            if pd.notna(value):
                lookup[(region, match_type)] = float(value)

    return lookup
    

def create_feature_matrix(keywords, opt_date=None, course_start_dts=None, base_dir=Path('data/gen_ai'), embedding_method='bert', course='gen_ai', raw_emb_map=None, svd_pipeline=None):
    """
    Create feature matrix for optimization.
    
    Args:
        keywords: List of keywords to create features for.
        opt_date: Optimization date (defaults to today).
        course_start_dts: List of course start dates.
        base_dir: Base directory for data files.
        embedding_method: 'bert' for BERT embeddings or 'llm' for LLM relevance scores.
        course: Course identifier ('gen_ai', 'ml', 'sys_eng') - used for LLM scoring.
        raw_emb_map: Optional dict {keyword: raw_embedding_vector}. When provided
                     with *svd_pipeline*, embeddings are computed on-the-fly instead
                     of loading the saved pipeline.
        svd_pipeline: Optional fitted SVD pipeline dict (from
                      ``fit_svd_pipeline``).  Used together with *raw_emb_map*.
    
    Returns:
        DataFrame with all features for optimization.
    """
    # Get all keyword combinations
    regions = ['USA', 'A', 'B'] # Removed region 'C' since low EPC
    match_types = ['Exact match', 'Phrase match', 'Broad match']
    combinations = list(product(keywords, regions, match_types))
    X = pd.DataFrame(combinations, columns=['Keyword', 'Region', 'Match type'])
    print(f"[Info] Created {len(X)} keyword combinations.")

    # Get date features
    # Use today's date if not provided
    if opt_date is None:
        opt_date = datetime.now()
    if course_start_dts is None:
        course_start_dts = COURSE_START_DATES
        
    X['Day'] = opt_date
    X = get_date_features(X, course_start_dts)
    
    # Get keyword stats from GKP
    gkp_df = get_gkp_data(gkp_dir=base_dir / 'gkp')
    gkp_df = impute_missing_data(gkp_df)
    X = merge_with_ads_data(X, gkp_df)

    # Filter out keywords with 0 historical searches
    rows_before = len(X)
    X = X[X['last_month_searches'] > 0].reset_index(drop=True) # np.log1p(0) = 0
    rows_removed = rows_before - len(X)
    print(f"[Info] Removed {rows_removed} rows with 0 historical searches ({rows_removed/rows_before*100:.2f}% of data).")

    # Get keyword embeddings or LLM scores based on method
    if embedding_method == 'llm':
        print(f"[Info] Using LLM relevance scores")
        cache_path = str(base_dir / 'clean/unique_keyword_embeddings_llm.csv')
        llm_df = get_llm_scores_cached(keywords, course=course, cache_path=cache_path)
        X = X.merge(llm_df, on='Keyword', how='left')
        # Fill missing scores with neutral value (3)
        X['llm_relevance_score'] = X['llm_relevance_score'].fillna(3)
        embedding_cols = ['llm_relevance_score']
    else:
        if raw_emb_map is not None and svd_pipeline is not None:
            # Use provided SVD pipeline (daily backtest / oracle evaluation)
            from utils.embeddings import apply_svd_pipeline
            unique_kw_in_X = X['Keyword'].unique().tolist()
            raw_matrix = np.array([raw_emb_map[kw] for kw in unique_kw_in_X])
            transformed = apply_svd_pipeline(raw_matrix, svd_pipeline)
            n_comp = transformed.shape[1]
            embedding_cols = [f'bert_{i}' for i in range(n_comp)]
            emb_df = pd.DataFrame(transformed, columns=embedding_cols)
            emb_df['Keyword'] = unique_kw_in_X
            X = X.merge(emb_df, on='Keyword', how='left')
            print(f"[Info] Using provided SVD pipeline (k={n_comp})")
        else:
            print(f"[Info] Using BERT embeddings")
            emb_df = get_emb_from_pipeline(keywords, base_dir)
            X = X.merge(emb_df, on='Keyword', how='left')
            embedding_cols = [col for col in X.columns if col.startswith('bert_')]

    # Features (and Keyword)
    features = [
        'Keyword',
        'Match type', 'Region', 'day_of_week', 'is_weekend', 'month',
        'is_public_holiday', 'days_to_next_course_start', 'last_month_searches',
        'three_month_avg', 'six_month_avg', 'mom_change', 'search_trend',
        'Competition (indexed value)', 'Top of page bid (low range)',
        'Top of page bid (high range)', 'Feature Space Distance', 'Leaf Uncertainty'
    ]
    features.extend(embedding_cols)

    X = X[features]

    return X

def embed_xgb(model, model_path, X, budget=400, base_dir=None):
    """
    Embed XGBoost model into Gurobi.
    """

    # 1. Load Model and Pipeline
    pipeline = joblib.load(model_path)
    booster = pipeline.named_steps['model'].get_booster()
    
    # Access the preprocessor specifically for metadata (names/scaling)
    preprocessor = pipeline.named_steps['preprocess']

    # Get Base Score
    config = json.loads(booster.save_config())
    base_score = float(config['learner']['learner_model_param']['base_score'])

    impression_multiplier_lookup = load_impression_multiplier_lookup(base_dir=base_dir)

    # 2. Filter Logic
    # We set Cost=0 to check the intercept (base validity)
    X['Cost'] = 0.0

    # Guard: NaN features will silently corrupt tree pruning (both
    # branches fail Python's comparison, causing over-pruning / infeasibility).
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    if nan_cols:
        nan_counts = {c: int(X[c].isna().sum()) for c in nan_cols}
        raise ValueError(
            f"embed_xgb received a feature matrix with NaN values.\n"
            f"Columns with NaNs: {nan_counts}\n"
            f"Fix upstream data (e.g. missing course start dates in config.py)."
        )

    # Use the full pipeline to predict (handles float32 cast automatically)
    pred_clicks_cost0 = pipeline.predict(X)
    
    valid_indices = [i for i, pred in enumerate(pred_clicks_cost0) if pred >= 0]
    print(f"[Info] Pruned {len(X) - len(valid_indices)} rows with negative predicted clicks at Cost=0.")
    X = X.iloc[valid_indices].reset_index(drop=True)

    # 3. Preprocess X (CRITICAL CHANGE)
    # We use pipeline[:-1] to run everything UP TO the model (Preprocessor + Float32 Cast)
    # This ensures X_proc is Float32 and matches the tree splits exactly.
    X_proc = pipeline[:-1].transform(X)

    # 4. Extract Metadata
    # We use the preprocessor step to get names/scales since the caster step might not store them
    feature_names = preprocessor.get_feature_names_out()
    cost_idx = list(feature_names).index('num__Cost')
    
    # Retrieve the scale for the Cost variable
    # Note: This assumes 'num' is the name of your numerical transformer in ColumnTransformer
    cost_scale = preprocessor.named_transformers_['num'].scale_[
        list(preprocessor.named_transformers_['num'].feature_names_in_).index('Cost')
    ]

    # 4. Parse Tree Structure
    def parse_single_tree(node, current_conds):
        """Helper to extract paths. Standard recursive parsing."""
        if 'leaf' in node:
            yield (current_conds, node['leaf'])
        else:
            try:
                feat_id = int(node['split'].replace('f', ''))
            except ValueError:
                return 
            
            threshold = node['split_condition']
            yes_id = node['yes'] 
            no_id = node['no']
            
            # Find children
            yes_child = next(c for c in node['children'] if c['nodeid'] == yes_id)
            no_child = next(c for c in node['children'] if c['nodeid'] == no_id)
            
            # Recurse Left (Yes)
            yield from parse_single_tree(yes_child, current_conds + [(feat_id, 'lt', threshold)])
            
            # Recurse Right (No)
            yield from parse_single_tree(no_child, current_conds + [(feat_id, 'ge', threshold)])

    def get_tree_paths(booster):
        tree_dumps = booster.get_dump(dump_format='json')
        all_paths = []
        for tree_json in tree_dumps:
            tree = json.loads(tree_json)
            paths = list(parse_single_tree(tree, []))
            all_paths.append(paths)
        return all_paths
    
    tree_paths = get_tree_paths(booster)

    # 5. Build Gurobi Constraints
    cost_vars = [] 
    pred_vars = [] 
    
    MAX_LHS = (budget / cost_scale) * 1.05 
    MIN_LHS = 0.0
    
    # Safety margin: Use a tiny epsilon ONLY for strict inequalities (<)
    # This prevents 'Dead Zones' where Gurobi can't find a valid path
    EPSILON = 1e-5 
    print(f"[Info] Using one-sided epsilon: {EPSILON} to handle strict inequalities.")
    
    K = len(X_proc)

    for i in tqdm(range(K), desc="Embedding Rows"):

        # A. Decision Variable 'x' (Cost)
        current_cost = model.addVar(lb=0.0, name=f"Cost_{i}")
        cost_vars.append(current_cost)

        # B. Prediction Variable
        pred_var = model.addVar(lb=-GRB.INFINITY, name=f"pred_{i}")
        tree_vars_sum = 0
        
        for t_idx, paths in enumerate(tree_paths):
            leaf_vars = []
            leaf_vals = []
            
            for leaf_idx, (conds, leaf_val) in enumerate(paths):
                
                is_feasible = True
                dynamic_conds = []
                
                for feat_idx, op, thr in conds:
                    if feat_idx == cost_idx:
                        # Dynamic Feature: Add constraint later
                        dynamic_conds.append((op, thr))
                    else:
                        # Static Feature: Prune immediately using Standard Math
                        val = X_proc[i, feat_idx]
                        
                        if op == 'lt' and not (val < thr): 
                            is_feasible = False; break
                        elif op == 'ge' and not (val >= thr): 
                            is_feasible = False; break
                
                if is_feasible:
                    z = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t_idx}_{leaf_idx}")
                    leaf_vars.append(z)
                    leaf_vals.append(leaf_val)
                    
                    # Big-M Constraints for Cost (with One-Sided Epsilon)
                    for op, thr in dynamic_conds:
                        lhs = current_cost / cost_scale
                        
                        if op == "lt":
                            # Logic: Cost < Threshold
                            # Gurobi Implementation: Cost <= Threshold - Epsilon
                            # Safe bound prevents negative values if thr is near zero
                            bound = max(0.0, thr - EPSILON)
                            
                            M = MAX_LHS - bound
                            model.addConstr(lhs <= bound + M * (1 - z), name=f"split_lt_{i}_{t_idx}")

                        elif op == "ge":
                            # Logic: Cost >= Threshold
                            # Gurobi Implementation: Cost >= Threshold (Exact Match)
                            # No margin needed because XGBoost is inclusive for >=
                            bound = thr
                            
                            M = bound - MIN_LHS
                            model.addConstr(lhs >= bound - M * (1 - z), name=f"split_ge_{i}_{t_idx}")

            # Tree Aggregation
            if leaf_vars:
                model.addConstr(gp.quicksum(leaf_vars) == 1, name=f"tree_active_{i}_{t_idx}")
                tree_vars_sum += gp.LinExpr(leaf_vals, leaf_vars)

        # Prediction Constraint
        model.addConstr(pred_var == tree_vars_sum + base_score, name=f"def_pred_{i}")
        pred_vars.append(pred_var)

        # Cap incremental clicks by an impression-adjusted daily search volume.
        match_type = X.iloc[i]['Match type']
        region = X.iloc[i]['Region']
        multiplier = impression_multiplier_lookup.get((region, match_type), 1.0)
        historical_searches = X.iloc[i]['last_month_searches']
        base_pred = pred_clicks_cost0[valid_indices[i]]
        daily_search_volume = (np.expm1(historical_searches) + 1.0) / 30.0
        model.addConstr(
            pred_var - base_pred <= multiplier * daily_search_volume,
            name=f"search_volume_cap_{i}"
        )
    
    model.update()
    return cost_vars, pred_vars, X

def optimize_bids(X, model_path, budget=400, kw_df=None, order_budget=False, max_purch=False, base_dir=None, min_spend=None, time_limit=600):
    """ Maximize clicks with embedded XGBoost model. 

    budget: total budget across all regions
    min_spend: If set, each active keyword must spend at least this amount (e.g. 1.0).
               Adds binary activation variables z_i with:
                 x_i <= budget * z_i,  x_i >= min_spend * z_i,  z_i in {0,1}
    time_limit: Gurobi solve time limit in seconds. None or 0 for no limit.
    
    Formulation:
        max   sum_i  g_i
        s.t.  sum_i  x_i <= budget
                g_i = Model_clicks(x_i, w_i)  for all i
                x_i >= 0  for all i
                g_i >= 0  for all i
    """

    model = gp.Model("max_clicks")
    # model.setParam('OutputFlag', 1)
    if time_limit:
        model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 0.01)

    cost_vars, pred_vars, X = embed_xgb(model, model_path, X, budget=budget, base_dir=base_dir)

    if kw_df is not None:
        X = X.merge(
            kw_df[['Keyword', 'Origin']],
            on='Keyword',
            how='left'
        )

    # Objective
    if max_purch:
        rates = get_conversion_rates(base_dir=base_dir)
        X = X.merge(rates, on='Region', how='left')
        X['Purch_rate'] = X['Purch_rate'].fillna(0)
        model.setObjective(gp.quicksum(pred_vars[i] * X.loc[i, 'Purch_rate'] for i in range(len(pred_vars))), GRB.MAXIMIZE)
    else:
        # Maximize clicks
        model.setObjective(gp.quicksum(pred_vars), GRB.MAXIMIZE)

    # Create regional budget variables
    regions = ['USA', 'A', 'B']
    region_budgets = {}
    for region in regions:
        region_budgets[region] = model.addVar(lb=0.0, name=f"Budget_{region}")

    # Total budget constraint: sum(regional_budgets) == budget
    model.addConstr(
        gp.quicksum(region_budgets.values()) == budget,
        name='total_budget_constraint'
    )

    # Regional budget constraints: sum(costs in region) <= regional_budget
    for region in regions:
        region_indices = X.index[X['Region'] == region].tolist()
        if region_indices:
            model.addConstr(
                gp.quicksum(cost_vars[i] for i in region_indices) <= region_budgets[region],
                name=f'budget_constraint_{region}'
            )
        else:
             # If no keywords for a region, its budget technically can be anything if not constrained otherwise,
             # but to avoid wasting budget, we can say budget >= 0 (implied) and usually 
             # the logic would just put 0 there if there's no utility.
             pass

    if order_budget:
        # Add ordering constraints: B_{USA} >= B_{A} >= B_{B}
        model.addConstr(region_budgets['USA'] >= region_budgets['A'], name='order_budget_USA_A')
        model.addConstr(region_budgets['A'] >= region_budgets['B'], name='order_budget_A_B')

    # Minimum-spend constraints: if keyword i is active (z_i=1), it must
    # spend at least min_spend dollars.
    if min_spend is not None:
        c_min = float(min_spend)
        print(f"[Info] Adding minimum-spend constraints: c_min = ${c_min:.2f}")
        for i, x_i in enumerate(cost_vars):
            z_i = model.addVar(vtype=GRB.BINARY, name=f"active_{i}")
            model.addConstr(x_i <= budget * z_i, name=f"min_spend_ub_{i}")
            model.addConstr(x_i >= c_min * z_i, name=f"min_spend_lb_{i}")

    # Optimize
    model.optimize()

    # If presolve returns INF_OR_UNBD, re-solve with DualReductions=0 to disambiguate.
    if model.status == GRB.INF_OR_UNBD:
        print("[Warning] Gurobi returned INF_OR_UNBD (status 4). Re-solving with DualReductions=0...")
        try:
            model.setParam('DualReductions', 0)
            model.setParam('InfUnbdInfo', 1)
            model.optimize()

            if model.status == GRB.INFEASIBLE:
                report_path = Path('opt_results/analysis/infeasibility_report.ilp')
                report_path.parent.mkdir(parents=True, exist_ok=True)
                print("[Info] Model is infeasible after disambiguation. Computing IIS...")
                model.computeIIS()
                model.write(str(report_path))
                print(f"[Info] Wrote IIS report to '{report_path}'.")
        except Exception as e:
            print(f"[Warning] Failed to disambiguate status 4: {type(e).__name__}: {e}")

    return model, cost_vars, pred_vars, X

def extract_solution(model, cost_vars, pred_vars, model_path, X):
    """
    Extracts solution from Gurobi and aligns it with the original Dataframe.
    
    Args:
        model: The optimized Gurobi model.
        cost_vars: List of Gurobi variables for Cost.
        pred_vars: List of Gurobi variables for Predicted Clicks.
        model_path: Path to the .joblib model file.
        X: The processed DataFrame (filtered for positive predictive clicks at cost=0) used for input (must contain metadata columns).
        
    Returns:
        pd.DataFrame: Results containing only the valid (optimized) rows.
    """
    
    # 1. Check Optimization Status
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print(f"[Error] Optimization failed or interrupted. Status: {model.status}")
        return None

    print(f"[Info] Optimization Success. Objective Value: {model.ObjVal:.4f}")

    # 2. Re-calculate Valid Indices (The Alignment Fix)
    # We must replicate the exact filtering logic from 'embed_xgb' to know 
    # which rows in X correspond to the variables in cost_vars.
    
    pipeline = joblib.load(model_path)
    
    # Create a temporary copy to check the 'Cost=0' condition
    X_temp = X.copy()
    X_temp['Cost'] = 0.0
    
    # Predict using the full pipeline
    # (This handles scaling, encoding, and the base_score automatically)
    base_preds = pipeline.predict(X_temp)

    # 4. Construct Results DataFrame
    # Pull metadata from the valid rows of X
    meta_cols = ['Keyword', 'Region', 'Match type', 'Origin']
    if 'Feature Space Distance' in X.columns:
        meta_cols.append('Feature Space Distance')
    results_df = X[meta_cols].copy()
    
    # Extract values from Gurobi variables
    results_df['Optimal Cost'] = [var.X for var in cost_vars]
    results_df['Gurobi Pred'] = [var.X for var in pred_vars]
    results_df['Gurobi Pred over Base'] = results_df['Gurobi Pred'] - base_preds
    # Filter out rows where Optimal Cost is zero (not selected)
    filt_opt_cost = results_df['Optimal Cost'] > 5e-4
    results_df = results_df[filt_opt_cost].reset_index(drop=True)
    print(f"[Info] Total clicks over base (cost=0): {results_df['Gurobi Pred over Base'].sum():.4f}")

    if results_df.empty:
        results_df['Actual Model Pred'] = pd.Series(dtype=float)
        results_df['Diff'] = pd.Series(dtype=float)
        print("[Info] Zero-bid optimum detected; returning empty optimization results.")
        return results_df

    # 5. Validation and Boundary Adjustment
    # Run the Optimal Costs back through the actual XGBoost model to verify accuracy
    # If discrepancy is large, slightly adjust costs to move away from tree boundaries
    X_validate = X.copy()[filt_opt_cost].reset_index(drop=True)
    X_validate['Cost'] = results_df['Optimal Cost'].values
    
    # The pipeline prediction includes the base_score naturally
    results_df['Actual Model Pred'] = pipeline.predict(X_validate)
    results_df['Diff'] = results_df['Gurobi Pred'] - results_df['Actual Model Pred']
    
    # Feature Space Distance and Leaf Uncertainty
    X_val_proc = pipeline[:-1].transform(X_validate)
    booster = pipeline.named_steps['model'].get_booster()
    if "cast" in pipeline.named_steps:
        import scipy.sparse as sp
        if sp.issparse(X_val_proc):
            X_val_proc_sparse = pipeline.named_steps['cast'].transform(X_val_proc)
        else:
            X_val_proc_sparse = X_val_proc
    else:
        X_val_proc_sparse = X_val_proc
        
    import xgboost as xgb
    dmatrix = xgb.DMatrix(X_val_proc_sparse)

    import joblib
    import os
    import numpy as np
    from pathlib import Path
    
    course = model_path.split(os.sep)[-2] 
    nn_path = Path(model_path).parent / "feature_nn.joblib"
    if nn_path.exists():
        nn = joblib.load(str(nn_path))
        X_val_num = X_validate.select_dtypes(include=["number"]).copy()
        X_val_num = X_val_num.fillna(0)
        distances, _ = nn.kneighbors(X_val_num)
        results_df['Feature Space Distance'] = distances.mean(axis=1)
    else:
        results_df['Feature Space Distance'] = np.nan

    num_trees = booster.best_iteration + 1 if hasattr(booster, 'best_iteration') and booster.best_iteration is not None else booster.num_boosted_rounds()
    tree_preds = [booster.predict(dmatrix, iteration_range=(i, i+1)) for i in range(num_trees)]
    results_df['Leaf Uncertainty'] = np.var(tree_preds, axis=0)

    # Identify rows with significant discrepancy and attempt adjustment
    DISCREPANCY_THRESHOLD = 0.1
    high_disc_mask = results_df['Diff'].abs() > DISCREPANCY_THRESHOLD
    n_high_disc = high_disc_mask.sum()
    
    if n_high_disc > 0:
        print(f"[Warning] Found {n_high_disc} rows with |Diff| > {DISCREPANCY_THRESHOLD}.")
    
    max_diff = results_df['Diff'].abs().max()
    mean_diff = results_df['Diff'].abs().mean()
    print(f"[Info] Max discrepancy between Gurobi and XGBoost: {max_diff:.6f}")
    print(f"[Info] Mean absolute discrepancy: {mean_diff:.6f}")

    print("[Info] Sample of Optimization Results:")
    print(results_df.head())

    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--course', required=True, help='Course name (e.g. gen_ai, ml, sys_eng, sys_think)')
    parser.add_argument('--embedding-method', default='bert', choices=['bert', 'llm'], help='Embedding method: bert or llm (default: bert)')
    parser.add_argument('--date', required=True, help='Optimization date (YYYY-MM-DD) used to select the exact dated model and output file')
    parser.add_argument('--budget', type=float, nargs='+', default=None, help='Total budget(s) to test (default: from config)')
    parser.add_argument('--order-budget', action='store_true', default=True, help='Use B_{USA} >= B_{A} >= B_{B}') # Default to True. Change here if want to remove.
    parser.add_argument('--max-purch', action='store_true', default=True, help='Use max purchases objective instead of clicks') # Default to True. Change here if want to remove.
    parser.add_argument('--min-spend', type=float, default=None, help='Minimum spend per active keyword (e.g. 1.0). If not set, no minimum-spend constraint is used.')
    args = parser.parse_args()

    opt_date = datetime.strptime(args.date, '%Y-%m-%d')

    if args.budget is None:

        from scripts.run_pipeline import calculate_daily_budget
        args.budget = [calculate_daily_budget(args.course)]

    embedding_method = args.embedding_method

    print(f"Optimizing bids for course: {args.course}")
    print(f"Embedding method: {embedding_method}")
    print(f"Budget(s): {args.budget}")
    print(f"Order budget: {args.order_budget}")
    print(f"Max purchases objective: {args.max_purch}")
    print(f"Min spend per keyword: {args.min_spend}")

    base_dir = Path(f'data/{args.course}')
    
    # Step 1: Create feature matrix with caching
    kw_df = pd.read_csv(base_dir / 'gkp/keywords_classified.csv')
    keywords = kw_df['Keyword'].tolist()

    raw_emb_cache = base_dir / 'cache' / 'raw_bert_embeddings.pkl'
    raw_emb_map = get_raw_bert_embeddings_cached(keywords, cache_path=raw_emb_cache)
    raw_matrix = np.array([raw_emb_map[kw] for kw in keywords])
    svd_pipeline = fit_svd_pipeline(raw_matrix, n_components=None)
    
    cache_dir = Path(f'opt_results/{args.course}/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    res_dir = Path(f'opt_results/{args.course}/bids')
    res_dir.mkdir(parents=True, exist_ok=True)

    date_str = opt_date.strftime('%Y%m%d')
    
    X = load_or_cache(
        create_feature_matrix,
        cache_dir / 'feature_matrix.parquet',
        False,  # use_cache (defaults to False, meaning it will recompute)
        keywords,
        opt_date,
        COURSE_START_DATES_MAP.get(args.course, []),
        base_dir,
        raw_emb_map=raw_emb_map,
        svd_pipeline=svd_pipeline,
    )

    X = X[X['Region'] != 'C']  # Filter out region C due to low EPC
    
    # Optimize bids using Gurobi
    model_path = Path(f'models/{args.course}/xgb_clicks_model_{embedding_method}_{date_str}.joblib')
    if not model_path.exists():
        raise FileNotFoundError(f"Expected model file not found: {model_path}")

    for b in args.budget:
        print(f"\n--- Budget: {b} ---")
        model, cost_vars, pred_vars, X_opt = optimize_bids(
            X.copy(), model_path, budget=b, kw_df=kw_df,
            order_budget=args.order_budget, max_purch=args.max_purch,
            base_dir=base_dir, min_spend=args.min_spend
        )

        # Extract solution and validate predictions
        results_df = extract_solution(model, cost_vars, pred_vars, model_path, X_opt)
        if results_df is not None:
            dated_out = res_dir / f"optimized_costs_{date_str}.csv"
            dated_out.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(dated_out, index=False)
            print(f"[Info] Optimization results saved to '{dated_out}'.")


if __name__ == '__main__':
    main()