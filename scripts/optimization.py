"""
Maximize clicks fora single day of data, based on a pre-trained XGB model.
1. Create feature matrix from raw data (and all keyword combos).
2. Load and embed pre-trained model.
3. Use Gurobi to maximize clicks under budget constraint.
"""

from datetime import datetime
import pickle
import sys
import joblib
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from itertools import product
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_pipeline import get_date_features, get_gkp_data, impute_missing_data, merge_with_ads_data
from utils.date_features import COURSE_START_DATES
from tidy_get_data import load_or_cache

def check_embeddings(embedding_df):
    '''Test consistency of embeddings'''

    # Load saved embeddings
    saved_emb = pd.read_csv('data/clean/unique_keyword_embeddings_bert.csv')

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


def get_emb_from_pipeline(keywords):

    # Read possible keywords and create embeddings for them
    emb_pipe_file = 'data/clean/bert_pipeline_50d.pkl'
    with open(emb_pipe_file, 'rb') as f:
        emb_pipeline = pickle.load(f)

    svd = emb_pipeline['svd']
    normalizer = emb_pipeline['normalizer']

    model_name = emb_pipeline['model_name']
    embedding_model = SentenceTransformer(model_name)

    embeddings = embedding_model.encode(
        keywords,
        convert_to_numpy=True,
    )

    # Apply post embedding steps: SVD + Normalization
    embeddings = svd.transform(embeddings)
    embeddings = normalizer.transform(embeddings)

    embedding_cols = [f'bert_{i}' for i in range(50)]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
    embedding_df['Keyword'] = keywords

    check_embeddings(embedding_df)

    return embedding_df
    

def create_feature_matrix(keywords, opt_date=None):
    
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
    X['Day'] = opt_date
    X = get_date_features(X, COURSE_START_DATES)
    
    # Get keyword stats from GKP
    gkp_df = get_gkp_data()
    gkp_df = impute_missing_data(gkp_df)
    X = merge_with_ads_data(X, gkp_df)

    # Get keyword embeddings
    emb_df = get_emb_from_pipeline(keywords)
    X = X.merge(emb_df, on='Keyword', how='left')

    # Features (and Keyword)
    features = [
        'Keyword',
        'Match type', 'Region', 'day_of_week', 'is_weekend', 'month',
        'is_public_holiday', 'days_to_next_course_start', 'last_month_searches',
        'three_month_avg', 'six_month_avg', 'mom_change', 'search_trend',
        'Competition (indexed value)', 'Top of page bid (low range)',
        'Top of page bid (high range)'
    ]
    bert_cols = [col for col in X.columns if col.startswith('bert_')]
    features.extend(bert_cols)

    X = X[features]

    return X

def embed_xgb(model, model_path, X, budget=400):
    """
    Embed XGBoost model into Gurobi.
    """

    # 1. Load Model and Preprocessor
    pipeline = joblib.load(model_path)
    booster = pipeline.named_steps['model'].get_booster()
    preprocessor = pipeline.named_steps['preprocess']

    # Get Base Score
    config = json.loads(booster.save_config())
    base_score = float(config['learner']['learner_model_param']['base_score'])

    # 2. Filter Logic
    X['Cost'] = 0.0
    X_cost0 = X.copy()
    pred_clicks_cost0 = pipeline.predict(X_cost0)
    valid_indices = [i for i, pred in enumerate(pred_clicks_cost0) if pred >= 0]
    print(f"[Info] Pruned {len(X) - len(valid_indices)} rows with negative predicted clicks at Cost=0.")
    X = X.iloc[valid_indices].reset_index(drop=True)

    # 3. Preprocess X
    X_proc = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    cost_idx = list(feature_names).index('num__Cost')
    cost_scale = preprocessor.named_transformers_['num'].scale_[cost_idx]

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
    margin = 5e-4 # Safety margin
    
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
                    
                    # Big-M Constraints for Cost (with Safety Margins)
                    for op, thr in dynamic_conds:
                        lhs = current_cost / cost_scale
                        if op == "lt":
                            # Cost <= Threshold - Margin
                            bound = thr - margin
                            M = MAX_LHS - bound
                            model.addConstr(lhs <= bound + M * (1 - z))
                        elif op == "ge":
                            # Cost >= Threshold + Margin
                            bound = thr + margin
                            M = bound - MIN_LHS
                            model.addConstr(lhs >= bound - M * (1 - z))

            # Tree Aggregation
            if leaf_vars:
                model.addConstr(gp.quicksum(leaf_vars) == 1, name=f"tree_active_{i}_{t_idx}")
                tree_vars_sum += gp.LinExpr(leaf_vals, leaf_vars)

        # Prediction Constraint
        model.addConstr(pred_var == tree_vars_sum + base_score, name=f"def_pred_{i}")
        pred_vars.append(pred_var) 
    
    model.update()
    return cost_vars, pred_vars, X

def optimize_bids(X, model_path, budget=400, x_max=None, kw_df=None, alpha=1.0):
    """ Maximize clicks with embedded XGBoost model. 
    
    Formulation:
        max   sum_i  g_i
        s.t.  sum_i  x_i <= budget
                g_i = Model_clicks(x_i, w_i)  for all i
                x_i >= 0  for all i
                g_i >= 0  for all i
    """

    model = gp.Model("max_clicks")
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 300)
    model.setParam('MIPGap', 0.02)

    cost_vars, pred_vars, X = embed_xgb(model, model_path, X)

    # Objective
    model.setObjective(gp.quicksum(pred_vars), GRB.MAXIMIZE)

    # Budget constraint
    model.addConstr(gp.quicksum(cost_vars) <= budget, name='budget_constraint')

    # Optional: x_max constraint (restrict max cost per keyword)
    if x_max is not None:
        model.addConstrs((cost_vars[i] <= x_max for i in range(len(cost_vars))), name='x_max_constraint')
    
    # Optional: New keyword budget constraint
    if kw_df is not None:
        X = X.merge(
            kw_df[['Keyword', 'Origin']],
            on='Keyword',
            how='left'
        )
        if alpha < 1.0:
            new_kw_indices = X.index[X['Origin'] != 'existing'].tolist()
            if new_kw_indices:
                model.addConstr(
                    gp.quicksum(cost_vars[i] for i in new_kw_indices) <= alpha * budget,
                    name='new_keyword_budget_constraint'
                )

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
    results_df = X[['Keyword', 'Region', 'Match type', 'Origin']].copy()
    
    # Extract values from Gurobi variables
    results_df['Optimal Cost'] = [var.X for var in cost_vars]
    results_df['Gurobi Pred'] = [var.X for var in pred_vars]
    results_df['Gurobi Pred over Base'] = results_df['Gurobi Pred'] - base_preds
    # Filter out rows where Optimal Cost is zero (not selected)
    filt_opt_cost = results_df['Optimal Cost'] > 5e-4
    results_df = results_df[filt_opt_cost].reset_index(drop=True)
    print(f"[Info] Total clicks over base (cost=0): {results_df['Gurobi Pred over Base'].sum():.4f}")

    # 5. Validation (Optional but Recommended)
    # Run the Optimal Costs back through the actual XGBoost model to verify accuracy
    X_validate = X.copy()[filt_opt_cost].reset_index(drop=True)
    X_validate['Cost'] = results_df['Optimal Cost']
    
    # The pipeline prediction includes the base_score naturally
    results_df['Actual Model Pred'] = pipeline.predict(X_validate)
    results_df['Diff'] = results_df['Gurobi Pred'] - results_df['Actual Model Pred']
    
    max_diff = results_df['Diff'].abs().max()
    print(f"[Info] Max discrepancy between Gurobi and XGBoost: {max_diff:.6f}")

    print("[Info] Sample of Optimization Results:")
    print(results_df.head())

    return results_df

def main():
    
    # Step 1: Create feature matrix with caching
    kw_df = pd.read_csv('data/gkp/keywords_classified.csv')
    keywords = kw_df['Keyword'].tolist()
    
    cache_dir = Path('opt_results/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    res_dir = Path('opt_results/bids')
    res_dir.mkdir(parents=True, exist_ok=True)
    
    X = load_or_cache(
        create_feature_matrix,
        cache_dir / 'feature_matrix.parquet',
        False,  # force_reload
        keywords
    )

    X = X[X['Region'] != 'C'][:10000]  # Filter out region C due to low EPC
    
    # Optimize bids using Gurobi
    model_path = 'models/xgb_clicks_model.joblib'
    X = X[:10]  # For testing with a smaller subset
    model, cost_vars, pred_vars, X = optimize_bids(X, model_path, kw_df=kw_df)

    # Extract solution and validate predictions
    results_df = extract_solution(model, cost_vars, pred_vars, model_path, X)
    if results_df is not None:
        results_df.to_csv(res_dir / 'optimized_costs.csv', index=False)
        print("[Info] Optimization results saved to 'opt_results/bids/optimized_costs.csv'.")


if __name__ == '__main__':
    main()