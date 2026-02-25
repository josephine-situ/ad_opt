#!/usr/bin/env python
"""
Interpret XGBoost clicks model using variable importance and SHAP values.

Loads the same SVD pipeline that was fitted alongside the XGBoost model
during backtesting, applies it to raw BERT embeddings, and then runs
SHAP / variable-importance analysis on the model.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.modeling import _to_float32_csr
from utils.embeddings import (
    get_raw_bert_embeddings_cached,
    replace_embeddings,
)

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--course', default='sys_think', help='Course name (default: ml)')
parser.add_argument('--k-policy', default='k_full', help='SVD k-policy folder name (default: k_full)')
parser.add_argument('--model-date', default='2025-12-01', help='Model date stamp (default: 2025-12-01)')
args = parser.parse_args()

base_dir = Path(f'data/{args.course}')
models_dir = Path(f'models/{args.course}/backtests/svd_sweep/{args.k_policy}')
MODEL_PATH = models_dir / f'xgb_clicks_model_{args.model_date}.joblib'
SVD_PATH = models_dir / f'svd_pipeline_{args.model_date}.joblib'
OUTPUT_DIR = Path('model_interpretability') / args.course
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 70)
print(f"XGBoost Clicks Model Interpretation ({args.course})")
print("=" * 70)

# ============================================================================
# Load data (full dataset – need 'Keyword' column for SVD embedding lookup)
# ============================================================================
print("\n1. Loading data...")

embedding_choice = 'bert'
full_file = base_dir / 'clean' / f'ad_opt_data_{embedding_choice}.csv'

df_full = pd.read_csv(full_file)
print(f"  Full data shape: {df_full.shape}")

# Use the latest 25 % of rows as "test" (mirrors prepare_train_test_split)
from sklearn.model_selection import train_test_split
_, X_test = train_test_split(df_full, test_size=0.25, random_state=42)
X_test = X_test.copy()

print(f"  Test split shape: {X_test.shape}")
print(f"  Columns: {list(X_test.columns)[:10]}...")
print(f"  Sample features: {list(X_test.columns[-10:])}")

# ============================================================================
# Load SVD pipeline & apply to test data
# ============================================================================
print("\n2. Loading SVD pipeline and transforming embeddings...")

svd_pipeline = None
if SVD_PATH.exists():
    svd_pipeline = joblib.load(SVD_PATH)
    n_comp = svd_pipeline['n_components']
    has_svd = svd_pipeline['svd'] is not None
    print(f"  Loaded SVD pipeline: {SVD_PATH.name}  (n_components={n_comp}, svd={'yes' if has_svd else 'no (normalize only)'})")

    # Get raw BERT embeddings for all keywords in the test set
    raw_emb_cache = base_dir / 'cache' / 'raw_bert_embeddings.pkl'
    unique_keywords = X_test['Keyword'].unique().tolist()
    raw_emb_map = get_raw_bert_embeddings_cached(
        unique_keywords, cache_path=raw_emb_cache,
    )
    print(f"  Raw BERT embeddings: {len(raw_emb_map)} keywords")

    # Replace the pre-computed bert_* columns with SVD-transformed embeddings
    X_test, emb_cols = replace_embeddings(X_test, raw_emb_map, svd_pipeline, prefix='bert')
    print(f"  After SVD replace_embeddings: {X_test.shape}  (embedding cols: {len(emb_cols)})")
else:
    print(f"  WARNING: SVD pipeline not found at {SVD_PATH} – using raw embedding columns from CSV")

# Drop non-feature columns before model inference
non_feature_cols = ['Day', 'Keyword', 'Clicks', 'Competition']
X_test = X_test.drop(columns=[c for c in non_feature_cols if c in X_test.columns])
print(f"  Final feature matrix shape: {X_test.shape}")

# ============================================================================
# Load XGBoost clicks model
# ============================================================================
print("\n3. Loading XGBoost clicks model...")

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"  Loaded clicks model: {MODEL_PATH.name}")
    
    # Extract the XGBoost model from the pipeline
    if hasattr(pipeline, 'named_steps'):
        # It's a Pipeline, extract the final estimator
        model = pipeline.named_steps.get('model') or pipeline.steps[-1][1]
        preprocessor = pipeline[:-1]  # All steps except the final model
        print(f"  Extracted XGBoost model from pipeline")
    else:
        # It's already a model
        model = pipeline
        preprocessor = None
    
except Exception as e:
    print(f"  Error loading XGBoost model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    preprocessor = None

# ============================================================================
# Variable Importance Analysis
# ============================================================================
print("\n4. Computing Variable Importance...")

if model is not None:
    try:
        # Get variable importance using sklearn if available, otherwise try XGBoost
        try:
            importance_dict = model.get_score(importance_type='weight')
        except AttributeError:
            # sklearn-style model
            try:
                # Try to get feature names from preprocessor
                if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                    importance_dict = {feature_names[i]: v for i, v in enumerate(model.feature_importances_)}
                else:
                    importance_dict = {f'Feature_{i}': v for i, v in enumerate(model.feature_importances_)}
            except Exception as e:
                print(f"    Warning: Could not extract feature names: {e}")
                importance_dict = {f'Feature_{i}': v for i, v in enumerate(model.feature_importances_)}
        
        imp_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=False)
        
        print("\n  Clicks Model - Top 15 Important Features (by frequency):")
        print("  " + "-" * 60)
        for feat, imp_val in imp_df.head(15).itertuples(index=False):
            print(f"    {feat:45s}: {imp_val:10.4f}")
            
        # Save to CSV
        imp_df.to_csv(OUTPUT_DIR / 'variable_importance_clicks.csv', index=False)
        print(f"\n  Saved to: {OUTPUT_DIR / 'variable_importance_clicks.csv'}")
            
    except Exception as e:
        print(f"  Error getting variable importance: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SHAP Values Analysis
# ============================================================================
print("\n4. Computing SHAP Values (this may take a moment)...")

try:
    if model is not None:
        print("  Computing SHAP for clicks model...")
        try:
            # Apply preprocessing if we have a pipeline
            if preprocessor is not None:
                print("  Applying preprocessing to test data...")
                X_for_shap = preprocessor.transform(X_test)
                if hasattr(X_for_shap, 'toarray'):
                    X_for_shap = X_for_shap.toarray()
                # Get feature names from preprocessor if available
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except AttributeError:
                    # If preprocessor doesn't have get_feature_names_out, try without last step
                    try:
                        feature_names = preprocessor[:-1].get_feature_names_out()
                    except (AttributeError, TypeError, IndexError):
                        feature_names = None
                
                if feature_names is not None:
                    X_for_shap = pd.DataFrame(X_for_shap, columns=feature_names)
                else:
                    X_for_shap = pd.DataFrame(X_for_shap, columns=[f'Feature_{i}' for i in range(X_for_shap.shape[1])])
                print(f"  Preprocessed data shape: {X_for_shap.shape}")
            else:
                X_for_shap = X_test
            
            # Create SHAP explainer for XGBoost using raw test data
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_for_shap)
            
            # Calculate mean absolute SHAP values
            shap_importance = pd.DataFrame({
                'Feature': [str(f) for f in X_for_shap.columns],
                'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
            }).sort_values('Mean_Abs_SHAP', ascending=False)
            
            print("\n  Clicks Model - SHAP-based Feature Importance (Top 15):")
            print("  " + "-" * 60)
            for idx, row in shap_importance.head(15).iterrows():
                print(f"    {row['Feature']:45s}: {row['Mean_Abs_SHAP']:10.6f}")

            # Save SHAP importance
            shap_importance.to_csv(OUTPUT_DIR / 'shap_importance_clicks.csv', index=False)
            print(f"\n  Saved to: {OUTPUT_DIR / 'shap_importance_clicks.csv'}")

            # Create SHAP summary plot
            print("    Creating SHAP summary plot...")
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_for_shap, show=False)
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / 'shap_summary_clicks.png', dpi=100, bbox_inches='tight')
                plt.close()
                print(f"  SHAP summary plot saved to: {OUTPUT_DIR / 'shap_summary_clicks.png'}")
            except Exception as e:
                print(f"  Error creating SHAP summary plot: {e}")
        except Exception as e:
            print(f"  Error computing SHAP: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  Warning: Model not loaded, skipping SHAP analysis")

except Exception as e:
    print(f"  Error in SHAP analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Interpretation complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)