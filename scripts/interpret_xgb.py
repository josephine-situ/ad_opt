#!/usr/bin/env python
"""
Interpret XGBoost models using variable importance and SHAP values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shap

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
EMBEDDING_METHOD = 'bert'  # or 'tfidf'
OUTPUT_DIR = Path(__file__).parent.parent / 'model_interpretability'
OUTPUT_DIR.mkdir(exist_ok=True)
model_dir = Path(__file__).parent.parent / 'models'

print("=" * 70)
print("XGBoost Model Interpretation")
print("=" * 70)

# ============================================================================
# Load test data
# ============================================================================
print("\n1. Loading test data...")

embedding_choice = 'bert' if EMBEDDING_METHOD == 'bert' else 'tfidf'
test_file = Path(__file__).parent.parent / 'data' / 'clean' / f'test_{embedding_choice}.csv'

X_test = pd.read_csv(test_file)

print(f"  Test data shape: {X_test.shape}")
print(f"  Columns: {list(X_test.columns)[:10]}...")
print(f"  Sample features: {list(X_test.columns[-10:])}")

# ============================================================================
# Load preprocessor and feature names
# ============================================================================
print("\n2b. Loading preprocessor...")

import joblib

try:
    preprocessor = joblib.load(model_dir / f'xgb_tweedie_{embedding_choice}_epc_preprocess.joblib')
    print(f"  Loaded preprocessor for {embedding_choice}")
    
    # Get feature names from the preprocessor
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        print(f"  Preprocessor output shape: {len(feature_names)} features")
    else:
        feature_names = None
        print(f"  Warning: Could not extract feature names from preprocessor")
except Exception as e:
    print(f"  Error loading preprocessor: {e}")
    preprocessor = None
    feature_names = None



# ============================================================================
# Load XGBoost models
# ============================================================================
print("\n3. Loading XGBoost Tweedie models...")

model_dir = Path(__file__).parent.parent / 'models'

# Load Tweedie models (EPC and clicks) using native XGBoost
try:
    import xgboost as xgb
    
    model_epc = xgb.Booster(model_file=str(model_dir / f'xgb_tweedie_{embedding_choice}_epc.json'))
    model_clicks = xgb.Booster(model_file=str(model_dir / f'xgb_tweedie_{embedding_choice}_clicks.json'))
    print(f"  Loaded EPC model: xgb_tweedie_{embedding_choice}_epc.json")
    print(f"  Loaded clicks model: xgb_tweedie_{embedding_choice}_clicks.json")
    
except Exception as e:
    print(f"  Error loading XGBoost models: {e}")
    model_epc = None
    model_clicks = None

# ============================================================================
# Variable Importance Analysis
# ============================================================================
print("\n4. Computing Variable Importance...")

if model_epc is not None:
    try:
        # Get variable importance for EPC model using native XGBoost
        importance_epc_dict = model_epc.get_score(importance_type='weight')
        imp_df_epc = pd.DataFrame(list(importance_epc_dict.items()), columns=['Feature', 'Importance'])
        imp_df_epc = imp_df_epc.sort_values('Importance', ascending=False)
        
        print("\n  EPC Model - Top 15 Important Features (by frequency):")
        print("  " + "-" * 60)
        for feat, imp_val in imp_df_epc.head(15).itertuples(index=False):
            print(f"    {feat:45s}: {imp_val:10.0f}")
            
        # Save to CSV
        imp_df_epc.to_csv(OUTPUT_DIR / f'variable_importance_epc_{embedding_choice}.csv', index=False)
        print(f"\n  Saved to: {OUTPUT_DIR / f'variable_importance_epc_{embedding_choice}.csv'}")
            
    except Exception as e:
        print(f"  Error getting variable importance for EPC: {e}")
        import traceback
        traceback.print_exc()

if model_clicks is not None:
    try:
        # Get variable importance for clicks model using native XGBoost
        importance_clicks_dict = model_clicks.get_score(importance_type='weight')
        imp_df_clicks = pd.DataFrame(list(importance_clicks_dict.items()), columns=['Feature', 'Importance'])
        imp_df_clicks = imp_df_clicks.sort_values('Importance', ascending=False)
        
        print("\n  Clicks Model - Top 15 Important Features (by frequency):")
        print("  " + "-" * 60)
        for feat, imp_val in imp_df_clicks.head(15).itertuples(index=False):
            print(f"    {feat:45s}: {imp_val:10.0f}")
            
        # Save to CSV
        imp_df_clicks.to_csv(OUTPUT_DIR / f'variable_importance_clicks_{embedding_choice}.csv', index=False)
        print(f"\n  Saved to: {OUTPUT_DIR / f'variable_importance_clicks_{embedding_choice}.csv'}")
            
    except Exception as e:
        print(f"  Error getting variable importance for clicks: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SHAP Values Analysis
# ============================================================================
print("\n5. Computing SHAP Values (this may take a moment)...")

try:
    if preprocessor is None:
        print("  Warning: Preprocessor not loaded, skipping SHAP analysis")
    else:
        # Apply preprocessor to test data to get the same shape as training
        print("  Applying preprocessor to test data...")
        X_preprocessed = preprocessor.transform(X_test)
        
        # Convert to dense if sparse
        if hasattr(X_preprocessed, 'toarray'):
            X_preprocessed = X_preprocessed.toarray()
        
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names if feature_names is not None else [f'f{i}' for i in range(X_preprocessed.shape[1])])
        print(f"  Preprocessed data shape: {X_preprocessed.shape}")
        
        if model_epc is not None:
            print("  Computing SHAP for EPC model...")
            try:
                # Create SHAP explainer for XGBoost using preprocessed data
                explainer_epc = shap.TreeExplainer(model_epc)
                shap_values_epc = explainer_epc.shap_values(X_preprocessed)
                
                # Calculate mean absolute SHAP values
                shap_importance_epc = pd.DataFrame({
                    'Feature': X_preprocessed.columns,
                    'Mean_Abs_SHAP': np.abs(shap_values_epc).mean(axis=0)
                }).sort_values('Mean_Abs_SHAP', ascending=False)
                
                print("\n  EPC Model - SHAP-based Feature Importance (Top 15):")
                print("  " + "-" * 60)
                for idx, row in shap_importance_epc.head(15).iterrows():
                    print(f"    {row['Feature']:45s}: {row['Mean_Abs_SHAP']:10.6f}")

                # Save SHAP importance
                shap_importance_epc.to_csv(OUTPUT_DIR / f'shap_importance_epc_{embedding_choice}.csv', index=False)
                print(f"\n  Saved to: {OUTPUT_DIR / f'shap_importance_epc_{embedding_choice}.csv'}")

                # Create SHAP summary plot
                print("    Creating SHAP summary plot...")
                try:
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values_epc, X_preprocessed, show=False)
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / f'shap_summary_epc_{embedding_choice}.png', dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"  SHAP summary plot saved to: {OUTPUT_DIR / f'shap_summary_epc_{embedding_choice}.png'}")
                except Exception as e:
                    print(f"  Error creating SHAP summary plot: {e}")
            except Exception as e:
                print(f"  Error computing SHAP for EPC: {e}")
                import traceback
                traceback.print_exc()

        if model_clicks is not None:
            print("  Computing SHAP for clicks model...")
            try:
                # Create SHAP explainer for XGBoost using preprocessed data
                explainer_clicks = shap.TreeExplainer(model_clicks)
                shap_values_clicks = explainer_clicks.shap_values(X_preprocessed)
                
                # Calculate mean absolute SHAP values
                shap_importance_clicks = pd.DataFrame({
                    'Feature': X_preprocessed.columns,
                    'Mean_Abs_SHAP': np.abs(shap_values_clicks).mean(axis=0)
                }).sort_values('Mean_Abs_SHAP', ascending=False)
                
                print("\n  Clicks Model - SHAP-based Feature Importance (Top 15):")
                print("  " + "-" * 60)
                for idx, row in shap_importance_clicks.head(15).iterrows():
                    print(f"    {row['Feature']:45s}: {row['Mean_Abs_SHAP']:10.6f}")

                # Save SHAP importance
                shap_importance_clicks.to_csv(OUTPUT_DIR / f'shap_importance_clicks_{embedding_choice}.csv', index=False)
                print(f"\n  Saved to: {OUTPUT_DIR / f'shap_importance_clicks_{embedding_choice}.csv'}")

                # Create SHAP summary plot
                print("    Creating SHAP summary plot...")
                try:
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values_clicks, X_preprocessed, show=False)
                    plt.tight_layout()
                    plt.savefig(OUTPUT_DIR / f'shap_summary_clicks_{embedding_choice}.png', dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"  SHAP summary plot saved to: {OUTPUT_DIR / f'shap_summary_clicks_{embedding_choice}.png'}")
                except Exception as e:
                    print(f"  Error creating SHAP summary plot: {e}")
            except Exception as e:
                print(f"  Error computing SHAP for clicks: {e}")
                import traceback
                traceback.print_exc()

except Exception as e:
    print(f"  Error in SHAP analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Interpretation complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)