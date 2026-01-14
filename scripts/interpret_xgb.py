#!/usr/bin/env python
"""
Interpret XGBoost clicks model using variable importance and SHAP values.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgb_clicks_model.joblib'
OUTPUT_DIR = Path(__file__).parent.parent / 'model_interpretability'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("XGBoost Clicks Model Interpretation")
print("=" * 70)

# ============================================================================
# Load test data
# ============================================================================
print("\n1. Loading test data...")

embedding_choice = 'bert'
test_file = Path(__file__).parent.parent / 'data' / 'clean' / f'test_{embedding_choice}.csv'

X_test = pd.read_csv(test_file)

print(f"  Test data shape: {X_test.shape}")
print(f"  Columns: {list(X_test.columns)[:10]}...")
print(f"  Sample features: {list(X_test.columns[-10:])}")

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