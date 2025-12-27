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
# Prepare data for model predictions
# ============================================================================
print("\n2. Preparing data for model predictions...")

# Convert categorical columns to pandas categorical type for IAI compatibility
categorical_cols = ['Match type', 'Region', 'day_of_week']
for col in categorical_cols:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')
        print(f"  Converted '{col}' to categorical type")

# Also identify and convert any other object-type columns that should be categorical
object_cols = X_test.select_dtypes(include=['object']).columns.tolist()
for col in object_cols:
    if col not in categorical_cols:
        print(f"  Converting object column '{col}' to categorical")
        X_test[col] = X_test[col].astype('category')

print(f"  Data dtypes: {X_test.dtypes.value_counts().to_dict()}")


# ============================================================================
# Load XGBoost models
# ============================================================================
print("\n3. Loading XGBoost models...")

model_dir = Path(__file__).parent.parent / 'models'

# Load models
try:
    # Try IAI read_json first
    try:
        from utils.iai_setup import iai
        model_conversion = iai.read_json(str(model_dir / f'xgb_{embedding_choice}_conversion.json'))
        model_clicks = iai.read_json(str(model_dir / f'xgb_{embedding_choice}_clicks.json'))
        print(f"  Loaded conversion model: xgb_{embedding_choice}_conversion.json")
        print(f"  Loaded clicks model: xgb_{embedding_choice}_clicks.json")
    except Exception as e:
        print(f"  Warning: Could not load models with IAI: {e}")
        print("  Skipping model-specific interpretation (fallback to feature matrix analysis)")
        model_conversion = None
        model_clicks = None
        
except Exception as e:
    print(f"  Error loading models: {e}")
    model_conversion = None
    model_clicks = None

# ============================================================================
# Variable Importance Analysis
# ============================================================================
print("\n4. Computing Variable Importance...")

if model_conversion is not None:
    try:
        # Get variable importance for conversion model
        importance_conv = model_conversion.variable_importance()
        print("\n  Conversion Model - Top 15 Important Features:")
        print("  " + "-" * 60)
        
        # Convert to DataFrame for consistent handling
        if isinstance(importance_conv, dict):
            sorted_imp = sorted(importance_conv.items(), key=lambda x: x[1], reverse=True)
            imp_df_conv = pd.DataFrame(list(sorted_imp), columns=['Feature', 'Importance'])
            for feat, imp_val in sorted_imp[:15]:
                print(f"    {feat:45s}: {imp_val:10.4f}")
        elif isinstance(importance_conv, pd.Series):
            imp_df_conv = importance_conv.to_frame('Importance').reset_index()
            imp_df_conv.columns = ['Feature', 'Importance']
            imp_df_conv = imp_df_conv.sort_values('Importance', ascending=False)
            for feat, imp_val in imp_df_conv.head(15).itertuples(index=False):
                print(f"    {feat:45s}: {imp_val:10.4f}")
        else:
            # Array format
            imp_df_conv = pd.DataFrame(importance_conv)
            print(imp_df_conv.head(15))
            
        # Save to CSV
        imp_df_conv.to_csv(OUTPUT_DIR / f'variable_importance_conversion_{embedding_choice}.csv', index=False)
        print(f"\n  Saved to: {OUTPUT_DIR / f'variable_importance_conversion_{embedding_choice}.csv'}")
            
    except Exception as e:
        print(f"  Error getting variable importance for conversion: {e}")

if model_clicks is not None:
    try:
        # Get variable importance for clicks model
        importance_clicks = model_clicks.variable_importance()
        print("\n  Clicks Model - Top 15 Important Features:")
        print("  " + "-" * 60)
        
        # Convert to DataFrame for consistent handling
        if isinstance(importance_clicks, dict):
            sorted_imp = sorted(importance_clicks.items(), key=lambda x: x[1], reverse=True)
            imp_df_clicks = pd.DataFrame(list(sorted_imp), columns=['Feature', 'Importance'])
            for feat, imp_val in sorted_imp[:15]:
                print(f"    {feat:45s}: {imp_val:10.4f}")
        elif isinstance(importance_clicks, pd.Series):
            imp_df_clicks = importance_clicks.to_frame('Importance').reset_index()
            imp_df_clicks.columns = ['Feature', 'Importance']
            imp_df_clicks = imp_df_clicks.sort_values('Importance', ascending=False)
            for feat, imp_val in imp_df_clicks.head(15).itertuples(index=False):
                print(f"    {feat:45s}: {imp_val:10.4f}")
        else:
            # Array format
            imp_df_clicks = pd.DataFrame(importance_clicks)
            print(imp_df_clicks.head(15))
            
        # Save to CSV
        imp_df_clicks.to_csv(OUTPUT_DIR / f'variable_importance_clicks_{embedding_choice}.csv', index=False)
        print(f"\n  Saved to: {OUTPUT_DIR / f'variable_importance_clicks_{embedding_choice}.csv'}")
            
    except Exception as e:
        print(f"  Error getting variable importance for clicks: {e}")

# ============================================================================
# SHAP Values Analysis
# ============================================================================
print("\n5. Computing SHAP Values (this may take a moment)...")

# Get model feature names upfront for both models
model_features_conv = None
model_features_clicks = None

if model_conversion is not None:

    X_conv = X_test.copy()
    X_conv.columns = X_conv.columns.str.replace('.', '_', regex=False)
    model_features_conv = model_conversion.get_features_used()
    X_conv = X_conv[model_features_conv]

if model_clicks is not None:
    X_clicks = X_test.copy()
    X_clicks.columns = X_clicks.columns.str.replace('.', '_', regex=False)
    model_features_clicks = model_clicks.get_features_used()
    X_clicks = X_clicks[model_features_clicks]

if model_conversion is not None:

    # Get SHAP values for conversion model
    print("  Computing SHAP for conversion model...")
    shap_conv = model_conversion.predict_shap(X_conv)

    # Calculate SHAP importance from feature DataFrame
    # shap_conv is a dict with keys: 'expected_value', 'features', 'shap_values'
    if isinstance(shap_conv, dict) and 'shap_values' in shap_conv:
        print("    Found shap_values in dict")
        shap_values = np.abs(shap_conv['shap_values']).mean(axis=0)
        feature_names = list(shap_conv['features'].columns)
        
        shap_importance_conv = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': shap_values
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        
        print("\n  Conversion Model - SHAP-based Feature Importance (Top 15):")
        print("  " + "-" * 60)
        for idx, row in shap_importance_conv.head(15).iterrows():
            print(f"    {row['Feature']:45s}: {row['Mean_Abs_SHAP']:10.6f}")

        # Save SHAP importance
        shap_importance_conv.to_csv(OUTPUT_DIR / f'shap_importance_conversion_{embedding_choice}.csv', index=False)
        print(f"\n  Saved to: {OUTPUT_DIR / f'shap_importance_conversion_{embedding_choice}.csv'}")

        # Create SHAP summary plot
        print("    Creating SHAP summary plot...")
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_conv['shap_values'], shap_conv['features'], show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'shap_summary_conversion_{embedding_choice}.png', dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  SHAP summary plot saved to: {OUTPUT_DIR / f'shap_summary_conversion_{embedding_choice}.png'}")
        except Exception as e:
            print(f"  Error creating SHAP summary plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    ERROR: No shap_values key found in dict")

if model_clicks is not None:

    # Get SHAP values for clicks model
    print("  Computing SHAP for clicks model...")
    shap_clicks = model_clicks.predict_shap(X_clicks)

    # Calculate SHAP importance from feature DataFrame
    # shap_clicks is a dict with keys: 'expected_value', 'features', 'shap_values'
    if isinstance(shap_clicks, dict) and 'shap_values' in shap_clicks:
        print("    Found shap_values in dict")
        shap_values = np.abs(shap_clicks['shap_values']).mean(axis=0)
        feature_names = list(shap_clicks['features'].columns)
        
        shap_importance_clicks = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': shap_values
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
            shap.summary_plot(shap_clicks['shap_values'], shap_clicks['features'], show=False)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'shap_summary_clicks_{embedding_choice}.png', dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  SHAP summary plot saved to: {OUTPUT_DIR / f'shap_summary_clicks_{embedding_choice}.png'}")
        except Exception as e:
            print(f"  Error creating SHAP summary plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    ERROR: No shap_values key found in dict")

print("\n" + "=" * 70)
print("Interpretation complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 70)