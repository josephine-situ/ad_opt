import pandas as pd
import joblib

# 1. Load your data and preprocessor
df = pd.read_csv("opt_results/feature_matrices/X_ort_bert_xgb_xgb.csv") # Adjust path
preprocessor = joblib.load("models/xgb_tweedie_bert_clicks_preprocess.joblib") # Adjust path

# 2. Inspect the problematic row
bad_row = df.iloc[[10679]] # Double bracket to keep it a DataFrame
print("Raw 'mom_change':", bad_row['mom_change'].values[0])

# 3. Transform and check for NaNs
try:
    transformed_data = preprocessor.transform(bad_row)
    
    # Check for NaNs in the output
    import numpy as np
    if np.isnan(transformed_data).any():
        print("!!! PREPROCESSOR PRODUCED NaNs !!!")
        # Find which column is NaN
        nan_indices = np.where(np.isnan(transformed_data))[1]
        print(f"NaNs found in transformed feature indices: {nan_indices}")
    else:
        print("Preprocessing successful. No NaNs.")
        
except Exception as e:
    print(f"Preprocessing crashed: {e}")