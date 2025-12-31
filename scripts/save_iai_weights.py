"""
Extract and save model weights and constants to CSV files.
This allows bid_optimization.py to run without requiring IAI.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.iai_setup import iai
except ImportError:
    print("ERROR: Could not set up IAI. Install with: pip install iai")
    sys.exit(1)


def extract_and_save_weights(embedding_method='bert', models_dir='models', output_dir='models'):
    """
    Extract weights and constants from IAI models and save to CSV files.
    
    Args:
        embedding_method: 'bert' or 'tfidf'
        models_dir: Directory containing trained model JSON files
        output_dir: Directory to save CSV files
    """
    print(f"\n{'='*70}")
    print(f"Extracting weights for embedding method: {embedding_method}")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    epc_model_path = Path(models_dir) / f'lr_{embedding_method}_epc.json'
    clicks_model_path = Path(models_dir) / f'lr_{embedding_method}_clicks.json'

    print(f"\nLoading models from {models_dir}...")
    print(f"  EPC: {epc_model_path}")
    print(f"  Clicks: {clicks_model_path}")

    if not epc_model_path.exists():
        raise FileNotFoundError(f"EPC model not found: {epc_model_path}")
    if not clicks_model_path.exists():
        raise FileNotFoundError(f"Clicks model not found: {clicks_model_path}")

    lnr_epc = iai.read_json(str(epc_model_path))
    lnr_clicks = iai.read_json(str(clicks_model_path))

    print(f"\nExtracting EPC model weights and constants...")
    weights_epc_tuple = lnr_epc.get_prediction_weights()
    epc_const = lnr_epc.get_prediction_constant()

    print(f"Extracting clicks model weights and constants...")
    weights_clicks_tuple = lnr_clicks.get_prediction_weights()
    clicks_const = lnr_clicks.get_prediction_constant()

    # Parse tuple format (continuous, categorical)
    if isinstance(weights_epc_tuple, tuple):
        weights_epc_numeric = weights_epc_tuple[0]
        weights_epc_categorical = weights_epc_tuple[1] if len(weights_epc_tuple) > 1 else {}
    else:
        weights_epc_numeric = weights_epc_tuple
        weights_epc_categorical = {}

    if isinstance(weights_clicks_tuple, tuple):
        weights_clicks_numeric = weights_clicks_tuple[0]
        weights_clicks_categorical = weights_clicks_tuple[1] if len(weights_clicks_tuple) > 1 else {}
    else:
        weights_clicks_numeric = weights_clicks_tuple
        weights_clicks_categorical = {}

    # Save EPC numeric weights
    print(f"\nSaving EPC model...")
    epc_numeric_df = pd.DataFrame(list(weights_epc_numeric.items()), columns=['feature', 'weight'])
    epc_numeric_file = output_dir / f'weights_{embedding_method}_epc_numeric.csv'
    epc_numeric_df.to_csv(epc_numeric_file, index=False)
    print(f"  Saved numeric weights to {epc_numeric_file}")

    if weights_epc_categorical:
        epc_cat_rows = []
        for feature, level_dict in weights_epc_categorical.items():
            for level, weight in level_dict.items():
                epc_cat_rows.append({'feature': feature, 'level': level, 'weight': weight})
        epc_cat_df = pd.DataFrame(epc_cat_rows)
        epc_cat_file = output_dir / f'weights_{embedding_method}_epc_categorical.csv'
        epc_cat_df.to_csv(epc_cat_file, index=False)
        print(f"  Saved categorical weights to {epc_cat_file}")

    epc_const_file = output_dir / f'weights_{embedding_method}_epc_constant.csv'
    pd.DataFrame({'constant': [epc_const]}).to_csv(epc_const_file, index=False)
    print(f"  Saved constant to {epc_const_file}")

    # Save clicks numeric weights
    print(f"\nSaving clicks model...")
    clicks_numeric_df = pd.DataFrame(list(weights_clicks_numeric.items()), columns=['feature', 'weight'])
    clicks_numeric_file = output_dir / f'weights_{embedding_method}_clicks_numeric.csv'
    clicks_numeric_df.to_csv(clicks_numeric_file, index=False)
    print(f"  Saved numeric weights to {clicks_numeric_file}")

    if weights_clicks_categorical:
        clicks_cat_rows = []
        for feature, level_dict in weights_clicks_categorical.items():
            for level, weight in level_dict.items():
                clicks_cat_rows.append({'feature': feature, 'level': level, 'weight': weight})
        clicks_cat_df = pd.DataFrame(clicks_cat_rows)
        clicks_cat_file = output_dir / f'weights_{embedding_method}_clicks_categorical.csv'
        clicks_cat_df.to_csv(clicks_cat_file, index=False)
        print(f"  Saved categorical weights to {clicks_cat_file}")

    clicks_const_file = output_dir / f'weights_{embedding_method}_clicks_constant.csv'
    pd.DataFrame({'constant': [clicks_const]}).to_csv(clicks_const_file, index=False)
    print(f"  Saved constant to {clicks_const_file}")
    
    print(f"\n✓ Weights and constants saved successfully!")
    print(f"{'='*70}\n")


def main():
    """Extract and save weights for both BERT and TF-IDF models."""
    try:
        print("\n" + "="*70)
        print("Model Weights & Constants Extraction")
        print("="*70)
        
        # Extract and save for both embedding methods
        for embedding_method in ['bert', 'tfidf']:
            extract_and_save_weights(embedding_method=embedding_method)
        
        print("="*70)
        print("✓ All weights and constants extracted and saved!")
        print("="*70)
        print("\nYou can now run bid_optimization.py without requiring IAI:")
        print("  python bid_optimization.py --embedding-method bert")
        print("  python bid_optimization.py --embedding-method tfidf")
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
