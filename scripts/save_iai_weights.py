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
    
    # Load conversion model
    conversion_model_path = Path(models_dir) / f'lr_{embedding_method}_conversion.json'
    clicks_model_path = Path(models_dir) / f'lr_{embedding_method}_clicks.json'
    
    print(f"\nLoading models from {models_dir}...")
    print(f"  Conversion: {conversion_model_path}")
    print(f"  Clicks: {clicks_model_path}")
    
    if not conversion_model_path.exists():
        raise FileNotFoundError(f"Conversion model not found: {conversion_model_path}")
    if not clicks_model_path.exists():
        raise FileNotFoundError(f"Clicks model not found: {clicks_model_path}")
    
    # Load models using IAI
    lnr_conv = iai.read_json(str(conversion_model_path))
    lnr_clicks = iai.read_json(str(clicks_model_path))
    
    # Extract weights and constants
    print(f"\nExtracting conversion model weights and constants...")
    weights_conv_tuple = lnr_conv.get_prediction_weights()
    conv_const = lnr_conv.get_prediction_constant()
    
    print(f"Extracting clicks model weights and constants...")
    weights_clicks_tuple = lnr_clicks.get_prediction_weights()
    clicks_const = lnr_clicks.get_prediction_constant()
    
    # Parse tuple format (continuous, categorical)
    if isinstance(weights_conv_tuple, tuple):
        weights_conv_numeric = weights_conv_tuple[0]
        weights_conv_categorical = weights_conv_tuple[1] if len(weights_conv_tuple) > 1 else {}
    else:
        weights_conv_numeric = weights_conv_tuple
        weights_conv_categorical = {}
    
    if isinstance(weights_clicks_tuple, tuple):
        weights_clicks_numeric = weights_clicks_tuple[0]
        weights_clicks_categorical = weights_clicks_tuple[1] if len(weights_clicks_tuple) > 1 else {}
    else:
        weights_clicks_numeric = weights_clicks_tuple
        weights_clicks_categorical = {}
    
    # Save conversion model weights
    print(f"\nSaving conversion model...")
    
    # Save numeric weights
    conv_numeric_df = pd.DataFrame(
        list(weights_conv_numeric.items()),
        columns=['feature', 'weight']
    )
    conv_numeric_file = output_dir / f'weights_{embedding_method}_conversion_numeric.csv'
    conv_numeric_df.to_csv(conv_numeric_file, index=False)
    print(f"  Saved numeric weights to {conv_numeric_file}")
    
    # Save categorical weights
    if weights_conv_categorical:
        conv_cat_rows = []
        for feature, level_dict in weights_conv_categorical.items():
            for level_name, weight in level_dict.items():
                conv_cat_rows.append({
                    'feature': feature,
                    'level': level_name,
                    'weight': weight
                })
        conv_cat_df = pd.DataFrame(conv_cat_rows)
        conv_cat_file = output_dir / f'weights_{embedding_method}_conversion_categorical.csv'
        conv_cat_df.to_csv(conv_cat_file, index=False)
        print(f"  Saved categorical weights to {conv_cat_file}")
    
    # Save constant
    conv_const_file = output_dir / f'weights_{embedding_method}_conversion_constant.csv'
    pd.DataFrame({'constant': [conv_const]}).to_csv(conv_const_file, index=False)
    print(f"  Saved constant to {conv_const_file}")
    
    # Save clicks model weights
    print(f"\nSaving clicks model...")
    
    # Save numeric weights
    clicks_numeric_df = pd.DataFrame(
        list(weights_clicks_numeric.items()),
        columns=['feature', 'weight']
    )
    clicks_numeric_file = output_dir / f'weights_{embedding_method}_clicks_numeric.csv'
    clicks_numeric_df.to_csv(clicks_numeric_file, index=False)
    print(f"  Saved numeric weights to {clicks_numeric_file}")
    
    # Save categorical weights
    if weights_clicks_categorical:
        clicks_cat_rows = []
        for feature, level_dict in weights_clicks_categorical.items():
            for level_name, weight in level_dict.items():
                clicks_cat_rows.append({
                    'feature': feature,
                    'level': level_name,
                    'weight': weight
                })
        clicks_cat_df = pd.DataFrame(clicks_cat_rows)
        clicks_cat_file = output_dir / f'weights_{embedding_method}_clicks_categorical.csv'
        clicks_cat_df.to_csv(clicks_cat_file, index=False)
        print(f"  Saved categorical weights to {clicks_cat_file}")
    
    # Save constant
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
