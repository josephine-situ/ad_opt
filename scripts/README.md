# Data Preparation Pipeline

Converts the Jupyter notebook workflow to a reusable Python script for preparing ad optimization data.

## Features

- Loads keyword data from a single combined raw export ("Search keyword - raw input to models.csv")
- Extracts temporal features (day of week, holidays, course start dates)
- Merges with Google Ads keyword planner data
- Generates keyword embeddings using either **TF-IDF** or **BERT**
- Applies TruncatedSVD for dimensionality reduction
- Normalizes embeddings for cosine similarity
- Prepares train/test splits for modeling

## Usage

### TF-IDF Embeddings (Fast, Interpretable)
```bash
python scripts/tidy_get_data.py --embedding-method tfidf --n-components 50
```

### BERT Embeddings (Semantic, Better Generalization)
```bash
python scripts/tidy_get_data.py --embedding-method bert --n-components 50
```

### Full Options
```bash
python scripts/tidy_get_data.py \
  --embedding-method tfidf \          # or 'bert'
  --n-components 50 \                 # embedding dimensions
  --output-dir data/clean \           # output directory
  --data-dir data/reports             # input data directory
```

## Output Files

Running the pipeline generates:
- `data/clean/ad_opt_data_{method}.csv` - Full dataset with embeddings
- `data/clean/train_{method}.csv` - Training set (75%)
- `data/clean/test_{method}.csv` - Test set (25%)

Where `{method}` is either `tfidf` or `bert`.

## Installation

### Requirements
- pandas, numpy, scikit-learn, holidays (for both methods)
- sentence-transformers (for BERT)

### Install sentence-transformers for BERT support
```bash
pip install sentence-transformers
```

## Architecture

### Main Pipeline Steps

1. **Load Data**: Load keyword report export (single raw CSV)
2. **Format**: Clean campaigns, regions, keywords
3. **Date Features**: Extract day_of_week, is_weekend, month, is_public_holiday, days_to_next_course_start
4. **Filter**: Remove early records (before 2024-11-03)
5. **Merge Ads Data**: Join with Google Keyword Planner metrics
6. **Clean**: Drop rows with missing ad data, convert percentages
7. **Embeddings**: Compute TF-IDF or BERT embeddings, reduce with TruncatedSVD, normalize
8. **Train/Test Split**: 75/25 random split

### Helper Functions (in `helpers.py`)

- `get_tfidf_embeddings()` - Generate TF-IDF embeddings with SVD reduction
- `get_bert_embeddings_pipeline()` - Generate BERT embeddings with SVD reduction

## Embedding Methods Comparison

| Aspect | TF-IDF | BERT |
|--------|--------|------|
| Speed | Fast | Slower (GPU optional) |
| Interpretability | High (ngrams visible) | Low (learned representations) |
| Semantics | Keyword-based | Contextual, semantic |
| Generalization | Good for exact matches | Better for paraphrasing/synonyms |
| Dependencies | sklearn | sentence-transformers |
| Dimensionality Post-SVD | 50 | 50 |

## Implementation Notes

- Both methods use `TruncatedSVD(n_components=50)` to reduce to 50 dimensions
- Embeddings are normalized to unit norm (L2) for cosine similarity
- TF-IDF uses unigrams + bigrams (1,2)-grams
- BERT uses the `all-MiniLM-L6-v2` model (384D → 50D)
- Pipeline is designed to be reproducible (fixed random_state=42)

## Example: Running Both Methods

```bash
# Generate TF-IDF version
python scripts/tidy_get_data.py --embedding-method tfidf

# Generate BERT version
python scripts/tidy_get_data.py --embedding-method bert

# Load and compare
import pandas as pd
df_tfidf = pd.read_csv('data/ad_opt_data_tfidf.csv')
df_bert = pd.read_csv('data/ad_opt_data_bert.csv')
print(f"TF-IDF shape: {df_tfidf.shape}")
print(f"BERT shape: {df_bert.shape}")
```

---

# Prediction Modeling (No IAI)

To train Tweedie-loss models without requiring an InterpretableAI (IAI) license:

```bash
python scripts/prediction_modeling_tweedie.py --target conversion --embedding-method tfidf
python scripts/prediction_modeling_tweedie.py --target clicks --embedding-method tfidf
```

This script exports linear-model weights to `models/weights_{embedding}_{target}_*.csv`, which are consumed by `scripts/bid_optimization.py`.

Optional: XGBoost Tweedie comparison model

```bash
pip install -e ".[ml_open]"
python scripts/prediction_modeling_tweedie.py --target conversion --embedding-method tfidf --models glm xgb
```

Optional: Tweedie-loss "random forest" (XGBoost RF mode)

```bash
pip install -e ".[ml_open]"
python scripts/prediction_modeling_tweedie.py --target conversion --embedding-method tfidf --models glm rf
```

---

# Ablation Studies

This repo includes a lightweight ablation runner that automates two experiment families:

1) **Embedding dimension ablation** (e.g., BERT 10D → 50D)
2) **Feature combination ablation** (baseline vs adding HHI/entropy/interaction features)

The runner will materialize datasets on-demand by calling `scripts/tidy_get_data.py` into per-experiment folders, so it will not overwrite your main `data/clean/*.csv` outputs.

## Usage

### Embedding dimension ablation

```bash
python scripts/run_ablation_studies.py embedding-dims \
  --embedding-method bert \
  --dims 10 20 30 40 50 \
  --targets conversion clicks epc
```

By default this uses the **baseline** feature set (no HHI/entropy/interaction) and runs `tidy_get_data` with `--diversity-mode off` to isolate the effect of embedding dimensionality.

### Feature combination ablation

```bash
python scripts/run_ablation_studies.py feature-combos \
  --embedding-method bert \
  --n-components 20 \
  --targets conversion epc
```

## Outputs

- Results CSVs are written to `opt_results/analysis/ablations/`.
- Per-dimension datasets are written under `data/clean/ablations/`.

## Note on caching

`scripts/tidy_get_data.py` caches embeddings separately per `(embedding_method, n_components)`.
