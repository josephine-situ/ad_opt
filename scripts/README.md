# Data Preparation Pipeline

Converts the Jupyter notebook workflow to a reusable Python script for preparing ad optimization data.

## Features

- Loads and combines keyword data from 2024 and 2025
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
  --output-dir data \                 # output directory
  --data-dir .                         # input data directory
```

## Output Files

Running the pipeline generates:
- `data/ad_opt_data_{method}.csv` - Full dataset with embeddings
- `data/train_{method}.csv` - Training set (75%)
- `data/test_{method}.csv` - Test set (25%)

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

1. **Load Data**: Combine 2024 and 2025 keyword reports
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
