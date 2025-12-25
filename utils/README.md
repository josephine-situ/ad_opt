# Utils Module Structure

Utility functions refactored from legacy `helpers.py` into focused, modular components.

## Directory Layout

```
utils/
├── __init__.py              # Central re-export point
├── data_cleaning.py         # Currency, percentage, text normalization
├── date_features.py         # Temporal features (holidays, weekends, course dates)
└── embeddings.py            # TF-IDF and BERT embeddings with dimensionality reduction
```

## Usage

### Option 1: Import from main utils package (recommended)
```python
from utils import (
    clean_currency,
    convert_percent_to_float,
    _region_to_country_code,
    _is_holiday,
    calculate_days_to_next,
    get_tfidf_embeddings,
    get_bert_embeddings_pipeline,
)
```

### Option 2: Import from specific submodules
```python
# Data cleaning
from utils.data_cleaning import clean_currency, convert_percent_to_float

# Date features
from utils.date_features import _is_holiday, calculate_days_to_next, _region_to_country_code

# Embeddings
from utils.embeddings import get_tfidf_embeddings, get_bert_embeddings_pipeline
```

### Option 3: Backward compatibility via helpers.py (legacy)
```python
# Still works, but not recommended for new code
from helpers import clean_currency, get_tfidf_embeddings
```

## Module Descriptions

### `data_cleaning.py`
Currency and percentage parsing utilities.

**Functions:**
- `clean_currency(x)` - Parse currency strings (e.g., "$1,234.56" → 1234.56)
- `convert_percent_to_float(value)` - Parse percentages (e.g., "50%" → 0.5, "∞" → inf)

### `date_features.py`
Temporal feature extraction and holiday detection.

**Functions:**
- `_region_to_country_code(r)` - Map region names to country codes (US/CA)
- `_get_holiday_calendars(country_codes, years)` - Build holiday calendars for countries
- `_is_holiday(row, holiday_calendars, ...)` - Check if a date is a public holiday
- `calculate_days_to_next(d, course_start_dts)` - Days until next course start date

### `embeddings.py`
Keyword embeddings with dimensionality reduction (consistent 50D output).

**Functions:**
- `get_tfidf_embeddings(texts, n_components=50, ...)` - TF-IDF → TruncatedSVD → L2 normalized
- `get_bert_embeddings_pipeline(texts, n_components=50, ...)` - BERT → TruncatedSVD → L2 normalized
- `get_bert_embedding(texts, model, tokenizer, device)` - Raw BERT CLS token embeddings (lower-level)

## Dimensionality Reduction

Both embedding methods use the same pipeline for consistency:
1. Generate embeddings (TF-IDF sparse vectors or BERT dense vectors)
2. Apply TruncatedSVD to reduce to 50 dimensions (configurable)
3. L2 normalize for cosine similarity

## Migration from `helpers.py`

If you're using the old `helpers.py`:

**Before (legacy):**
```python
from helpers import clean_currency, get_tfidf_embeddings
```

**After (recommended):**
```python
from utils import clean_currency, get_tfidf_embeddings
# or
from utils.data_cleaning import clean_currency
from utils.embeddings import get_tfidf_embeddings
```

The old `helpers.py` still works via re-exports but should be considered deprecated.
