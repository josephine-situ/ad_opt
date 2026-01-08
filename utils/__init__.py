"""
Utilities for ad optimization data pipeline.

Modules:
- data_cleaning: Currency, percentage, and text cleaning functions
- date_features: Temporal feature extraction (weekends, holidays, course dates)
- embeddings: TF-IDF and BERT embedding generation with dimensionality reduction
- data_pipeline: High-level pipeline functions (load, format, merge, split, save)
"""

from .data_cleaning import (
    clean_currency,
    convert_percent_to_float,
)

from .date_features import (
    _region_to_country_code,
    _get_holiday_calendars,
    _is_holiday,
    calculate_days_to_next,
)

from .embeddings import (
    get_tfidf_embeddings,
    get_bert_embeddings_pipeline,
    get_bert_embedding,
)

from .data_pipeline import (
    load_and_combine_keyword_data,
    format_keyword_data,
    extract_date_features,
    filter_data_by_date,
    get_gkp_data,
    impute_missing_data,
    merge_with_ads_data,
    add_embeddings,
    prepare_train_test_split,
    save_outputs,
    load_embeddings,
)

from .tee_logging import (
    setup_tee_logging,
)

from .keyword_matching import (
    normalize_kw_basic,
    normalize_kw_similar_words,
    load_keyword_origin_map,
    fuzzy_fill_from_gkp,
)

__all__ = [
    # Data cleaning
    'clean_currency',
    'convert_percent_to_float',
    # Date features
    '_region_to_country_code',
    '_get_holiday_calendars',
    '_is_holiday',
    'calculate_days_to_next',
    # Embeddings
    'get_tfidf_embeddings',
    'get_bert_embeddings_pipeline',
    'get_bert_embedding',
    # Data pipeline
    'load_and_combine_keyword_data',
    'format_keyword_data',
    'extract_date_features',
    'filter_data_by_date',
    'get_gkp_data',
    'impute_missing_data',
    'merge_with_ads_data',
    'add_embeddings',
    'prepare_train_test_split',
    'save_outputs',
    'load_embeddings',
    # Logging
    'setup_tee_logging',
    # Keyword matching
    'normalize_kw_basic',
    'normalize_kw_similar_words',
    'load_keyword_origin_map',
    'fuzzy_fill_from_gkp',
]
