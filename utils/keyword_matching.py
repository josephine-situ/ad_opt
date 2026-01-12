"""Keyword normalization + fuzzy matching helpers.

These utilities are intentionally lightweight (no external NLP deps) and are
used across the data pipeline and bid optimization to improve join rates.

Python: 3.8+
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


_ORIGIN_FILE_DEFAULT = Path('data/gkp/keywords_classified.csv')


def normalize_kw_basic(s: str) -> str:
    return str(s).lower().strip()


def _normalize_token(token: str) -> str:
    t = token

    # Fast path for common pairs where naive stemming is error-prone.
    # (e.g., courses -> course; removing 'es' first produces 'cour'.)
    common_map = {
        'course': 'course',
        'courses': 'course',
        'class': 'class',
        'classes': 'class',
        'business': 'business',
        'businesses': 'business',
    }
    if t in common_map:
        return common_map[t]

    # Canonicalize AI.
    if t in {'ai', 'a.i'}:
        return 'ai'

    # Canonicalize engineering/engineers.
    if t in {'engineering', 'engineers', 'engineer'}:
        return 'engineer'

    # Canonicalize certificate/certification.
    if t in {'certificate', 'certificates', 'certification', 'certifications'}:
        return 'certification'

    # Plural normalization.
    # businesses -> business
    if t.endswith('ies') and len(t) > 3:
        t = t[:-3] + 'y'

    # -es plural normalization for a few endings where -s stripping isn't enough.
    # NOTE: don't include generic '*ses' here (it breaks 'courses' -> 'cour').
    if t.endswith('es') and len(t) > 3:
        if t.endswith(('sses', 'ches', 'shes', 'xes', 'zes')):
            t = t[:-2]

    # Generic trailing 's' plural.
    if t.endswith('s') and not t.endswith('ss') and len(t) > 3:
        t = t[:-1]

    # Re-apply canonical maps after plural stripping.
    if t in {'engineering', 'engineers', 'engineer'}:
        return 'engineer'
    if t in {'certificate', 'certificates', 'certification', 'certifications'}:
        return 'certification'
    if t in {'ai', 'a.i'}:
        return 'ai'

    return t


def normalize_kw_similar_words(keyword: str) -> str:
    """Normalize keyword for a fuzzy-ish merge key.

    Handles:
    - punctuation differences
    - simple pluralization
    - common synonym folds:
      - artificial intelligence <-> ai (canonical: ai)
      - engineering <-> engineers (canonical: engineer)
      - certificate <-> certification(s) (canonical: certification)

    Returns a normalized string key.
    """
    kw = normalize_kw_basic(keyword)

    # Keep letters/numbers/spaces/apostrophes; turn other punctuation into spaces.
    kw = re.sub(r"[^a-z0-9\s']+", ' ', kw)
    tokens = [t for t in kw.split() if t]

    # Phrase normalization: artificial intelligence -> ai
    out: List[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == 'artificial' and tokens[i + 1] == 'intelligence':
            out.append('ai')
            i += 2
            continue
        out.append(tokens[i])
        i += 1

    out = [_normalize_token(t) for t in out]
    return ' '.join(out).strip()


def load_keyword_origin_map(origin_file: Path = _ORIGIN_FILE_DEFAULT) -> Dict[str, str]:
    """Load Keyword->Origin mapping from keywords_classified.csv.

    Keys are normalized using lowercase + strip.
    Values are lowercase Origin strings.
    """
    try:
        if not origin_file.exists():
            return {}
        df = pd.read_csv(origin_file)
        if 'Keyword' not in df.columns or 'Origin' not in df.columns:
            return {}
        return {
            normalize_kw_basic(k): str(o).strip().lower()
            for k, o in zip(df['Keyword'].astype(str), df['Origin'].astype(str))
        }
    except Exception:
        return {}


def _best_gkp_rows_by_key(
    gkp_df: pd.DataFrame,
    *,
    gkp_keyword_col: str,
    value_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    gkp_keyed = gkp_df.copy()
    gkp_keyed['_kw_key'] = gkp_keyed[gkp_keyword_col].astype(str).map(normalize_kw_similar_words)

    score_cols = [c for c in value_cols if c.startswith('searches_')]
    if score_cols:
        gkp_keyed['_score'] = gkp_keyed[score_cols].fillna(0).sum(axis=1)
    else:
        gkp_keyed['_score'] = 0

    # Representative row per key: prefer highest overall searches.
    gkp_best = gkp_keyed.sort_values('_score', ascending=False).drop_duplicates(subset=['_kw_key'], keep='first')

    best_keyword_by_key = gkp_best.set_index('_kw_key')[gkp_keyword_col]
    best_values_by_key = gkp_best.set_index('_kw_key')[value_cols]
    return gkp_best, best_keyword_by_key, best_values_by_key


def fuzzy_fill_from_gkp(
    merged_df: pd.DataFrame,
    *,
    keyword_col: str,
    gkp_df: pd.DataFrame,
    gkp_keyword_col: str,
    value_cols: Optional[List[str]] = None,
    verbose: bool = True,
    source_display_col: Optional[str] = None,
    target_display_col: Optional[str] = None,
    print_all_mappings: bool = True,
) -> int:
    """Fill missing GKP columns in merged_df using a fuzzy-ish keyword key.

    Assumes merged_df already contains the GKP columns (from an earlier exact merge)
    and fills remaining NaNs based on normalize_kw_similar_words().

    Returns:
        Number of rows newly filled (measured by the first value column).
    """
    if merged_df.empty or gkp_df is None or len(gkp_df) == 0:
        return 0

    if keyword_col not in merged_df.columns:
        return 0
    if gkp_keyword_col not in gkp_df.columns:
        return 0

    if value_cols is None:
        value_cols = [c for c in gkp_df.columns if c != gkp_keyword_col]

    value_cols = [c for c in value_cols if c in merged_df.columns]
    if not value_cols:
        return 0

    first_val = value_cols[0]
    unmatched_mask = merged_df[first_val].isnull()
    if not unmatched_mask.any():
        return 0

    gkp_best, best_keyword_by_key, best_values_by_key = _best_gkp_rows_by_key(
        gkp_df,
        gkp_keyword_col=gkp_keyword_col,
        value_cols=value_cols,
    )

    best_display_by_key = best_keyword_by_key
    if target_display_col is not None and target_display_col in gkp_best.columns:
        best_display_by_key = gkp_best.set_index('_kw_key')[target_display_col]

    merged_df.loc[unmatched_mask, '_kw_key'] = merged_df.loc[unmatched_mask, keyword_col].astype(str).map(normalize_kw_similar_words)
    mapped_keys = merged_df.loc[unmatched_mask, '_kw_key']

    before_unmatched = int(unmatched_mask.sum())
    for col in value_cols:
        merged_df.loc[unmatched_mask, col] = merged_df.loc[unmatched_mask, col].fillna(mapped_keys.map(best_values_by_key[col]))

    after_unmatched = int(merged_df[first_val].isnull().sum())
    filled = before_unmatched - after_unmatched

    if verbose and filled > 0:
        print(f"  Fuzzy-matched {filled} rows using similar-word normalization")
        newly_filled_mask = unmatched_mask & merged_df[first_val].notnull()
        src_col = source_display_col if (source_display_col is not None and source_display_col in merged_df.columns) else keyword_col

        mapped = (
            merged_df.loc[newly_filled_mask, [src_col, '_kw_key']]
            .dropna()
            .drop_duplicates()
        )
        mappings = []
        for _, r in mapped.iterrows():
            k = r['_kw_key']
            gkp_kw = best_display_by_key.get(k)
            if isinstance(gkp_kw, str) and gkp_kw:
                mappings.append(f"{r[src_col]} -> {gkp_kw}")

        if mappings:
            mappings = sorted(set(mappings))
            if print_all_mappings:
                print(f"  Fuzzy mappings ({len(mappings)}):")
                for m in mappings:
                    print(f"    {m}")
            else:
                print(f"  Example fuzzy mappings: {mappings[:5]}")

    return filled
