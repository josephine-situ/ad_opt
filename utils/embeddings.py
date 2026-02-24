"""
Embedding generation: TF-IDF and BERT with dimensionality reduction via TruncatedSVD.
Supports consistent 50D output regardless of input method.
"""

import pandas as pd
import numpy as np
import torch


def get_bert_embedding(text_list, model, tokenizer, device):
    """
    Generates BERT embeddings (CLS token) for a list of texts using transformers library.
    
    Args:
    - text_list (list): List of text strings to embed.
    - model: HuggingFace transformer model.
    - tokenizer: HuggingFace tokenizer.
    - device: Torch device ('cpu' or 'cuda').
    
    Returns:
    - np.ndarray: Shape (n_texts, hidden_size) of CLS token embeddings.
    """
    if text_list is None or len(text_list) == 0:
        return np.array([])
        
    # Ensure it's a list, not a Series
    if not isinstance(text_list, list):
        text_list = text_list.tolist()
    
    # Ensure tokenizer has pad_token set (required for padding=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Tokenize: Add special tokens ([CLS], [SEP]), pad/truncate to max length
    encoded_input = tokenizer(
        text_list, 
        padding=True, 
        truncation=True, 
        return_tensors='pt', 
        max_length=64  # Keywords are short
    )
    
    # Move inputs to the same device as the model
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    
    # Generate embeddings (no gradient needed for inference)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Extract the embedding of the [CLS] token (first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    
    return cls_embeddings.cpu().numpy()


def get_tfidf_embeddings(unique_texts, n_components=50, ngram_range=(1, 2), min_df=1, return_model=False):
    """
    Generate TF-IDF embeddings reduced to n_components via TruncatedSVD and L2 normalized.
    
    Pipeline: TF-IDF → TruncatedSVD → L2 Normalization
    
    Args:
    - unique_texts (list or array): Unique text strings (e.g., keywords).
    - n_components (int): Target embedding dimension. Default 50.
    - ngram_range (tuple): (min_n, max_n) for n-grams. Default (1, 2).
    - min_df (int): Minimum document frequency. Default 1 (keep all terms).
    - return_model (bool): If True, return tuple (embeddings_df, vectorizer, svd, normalizer). Default False.
    
    Returns:
    - pd.DataFrame: Columns ['tfidf_0', 'tfidf_1', ..., 'text'] with shape (n_texts, n_components + 1).
    - (Optional) tuple: (embeddings_df, vectorizer, svd, normalizer) if return_model=True
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    
    # Vectorize
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    X_tfidf = vectorizer.fit_transform(unique_texts)
    
    # Reduce to n_components
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    
    # Normalize to unit length (L2 norm) for cosine similarity
    normalizer = Normalizer(norm='l2')
    X_normalized = normalizer.fit_transform(X_svd)
    
    # Create DataFrame
    embedding_df = pd.DataFrame(
        X_normalized,
        columns=[f'tfidf_{i}' for i in range(n_components)]
    )
    embedding_df['text'] = unique_texts
    
    if return_model:
        return embedding_df, {'vectorizer': vectorizer, 'svd': svd, 'normalizer': normalizer}
    return embedding_df


def get_bert_embeddings_pipeline(unique_texts, n_components=50, model_name='all-MiniLM-L6-v2', batch_size=32, return_model=False):
    """
    Generate BERT embeddings (via sentence-transformers) reduced to n_components via TruncatedSVD and L2 normalized.
    
    Pipeline: BERT (sentence-transformers) → TruncatedSVD → L2 Normalization
    
    Args:
    - unique_texts (list or array): Unique text strings (e.g., keywords).
    - n_components (int): Target embedding dimension. Default 50.
    - model_name (str): Sentence-transformers model identifier. Default 'all-MiniLM-L6-v2' (fast, 384D).
    - batch_size (int): Batch size for encoding. Default 32.
    - return_model (bool): If True, return tuple (embeddings_df, model_dict). Default False.
    
    Returns:
    - pd.DataFrame: Columns ['bert_0', 'bert_1', ..., 'text'] with shape (n_texts, n_components + 1).
    - (Optional) tuple: (embeddings_df, model_dict) if return_model=True
        where model_dict = {'transformer': model, 'svd': svd, 'normalizer': normalizer}
    
    Notes:
    - Requires: pip install sentence-transformers transformers>=4.35.2
    - First run downloads the model (~100MB for MiniLM).
    - Faster with GPU, but CPU works fine for small batches.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer
    
    # Import with better error handling for Python 3.9 compatibility
    try:
        from sentence_transformers import SentenceTransformer
    except TypeError as e:
        if "unsupported operand type(s) for |" in str(e):
            raise RuntimeError(
                "Python 3.9 compatibility issue with transformers library.\n"
                "Fix: pip install --upgrade transformers==4.35.2\n"
                "Or: Use Python 3.10+ for latest versions"
            ) from e
        raise
    
    # Ensure it's a list
    if not isinstance(unique_texts, list):
        unique_texts = list(unique_texts)
    
    # Load model and encode
    model = SentenceTransformer(model_name)
    X_bert = model.encode(
        unique_texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    # Reduce to n_components
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_bert)
    
    # Normalize to unit length (L2 norm) for cosine similarity
    normalizer = Normalizer(norm='l2')
    X_normalized = normalizer.fit_transform(X_svd)
    
    # Create DataFrame
    embedding_df = pd.DataFrame(
        X_normalized,
        columns=[f'bert_{i}' for i in range(n_components)]
    )
    embedding_df['text'] = unique_texts
    
    if return_model:
        return embedding_df, {'transformer': model, 'svd': svd, 'normalizer': normalizer}
    return embedding_df


# ---------------------------------------------------------------------------
# Raw embedding cache + SVD fit / transform utilities
# ---------------------------------------------------------------------------

def get_raw_bert_embeddings_cached(unique_texts, model_name='all-MiniLM-L6-v2',
                                   batch_size=32, cache_path=None):
    """
    Return raw BERT embeddings (no SVD) with optional file caching.

    Args:
        unique_texts: Iterable of text strings.
        model_name: Sentence-transformers model name.
        batch_size: Encoding batch size.
        cache_path: Path to a pickle cache file.  If *None*, no caching.

    Returns:
        dict: ``{text: np.ndarray}`` mapping each text to its raw embedding.
    """
    import pickle
    from pathlib import Path

    if not isinstance(unique_texts, list):
        unique_texts = list(unique_texts)

    cached: dict = {}
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)

    missing = [t for t in unique_texts if t not in cached]

    if missing:
        from sentence_transformers import SentenceTransformer

        print(f"  [RawBERT] Encoding {len(missing)} texts with {model_name} ...")
        model = SentenceTransformer(model_name)
        raw_embs = model.encode(
            missing, batch_size=batch_size,
            show_progress_bar=True, convert_to_numpy=True,
        )
        for text, emb in zip(missing, raw_embs):
            cached[text] = emb

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f)
            print(f"  [RawBERT] Saved {len(cached)} embeddings → {cache_path}")

    return {t: cached[t] for t in unique_texts if t in cached}


def fit_svd_pipeline(raw_embeddings_matrix, n_components=50):
    """
    Fit TruncatedSVD + L2 normalizer on a matrix of raw embeddings.

    Args:
        raw_embeddings_matrix: ``(n_samples, embedding_dim)`` numpy array.
        n_components: Number of SVD components.
            ``None`` or ``>= embedding_dim`` → skip SVD, normalise only.

    Returns:
        dict with keys ``'svd'``, ``'normalizer'``, ``'n_components'``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer

    dim = raw_embeddings_matrix.shape[1]

    if n_components is None or n_components >= dim:
        normalizer = Normalizer(norm='l2')
        normalizer.fit(raw_embeddings_matrix)
        return {'svd': None, 'normalizer': normalizer, 'n_components': dim}

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(raw_embeddings_matrix)

    normalizer = Normalizer(norm='l2')
    normalizer.fit(X_svd)

    return {'svd': svd, 'normalizer': normalizer, 'n_components': n_components}


def apply_svd_pipeline(raw_embeddings_matrix, pipeline_dict):
    """
    Transform raw embeddings through a pre-fitted SVD + normalizer pipeline.

    Returns:
        numpy array of shape ``(n_samples, n_components)``.
    """
    X = raw_embeddings_matrix
    if pipeline_dict['svd'] is not None:
        X = pipeline_dict['svd'].transform(X)
    X = pipeline_dict['normalizer'].transform(X)
    return X


def replace_embeddings(df, raw_emb_map, svd_pipeline, prefix='bert'):
    """
    Replace (or add) embedding columns in *df* using raw embeddings and a
    fitted SVD pipeline.

    Args:
        df: DataFrame with a ``'Keyword'`` column.
        raw_emb_map: ``{keyword: raw_vector}`` dict.
        svd_pipeline: Output of :func:`fit_svd_pipeline`.
        prefix: Column-name prefix (default ``'bert'``).

    Returns:
        ``(df_with_new_emb_cols, list_of_new_col_names)``
    """
    # Drop old embedding columns
    old_cols = [c for c in df.columns if c.startswith(f'{prefix}_')]
    df = df.drop(columns=old_cols, errors='ignore')

    unique_kw = df['Keyword'].unique()
    raw_matrix = np.array([raw_emb_map[kw] for kw in unique_kw])

    transformed = apply_svd_pipeline(raw_matrix, svd_pipeline)
    n_comp = transformed.shape[1]

    emb_cols = [f'{prefix}_{i}' for i in range(n_comp)]
    emb_df = pd.DataFrame(transformed, columns=emb_cols)
    emb_df['Keyword'] = unique_kw

    df = df.merge(emb_df, on='Keyword', how='left')

    return df, emb_cols
