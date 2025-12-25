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


def get_tfidf_embeddings(unique_texts, n_components=50, ngram_range=(1, 2), min_df=1):
    """
    Generate TF-IDF embeddings reduced to n_components via TruncatedSVD and L2 normalized.
    
    Pipeline: TF-IDF → TruncatedSVD → L2 Normalization
    
    Args:
    - unique_texts (list or array): Unique text strings (e.g., keywords).
    - n_components (int): Target embedding dimension. Default 50.
    - ngram_range (tuple): (min_n, max_n) for n-grams. Default (1, 2).
    - min_df (int): Minimum document frequency. Default 1 (keep all terms).
    
    Returns:
    - pd.DataFrame: Columns ['tfidf_0', 'tfidf_1', ..., 'text'] with shape (n_texts, n_components + 1).
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
    
    return embedding_df


def get_bert_embeddings_pipeline(unique_texts, n_components=50, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Generate BERT embeddings (via sentence-transformers) reduced to n_components via TruncatedSVD and L2 normalized.
    
    Pipeline: BERT (sentence-transformers) → TruncatedSVD → L2 Normalization
    
    Args:
    - unique_texts (list or array): Unique text strings (e.g., keywords).
    - n_components (int): Target embedding dimension. Default 50.
    - model_name (str): Sentence-transformers model identifier. Default 'all-MiniLM-L6-v2' (fast, 384D).
    - batch_size (int): Batch size for encoding. Default 32.
    
    Returns:
    - pd.DataFrame: Columns ['bert_0', 'bert_1', ..., 'text'] with shape (n_texts, n_components + 1).
    
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
    
    return embedding_df
