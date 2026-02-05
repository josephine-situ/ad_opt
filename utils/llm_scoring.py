"""
LLM-based keyword relevance scoring using Prometheus.

This module provides functions to evaluate keyword relevance using a large language model
instead of BERT embeddings. The LLM scores keywords on a 1-5 scale based on their
relevance to a specific course.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json
import re


# Course name mappings for the rubric
COURSE_NAME_MAP = {
    'gen_ai': 'Generative AI',
    'ml': 'Machine Learning',
    'sys_eng': 'Systems Engineering',
}


def get_relevance_prompt(keyword: str, course_name: str) -> str:
    """
    Generate the prompt for evaluating keyword relevance.
    
    Args:
        keyword: The search keyword to evaluate.
        course_name: The course name (e.g., 'Generative AI', 'Machine Learning').
    
    Returns:
        The formatted prompt string.
    """
    prompt = f"""Role: You are a Digital Marketing Specialist for a paid MIT {course_name} course.
Task: Calculate a relevance score (1-5) for the keyword below.
Scoring Logic:

Start with a Base Score of 3.
Apply Modifiers (Cumulative):
-1 points: Keyword contains "free", "cheap", "youtube", "login", or is unrelated to {course_name.lower()}.
-1 point: Keyword is purely informational (definitions, "what is", news).
+1 point: Keyword implies learning intent ("course", "training", "tutorial", "education").
+1 points: Keyword implies high purchase intent ("certification", "bootcamp", "university", "MIT", "executive", "paid").

Keyword: {keyword}
Output: Output ONLY the final calculated score number."""
    return prompt


def parse_llm_score(response: str) -> int:
    """
    Parse the LLM response to extract the numeric score.
    
    Args:
        response: The raw LLM response text.
    
    Returns:
        Integer score between 1 and 5, or 3 as default if parsing fails.
    """
    # Clean the response
    response = response.strip()
    
    # Try to extract a number from the response
    # First, try direct conversion
    try:
        score = int(response)
        return max(1, min(5, score))  # Clamp to 1-5
    except ValueError:
        pass
    
    # Try to find a number in the response
    numbers = re.findall(r'\b([1-5])\b', response)
    if numbers:
        return int(numbers[0])
    
    # Last resort: look for any digit
    digits = re.findall(r'\d', response)
    if digits:
        score = int(digits[0])
        return max(1, min(5, score))
    
    # Default to neutral score
    return 3


def load_prometheus_model(model_name: str = "prometheus-eval/prometheus-7b-v2.0"):
    """
    Load the Prometheus model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier for Prometheus.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  Loading Prometheus model: {model_name}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    return model, tokenizer, device


def score_keyword_batch(
    keywords: list,
    model,
    tokenizer,
    device: str,
    course_name: str,
    batch_size: int = 1,
    max_new_tokens: int = 10,
) -> list:
    """
    Score a batch of keywords using the Prometheus model.
    
    Args:
        keywords: List of keywords to score.
        model: The loaded Prometheus model.
        tokenizer: The tokenizer.
        device: The device ('cuda' or 'cpu').
        course_name: The course name for the rubric.
        batch_size: Number of keywords to process at once.
        max_new_tokens: Maximum tokens to generate for response.
    
    Returns:
        List of integer scores (1-5).
    """
    import torch
    
    scores = []
    
    for i in tqdm(range(0, len(keywords), batch_size), desc="Scoring keywords"):
        batch = keywords[i:i + batch_size]
        prompts = [get_relevance_prompt(kw, course_name) for kw in batch]
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode responses
        for j, output in enumerate(outputs):
            # Get only the generated tokens (after the prompt)
            prompt_length = inputs['input_ids'][j].shape[0]
            generated_tokens = output[prompt_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            score = parse_llm_score(response)
            scores.append(score)
    
    return scores


def get_llm_keyword_scores(
    unique_texts: list,
    course: str = 'gen_ai',
    model_name: str = "prometheus-eval/prometheus-7b-v2.0",
    batch_size: int = 1,
    cache_path: str = None,
    return_model: bool = False,
) -> pd.DataFrame:
    """
    Generate LLM-based relevance scores for keywords.
    
    This replaces BERT embeddings with a single relevance score (1-5) for each keyword.
    
    Args:
        unique_texts: List of unique keyword strings.
        course: Course identifier ('gen_ai', 'ml', 'sys_eng').
        model_name: HuggingFace model name for Prometheus.
        batch_size: Batch size for inference.
        cache_path: Optional path to cache results.
        return_model: If True, return tuple (df, model_info).
    
    Returns:
        pd.DataFrame with columns ['Keyword', 'llm_relevance_score'].
        Optionally returns tuple (df, model_info) if return_model=True.
    """
    # Convert to list if needed
    if not isinstance(unique_texts, list):
        unique_texts = list(unique_texts)
    
    course_name = COURSE_NAME_MAP.get(course, course)
    
    # Check for cached results
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            print(f"  [Cache] Loading LLM scores from {cache_file.name}")
            cached_df = pd.read_csv(cache_file)
            
            # Find keywords not in cache
            cached_keywords = set(cached_df['Keyword'].tolist())
            new_keywords = [kw for kw in unique_texts if kw not in cached_keywords]
            
            if not new_keywords:
                print(f"  All {len(unique_texts)} keywords found in cache")
                result_df = cached_df[cached_df['Keyword'].isin(unique_texts)].copy()
                if return_model:
                    return result_df, {'model_name': model_name, 'course': course}
                return result_df
            
            print(f"  Found {len(new_keywords)} new keywords not in cache")
            unique_texts = new_keywords
    
    print(f"  Scoring {len(unique_texts)} keywords with Prometheus LLM...")
    print(f"  Course: {course_name}")
    
    # Load model
    model, tokenizer, device = load_prometheus_model(model_name)
    
    # Score keywords
    scores = score_keyword_batch(
        unique_texts,
        model,
        tokenizer,
        device,
        course_name,
        batch_size=batch_size,
    )
    
    # Create DataFrame
    result_df = pd.DataFrame({
        'Keyword': unique_texts,
        'llm_relevance_score': scores,
    })
    
    # Merge with cached results if any
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            cached_df = pd.read_csv(cache_file)
            result_df = pd.concat([cached_df, result_df], ignore_index=True)
            result_df = result_df.drop_duplicates(subset=['Keyword'], keep='last')
        
        # Save updated cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(cache_file, index=False)
        print(f"  Saved LLM scores cache to {cache_file}")
    
    # Filter to requested keywords
    all_keywords = set(unique_texts)
    result_df = result_df[result_df['Keyword'].isin(all_keywords)].copy()
    
    if return_model:
        return result_df, {'model_name': model_name, 'course': course}
    
    return result_df


def add_llm_scores_to_df(
    df: pd.DataFrame,
    course: str = 'gen_ai',
    model_name: str = "prometheus-eval/prometheus-7b-v2.0",
    batch_size: int = 1,
    cache_dir: str = None,
) -> pd.DataFrame:
    """
    Add LLM relevance scores to a DataFrame.
    
    Args:
        df: DataFrame with a 'Keyword' column.
        course: Course identifier.
        model_name: Prometheus model name.
        batch_size: Batch size for inference.
        cache_dir: Directory for caching LLM scores.
    
    Returns:
        DataFrame with 'llm_relevance_score' column added.
    """
    unique_keywords = df['Keyword'].unique().tolist()
    
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f'llm_scores_{course}.csv'
    
    scores_df = get_llm_keyword_scores(
        unique_keywords,
        course=course,
        model_name=model_name,
        batch_size=batch_size,
        cache_path=cache_path,
    )
    
    # Merge scores into original DataFrame
    result_df = df.merge(scores_df, on='Keyword', how='left')
    
    # Fill any missing scores with neutral value
    result_df['llm_relevance_score'] = result_df['llm_relevance_score'].fillna(3)
    
    return result_df
