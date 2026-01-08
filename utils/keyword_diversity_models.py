"""Keyword diversity (entropy/HHI) models.

This module supports training lightweight regression models that predict per-keyword
search-term diversity metrics from keyword embeddings.

Intended usage:
- In "training" mode: compute targets from a search-term performance report and
  generate out-of-fold (OOF) predictions per keyword to avoid leakage when used as
  stacked features.
- In "prod" mode: load saved models and predict for all keywords.

The resulting predictions are merged back into the main feature matrix as:
- keyword_entropy_pred
- keyword_hhi_pred

Expected inputs:
- Main feature df has at least: 'Keyword' and embedding columns (bert_* or tfidf_*).
- Search-term performance df has at least:
    'Keyword', 'Search Term', 'Clicks', 'Conv. value'

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception as e:  # pragma: no cover
    raise ImportError("joblib is required (normally installed with scikit-learn)") from e


DEFAULT_ENTROPY_COL = "keyword_entropy_pred"
DEFAULT_HHI_COL = "keyword_hhi_pred"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()
    return out


def calculate_entropy_targets(
    df: pd.DataFrame,
    *,
    keyword_col: str = "Keyword",
    term_col: str = "Search Term",
    clicks_col: str = "Clicks",
    out_col: str = "keyword_entropy_target",
) -> pd.DataFrame:
    """Compute per-keyword click entropy over search terms."""

    df = _normalize_columns(df)
    required = {keyword_col, term_col, clicks_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Entropy target requires columns: {sorted(missing)}")

    # Normalize text fields
    df[keyword_col] = df[keyword_col].astype(str).str.replace(r"[\"\[\]]", "", regex=True).str.lower().str.strip()
    df[term_col] = df[term_col].astype(str).str.lower().str.strip()

    term_stats = (
        df.groupby([keyword_col, term_col], dropna=False)[clicks_col]
        .sum()
        .reset_index()
    )

    # Avoid division by 0: keywords with zero total clicks will get NaN target.
    keyword_totals = term_stats.groupby(keyword_col)[clicks_col].transform("sum")
    term_stats["p_i"] = term_stats[clicks_col] / keyword_totals

    # Numerical safety: ignore p=0 contributions.
    p = term_stats["p_i"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = -p * np.log(p)
    contrib[~np.isfinite(contrib)] = 0.0
    term_stats["entropy_contribution"] = contrib

    y = term_stats.groupby(keyword_col)["entropy_contribution"].sum().reset_index()
    y = y.rename(columns={"entropy_contribution": out_col, keyword_col: "Keyword"})
    return y


def calculate_hhi_targets(
    df: pd.DataFrame,
    *,
    keyword_col: str = "Keyword",
    term_col: str = "Search Term",
    value_col: str = "Conv. value",
    out_col: str = "keyword_hhi_target",
) -> pd.DataFrame:
    """Compute per-keyword HHI over search terms using Conv. value shares.

    HHI is computed as: sum_i p_i^2 where p_i is the share of conv value
    attributable to search term i within a keyword.

    Notes:
    - Keywords with total conv value == 0 get NaN target (insufficient signal).
    """

    df = _normalize_columns(df)
    required = {keyword_col, term_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"HHI target requires columns: {sorted(missing)}")

    # Normalize text fields
    df[keyword_col] = df[keyword_col].astype(str).str.replace(r"[\"\[\]]", "", regex=True).str.lower().str.strip()
    df[term_col] = df[term_col].astype(str).str.lower().str.strip()

    term_stats = (
        df.groupby([keyword_col, term_col], dropna=False)[value_col]
        .sum()
        .reset_index()
    )

    keyword_totals = term_stats.groupby(keyword_col)[value_col].transform("sum")
    term_stats["p_i"] = term_stats[value_col] / keyword_totals

    p = term_stats["p_i"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        hhi_contrib = p**2
    hhi_contrib[~np.isfinite(hhi_contrib)] = 0.0

    y = pd.DataFrame({"Keyword": term_stats[keyword_col], "_hhi": hhi_contrib})
    y = y.groupby("Keyword")["_hhi"].sum().reset_index().rename(columns={"_hhi": out_col})

    # Mark zero-total-conv keywords as NaN target (not informative for training)
    totals = term_stats.groupby(keyword_col)[value_col].sum().reset_index()
    totals = totals.rename(columns={keyword_col: "Keyword", value_col: "_total"})
    y = y.merge(totals, on="Keyword", how="left")
    y.loc[y["_total"].fillna(0.0) <= 0.0, out_col] = np.nan
    y = y.drop(columns=["_total"], errors="ignore")

    return y


def build_default_regressor(random_state: int = 42) -> Pipeline:
    """A small, stable regressor for embeddings."""

    # RidgeCV is fast and works well for dense embeddings.
    model = RidgeCV(alphas=np.logspace(-4, 4, 25))
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _embedding_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("bert_") or c.startswith("tfidf_")]
    if not cols:
        cols = [c for c in df.columns if "bert" in c or "tfidf" in c]
    return cols


def _safe_n_splits(n_splits: int, n_groups: int) -> int:
    return int(max(2, min(int(n_splits), int(n_groups))))


def _stacked_oof_predictions(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    base_estimator: Pipeline,
    n_splits: int,
) -> Tuple[np.ndarray, List[Pipeline]]:
    """Return OOF preds and the list of fitted fold models."""

    unique_groups = pd.Index(groups).unique()
    n_splits_eff = _safe_n_splits(n_splits, len(unique_groups))

    splitter = GroupKFold(n_splits=n_splits_eff)
    oof = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)
    models: List[Pipeline] = []

    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        m = clone(base_estimator)
        m.fit(X[train_idx], y[train_idx])
        oof[val_idx] = m.predict(X[val_idx])
        models.append(m)

    return oof, models


@dataclass(frozen=True)
class DiversityModelArtifacts:
    entropy_model_path: Path
    hhi_model_path: Path


def default_artifact_paths(model_dir: str | Path, embedding_method: str, n_components: int) -> DiversityModelArtifacts:
    model_dir = Path(model_dir)
    suffix = f"{embedding_method}_{n_components}d"
    return DiversityModelArtifacts(
        entropy_model_path=model_dir / f"diversity_entropy_model_{suffix}.joblib",
        hhi_model_path=model_dir / f"diversity_hhi_model_{suffix}.joblib",
    )


def train_or_load_and_predict(
    *,
    main_df: pd.DataFrame,
    search_term_df: Optional[pd.DataFrame],
    embedding_method: str,
    n_components: int,
    mode: str,
    model_dir: str | Path = "models",
    n_splits: int = 5,
    random_state: int = 42,
    entropy_pred_col: str = DEFAULT_ENTROPY_COL,
    hhi_pred_col: str = DEFAULT_HHI_COL,
) -> pd.DataFrame:
    """Train/load diversity models and produce per-keyword predictions.

    Returns a dataframe with columns: Keyword, entropy_pred_col, hhi_pred_col.

    mode:
      - "training": requires search_term_df; produces OOF preds for keywords with targets.
      - "prod": loads saved models; if missing, trains using provided search_term_df.
    """

    if mode not in {"training", "prod"}:
        raise ValueError("mode must be 'training' or 'prod'")

    main_df = _normalize_columns(main_df)
    if "Keyword" not in main_df.columns:
        raise KeyError("main_df must have a 'Keyword' column")

    emb_cols = _embedding_columns(main_df)
    if not emb_cols:
        raise ValueError("No embedding columns found in main_df (expected bert_* or tfidf_*)")

    # One row per keyword for embedding inputs.
    kw_embed = main_df[["Keyword", *emb_cols]].drop_duplicates(subset=["Keyword"]).copy()
    kw_embed = kw_embed.dropna(subset=emb_cols)

    artifacts = default_artifact_paths(model_dir, embedding_method=embedding_method, n_components=n_components)
    artifacts.entropy_model_path.parent.mkdir(parents=True, exist_ok=True)

    entropy_model: Optional[Pipeline] = None
    hhi_model: Optional[Pipeline] = None

    if mode == "prod":
        if artifacts.entropy_model_path.exists() and artifacts.hhi_model_path.exists():
            entropy_model = joblib.load(artifacts.entropy_model_path)
            hhi_model = joblib.load(artifacts.hhi_model_path)
        elif search_term_df is None:
            raise FileNotFoundError(
                "Diversity models not found for prod mode and no search_term_df was provided to train them. "
                "Provide the raw report dataframe once to train and persist models."
            )

    if entropy_model is None or hhi_model is None:
        if search_term_df is None:
            raise ValueError("search_term_df is required to train diversity models")

        search_term_df = _normalize_columns(search_term_df)

        # Restrict target calculation to phrase/broad only (as requested), if Match type is present.
        if 'Match type' in search_term_df.columns:
            allowed = {"Broad match", "Phrase match"}
            search_term_df = search_term_df[search_term_df['Match type'].isin(allowed)].copy()

        # Compute targets from search terms.
        entropy_y = calculate_entropy_targets(search_term_df)
        hhi_y = calculate_hhi_targets(search_term_df)

        targets = entropy_y.merge(hhi_y, on="Keyword", how="outer")

        # Join embeddings.
        train_df = targets.merge(kw_embed, on="Keyword", how="inner")

        # Filter training rows: need entropy target; HHI may be NaN (skip those rows for HHI model)
        if "Match type" in main_df.columns:
            # Restrict targets to phrase/broad only as requested.
            kw_match = main_df[["Keyword", "Match type"]].drop_duplicates(subset=["Keyword"]).copy()
            train_df = train_df.merge(kw_match, on="Keyword", how="left")
            allowed = {"Broad match", "Phrase match"}
            train_df = train_df[train_df["Match type"].isin(allowed)].copy()

        X = train_df[emb_cols].to_numpy(dtype=float)

        base = build_default_regressor(random_state=random_state)

        # Entropy model
        y_entropy = train_df["keyword_entropy_target"].to_numpy(dtype=float)
        entropy_model = clone(base).fit(X, y_entropy)

        # HHI model: drop NaN targets
        hhi_mask = train_df["keyword_hhi_target"].notna().to_numpy()
        if int(hhi_mask.sum()) < 10:
            # Too few examples; fall back to predicting the global mean.
            class _MeanModel:
                def __init__(self, mean: float):
                    self.mean = float(mean)

                def predict(self, X):
                    return np.full(shape=(np.asarray(X).shape[0],), fill_value=self.mean, dtype=float)

            mean_hhi = float(np.nanmean(train_df["keyword_hhi_target"].to_numpy(dtype=float)))
            if not np.isfinite(mean_hhi):
                mean_hhi = 0.0
            hhi_model = _MeanModel(mean_hhi)  # type: ignore[assignment]
        else:
            y_hhi = train_df.loc[hhi_mask, "keyword_hhi_target"].to_numpy(dtype=float)
            X_hhi = train_df.loc[hhi_mask, emb_cols].to_numpy(dtype=float)
            hhi_model = clone(base).fit(X_hhi, y_hhi)

        # Persist models for prod.
        joblib.dump(entropy_model, artifacts.entropy_model_path)
        joblib.dump(hhi_model, artifacts.hhi_model_path)

    # Produce predictions.
    X_all = kw_embed[emb_cols].to_numpy(dtype=float)
    pred_entropy = np.asarray(entropy_model.predict(X_all), dtype=float)
    pred_hhi = np.asarray(hhi_model.predict(X_all), dtype=float)

    out = pd.DataFrame({
        "Keyword": kw_embed["Keyword"].to_numpy(),
        entropy_pred_col: pred_entropy,
        hhi_pred_col: pred_hhi,
    })

    if mode == "training":
        if search_term_df is None:
            raise ValueError("training mode requires search_term_df")

        # Replace entropy/hhi predictions for in-sample keywords with OOF predictions.
        search_term_df = _normalize_columns(search_term_df)
        if 'Match type' in search_term_df.columns:
            allowed = {"Broad match", "Phrase match"}
            search_term_df = search_term_df[search_term_df['Match type'].isin(allowed)].copy()

        entropy_y = calculate_entropy_targets(search_term_df)
        hhi_y = calculate_hhi_targets(search_term_df)
        targets = entropy_y.merge(hhi_y, on="Keyword", how="outer")

        train_df = targets.merge(kw_embed, on="Keyword", how="inner")
        if "Match type" in main_df.columns:
            kw_match = main_df[["Keyword", "Match type"]].drop_duplicates(subset=["Keyword"]).copy()
            train_df = train_df.merge(kw_match, on="Keyword", how="left")
            allowed = {"Broad match", "Phrase match"}
            train_df = train_df[train_df["Match type"].isin(allowed)].copy()

        # Entropy OOF
        X = train_df[emb_cols].to_numpy(dtype=float)
        groups = train_df["Keyword"].astype(str).tolist()
        y_entropy = train_df["keyword_entropy_target"].to_numpy(dtype=float)

        base = build_default_regressor(random_state=random_state)
        oof_entropy, _ = _stacked_oof_predictions(
            X=X, y=y_entropy, groups=groups, base_estimator=base, n_splits=n_splits
        )
        oof_entropy_df = pd.DataFrame({"Keyword": train_df["Keyword"].to_numpy(), entropy_pred_col: oof_entropy})

        # HHI OOF (only where target present)
        hhi_mask = train_df["keyword_hhi_target"].notna().to_numpy()
        if int(hhi_mask.sum()) >= 2:
            X_hhi = train_df.loc[hhi_mask, emb_cols].to_numpy(dtype=float)
            y_hhi = train_df.loc[hhi_mask, "keyword_hhi_target"].to_numpy(dtype=float)
            groups_hhi = train_df.loc[hhi_mask, "Keyword"].astype(str).tolist()
            oof_hhi, _ = _stacked_oof_predictions(
                X=X_hhi, y=y_hhi, groups=groups_hhi, base_estimator=base, n_splits=n_splits
            )
            oof_hhi_df = pd.DataFrame({"Keyword": train_df.loc[hhi_mask, "Keyword"].to_numpy(), hhi_pred_col: oof_hhi})
        else:
            oof_hhi_df = pd.DataFrame(columns=["Keyword", hhi_pred_col])

        out = out.merge(oof_entropy_df, on="Keyword", how="left", suffixes=("", "_oof"))
        # If we have OOF, overwrite pred with it
        out[entropy_pred_col] = out[f"{entropy_pred_col}_oof"].combine_first(out[entropy_pred_col])
        out = out.drop(columns=[f"{entropy_pred_col}_oof"], errors="ignore")

        out = out.merge(oof_hhi_df, on="Keyword", how="left", suffixes=("", "_oof"))
        out[hhi_pred_col] = out[f"{hhi_pred_col}_oof"].combine_first(out[hhi_pred_col])
        out = out.drop(columns=[f"{hhi_pred_col}_oof"], errors="ignore")

    return out
