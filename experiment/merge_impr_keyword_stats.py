#!/usr/bin/env python3
"""Merge experiment impressions with GKP keyword stats.

This script:
1. Loads the experiment impressions export and aggregates impressions by
   search keyword, region, and match type.
2. Loads the saved keyword stats export, finds the rightmost monthly search
   column, and sums that month's searches by keyword.
3. Merges both datasets on normalized keyword text.
4. Computes the multiplier impr / last month searches per merged keyword row.
5. Summarizes the average multiplier by region and match type.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_csv_header_row(path: Path, header_prefixes: tuple[str, ...]) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for idx, line in enumerate(handle):
            stripped = line.strip()
            if any(stripped.startswith(prefix) for prefix in header_prefixes):
                return idx
            if idx >= 50:
                break
    raise ValueError(f"Could not find a CSV header row in {path}")


def normalize_keyword(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def extract_region(campaign: object) -> str:
    if pd.isna(campaign):
        return ""
    parts = str(campaign).split(" - ")
    return parts[-1].strip() if parts else str(campaign).strip()


def load_experiment_impressions(path: Path) -> pd.DataFrame:
    header_idx = find_csv_header_row(path, ("Day,Search keyword", "Search keyword,"))
    df = pd.read_csv(path, skiprows=header_idx)
    required = {"Search keyword", "Search keyword match type", "Campaign", "Impr."}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected experiment columns in {path}: {sorted(missing)}")

    df = df.copy()
    df["Search keyword"] = df["Search keyword"].astype(str)
    df["keyword_norm"] = df["Search keyword"].map(normalize_keyword)
    df["Region"] = df["Campaign"].map(extract_region)
    df["Impr."] = pd.to_numeric(df["Impr."], errors="coerce").fillna(0)

    grouped = (
        df.groupby(["Region", "Search keyword match type", "Search keyword", "keyword_norm"], as_index=False)
        .agg(Impressions=("Impr.", "sum"))
    )
    return grouped


def load_keyword_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    if "Keyword" not in df.columns:
        raise ValueError(f"Missing 'Keyword' column in {path}")

    search_cols = [col for col in df.columns if str(col).startswith("Searches:")]
    if not search_cols:
        raise ValueError(f"No monthly 'Searches:' columns found in {path}")
    last_month_col = search_cols[-1]

    df = df.loc[df["Keyword"].notna(), ["Keyword", last_month_col]].copy()
    df["keyword_norm"] = df["Keyword"].map(normalize_keyword)
    df[last_month_col] = pd.to_numeric(df[last_month_col], errors="coerce").fillna(0)

    grouped = (
        df.groupby(["Keyword", "keyword_norm"], as_index=False)
        .agg(**{"Last month searches": (last_month_col, "sum")})
    )
    return grouped


def build_outputs(impr_path: Path, stats_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    impr_df = load_experiment_impressions(impr_path)
    stats_df = load_keyword_stats(stats_path)

    merged = impr_df.merge(stats_df, on="keyword_norm", how="left", suffixes=("", "_stats"))
    merged["Multiplier"] = merged["Impressions"] / (merged["Last month searches"].fillna(0) + 1) / 30 # Avoid divide by 0, daily
    # merged.loc[merged["Last month searches"] == 0, "Multiplier"] = pd.NA

    summary = (
        merged.groupby(["Region", "Search keyword match type"], as_index=False)
        .agg(
            Average_multiplier=("Multiplier", "mean"),
            Merged_keywords=("keyword_norm", "nunique"),
            Total_impressions=("Impressions", "sum"),
            Total_last_month_searches=("Last month searches", "sum"),
        )
    )
    return merged, summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gkp_dir = repo_root / "data" / "sys_think" / "gkp"

    parser = argparse.ArgumentParser(
        description="Merge experiment impressions with keyword stats and summarize impr/search multipliers."
    )
    parser.add_argument(
        "--impressions",
        type=Path,
        default=repo_root / "experiment" / "Experiment - impr.csv",
        help="Experiment impressions export",
    )
    parser.add_argument(
        "--keyword-stats",
        type=Path,
        default=gkp_dir / "Saved Keywords Stats 2026-02-07 at 11_18_23.csv",
        help="GKP saved keyword stats export",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=repo_root / "experiment" / "impr_keyword_stats_merged.csv",
        help="Path for the merged detailed output",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repo_root / "experiment" / "impr_keyword_stats_summary.csv",
        help="Path for the region/match-type summary output",
    )
    args = parser.parse_args()

    merged_df, summary_df = build_outputs(args.impressions, args.keyword_stats)

    args.merged_output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(args.merged_output, index=False)
    summary_df.to_csv(args.summary_output, index=False)

    print(f"Wrote merged detail to {args.merged_output}")
    print(f"Wrote summary to {args.summary_output}")
    print(summary_df.sort_values(["Region", "Search keyword match type"]).to_string(index=False))


if __name__ == "__main__":
    main()