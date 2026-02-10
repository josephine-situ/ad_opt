# Ad Bid Optimizer

Optimizes daily Google Ads spend per keyword, region, and match type using XGBoost and Gurobi mixed-integer programming. The pipeline ingests Google Ads performance data and Google Keyword Planner statistics, trains a clicks prediction model, embeds that model into a Gurobi MIP to maximize predicted clicks (or purchases) subject to a budget constraint, and outputs daily bids, budgets, and bid adjustments ready for upload.

## Pipeline Overview

```
Google Ads reports ──┐
Semrush keywords ────┤
Google Keyword       ├── tidy_get_data.py ── modeling.py ── optimization.py ── bid_post_processing.py
Planner stats ───────┘        │                  │                │                    │
                         Clean & embed      Train XGBoost    Embed model in       Base bids,
                         keyword data       clicks model     Gurobi MIP &         bid adjustments,
                                                             solve                daily budgets
```

### Steps

1. **Keyword curation** — `compare_keywords.py` merges new Semrush keywords with existing Google Ads search terms and keywords, deduplicates, and classifies each as `existing`, `existing searches`, or `new`.
2. **Google Keyword Planner** — Paste the keyword list into GKP to obtain historical search volume, competition index, and bid ranges.
3. **Configuration** — Set course start dates, minimum data dates, and budgets in `config.py`.
4. **Data preparation** — `tidy_get_data.py` loads reports, cleans data, extracts temporal features, merges GKP statistics, generates keyword embeddings (BERT 50-d or LLM relevance scores), imputes missing values, and creates train/test splits.
5. **Modeling** — `modeling.py` trains an XGBoost regressor (with CV-tuned hyperparameters) to predict clicks from keyword features + cost. `interpret_xgb.py` produces SHAP-based variable importance.
6. **Optimization** — `optimization.py` embeds the trained XGBoost tree structure into a Gurobi MIP and solves for cost allocations that maximize total predicted clicks (or expected purchases) under a total budget constraint with regional sub-budgets.
7. **Post-processing** — `bid_post_processing.py` converts optimal costs into base bids and computes bid adjustments by hour, device, location, and age segment based on conversion rate ratios.

### Backtesting

A rolling daily backtest validates out-of-sample performance:

1. `backtest_daily.py` — For each day *t*, trains a model on data up to *t−1* and optimizes bids for day *t*. Supports masked keywords (randomly hiding a fraction of known keywords to simulate new keywords).
2. `backtest_eval.py` — Re-evaluates each day's optimized bids using a model trained on the full dataset.
3. `analyze_backtest_results.py` — Aggregates metrics across days and budgets, producing summary CSVs and LaTeX tables.
4. `sensitivity_analysis.py` — Drops random fractions of keywords and rescales bids to measure concentration risk.

## Repository Structure

```
├── config.py                   # Course start dates, min dates, budgets
├── pyproject.toml              # Package metadata and dependencies
├── scripts/
│   ├── compare_keywords.py     # Merge & classify Semrush + existing keywords
│   ├── tidy_get_data.py        # Full data preparation pipeline
│   ├── modeling.py             # XGBoost training & evaluation
│   ├── interpret_xgb.py        # SHAP variable importance
│   ├── optimization.py         # Gurobi MIP bid optimization
│   ├── bid_post_processing.py  # Base bids & segment bid adjustments
│   ├── format_bids_excel.py    # Export bids to formatted Excel
│   ├── backtest_daily.py       # Rolling daily backtest
│   ├── backtest_eval.py        # Backtest evaluation
│   ├── analyze_backtest_results.py  # Backtest summary tables
│   └── sensitivity_analysis.py # Keyword concentration risk analysis
├── utils/
│   ├── data_cleaning.py        # Currency, percentage, text cleaning
│   ├── data_pipeline.py        # Load, format, merge, embed, split
│   ├── date_features.py        # Holidays, weekends, days-to-course-start
│   ├── embeddings.py           # TF-IDF and BERT embedding generation
│   ├── keyword_matching.py     # Fuzzy keyword normalization & matching
│   ├── llm_scoring.py          # LLM-based keyword relevance scoring (Qwen3)
│   └── tee_logging.py          # Tee stdout/stderr to log files
├── data/<course>/
│   ├── reports/                # Raw Google Ads CSV exports
│   ├── gkp/                    # Google Keyword Planner & Semrush data
│   ├── cache/                  # Intermediate pipeline caches (parquet)
│   └── clean/                  # Processed train/test datasets
├── models/                     # Saved XGBoost pipelines (.joblib)
├── model_interpretability/     # SHAP & variable importance CSVs
├── opt_results/<course>/
│   ├── bids/                   # Optimized cost allocations
│   ├── bid_adjustments/        # Segment bid adjustments
│   ├── backtests/              # Backtest experiment outputs
│   ├── cache/                  # Optimization feature matrix caches
│   └── eval_models/            # Evaluation models for backtests
└── logs/                       # Timestamped run logs
```

Courses supported: `gen_ai`, `ml`, `sys_eng`, `sys_think`.

## Features

| Category | Features |
|----------|----------|
| **Temporal** | Day of week, weekend flag, month, public holiday flag, days to next course start |
| **Search popularity** | Last-month searches, 3-month avg, 6-month avg, month-over-month change, search trend |
| **Competitiveness** | Competition index, top-of-page bid (low range), top-of-page bid (high range) |
| **Keyword representation** | BERT embeddings (50-d via SVD) or LLM relevance score (1–5 scale) |
| **Categorical** | Match type (Exact, Phrase, Broad), Region (USA, A, B) |
| **Decision variable** | Cost (daily spend per keyword-region-match type combination) |

**Target:** Clicks (daily).

## Setup

```bash
conda create -n mlopt python=3.10 && conda activate mlopt
git clone https://github.com/josephine-situ/ad_opt.git && cd ad_opt
pip install -e ".[bert,ml_open,optimization]"
```

Optional extras: `bert` (sentence-transformers), `llm` (Qwen3 scoring), `ml_open` (XGBoost), `optimization` (Gurobi), `dev` (pytest, black, pylint, mypy).

**Gurobi license** required for optimization — see https://www.gurobi.com/.