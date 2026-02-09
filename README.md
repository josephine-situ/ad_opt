# ML Optimization for Ads - Project Structure

## Overview

This project implements a machine learning pipeline for optimizing daily spend per keyword, region, and match type in Google Ads. It includes data preprocessing, model training, and bid optimization using linear programming.

## Steps
1. Get reports from Google Ads (Search keyword - raw input to models.csv (saved), Search keyword - search terms (saved, use conv > 0), Purchase report, and Location report (for geographic click distribution in evaluation) from Audiences, keywords and content > Locations). Make sure to use 'All conv.' when getting conversion reports (includes when purchases isn't a primary conversion action). Get Semrush keywords (Semrush > SEO > Keyword Magic Tool).
2. Run compare_keywords.py to combine new and existing keywords and search terms.
3. Copy and paste these keywords into Google Keyword Planner (change to Google and search partners) to get historical search popularity and competitiveness indices (make sure to change date range to 6 months before course start date).
4. Adjust the course start dates, course min dates, and budgets in `config.py`.
5. Run `python scripts/tidy_get_data.py --course gen_ai --force-reload`
6. Run `modeling.py`. Here, you can run `python scripts/interpret_xgb.py --course gen_ai` to make sure the model makes sense.
7. Run `optimization.py` (through `submit_bid_optimization_job.sh` or do a small run)

For the backtest, after step 5:
6. Run `backtest_daily.py` (through `submit_backtest.sh`)
7. Run `backtest_eval.py`
8. Run `analyze_backtest_results.py`

## Quick Start

### 1. Environment Setup

#### Option A: Using Conda (Recommended)

```bash
# Create conda environment with Python 3.10+
conda create -n mlopt python=3.10

# Activate environment
conda activate mlopt

# Clone repository
git clone https://github.com/josephine-situ/ad_opt.git
cd ad_opt

# Install all dependencies (core + optional)
pip install -e ".[bert,ml_open,optimization]"
```

#### Option B: Using venv

```bash
# Clone repository
git clone https://github.com/josephine-situ/ad_opt.git
cd ad_opt

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -e ".[bert,ml_open,optimization]"
```

**Dependency Installation Details:**

```bash
# Core only (data processing, embeddings)
pip install -e .

# Optional packages:
pip install -e ".[bert]"           # BERT embeddings (sentence-transformers)
pip install -e ".[ml]"             # IAI model training (requires license)
pip install -e ".[ml_open]"        # Open-source model training (no license)
pip install -e ".[optimization]"   # Gurobi optimization (requires license)

# All optional packages (recommended for full pipeline)
pip install -e ".[bert,ml_open,optimization]"

# Development tools
pip install -e ".[dev]"            # pytest, black, pylint, mypy
```

**Note on Commercial Licenses:**
- **Gurobi:** Required for bid optimization. Get license from https://www.gurobi.com/

### 3. Data Preparation Details

The `tidy_get_data.py` script handles the full data pipeline:

```bash
python scripts/tidy_get_data.py --embedding-method tfidf --force-reload
```

This script:
- Loads and combines 2024 and 2025 keyword data from `data/reports/`
- Cleans currency, percentages, and text
- Extracts temporal features (holidays, weekends, days to course start)
- Retrieves monthly search volume from Google Keyword Planner data
- Calculates time series statistics (3-month/6-month averages, momentum)
- Generates keyword embeddings (TF-IDF or BERT)
- Imputes missing values
- Removes rows with remaining missing values
- Creates train/test splits (75/25)
- Saves processed datasets and embedding models to `data/clean/` and `models/`

**Output files:**
```
data/clean/
├── ad_opt_data_tfidf.csv           # Full dataset (~71k rows × 60 cols)
├── train_tfidf.csv                 # Training set (~53k rows)
├── test_tfidf.csv                  # Test set (~18k rows)
├── unique_keyword_embeddings_tfidf.csv

models/
├── tfidf_pipeline_50d.pkl          # Saved TF-IDF vectorizer, SVD, normalizer
└── (embedding pipeline for inference)
```

Uses Gurobi linear programming to maximize profit:
- Loads new keywords from `data/gkp/keywords_classified.csv`
- Generates embeddings using saved pipeline
- Loads trained models for conversion and clicks prediction
- Solves optimization problem subject to budget and bid constraints

**Requirements:** Gurobi solver (included in `pip install -e ".[optimization]"`)

**Output:** Optimized bids saved to `opt_results/` directory

## Modules

### `utils/data_cleaning.py`
Utilities for parsing and normalizing data:
- `clean_currency()` - Parse currency strings
- `convert_percent_to_float()` - Parse percentage values

### `utils/date_features.py`
Temporal feature extraction:
- `_is_holiday()` - Detect public holidays
- `calculate_days_to_next()` - Days until next event
- `_region_to_country_code()` - Map regions to country codes

### `utils/embeddings.py`
Keyword embedding generation:
- `get_tfidf_embeddings()` - TF-IDF with TruncatedSVD (fast, interpretable)
- `get_bert_embeddings_pipeline()` - BERT with TruncatedSVD (semantic, accurate)
- `get_bert_embedding()` - Raw BERT embeddings (low-level)

### `utils/data_pipeline.py`
High-level pipeline functions (called by `scripts/tidy_get_data.py`):
- `load_and_combine_keyword_data()`
- `format_keyword_data()`
- `extract_date_features()`
- `merge_with_ads_data()`
- `add_embeddings()`
- `prepare_train_test_split()`
- `save_outputs()`

## Features

### Input Data
- Google Ads performance data
- Google Keyword Planner data (search volume, competition, bid ranges)

### Engineered Features
**Temporal:**
- Day of week
- Weekend indicator
- Month
- Public holiday indicator
- Days to next course start

**Text/Semantic:**
- TF-IDF embeddings (50 dimensions)
- Or BERT embeddings (50 dimensions after SVD)

**Bid Features:**
- Average CPC
- Average bid (low + high range / 2)

**Categorical:**
- Match type (broad, exact, phrase)
- Region (USA, Region A, B, C)

### Target Variables
- **Clicks:** Number of ad clicks

## Dependencies

Dependencies are managed via `pyproject.toml`. Install using pip with extras:

```bash
# Core dependencies only (data processing, embeddings)
pip install -e .

# Add optional packages as needed
pip install -e ".[bert]"           # BERT embeddings
pip install -e ".[ml]"             # IAI model training
pip install -e ".[optimization]"   # Gurobi optimization
pip install -e ".[dev]"            # Development tools

# All optional packages
pip install -e ".[bert,ml,optimization,dev]"
```

**Core Packages:**
```
pandas>=1.5.3          # Data manipulation
numpy>=1.24.3          # Numerical computing
scikit-learn>=1.2.2    # ML utilities (TF-IDF, train/test split)
scipy>=1.10.1          # Scientific computing
openpyxl>=3.0.10       # Excel file reading
holidays>=0.83         # Holiday calendar detection
torch>=2.0.0           # Deep learning backend
```

**Optional Packages:**
```
# BERT embeddings
sentence-transformers>=5.1.2       # Pre-trained BERT models
transformers>=4.35.2               # HuggingFace transformers (Python 3.9 compatible)

# Model training
iai>=2.11.2                        # InterpretableAI (requires commercial license)

# Bid optimization
gurobipy>=10.0.1                   # Gurobi solver (requires commercial license)

# Development
pytest>=7.3.1                      # Testing
black>=23.3.0                      # Code formatting
pylint>=2.12.2                     # Linting
mypy>=1.0.0                        # Type checking
```

**Note on Commercial Licenses:**
- **Gurobi:** https://www.gurobi.com/