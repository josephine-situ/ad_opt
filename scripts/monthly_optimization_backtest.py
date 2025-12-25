#!/usr/bin/env python3
"""
Per-day optimization over a specified month and comparison vs actuals.

For the specified month (YYYY-MM), this script:
 - Loads training data `clean_data/ad_opt_data_{embedding}.csv`
 - Loads embeddings `clean_data/unique_keyword_embeddings_{embedding}.csv`
 - Loads model weights from `models/weights_{embedding}_*` CSVs
 - For each calendar day in the month:
     - Builds the feature matrix (using `create_feature_matrix` from `bid_optimization`) for that day
     - Runs `optimize_bids_embedded` with budget = (total_cost_for_month / days_in_month)
     - Saves all active bids and computes expected daily profit = sum(pred_conv - bid * pred_clicks)
 - Sums expected daily profit across the month to get `expected_profit_month`
 - Computes `actual_profit_month = total_conv_value_for_month - total_cost_for_month`
 - Prints and saves the difference: `expected_profit_month - actual_profit_month`

Usage:
    python3 scripts/monthly_optimization_backtest.py --year-month 2024-11 --embedding-method bert

"""
import sys
from pathlib import Path
import argparse
from datetime import datetime, date
import calendar
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.bid_optimization import (
    create_feature_matrix,
    load_weights_from_csv,
    optimize_bids_embedded,
)


def parse_year_month(s: str):
    try:
        dt = datetime.strptime(s, '%Y-%m')
        return dt.year, dt.month
    except Exception:
        raise argparse.ArgumentTypeError("--year-month must be in YYYY-MM format")


def compute_model_predictions(X, weights_dict):
    """Compute conv_predictions and clicks_predictions arrays for X using weights_dict.
    Mirrors the logic inside `optimize_bids`.
    Returns: conv_predictions, clicks_predictions, conv_cpc_weight, clicks_cpc_weight
    """
    n = len(X)
    conv_const = weights_dict['conv_const']
    conv_weights = weights_dict['conv_weights']
    clicks_const = weights_dict['clicks_const']
    clicks_weights = weights_dict['clicks_weights']

    conv_predictions = np.full(n, conv_const, dtype=float)
    clicks_predictions = np.full(n, clicks_const, dtype=float)

    # Apply conversion weights
    for feature_name, weight in conv_weights.items():
        feature_str = str(feature_name).strip()
        if isinstance(weight, dict):
            # categorical
            for level_name, level_weight in weight.items():
                ohe_name = f"{feature_str}_{level_name}"
                if ohe_name in X.columns:
                    conv_predictions += level_weight * X[ohe_name].values
        else:
            if feature_str in X.columns:
                conv_predictions += weight * X[feature_str].values
            else:
                # try substring match
                matching = [c for c in X.columns if feature_str.lower() in c.lower()]
                if matching:
                    for c in matching:
                        conv_predictions += weight * X[c].values

    # Apply clicks weights
    for feature_name, weight in clicks_weights.items():
        feature_str = str(feature_name).strip()
        if isinstance(weight, dict):
            for level_name, level_weight in weight.items():
                ohe_name = f"{feature_str}_{level_name}"
                if ohe_name in X.columns:
                    clicks_predictions += level_weight * X[ohe_name].values
        else:
            if feature_str in X.columns:
                clicks_predictions += weight * X[feature_str].values
            else:
                matching = [c for c in X.columns if feature_str.lower() in c.lower()]
                if matching:
                    for c in matching:
                        clicks_predictions += weight * X[c].values

    conv_cpc_weight = conv_weights.get('Avg. CPC', conv_weights.get('Avg_ CPC', 0.0))
    clicks_cpc_weight = clicks_weights.get('Avg. CPC', clicks_weights.get('Avg_ CPC', 0.0))

    return conv_predictions, clicks_predictions, conv_cpc_weight, clicks_cpc_weight


def main():
    parser = argparse.ArgumentParser(description='Per-day optimizations for a month and compare expected profit vs actual')
    parser.add_argument('--year-month', type=parse_year_month, default='2025-10', help='Month to run in YYYY-MM')
    parser.add_argument('--embedding-method', type=str, default='bert', choices=['bert','tfidf'])
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--max-bid', type=float, default=50.0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    year, month = args.year_month
    root = Path(__file__).resolve().parent.parent

    # Load training data
    training_file = root / 'clean_data' / f'ad_opt_data_{args.embedding_method}.csv'
    if not training_file.exists():
        print(f"Training data not found: {training_file}")
        sys.exit(1)
    training_df = pd.read_csv(training_file)
    training_df['Day'] = pd.to_datetime(training_df['Day'])

    # Check month exists in data
    month_mask = (training_df['Day'].dt.year == year) & (training_df['Day'].dt.month == month)
    if month_mask.sum() == 0:
        print(f"No data for month {year}-{month:02d} in training data")
        sys.exit(1)

    # Compute month totals
    total_cost_month = training_df.loc[month_mask, 'Cost'].sum()
    total_conv_month = training_df.loc[month_mask, 'Conv. value'].sum()
    actual_profit_month = total_conv_month - total_cost_month

    # Days in month (calendar days)
    days_in_month = calendar.monthrange(year, month)[1]
    daily_budget = float(total_cost_month) / days_in_month if days_in_month > 0 else 0.0

    print(f"Month: {year}-{month:02d}")
    print(f"Total cost (month): ${total_cost_month:,.2f}")
    print(f"Total conversion value (month): ${total_conv_month:,.2f}")
    print(f"Actual profit (month): ${actual_profit_month:,.2f}")
    print(f"Days in month: {days_in_month}, daily budget = ${daily_budget:,.2f}")

    if args.dry_run:
        print("Dry run; exiting")
        return

    # Load embeddings
    emb_file = root / 'clean_data' / f'unique_keyword_embeddings_{args.embedding_method}.csv'
    if not emb_file.exists():
        print(f"Embeddings file not found: {emb_file}")
        sys.exit(1)
    keyword_df = pd.read_csv(emb_file)

    # Load weights
    weights = load_weights_from_csv(args.embedding_method, args.models_dir)

    expected_profit_month = 0.0
    daily_details = []
    all_bids = []

    # Iterate each calendar day
    for day in range(1, days_in_month+1):
        target_day = date(year, month, day)
        print(f"\nProcessing day: {target_day}")

        try:
            X, kw_idx_list, region_list, match_list = create_feature_matrix(keyword_df, embedding_method=args.embedding_method, target_day=str(target_day), training_data_path=str(training_file), weights_dict=weights)
        except Exception as e:
            print(f"  create_feature_matrix error for {target_day}: {e}")
            continue

        if X is None or len(X) == 0:
            print(f"  No feature rows for {target_day}; skipping")
            continue

        # Compute model predictions arrays
        conv_preds, clicks_preds, conv_cpc_w, clicks_cpc_w = compute_model_predictions(X, weights)

        # Run optimize_bids for today's combos with daily budget
        try:
            model, b_var, z_var, y_var, f_eff_var, g_eff_var = optimize_bids_embedded(X, weights, budget=daily_budget, max_bid=args.max_bid)
        except Exception as e:
            print(f"  optimize_bids error for {target_day}: {e}")
            continue

        if model is None:
            print(f"  No model returned for {target_day}")
            continue

        # Ensure model has solution
        try:
            b_vals = b_var.X
            z_vals = z_var.X
        except Exception as e:
            print(f"  No solution for {target_day}: {e}")
            continue

        # Compute predicted conv and clicks per row using chosen bids
        pred_conv = conv_preds + conv_cpc_w * b_vals
        pred_clicks = clicks_preds + clicks_cpc_w * b_vals

        # Use all active bids (where bid > 0)
        active_idx = np.where(b_vals > 0)[0]
        # Compute expected profit for all active bids: sum(pred_conv - bid * pred_clicks)
        day_expected_profit = float(np.sum(pred_conv[active_idx] - b_vals[active_idx] * pred_clicks[active_idx]))
        expected_profit_month += day_expected_profit

        # Build and save all active bids for this day
        try:
            kw_names = [keyword_df.iloc[kw_idx_list[i]]['Keyword'] for i in active_idx]
        except Exception:
            kw_names = [keyword_df.iloc[kw_idx_list[i]]['Keyword'] if i < len(kw_idx_list) else '' for i in active_idx]

        for idx_pos, i in enumerate(active_idx):
            all_bids.append({
                'day': str(target_day),
                'rank': idx_pos + 1,
                'keyword': kw_names[idx_pos],
                'region': region_list[i],
                'match': match_list[i],
                'bid': float(b_vals[i]),
                'pred_conv': float(pred_conv[i]),
                'pred_clicks': float(pred_clicks[i]),
                'predicted_profit': float(pred_conv[i] - b_vals[i] * pred_clicks[i]),
            })

        daily_details.append({
            'day': str(target_day),
            'rows': len(X),
            'budget': daily_budget,
            'budget_used': float(np.sum(b_vals)),
            'active_bids_count': len(active_idx),
            'expected_profit': day_expected_profit,
            'sum_pred_conv': float(np.sum(pred_conv[active_idx])),
            'sum_pred_clicks': float(np.sum(pred_clicks[active_idx])),
        })

        print(f"  Rows: {len(X)}, budget used: ${np.sum(b_vals):,.2f}, active bids: {len(active_idx)}, expected profit: ${day_expected_profit:,.2f}")

    # After all days
    print("\n===== Monthly Summary =====")
    print(f"Expected profit (sum of daily optimized bids): ${expected_profit_month:,.2f}")
    print(f"Actual profit (month): ${actual_profit_month:,.2f}")
    diff = expected_profit_month - actual_profit_month
    print(f"Difference (expected - actual): ${diff:,.2f}")

    # Save details
    out_dir = root / 'opt_results'
    out_dir.mkdir(exist_ok=True)
    details_file = out_dir / f'month_daily_summary_{args.embedding_method}_{year}-{month:02d}.csv'
    pd.DataFrame(daily_details).to_csv(details_file, index=False)
    print(f"Daily details saved to: {details_file}")
    # Save all bids across the month
    bids_file = out_dir / f'month_all_bids_{args.embedding_method}_{year}-{month:02d}.csv'
    if all_bids:
        pd.DataFrame(all_bids).to_csv(bids_file, index=False)
        print(f"All bids (per day) saved to: {bids_file}")
    else:
        print("No bids to save.")

    # Save a simple summary log with expected vs actual
    summary_file = out_dir / f'month_summary_{args.embedding_method}_{year}-{month:02d}.txt'
    with open(summary_file, 'w') as sf:
        sf.write(f"Month: {year}-{month:02d}\n")
        sf.write(f"Total cost (month): ${total_cost_month:,.2f}\n")
        sf.write(f"Total conversion value (month): ${total_conv_month:,.2f}\n")
        sf.write(f"Actual profit (month): ${actual_profit_month:,.2f}\n")
        sf.write(f"Expected profit (sum of daily optimized bids): ${expected_profit_month:,.2f}\n")
        sf.write(f"Difference (expected - actual): ${expected_profit_month - actual_profit_month:,.2f}\n")
        sf.write(f"Days in month: {days_in_month}\n")
        sf.write(f"Daily budget: ${daily_budget:,.2f}\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
