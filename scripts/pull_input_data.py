#!/usr/bin/env python3
"""
Script to pull input data from various sources (ads reports, keyword planning, SEMrush).
"""

import argparse
import os
import sys

from google.ads.googleads.client import GoogleAdsClient

ADS_REPORTS = 'ads_reports'
KEYWORD_PLANNING = 'keyword_planning'
SEMRUSH = 'semrush'
VALID_DATASETS = {ADS_REPORTS, KEYWORD_PLANNING, SEMRUSH}

# TODO: Put queries into their own module.
RAW_INPUT_TO_MODELS_QUERY = """
    SELECT
        segments.date,
        search_term_view.search_term,
        segments.search_term_match_type,
        campaign.name,
        metrics.clicks,
        metrics.conversions_value,
        customer.currency_code,
        metrics.cost_micros
    FROM search_term_view
    WHERE segments.date BETWEEN '2024-07-01' AND '2026-01-11'
    ORDER BY segments.date
"""

def validate_environment_variables(datasets):
    """Validate that required environment variables are set for the given dataset."""
    required_vars = []
    for dataset in datasets:
        if dataset in [ADS_REPORTS, KEYWORD_PLANNING]:
            required_vars.extend([
                'GOOGLE_ADS_CUSTOMER_ID',
                'GOOGLE_ADS_YAML_PATH'
            ])

        if dataset == SEMRUSH:
            required_vars.append('SEMRUSH_API_KEY')

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    return True


def pull_ads_reports(google_ads_client: GoogleAdsClient, customer_id: str = None):
    """Pull ads reports data from Google Ads."""
    
    print(f"Pulling ads reports data...")
    print(f"Customer ID: {customer_id}")
    # TODO: This is completely untested. I think the query is approximately correct,
    # but we need to run it against an account with real data and compare results

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=RAW_INPUT_TO_MODELS_QUERY)

    # Process results
    # TODO: Once we've validated this, we'll need to write it to a file instead of printing
    # We will also need to add the other GAQL queries
    for batch in stream:
        for row in batch.results:
            date = row.segments.date
            search_term = row.search_term_view.search_term
            match_type = row.segments.search_term_match_type.name
            campaign_name = row.campaign.name
            clicks = row.metrics.clicks
            conv_value = row.metrics.conversions_value
            currency = row.customer.currency_code
            cost = row.metrics.cost_micros / 1_000_000  # Convert from micros https://groups.google.com/g/adwords-scripts/c/mSl5bxSkwec

            print(f"{date},{search_term},{match_type},{campaign_name},{clicks},{conv_value},{currency},{cost}")
    

def pull_keyword_planning(google_ads_client: GoogleAdsClient, customer_id: str = None):
    """Pull keyword planning data from Google Ads."""
    customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID')
    
    print(f"Pulling keyword planning data...")
    print(f"Customer ID: {customer_id}")
    # TODO: Implement keyword planning pull logic.


def pull_semrush():
    """Pull data from SEMrush API."""
    api_key = os.getenv('SEMRUSH_API_KEY')
    
    print(f"Pulling SEMrush data...")
    # TODO: Implement SEMrush pull logic - blocked on API access


def main():
    parser = argparse.ArgumentParser(
        description='Pull input data from various sources'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='',
        help='Comma-separated list of datasets to pull (choices: ads_reports, keyword_planning, semrush)'
    )
    
    args = parser.parse_args()
    
    # Parse comma-separated datasets into a set
    requested_datasets = {ds.strip() for ds in args.datasets.split(',')}

    # Validate dataset choices
    invalid_datasets = requested_datasets - VALID_DATASETS
    if invalid_datasets:
        print(f"Error: Invalid dataset(s): {', '.join(invalid_datasets)}")
        print(f"Valid choices are: {', '.join(sorted(VALID_DATASETS))}")
        sys.exit(1)
    
    # Ensure we have necessary credentials set for the requested datasets
    validate_environment_variables(requested_datasets)

    yaml_path = os.getenv('GOOGLE_ADS_YAML_PATH')
    google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)
    customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID')

    if ADS_REPORTS in requested_datasets:
        pull_ads_reports(google_ads_client, customer_id)
        print(f"Successfully pulled ads_reports data")
    
    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning(google_ads_client, customer_id)
        print(f"Successfully pulled keyword_planning data")
    
    if SEMRUSH in requested_datasets:
        pull_semrush()
        print(f"Successfully pulled semrush data")

    print(f"All requested datasets pulled successfully")


if __name__ == '__main__':
    main()
