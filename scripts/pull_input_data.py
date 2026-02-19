#!/usr/bin/env python3
"""
Script to pull input data from various sources (ads reports, keyword planning, SEMrush).
"""

import argparse
import os
import sys

ADS_REPORTS = 'ads_reports'
KEYWORD_PLANNING = 'keyword_planning'
SEMRUSH = 'semrush'
VALID_DATASETS = {ADS_REPORTS, KEYWORD_PLANNING, SEMRUSH}

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


def pull_ads_reports():
    """Pull ads reports data from Google Ads."""
    customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID')
    yaml_path = os.getenv('GOOGLE_ADS_YAML_PATH')
    
    print(f"Pulling ads reports data...")
    print(f"Customer ID: {customer_id}")
    print(f"Config file: {yaml_path}")
    # TODO: Implement ads reports pull logic - slightly blocked on approach, but can start with using sheets as input.
    

def pull_keyword_planning():
    """Pull keyword planning data from Google Ads."""
    customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID')
    yaml_path = os.getenv('GOOGLE_ADS_YAML_PATH')
    
    print(f"Pulling keyword planning data...")
    print(f"Customer ID: {customer_id}")
    print(f"Config file: {yaml_path}")
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

    if ADS_REPORTS in requested_datasets:
        pull_ads_reports()
        print(f"Successfully pulled ads_reports data")
    
    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning()
        print(f"Successfully pulled keyword_planning data")
    
    if SEMRUSH in requested_datasets:
        pull_semrush()
        print(f"Successfully pulled semrush data")

    print(f"All requested datasets pulled successfully")


if __name__ == '__main__':
    main()
