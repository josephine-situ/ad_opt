#!/usr/bin/env python3
"""
Script to pull input data from various sources (ads reports, keyword planning, SEMrush).
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

from google.ads.googleads.client import GoogleAdsClient

from utils.gaql_queries import RAW_INPUT_TO_MODELS_QUERY

ADS_REPORTS = "ads_reports"
KEYWORD_PLANNING = "keyword_planning"
SEMRUSH = "semrush"
VALID_DATASETS = {ADS_REPORTS, KEYWORD_PLANNING, SEMRUSH}

def write_to_file(header_parts, row_generator, output_file, delimiter="\t"):
    """Write data to a file with the given header and rows from a generator."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter)
        # Write header
        writer.writerow(header_parts)
        for row in row_generator:
            writer.writerow(row)

def validate_environment_variables(datasets):
    """Validate that required environment variables are set for the given dataset."""
    required_vars = []
    for dataset in datasets:
        if dataset in [ADS_REPORTS, KEYWORD_PLANNING]:
            required_vars.extend(["GOOGLE_ADS_CUSTOMER_ID", "GOOGLE_ADS_YAML_PATH"])

        if dataset == SEMRUSH:
            required_vars.append("SEMRUSH_API_KEY")

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return True


def pull_ads_reports(google_ads_client: GoogleAdsClient, customer_id: str):
    """Pull ads reports data from Google Ads."""

    print(f"Pulling ads reports data...")
    print(f"Customer ID: {customer_id}")
    # TODO: This is completely untested. I think the query is approximately correct,
    # but since I have no actual data in my account, I can only validate that the query runs without error.
    query = RAW_INPUT_TO_MODELS_QUERY.format(start_date=(datetime.now() - relativedelta(months=12)).strftime("%Y-%m-%d"), end_date=datetime.now().strftime("%Y-%m-%d"))

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    # TODO: Once we've validated this, we'll need to write it to a file instead of printing
    for batch in stream:
        for row in batch.results:
            date = row.segments.date
            search_term = row.search_term_view.search_term
            match_type = row.segments.search_term_match_type.name
            campaign_name = row.campaign.name
            clicks = row.metrics.clicks
            conv_value = row.metrics.conversions_value
            currency = row.customer.currency_code
            cost = (
                row.metrics.cost_micros / 1_000_000
            )  # Convert from micros https://groups.google.com/g/adwords-scripts/c/mSl5bxSkwec

            print(
                "\t".join(
                    [
                        date,
                        search_term,
                        match_type,
                        campaign_name,
                        clicks,
                        conv_value,
                        currency,
                        cost,
                    ]
                )
            )

def generate_rows_from_gkp_response(response):
    # Write data rows
    for result in response.results:
        metrics = result.keyword_metrics
        keyword = result.text
        avg_monthly_searches = metrics.avg_monthly_searches if metrics.avg_monthly_searches else ""
        competition = metrics.competition.name.capitalize() if metrics.competition else ""
        competition_index = metrics.competition_index if metrics.competition_index else ""
        low_bid = (
            metrics.low_top_of_page_bid_micros / 1_000_000
            if metrics.low_top_of_page_bid_micros
            else ""
        )
        high_bid = (
            metrics.high_top_of_page_bid_micros / 1_000_000
            if metrics.high_top_of_page_bid_micros
            else ""
        )

        # Build row with base columns
        row_parts = [
            keyword,
            avg_monthly_searches,
            competition,
            competition_index,
            low_bid,
            high_bid,
        ]

        # Add monthly search volumes (one column per month)
        if metrics.monthly_search_volumes:
            for monthly_vol in metrics.monthly_search_volumes:
                row_parts.append(
                    monthly_vol.monthly_searches if monthly_vol.monthly_searches else 0
                )
        else:
            row_parts.extend(
                ["" * 12]
            )  # Add empty columns if no monthly data is available to match examples more closely
        yield row_parts

def pull_keyword_planning(
    google_ads_client: GoogleAdsClient, customer_id: str, keyword_planning_input_file: str, output_course: str
):
    """Pull keyword planning data from Google Ads using generate_keyword_historical_metrics.

    Args:
        google_ads_client: GoogleAdsClient instance
        customer_id: Google Ads customer ID
        keyword_planning_input_file: Path to file containing keywords (one per line)
        output_course: {gen_ai, ml, sys_eng, sys_think} - determines output location for the pulled data
    """
    # Read keywords from file
    if not keyword_planning_input_file:
        print(
            f"Error: --keyword-planning-input-file is required when pulling keyword_planning dataset"
        )
        sys.exit(1)
    keywords = Path(keyword_planning_input_file).read_text().splitlines()

    print(f"Pulling keyword planning data...")
    print(f"Customer ID: {customer_id}")
    print(f"Keywords file: {keyword_planning_input_file}")
    num_keywords = len(keywords)
    if num_keywords > 10_000:
        print(
            f"Warning: {num_keywords} keywords provided, but the API only supports up to 10,000 keywords per request."
        )
        sys.exit(1)
    print(f"Loaded {num_keywords} keywords")

    # Initialize keyword plan idea service
    keyword_plan_idea_service = google_ads_client.get_service("KeywordPlanIdeaService")

    # Set up request for historical metrics
    request = google_ads_client.get_type("GenerateKeywordHistoricalMetricsRequest")
    request.customer_id = customer_id
    request.keywords = keywords
    # Not sure if this is actually required/desirable?
    request.keyword_plan_network = google_ads_client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

    print("Fetching historical metrics from Google Ads...")
    response = keyword_plan_idea_service.generate_keyword_historical_metrics(request=request)

    print(f"\nReceived {len(response.results)} keyword results:\n")

    header_parts = [
        "Keyword",
        "Avg. monthly searches",
        "Competition",
        "Competition (indexed value)",
        "Top of page bid (low range)",
        "Top of page bid (high range)",
    ]

    # Add monthly search volume headers for trailing 12 months. This matches the default timeframe we search for.
    # We can do this based on API output, but I'm not sure what's conditionally included and what isn't yet.
    current_date = datetime.now()
    for i in range(11, -1, -1):
        month_date = current_date - relativedelta(months=i)
        month_name = month_date.strftime("%b")
        year = month_date.year
        header_parts.append(f"Searches: {month_name} {year}")

    # Create output directory and filename
    output_dir = Path(f"data/{output_course}/gkp")
    
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    # I know this is a TSV, but the rest of the code picks up csvs. We can change that later
    output_file = output_dir / f"Saved Keyword Stats {date_str} at {time_str}.csv"

    write_to_file(header_parts, generate_rows_from_gkp_response(response), output_file)
    print(f"Keyword planning data written to: {output_file}")


def pull_semrush():
    """Pull data from SEMrush API."""
    api_key = os.getenv("SEMRUSH_API_KEY")

    print(f"Pulling SEMrush data...")
    # TODO: Implement SEMrush pull logic - blocked on API access atm


# TODO: Need to figure out if all courses are in one account (and if so, how they're organized) or if we have one course per like in the example I've got
def main():
    parser = argparse.ArgumentParser(description="Pull input data from various sources")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of datasets to pull (choices: ads_reports, keyword_planning, semrush)",
    )
    parser.add_argument(
        "--keyword-planning-input-file",
        type=str,
        default="",
        help="Location of a list of keywords to pull planning data for (required if keyword_planning dataset is selected). File should be a single keyword per line",
    )
    parser.add_argument(
        "--output-course",
        type=str,
        default="",
        choices=['gen_ai', 'ml', 'sys_eng', 'sys_think'],
        required=True,
        help="The course to pull data for, determines the location of the file outputs."
    )

    args = parser.parse_args()

    # Parse comma-separated datasets into a set
    requested_datasets = {ds.strip() for ds in args.datasets.split(",")}

    # Validate dataset choices
    invalid_datasets = requested_datasets - VALID_DATASETS
    if invalid_datasets:
        print(f"Error: Invalid dataset(s): {', '.join(invalid_datasets)}")
        print(f"Valid choices are: {', '.join(sorted(VALID_DATASETS))}")
        sys.exit(1)

    # Ensure we have necessary credentials set for the requested datasets
    validate_environment_variables(requested_datasets)

    yaml_path = os.getenv("GOOGLE_ADS_YAML_PATH")
    google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID")

    if ADS_REPORTS in requested_datasets:
        pull_ads_reports(google_ads_client, customer_id)
        print(f"Successfully pulled ads_reports data")

    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning(google_ads_client, customer_id, args.keyword_planning_input_file, args.output_course)
        print(f"Successfully pulled keyword_planning data")

    if SEMRUSH in requested_datasets:
        pull_semrush()
        print(f"Successfully pulled semrush data")

    print(f"All requested datasets pulled successfully")


if __name__ == "__main__":
    main()
