#!/usr/bin/env python3
"""
Script to pull input data from various sources (ads reports, keyword planning, SEMrush).
"""

import argparse
import os
import sys
from pathlib import Path

from google.ads.googleads.client import GoogleAdsClient

ADS_REPORTS = "ads_reports"
KEYWORD_PLANNING = "keyword_planning"
SEMRUSH = "semrush"
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

# Query to check if keywords exist in account - could provide "In Account" field for GKP output
KEYWORDS_IN_ACCOUNT_QUERY = """
    SELECT ad_group_criterion.keyword.text
    FROM keyword_view
    WHERE ad_group_criterion.type = KEYWORD
"""

# Query to get ad impression share for keywords in account - could provide "Ad impression share" field for GKP output
KEYWORD_IMPRESSION_SHARE_QUERY = """
    SELECT 
        ad_group_criterion.keyword.text,
        metrics.search_impression_share
    FROM keyword_view
    WHERE segments.date DURING LAST_30_DAYS
    AND ad_group_criterion.type = KEYWORD
"""


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
    # but since I have no actual data in my account, I can't actually validate anything.

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=RAW_INPUT_TO_MODELS_QUERY)

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


def pull_keyword_planning(
    google_ads_client: GoogleAdsClient, customer_id: str, keyword_planning_input_file: str
):
    """Pull keyword planning data from Google Ads using generate_keyword_historical_metrics.

    Args:
        google_ads_client: GoogleAdsClient instance
        customer_id: Google Ads customer ID
        keyword_planning_input_file: Path to file containing keywords (one per line)
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
    # TODO: Limit to 10k keywords. That's the per-request limit
    print(f"Loaded {len(keywords)} keywords")

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

    """
    The following fields are in our example data, but absent from the API response.
     - three_month_change - We could technically calculate this from the monthly volumes we pull by default.
     - yoy_change - I think we'd have to calculate this ourselves by pulling more historical data and comparing volumes per row.
     - organic_avg_position - Search console data, but blank in example files. Is this used?
     - organic_impression_share - Search console data, but blank in example files. Is this used?
     
     The following fields are not in our example data, but should be available using a different API.
     See above GAQL queries for theoretical implementation ideas.
     - in_account - Not available from this API (would need to check if keyword exists in account separately)
     - ad_impression_share - Not available from this API endpoint (this would come from actual campaign data)
     
     Segmentation and currency data are set as USD and "All" for now, since those are representative of the examples I have. This may need to change
    """

    # Output header row to match GKP CSV format
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

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

    print("\t".join(header_parts))

    result_data = []
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

        print("\t".join(str(p) for p in row_parts))


def pull_semrush():
    """Pull data from SEMrush API."""
    api_key = os.getenv("SEMRUSH_API_KEY")

    print(f"Pulling SEMrush data...")
    # TODO: Implement SEMrush pull logic - blocked on API access atm


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
        pull_keyword_planning(google_ads_client, customer_id, args.keyword_planning_input_file)
        print(f"Successfully pulled keyword_planning data")

    if SEMRUSH in requested_datasets:
        pull_semrush()
        print(f"Successfully pulled semrush data")

    print(f"All requested datasets pulled successfully")


if __name__ == "__main__":
    main()
