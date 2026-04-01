#!/usr/bin/env python3
"""
Script to pull input data from various sources (ads reports, keyword planning, SEMrush).
"""

import argparse
import os
import sys
from datetime import datetime

import requests
from dateutil.relativedelta import relativedelta

from utils.ads_reporting import *
from utils.report_row_generators import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COURSE_CONFIG

ADS_REPORTS = "ads_reports"
KEYWORD_PLANNING = "keyword_planning"
SEMRUSH = "semrush"
VALID_DATASETS = {ADS_REPORTS, KEYWORD_PLANNING, SEMRUSH}

SEMRUSH_HOST = "https://api.semrush.com/{}"
SEMRUSH_PHRASE_MAPPING = {
    "gen_ai": "generative ai course",
    "ml": "machine learning course",
    "sys_eng": "systems engineering course",
    "sys_think": "systems thinking course",
    "quant_comp": "quantum computing course",
    "dai": "deploying ai course",
}

def _gkp_month_header_sort_key(header: str) -> tuple[int, int]:
    month_str, year_str = header.replace("Searches: ", "", 1).rsplit(" ", 1)
    month_number = datetime.strptime(month_str, "%b").month
    return int(year_str), month_number


def validate_environment_variables(datasets: Iterable[str]) -> bool:
    """Validate that required environment variables are set for the given dataset."""
    required_vars = []
    for dataset in datasets:
        if dataset == SEMRUSH:
            required_vars.append("SEMRUSH_API_KEY")

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return True

def pull_ads_reports(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Pull all ads reports data from Google Ads for a given course."""

    # Default to last 12 months if not specified
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = COURSE_CONFIG[output_course]["min_date"]

    print(f"Pulling ads reports for course '{output_course}'...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Customer ID: {customer_id}")

    report_functions: list[ReportFunction] = [
        generate_search_keyword_report,
        generate_search_terms_report,
        generate_purchase_report,
        generate_hod_clicks_and_conversion_report,
        generate_age_clicks_and_conversion_report,
        generate_device_clicks_and_conversion_report,
        generate_loc_clicks_and_conversion_report
    ]

    for report in report_functions:
        report(google_ads_client, customer_id, output_course, start_date, end_date)

    print(f"Successfully generated all reports for {output_course}")


def generate_rows_from_gkp_response(
    response: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    monthly_headers = set()

    for result in response.results:
        metrics = result.keyword_metrics
        keyword = result.text
        avg_monthly_searches = metrics.avg_monthly_searches if metrics.avg_monthly_searches else ""
        competition = metrics.competition.name.capitalize() if metrics.competition else ""
        competition_index = metrics.competition_index if metrics.competition_index else ""
        low_bid = (
            Decimal(metrics.low_top_of_page_bid_micros) / 1_000_000
            if metrics.low_top_of_page_bid_micros
            else ""
        )
        high_bid = (
            Decimal(metrics.high_top_of_page_bid_micros) / 1_000_000
            if metrics.high_top_of_page_bid_micros
            else ""
        )

        row_parts = {
            "Keyword": keyword,
            "Avg. monthly searches": avg_monthly_searches,
            "Competition": competition,
            "Competition (indexed value)": competition_index,
            "Top of page bid (low range)": low_bid,
            "Top of page bid (high range)": high_bid,
        }

        # Add monthly search volumes (one column per month). If no monthly search volume is available, we don't zero pad it for consistency w/ the UI
        if metrics.monthly_search_volumes:
            for monthly_vol in metrics.monthly_search_volumes:
                header = f"Searches: {monthly_vol.month.name[:3].capitalize()} {monthly_vol.year}"
                monthly_headers.add(header)
                row_parts[header] = monthly_vol.monthly_searches if monthly_vol.monthly_searches else 0

        rows.append(row_parts)

    return rows, sorted(monthly_headers, key=_gkp_month_header_sort_key)


def pull_keyword_planning(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    keyword_planning_input_file: str,
    output_course: str,
) -> None:
    """Pull keyword planning data from Google Ads using generate_keyword_historical_metrics.

    Args:
        google_ads_client: GoogleAdsClient instance
        customer_id: Google Ads customer ID
        keyword_planning_input_file: Path to file containing keywords (one per line). Defaults to
            data/{output_course}/gkp/keywords_classified.csv when not provided.
        output_course: {gen_ai, ml, sys_eng, sys_think} - determines output location for the pulled data
    """
    # Read keywords from file
    if not keyword_planning_input_file:
        keyword_planning_input_file = f"data/{output_course}/gkp/keywords_classified.csv"

    keywords = []
    with open(keyword_planning_input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keyword = row.get("Keyword", "").strip()
            if keyword:
                keywords.append(keyword)

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

    keyword_plan_idea_service = google_ads_client.get_service("KeywordPlanIdeaService")

    # Set up request for historical metrics
    request = google_ads_client.get_type("GenerateKeywordHistoricalMetricsRequest")
    request.customer_id = customer_id
    request.keywords = keywords
    request.keyword_plan_network = google_ads_client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

    historical_metrics_options = google_ads_client.get_type("HistoricalMetricsOptions")
    current_date = datetime.now()
    # Start from 6 months before the course minimum date
    start_date = datetime.strptime(
        COURSE_CONFIG[output_course]["min_date"], "%Y-%m-%d"
    ) - relativedelta(months=6)
    # End date is last month (since current month data isn't complete)
    end_date = current_date - relativedelta(months=1)

    month_of_year_enum = google_ads_client.enums.MonthOfYearEnum
    historical_metrics_options.year_month_range.start.year = start_date.year
    historical_metrics_options.year_month_range.start.month = getattr(
        month_of_year_enum, start_date.strftime("%B").upper()
    )
    historical_metrics_options.year_month_range.end.year = end_date.year
    historical_metrics_options.year_month_range.end.month = getattr(
        month_of_year_enum, end_date.strftime("%B").upper()
    )
    request.historical_metrics_options = historical_metrics_options

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

    rows, date_header_parts = generate_rows_from_gkp_response(response)
    header_parts.extend(date_header_parts)

    # Create output directory and filename
    output_dir = Path(f"data/{output_course}/gkp")

    date_str = current_date.strftime("%Y-%m-%d")
    time_str = current_date.strftime("%H-%M-%S")
    # This is technically a TSV, but the rest of the code picks up csvs. We can change that later
    output_file = output_dir / f"Saved Keyword Stats {date_str} at {time_str}.csv"

    write_to_file(header_parts, rows, output_file)
    print(f"Keyword planning data written to: {output_file}")


def pull_semrush(output_course: str, num_keywords: int = 100) -> None:
    """Pull data from SEMrush API."""
    api_key = os.getenv("SEMRUSH_API_KEY")
    phrase = SEMRUSH_PHRASE_MAPPING[output_course]
    output_dir = Path(f"data/{output_course}/gkp")
    output_file = output_dir / f"semrush_new_kws.csv"

    print(f"Pulling SEMrush data...")
    query_params = {
        "type": "phrase_related",
        "key": api_key,
        "phrase": phrase,
        "database": "us",
        "export_columns": "Ph",
        # TODO: This is the limit returned by the API. It says it defaults to 10k in the docs but in practice it returns 100.
        # We currently only have 100k units, and this costs 40 units per keyword.
        "display_limit": num_keywords,
    }
    print(f"Executing SEMrush API request for phrase '{phrase}' with limit {num_keywords}...")
    response = requests.get(SEMRUSH_HOST, params=query_params)
    response.raise_for_status()
    rows = [{"Keyword": line} for line in response.text.splitlines()]
    # Skip the first line since it's just the header "Keyword"
    # This outputs a CSV already, so we don't need to do much to the response.
    write_to_file(["Keyword"], rows[1:], output_file)


def main() -> None:
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
        help="Location of a list of keywords to pull planning data for. Defaults to data/<output-course>/gkp/keywords_classified.csv when keyword_planning is selected. File should be a single keyword per line",
    )
    parser.add_argument(
        "--output-course",
        type=str,
        default="",
        choices=["gen_ai", "ml", "sys_eng", "sys_think", "quant_comp", "dai"],
        required=True,
        help="The course to pull data for, determines the location of the file outputs.",
    )
    parser.add_argument(
        "--google-ads-yaml",
        type=str,
        help="Path to Google Ads YAML configuration file",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        help="Google Ads customer ID",
    )
    parser.add_argument(
        "--num-keywords",
        type=int,
        help="Number of keywords to pull from SEMRush (default 100, max 10,000). Each keyword consumes 40 api units",
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
    yaml_path = args.google_ads_yaml

    if yaml_path:
        google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)
    else:
        google_ads_client = None

    customer_id = args.customer_id

    if ADS_REPORTS in requested_datasets:
        pull_ads_reports(google_ads_client, customer_id, args.output_course)
        print(f"Successfully pulled ads_reports data")

    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning(
            google_ads_client, customer_id, args.keyword_planning_input_file, args.output_course
        )
        print(f"Successfully pulled keyword_planning data")

    if SEMRUSH in requested_datasets:
        pull_semrush(args.output_course, args.num_keywords)
        print(f"Successfully pulled semrush data")

    print(f"All requested datasets pulled successfully")


if __name__ == "__main__":
    main()
