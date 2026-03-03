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

from utils.bid_adjustments import AGE_RANGE_MAP, DEVICE_MAP

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.gaql_queries import (
    SEARCH_KEYWORD_REPORT_QUERY,
    PURCHASE_REPORT_QUERY,
    LOCATION_REPORT_QUERY,
    HOD_CLICKS_REPORT_QUERY,
    AGE_CLICKS_REPORT_QUERY,
    DEVICE_CLICKS_REPORT_QUERY,
    LOC_CLICKS_REPORT_QUERY,
)
from config import COURSE_CONFIG

ADS_REPORTS = "ads_reports"
KEYWORD_PLANNING = "keyword_planning"
SEMRUSH = "semrush"
VALID_DATASETS = {ADS_REPORTS, KEYWORD_PLANNING, SEMRUSH}


def write_to_file(header_parts, row_generator, output_file, delimiter="\t", restval="0"):
    """Write data to a file with the given header and rows from a generator."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_parts, delimiter=delimiter, restval=restval)
        # Write header
        writer.writeheader()
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


def _generate_search_keyword_rows(stream):
    """Generate rows for search keyword report."""
    for batch in stream:
        for row in batch.results:
            yield {
                'Day': row.segments.date,
                'Search keyword': row.search_term_view.search_term,
                'Search keyword match type': row.segments.search_term_match_type.name.replace('_', ' ').title(),
                'Campaign': row.campaign.name,
                'Clicks': row.metrics.clicks,
                'Conv. value': f"{row.metrics.conversions_value:.2f}",
                'Currency code': row.customer.currency_code,
                'Cost': f"{row.metrics.cost_micros / 1_000_000:.2f}"
            }


def _generate_purchase_report_rows(stream):
    """Generate rows for purchase report."""
    for batch in stream:
        for row in batch.results:
            yield {
                'Campaign': row.campaign.name,
                'Conversion action': row.segments.conversion_action_name,
                'Conversions': f"{row.metrics.conversions:.2f}"
            }


def _generate_location_report_rows(stream):
    """Generate rows for location report."""
    for batch in stream:
        for row in batch.results:
            clicks = row.metrics.clicks
            conversions = row.metrics.conversions
            cost = row.metrics.cost_micros / 1_000_000
            conv_rate = (conversions / clicks * 100) if clicks > 0 else 0
            cost_per_conv = (cost / conversions) if conversions > 0 else 0
            
            yield {
                'Location': row.geographic_view.location_type,
                'Campaign': row.campaign.name,
                'Bid adj.': '--',
                'Clicks': clicks,
                'Currency code': row.customer.currency_code,
                'Cost': f"{cost:.2f}",
                'Conv. rate': f"{conv_rate:.2f}%",
                'Conversions': f"{conversions:.2f}",
                'Cost / conv.': f"{cost_per_conv:.2f}"
            }


def _generate_hod_clicks_rows(stream):
    """Generate rows for hour-of-day clicks report."""
    for batch in stream:
        for row in batch.results:
            yield {
                'Campaign': row.campaign.name,
                'Hour of the day': row.segments.hour,
                'Clicks': row.metrics.clicks
            }


def _generate_age_clicks_rows(stream):
    """Generate rows for age clicks report."""
    # Map age range enum values to display names
    
    for batch in stream:
        for row in batch.results:
            age_type = row.ad_group_criterion.age_range.type_.name
            age_display = AGE_RANGE_MAP.get(age_type, "")
            
            yield {
                'Campaign': row.campaign.name,
                'Age': age_display,
                'Clicks': row.metrics.clicks
            }


def _generate_device_clicks_rows(stream):
    """Generate rows for device clicks report."""
    # Map device enum values to display names
    
    for batch in stream:
        for row in batch.results:
            device_type = row.segments.device.name
            device_display = DEVICE_MAP.get(device_type, "")
            
            yield {
                'Campaign': row.campaign.name,
                'Device': device_display,
                'Clicks': row.metrics.clicks
            }


def _generate_loc_clicks_rows(stream):
    """Generate rows for location clicks report."""
    for batch in stream:
        for row in batch.results:
            yield {
                'Campaign': row.campaign.name,
                'Targeted location': row.geographic_view.location_type,
                'Clicks': row.metrics.clicks
            }


def generate_search_keyword_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate 'Search keyword - raw input to models' report."""
    output_path = Path(f"data/{output_course}/reports/Search keyword - raw input to models.csv")
    
    query = SEARCH_KEYWORD_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Day', 'Search keyword', 'Search keyword match type', 'Campaign', 'Clicks', 'Conv. value', 'Currency code', 'Cost']
    write_to_file(header_parts, _generate_search_keyword_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def generate_purchase_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate 'Purchase report' with conversion data."""
    output_path = Path(f"data/{output_course}/reports/Purchase report.csv")
    
    query = PURCHASE_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Campaign', 'Conversion action', 'Conversions']
    write_to_file(header_parts, _generate_purchase_report_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def generate_location_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate 'Location report' with geographic performance data."""
    output_path = Path(f"data/{output_course}/reports/Location report.csv")
    
    query = LOCATION_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Location', 'Campaign', 'Bid adj.', 'Clicks', 'Currency code', 'Cost', 'Conv. rate', 'Conversions', 'Cost / conv.']
    write_to_file(header_parts, _generate_location_report_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def generate_hod_clicks_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate hour-of-day clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/hod_clicks.csv")
    
    query = HOD_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Campaign', 'Hour of the day', 'Clicks']
    write_to_file(header_parts, _generate_hod_clicks_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def generate_age_clicks_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate age demographics clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/age_clicks.csv")
    
    query = AGE_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Campaign', 'Age', 'Clicks']
    write_to_file(header_parts, _generate_age_clicks_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def generate_device_clicks_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate device clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/device_clicks.csv")
    
    query = DEVICE_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Campaign', 'Device', 'Clicks']
    write_to_file(header_parts, _generate_device_clicks_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def generate_loc_clicks_report(ads_service, customer_id, output_course, start_date, end_date):
    """Generate location clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/loc_clicks.csv")
    
    query = LOC_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        course_title=COURSE_CONFIG[output_course]['course_title_base']
    )

    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    
    header_parts = ['Campaign', 'Targeted location', 'Clicks']
    write_to_file(header_parts, _generate_loc_clicks_rows(stream), output_path, delimiter=',')
    print(f"Generated: {output_path}")


def pull_ads_reports(google_ads_client, customer_id, output_course, start_date=None, end_date=None):
    """Pull all ads reports data from Google Ads for a given course."""
    
    # Default to last 12 months if not specified
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.now() - relativedelta(months=12)).strftime("%Y-%m-%d")
    
    print(f"Pulling ads reports for course '{output_course}'...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Customer ID: {customer_id}")
    ads_service = google_ads_client.get_service("GoogleAdsService")

    # Generate all reports
    generate_search_keyword_report(ads_service, customer_id, output_course, start_date, end_date)
    generate_purchase_report(ads_service, customer_id, output_course, start_date, end_date)
    generate_location_report(ads_service, customer_id, output_course, start_date, end_date)
    generate_hod_clicks_report(ads_service, customer_id, output_course, start_date, end_date)
    generate_age_clicks_report(ads_service, customer_id, output_course, start_date, end_date)
    generate_device_clicks_report(ads_service, customer_id, output_course, start_date, end_date)
    generate_loc_clicks_report(ads_service, customer_id, output_course, start_date, end_date)
    
    print(f"Successfully generated all reports for {output_course}")


def generate_rows_from_gkp_response(response, date_headers):
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
                row_parts[
                    f"Searches: {monthly_vol.month.name[:3].capitalize()} {monthly_vol.year}"
                ] = (monthly_vol.monthly_searches if monthly_vol.monthly_searches else 0)
        else:
            for header in date_headers:
                row_parts[header] = ""

        yield row_parts


def pull_keyword_planning(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    keyword_planning_input_file: str,
    output_course: str,
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

    keyword_plan_idea_service = google_ads_client.get_service("KeywordPlanIdeaService")

    # Set up request for historical metrics
    request = google_ads_client.get_type("GenerateKeywordHistoricalMetricsRequest")
    request.customer_id = customer_id
    request.keywords = keywords
    # Not sure if this is actually required/desirable?
    request.keyword_plan_network = google_ads_client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH

    # Set historical metrics options to get trailing 12 months
    historical_metrics_options = google_ads_client.get_type("HistoricalMetricsOptions")
    current_date = datetime.now()
    start_date = current_date - relativedelta(months=12)
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

    # This is a bit squirrely. We observed that the data coming out of the API is pretty sparse
    # We can't rely on it having an entry for each month in the range we've asked it for, so instead we construct all
    # possible headers and rely on the DictWriter to fill in missing values with 0s as a restval.
    date_header_parts = []
    for i in range(12, 0, -1):
        month_date = current_date - relativedelta(months=i)
        month_name = month_date.strftime("%b")
        year = month_date.year
        date_header_parts.append(f"Searches: {month_name} {year}")

    header_parts.extend(date_header_parts)

    # Create output directory and filename
    output_dir = Path(f"data/{output_course}/gkp")

    date_str = current_date.strftime("%Y-%m-%d")
    time_str = current_date.strftime("%H-%M-%S")
    # This is technically a TSV, but the rest of the code picks up csvs. We can change that later
    output_file = output_dir / f"Saved Keyword Stats {date_str} at {time_str}.csv"

    write_to_file(
        header_parts, generate_rows_from_gkp_response(response, date_header_parts), output_file
    )
    print(f"Keyword planning data written to: {output_file}")


def pull_semrush():
    """Pull data from SEMrush API."""
    api_key = os.getenv("SEMRUSH_API_KEY")

    print(f"Pulling SEMrush data...")
    raise NotImplementedError
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
        choices=["gen_ai", "ml", "sys_eng", "sys_think"],
        required=True,
        help="The course to pull data for, determines the location of the file outputs.",
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
        pull_ads_reports(google_ads_client, customer_id, args.output_course)
        print(f"Successfully pulled ads_reports data")

    if KEYWORD_PLANNING in requested_datasets:
        pull_keyword_planning(
            google_ads_client, customer_id, args.keyword_planning_input_file, args.output_course
        )
        print(f"Successfully pulled keyword_planning data")

    if SEMRUSH in requested_datasets:
        pull_semrush()
        print(f"Successfully pulled semrush data")

    print(f"All requested datasets pulled successfully")


if __name__ == "__main__":
    main()
