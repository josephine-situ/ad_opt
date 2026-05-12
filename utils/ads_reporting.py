import csv
from pathlib import Path
from typing import Protocol, Iterable

from config import COURSE_CONFIG
from utils.gaql_queries import (
    SEARCH_KEYWORD_REPORT_QUERY,
    PURCHASE_REPORT_QUERY,
    HOD_CLICKS_REPORT_QUERY,
    AGE_CLICKS_REPORT_QUERY,
    DEVICE_CLICKS_REPORT_QUERY,
    LOC_CLICKS_REPORT_QUERY,
    HOD_CONVERSIONS_REPORT_QUERY,
    AGE_CONVERSIONS_REPORT_QUERY,
    DEVICE_CONVERSIONS_REPORT_QUERY,
    LOC_CONVERSIONS_REPORT_QUERY,
    SEARCH_TERM_REPORT_QUERY,
)
from utils.google_ads_api import get_location_resource_names_for_countries
from utils.metrics import GoogleAdsMetricsClient, google_ads_metrics_client
from utils.report_row_generators import *

"""
This module contains functions to generate Google Ads reports which are used as input to the optimization model.
These functions are "standard" in that they follow a common pattern of deriving their output path, 
performing one or more queries via Google Ads API, and persisting the results to a file.
They all adhere to a common interface defined by the ReportFunction protocol.
"""


def write_to_file(
    header_parts: list[str],
    row_generator: Iterable[dict[str, Any]],
    output_file: Path,
    delimiter: str = "\t",
    restval: str = "0",
) -> None:
    """Write data to a file with the given header and rows from a generator."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header_parts, delimiter=delimiter, restval=restval)
        # Write header
        writer.writeheader()
        for row in row_generator:
            writer.writerow(row)

class ReportFunction(Protocol):
    """
    Protocol defining the common interface for all ads report generator functions.
    Each function is responsible for deriving it output path,
    performing any queries necessary and persisting the results to a file
    """

    def __call__(
        self,
        google_ads_client: GoogleAdsClient,
        customer_id: str,
        output_course: str,
        start_date: str,
        end_date: str,
    ) -> None: ...

def generate_search_keyword_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Generate 'Search keyword - raw input to models' report."""
    output_path = Path(f"data/{output_course}/reports/Search keyword - raw input to models.csv")

    query = SEARCH_KEYWORD_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    header_parts = [
        "Day",
        "Search keyword",
        "Search keyword match type",
        "Campaign",
        "Clicks",
        "Conv. value",
        "Currency code",
        "Cost",
    ]
    write_to_file(header_parts, generate_search_keyword_rows(stream), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 1)
    print(f"Generated: {output_path}")


def generate_search_terms_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Generate 'Search keyword - search terms' report."""
    output_path = Path(f"data/{output_course}/reports/Search keyword - search terms.csv")
    query = SEARCH_TERM_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        conversion_action_list="', '".join(COURSE_CONFIG[output_course]["conversion_actions"]),
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)
    header_parts = [
        "Search keyword",
        "Search keyword match type",
        "Search term",
        "Conversion action",
        "Conversions",
    ]
    write_to_file(header_parts, generate_search_terms_row(stream), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 1)
    print(f"Generated: {output_path}")


def generate_purchase_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Generate 'Purchase report' with conversion data."""
    output_path = Path(f"data/{output_course}/reports/Purchase report.csv")

    query = PURCHASE_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    header_parts = ["Campaign", "Conversion action", "All conv."]
    write_to_file(header_parts, generate_purchase_report_rows(stream), output_path, delimiter=",")
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 1)
    print(f"Generated: {output_path}")


def generate_hod_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
    output_suffix: str = "",
) -> None:
    """Generate hour-of-day clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/hod_clicks{output_suffix}.csv")

    query = HOD_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    header_parts = ["Campaign", "Hour of the day", "Clicks"]
    write_to_file(header_parts, generate_hod_clicks_rows(stream), output_path, delimiter=",")
    print(f"Generated: {output_path}")

    # Generate conversions report
    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/hod_conv{output_suffix}.csv")
    query_conv = HOD_CONVERSIONS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
    )
    stream_conv = ads_service.search_stream(customer_id=customer_id, query=query_conv)
    header_parts_conv = ["Campaign", "Conversion action", "Hour of the day", "All conv."]
    write_to_file(
        header_parts_conv,
        generate_hod_conversions_rows(stream_conv),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 2)
    print(f"Generated: {output_path_conv}")


def generate_age_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Generate age demographics clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/age_clicks.csv")

    query = AGE_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    header_parts = ["Campaign", "Age", "Clicks"]
    write_to_file(header_parts, generate_age_clicks_rows(stream), output_path, delimiter=",")
    print(f"Generated: {output_path}")

    # Generate conversions report
    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/age_conv.csv")
    query_conv = AGE_CONVERSIONS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
    )
    stream_conv = ads_service.search_stream(customer_id=customer_id, query=query_conv)
    header_parts_conv = ["Campaign", "Conversion action", "Age", "All conv."]
    write_to_file(
        header_parts_conv,
        generate_age_conversions_rows(stream_conv),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 2)
    print(f"Generated: {output_path_conv}")


def generate_device_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Generate device clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/device_clicks.csv")

    query = DEVICE_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    header_parts = ["Campaign", "Device", "Clicks"]
    write_to_file(header_parts, generate_device_clicks_rows(stream), output_path, delimiter=",")
    print(f"Generated: {output_path}")

    # Generate conversions report
    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/device_conv.csv")
    query_conv = DEVICE_CONVERSIONS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        purchase_action_list="', '".join(COURSE_CONFIG[output_course]["purchase_actions"]),
    )
    stream_conv = ads_service.search_stream(customer_id=customer_id, query=query_conv)
    header_parts_conv = ["Campaign", "Conversion action", "Device", "All conv."]
    write_to_file(
        header_parts_conv,
        generate_device_conversions_rows(stream_conv),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 2)
    print(f"Generated: {output_path_conv}")


def generate_loc_clicks_and_conversion_report(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    start_date: str,
    end_date: str,
) -> None:
    """Generate location clicks report for bid adjustments."""
    output_path = Path(f"data/{output_course}/reports/bid_adj/loc_clicks.csv")

    regions = COURSE_CONFIG[output_course]["regions"]
    all_locations = []
    for locs in regions.values():
        all_locations.extend(locs)
    location_resource_names = get_location_resource_names_for_countries(google_ads_client, all_locations)
    country_criterion_ids = ", ".join(
        name.split("/")[-1] for name in location_resource_names.values()
    )

    query = LOC_CLICKS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        country_criterion_ids=country_criterion_ids,
    )

    ads_service = google_ads_client.get_service("GoogleAdsService")
    stream = ads_service.search_stream(customer_id=customer_id, query=query)

    header_parts = ["Campaign", "Targeted location", "Clicks"]
    write_to_file(
        header_parts,
        generate_loc_clicks_rows(stream, google_ads_client, customer_id),
        output_path,
        delimiter=",",
    )
    print(f"Generated: {output_path}")

    # Generate conversions report
    purchase_actions = COURSE_CONFIG[output_course]["purchase_actions"]
    output_path_conv = Path(f"data/{output_course}/reports/bid_adj/loc_conv.csv")
    query_conv = LOC_CONVERSIONS_REPORT_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        purchase_action_list="', '".join(purchase_actions),
        country_criterion_ids=country_criterion_ids,
    )
    stream_conv = ads_service.search_stream(customer_id=customer_id, query=query_conv)
    header_parts_conv = ["Campaign", "Conversion action", "Targeted location", "All conv."]
    write_to_file(
        header_parts_conv,
        generate_loc_conversions_rows(stream_conv, google_ads_client, customer_id),
        output_path_conv,
        delimiter=",",
    )
    google_ads_metrics_client.track_google_ads_operation_count('search_stream', 2)
    print(f"Generated: {output_path_conv}")