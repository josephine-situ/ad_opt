from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterator

from google.ads.googleads.client import GoogleAdsClient

from utils.bid_adjustments import AGE_ENUM_TO_RANGE, DEVICE_ENUM_TO_NAME
from utils.google_ads_api import build_location_cache, get_from_location_cache


"""
This file contains "standard" row generator functions. 
They take in a stream of Google Ads API results and yield dictionaries representing rows for the output CSVs. 
Each function corresponds to a specific report type and is responsible for transforming the raw API data into 
the desired output format, including any necessary transformations or lookups. 
They're designed to be mostly stateless mapping functions, but that may not always be possible while being adequately performant
"""

def generate_search_keyword_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for search keyword report."""
    for batch in stream:
        for row in batch.results:
            first_page_bid = ""
            if row.ad_group_criterion.position_estimates.first_page_cpc_micros:
                first_page_bid = f"{Decimal(row.ad_group_criterion.position_estimates.first_page_cpc_micros) / 1_000_000:.2f}"
            
            yield {
                "Day": row.segments.date,
                "Search keyword": row.ad_group_criterion.keyword.text,
                "Search keyword match type": row.ad_group_criterion.keyword.match_type.name.replace(
                    "_", " "
                ).title(),
                "Campaign": row.campaign.name,
                "Clicks": row.metrics.clicks,
                "Conv. value": f"{row.metrics.all_conversions_value:.2f}",
                "Currency code": row.customer.currency_code,
                "Cost": f"{Decimal(row.metrics.cost_micros) / 1_000_000:.2f}",
                "First page CPC": first_page_bid,
            }


def generate_purchase_report_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for purchase report."""
    for batch in stream:
        for row in batch.results:
            yield {
                "Campaign": row.campaign.name,
                "Conversion action": row.segments.conversion_action_name,
                "All conv.": f"{row.metrics.all_conversions:.2f}",
            }


def generate_hod_clicks_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for hour-of-day clicks report."""
    for batch in stream:
        for row in batch.results:
            yield {
                "Campaign": row.campaign.name,
                "Hour of the day": row.segments.hour,
                "Clicks": row.metrics.clicks,
            }


def generate_age_clicks_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for age clicks report."""
    # Aggregate by campaign and age range (since age_range_view segments by ad group)
    aggregated: defaultdict[tuple, int] = defaultdict(int)

    for batch in stream:
        for row in batch.results:
            age_type = row.ad_group_criterion.age_range.type_
            age_display = AGE_ENUM_TO_RANGE.get(age_type, "")
            key = (row.campaign.name, age_display)
            aggregated[key] += row.metrics.clicks

    for (campaign, age), clicks in sorted(aggregated.items()):
        yield {"Campaign": campaign, "Age": age, "Clicks": clicks}


def generate_device_clicks_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for device clicks report."""
    # Map device enum values to display names

    for batch in stream:
        for row in batch.results:
            device_type = row.segments.device
            device_display = DEVICE_ENUM_TO_NAME.get(device_type, "")

            yield {
                "Campaign": row.campaign.name,
                "Device": device_display,
                "Clicks": row.metrics.clicks,
            }


def generate_loc_clicks_rows(
    stream: Any,
    google_ads_client: GoogleAdsClient,
    customer_id: str,
) -> Iterator[dict[str, Any]]:
    """Generate rows for location clicks report."""
    # First pass: collect all rows and criterion IDs
    rows_data = []
    criterion_ids = set()

    for batch in stream:
        for row in batch.results:
            rows_data.append(row)
            if row.geographic_view.country_criterion_id:
                criterion_ids.add(row.geographic_view.country_criterion_id)

    # Build location cache with a single bulk query (only for new IDs)
    build_location_cache(google_ads_client, customer_id, criterion_ids)

    # Second pass: generate output rows with location names
    for row in rows_data:
        # Get human-readable location name
        location_name = get_from_location_cache(row.geographic_view.country_criterion_id)

        yield {
            "Campaign": row.campaign.name,
            "Targeted location": location_name,
            "Clicks": row.metrics.clicks,
        }


def generate_hod_conversions_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for hour-of-day conversions report."""
    for batch in stream:
        for row in batch.results:
            yield {
                "Campaign": row.campaign.name,
                "Conversion action": row.segments.conversion_action_name,
                "Hour of the day": row.segments.hour,
                "All conv.": f"{row.metrics.all_conversions:.2f}",
            }


def generate_age_conversions_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for age demographics conversions report."""
    # Aggregate by campaign, conversion action, and age range
    from collections import defaultdict

    aggregated: defaultdict[tuple, float] = defaultdict(float)

    for batch in stream:
        for row in batch.results:
            age_type = row.ad_group_criterion.age_range.type
            age_display = AGE_ENUM_TO_RANGE.get(age_type, "")
            key = (row.campaign.name, row.segments.conversion_action_name, age_display)
            aggregated[key] += row.metrics.all_conversions

    for (campaign, conversion_action, age), conversions in sorted(aggregated.items()):
        yield {
            "Campaign": campaign,
            "Conversion action": conversion_action,
            "Age": age,
            "All conv.": f"{conversions:.2f}",
        }


def generate_device_conversions_rows(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for device conversions report."""
    for batch in stream:
        for row in batch.results:
            device_type = row.segments.device
            device_display = DEVICE_ENUM_TO_NAME.get(device_type, "")

            yield {
                "Campaign": row.campaign.name,
                "Conversion action": row.segments.conversion_action_name,
                "Device": device_display,
                "All conv.": f"{row.metrics.all_conversions:.2f}",
            }


def generate_loc_conversions_rows(
    stream: Any,
    google_ads_client: GoogleAdsClient,
    customer_id: str,
) -> Iterator[dict[str, Any]]:
    """Generate rows for location conversions report."""
    # First pass: collect all rows and criterion IDs
    rows_data = []
    criterion_ids = set()

    for batch in stream:
        for row in batch.results:
            rows_data.append(row)
            if row.geographic_view.country_criterion_id:
                criterion_ids.add(row.geographic_view.country_criterion_id)

    # Build location cache with a single bulk query (only for new IDs)
    build_location_cache(google_ads_client, customer_id, criterion_ids)

    # Second pass: generate output rows with location names
    for row in rows_data:
        # Get human-readable location name
        location_name = get_from_location_cache(row.geographic_view.country_criterion_id)

        yield {
            "Campaign": row.campaign.name,
            "Conversion action": row.segments.conversion_action_name,
            "Targeted location": location_name,
            "All conv.": f"{row.metrics.all_conversions:.2f}",
        }


def generate_search_terms_row(stream: Any) -> Iterator[dict[str, Any]]:
    """Generate rows for search terms report."""
    for batch in stream:
        for row in batch.results:
            yield {
                "Search keyword": row.segments.keyword.info.text,
                "Search keyword match type": row.segments.keyword.info.match_type.name.replace(
                    "_", " "
                ).title(),
                "Search term": row.search_term_view.search_term,
                "Conversion action": row.segments.conversion_action_name,
                "Conversions": f"{row.metrics.all_conversions:.2f}",
            }

