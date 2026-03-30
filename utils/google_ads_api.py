import csv
import re
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

from google.ads.googleads.client import GoogleAdsClient
from google.api_core import protobuf_helpers

from config import COURSE_CONFIG
from utils.gaql_queries import (
    GET_ENABLED_CAMPAIGNS_FOR_COURSE,
    GET_CRITERIA_FOR_CAMPAIGNS,
    GET_AGE_CRITERIA_FOR_CAMPAIGNS,
    GET_CAMPAIGN_BUDGETS_BY_NAMES,
    SELECT_AD_GROUPS_FOR_ENABLED_CAMPAIGNS,
)


def normalize_bid_adjustment(bid_adj: str | float | Decimal) -> Decimal:
    """Normalize bid adjustment to be within Google Ads limits (0.1 to 10.0, or exactly 0 for device criteria)."""
    bid_adj_decimal = Decimal(bid_adj)
    return Decimal(1.0) + bid_adj_decimal


def should_skip_campaign(
    campaign_name: str,
    check_course: str | None = None,
    check_region: str | None = None,
    check_match_type: str | None = None,
) -> bool:
    """Determine whether to skip a campaign based on its name and the specified region, match_type or course name.

    Campaign name format: "{course_title} - {region} - {match_type}"
    Uses regex to validate exact position of each component.
    """
    # Parse campaign name using regex
    # Pattern: "{any course title} - {region} - {match type}"
    # TODO: We may need to make this more flexible. The titles that exist at the moment are a bit less well-structured
    pattern = r"^(Course|Program) - (.+?) - (.+?) - (.+?)$"
    match = re.match(pattern, campaign_name)

    if not match:
        # Campaign name doesn't match expected format, skip it
        return True

    _, course_title, region, match_type = match.groups()

    # Check each component if specified
    if check_course and check_course not in course_title:
        return True
    if check_region and check_region != region:
        return True
    if check_match_type and check_match_type != match_type:
        return True

    return False

def get_enabled_campaigns_for_course(
    google_ads_service: Any, customer_id: str, output_course: str
) -> dict[str, int]:
    """Get all campaign IDs for a given course."""
    course_title = COURSE_CONFIG[output_course]["course_title_base"]

    query = GET_ENABLED_CAMPAIGNS_FOR_COURSE.format(course_title=course_title)

    stream = google_ads_service.search_stream(customer_id=customer_id, query=query)
    
    campaigns = {}
    for batch in stream:
        for row in batch.results:
            campaigns[row.campaign.name] = row.campaign.id
    return campaigns


def get_existing_campaign_criteria(
    google_ads_service: Any, customer_id: str, campaign_ids: Iterable[int | str]
) -> dict[str, dict[Any, int]]:
    """Get all existing campaign criteria for the specified campaigns."""
    campaign_id_list = "', '".join(
        [f"customers/{customer_id}/campaigns/{cid}" for cid in campaign_ids]
    )

    query = GET_CRITERIA_FOR_CAMPAIGNS.format(campaign_id_list=campaign_id_list)

    stream = google_ads_service.search_stream(customer_id=customer_id, query=query)

    # Organize criteria by type and campaign
    criteria: dict[str, dict[tuple, int]] = {
        "device": {},  # (campaign_id, device_type) -> criterion_id
        "schedule": {},  # (campaign_id, day, start_hour, end_hour) -> criterion_id
        "location": {},  # (campaign_id, geo_target) -> criterion_id
    }

    for batch in stream:
        for row in batch.results:
            campaign_path = row.campaign_criterion.campaign
            campaign_id = campaign_path.split("/")[-1]
            criterion_id = row.campaign_criterion.criterion_id
            criterion_type = row.campaign_criterion.type_

            if criterion_type.name == "DEVICE":
                device_type = row.campaign_criterion.device.type_
                criteria["device"][(campaign_id, device_type)] = criterion_id
            elif criterion_type.name == "AD_SCHEDULE":
                day = row.campaign_criterion.ad_schedule.day_of_week
                start_hour = row.campaign_criterion.ad_schedule.start_hour
                end_hour = row.campaign_criterion.ad_schedule.end_hour
                criteria["schedule"][(campaign_id, day, start_hour, end_hour)] = criterion_id
            elif criterion_type.name == "LOCATION":
                geo_target = row.campaign_criterion.location.geo_target_constant
                criteria["location"][(campaign_id, geo_target)] = criterion_id

    return criteria


def get_existing_ad_group_age_for_campaigns(
    google_ads_service: Any, customer_id: str, campaign_ids: Iterable[int | str]
) -> dict[tuple[str, Any], tuple[str, int]]:
    """Get existing ad group age criteria for the specified campaigns.

    Returns:
        dict: Map of (campaign_id, age_range_type) -> (ad_group_id, criterion_id)
    """

    query = GET_AGE_CRITERIA_FOR_CAMPAIGNS.format(campaign_ids=",".join(str(cid) for cid in campaign_ids))

    stream = google_ads_service.search_stream(customer_id=customer_id, query=query)

    # Organize criteria by (campaign_id, age_range_type) -> (ad_group_id, criterion_id)
    criteria = {}

    for batch in stream:
        for row in batch.results:
            campaign_id = str(row.campaign.id)
            ad_group_id = str(row.ad_group.id)
            criterion_id = row.ad_group_criterion.criterion_id
            age_range_type = row.ad_group_criterion.age_range.type_

            key = (campaign_id, age_range_type)
            criteria[key] = (ad_group_id, criterion_id)

    return criteria


def get_location_bid_adjustments(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    campaigns: dict[str, int],
    existing_criteria: dict[str, dict[Any, int]],
    adj_location_filepath: str | Path,
) -> list[Any]:
    """Push location bid adjustments to Google Ads."""
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    geo_target_service = google_ads_client.get_service("GeoTargetConstantService")
    operations = []

    # Cache for geo target lookups to avoid repeated API calls
    geo_target_cache: dict[str, str | None] = {}

    with open(adj_location_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row["Region"]
            location = row["Targeted location"]
            bid_adj_decimal = row.get("BidAdjustment", "")

            # Skip if no bid adjustment calculated
            if not bid_adj_decimal:
                continue

            bid_adjustment = normalize_bid_adjustment(bid_adj_decimal)

            # Look up geo target constant for this location
            if location not in geo_target_cache:
                try:
                    # Create request with location names
                    request = google_ads_client.get_type("SuggestGeoTargetConstantsRequest")
                    request.location_names.names.append(location)
                    request.locale = "en"

                    suggestions = geo_target_service.suggest_geo_target_constants(request=request)

                    if not suggestions.geo_target_constant_suggestions:
                        print(
                            f"Warning: Could not find geo target for location '{location}', skipping"
                        )
                        geo_target_cache[location] = None
                        continue

                    # Use the first (best) suggestion
                    geo_target_constant = suggestions.geo_target_constant_suggestions[
                        0
                    ].geo_target_constant.resource_name
                    geo_target_cache[location] = geo_target_constant

                except Exception as e:
                    print(f"Error looking up geo target for location '{location}': {e}")
                    geo_target_cache[location] = None
                    continue

            geo_target_constant = geo_target_cache[location]
            if geo_target_constant is None:
                continue

            # Apply to all campaigns for this region
            for campaign_name, campaign_id in campaigns.items():
                if should_skip_campaign(campaign_name, check_region=region):
                    continue

                # Check if criterion exists
                key = (str(campaign_id), geo_target_constant)
                if key in existing_criteria["location"]:
                    criterion_id = existing_criteria["location"][key]

                    # Update existing criterion
                    operation = google_ads_client.get_type("CampaignCriterionOperation")
                    criterion = operation.update
                    criterion.resource_name = campaign_criterion_service.campaign_criterion_path(
                        customer_id, campaign_id, criterion_id
                    )
                    criterion.bid_modifier = bid_adjustment

                    # Set field mask
                    field_mask = protobuf_helpers.field_mask(None, criterion._pb)
                    google_ads_client.copy_from(operation.update_mask, field_mask)

                    operations.append(operation)
                    print(
                        f"Prepared location adjustment: {campaign_name}, {location} -> {bid_adjustment:.2%}"
                    )
                else:
                    print(
                        f"Warning: Location criterion not found for {campaign_name}, {location} - skipping"
                    )

    return operations


def get_campaign_budget_info(
    google_ads_service: Any, customer_id: str, campaign_names_list: Iterable[str]
) -> dict[str, dict[str, Any]]:
    campaign_names = "', '".join(campaign_names_list)
    query = GET_CAMPAIGN_BUDGETS_BY_NAMES.format(campaign_names=campaign_names)

    stream = google_ads_service.search_stream(customer_id=customer_id, query=query)
    campaign_budget_resources = {}

    for batch in stream:
        for row in batch.results:
            campaign_name = row.campaign.name
            # TODO: Starting to look like this may want to be a dataclass
            campaign_budget_resources[campaign_name] = {
                "current_budget_amount": row.campaign_budget.amount_micros / 1_000_000,
                "budget_resource_id": row.campaign.campaign_budget,
            }
    return campaign_budget_resources


def get_ad_groups_for_enabled_campaigns(
    google_ads_service: Any, customer_id: str, campaign_names: Iterable[str]
) -> dict[str, int]:
    campaign_list = "', '".join(campaign_names)
    query = SELECT_AD_GROUPS_FOR_ENABLED_CAMPAIGNS.format(campaign_list=campaign_list)

    stream = google_ads_service.search_stream(customer_id=customer_id, query=query)
    campaign_to_ad_group = {}

    for batch in stream:
        for result in batch.results:
            campaign_name = result.campaign.name
            ad_group_id = result.ad_group.id
            # Take first ad group per campaign
            if campaign_name not in campaign_to_ad_group:
                campaign_to_ad_group[campaign_name] = ad_group_id
    return campaign_to_ad_group
