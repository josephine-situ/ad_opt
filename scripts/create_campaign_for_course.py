#!/usr/bin/env python3
"""
Script to create Google Ads campaigns and ad groups for a given course.
Creates campaigns following the structure: one campaign per (Course, Region, Match Type) tuple.
Each campaign contains exactly one ad group.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.v23.services.types.campaign_budget_service import CampaignBudgetOperation
from google.ads.googleads.v23.services.types.campaign_service import CampaignOperation
from google.ads.googleads.v23.services.types.ad_group_service import AdGroupOperation
from google.ads.googleads.v23.services.types.campaign_criterion_service import CampaignCriterionOperation

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COURSE_CONFIG


# We may need to change this in the future, but for now this encapsulates the related resources we create for a campaign
# We'll need to probably add the notion of run, but for now, this will scaffold out campaigns acceptably.
@dataclass
class CampaignSpec:
    """Specification for a campaign and its associated resources."""

    campaign_name: str
    ad_group_name: str
    budget_name: str
    default_budget: int
    region_label: str
    countries: list[str]
    budget_resource_name: Optional[str] = None
    campaign_resource_name: Optional[str] = None
    ad_group_resource_name: Optional[str] = None


def find_spec_by_name(specs: list[CampaignSpec], name: str, field: str) -> Optional[CampaignSpec]:
    """
    Find a CampaignSpec by searching for a name in the specified field.

    Args:
        specs: List of CampaignSpec objects to search
        name: The name to search for
        field: The field to search in ('budget_name', 'campaign_name', or 'ad_group_name')

    Returns:
        The matching CampaignSpec or None if not found
    """
    for spec in specs:
        if getattr(spec, field) == name:
            return spec
    return None


def create_campaign_budget_operation(
    google_ads_client: GoogleAdsClient, budget_name: str, daily_budget_micros: int
) -> CampaignBudgetOperation:
    """Create a campaign budget operation."""
    operation = google_ads_client.get_type("CampaignBudgetOperation")
    campaign_budget = operation.create
    campaign_budget.name = budget_name
    campaign_budget.amount_micros = daily_budget_micros
    campaign_budget.delivery_method = google_ads_client.enums.BudgetDeliveryMethodEnum.STANDARD
    # By default, budgets are created as "shared", which doesn't appear to be how the existing ones are set up.
    # We'll match that pattern for the time being, but it's easy enough to change in the future.
    # See https://developers.google.com/google-ads/api/reference/rpc/v23/CampaignBudget#explicitly_shared for more info
    campaign_budget.explicitly_shared = False
    return operation


def create_campaign_operation(
    google_ads_client: GoogleAdsClient, campaign_name: str, budget_resource_name: str
) -> CampaignOperation:
    """Create a Search campaign operation."""
    operation = google_ads_client.get_type("CampaignOperation")
    campaign = operation.create
    campaign.name = campaign_name
    campaign.advertising_channel_type = google_ads_client.enums.AdvertisingChannelTypeEnum.SEARCH
    campaign.status = google_ads_client.enums.CampaignStatusEnum.PAUSED
    campaign.campaign_budget = budget_resource_name

    # Set manual CPC bidding strategy
    campaign.manual_cpc.enhanced_cpc_enabled = False

    # Set network settings
    # TODO: Not quite sure if this what we want, but it's good enough for now.
    campaign.network_settings.target_google_search = True
    campaign.network_settings.target_search_network = True
    campaign.network_settings.target_content_network = False
    campaign.network_settings.target_partner_search_network = False
    campaign.contains_eu_political_advertising = (
        google_ads_client.enums.EuPoliticalAdvertisingStatusEnum.DOES_NOT_CONTAIN_EU_POLITICAL_ADVERTISING
    )
    return operation


def create_ad_group_operation(
    google_ads_client: GoogleAdsClient, ad_group_name: str, campaign_resource_name: str
) -> AdGroupOperation:
    """Create an ad group operation."""
    operation = google_ads_client.get_type("AdGroupOperation")
    ad_group = operation.create
    ad_group.name = ad_group_name
    ad_group.campaign = campaign_resource_name
    ad_group.status = google_ads_client.enums.AdGroupStatusEnum.ENABLED
    ad_group.type_ = google_ads_client.enums.AdGroupTypeEnum.SEARCH_STANDARD
    return operation


def create_ad_schedule_operations(
    google_ads_client: GoogleAdsClient, campaign_resource_name: str
) -> list[CampaignCriterionOperation]:
    """
    Create ad schedule operations for a campaign.
    Creates 6 four-hour windows starting at 00:00 (covering the full day).
    """
    operations = []
    
    # Create 6 four-hour windows: 0-4, 4-8, 8-12, 12-16, 16-20, 20-24
    for window in range(6):
        start_hour = window * 4
        end_hour = (window + 1) * 4

        # Create ad schedule for each day of the week
        days_of_week = [
            google_ads_client.enums.DayOfWeekEnum.MONDAY,
            google_ads_client.enums.DayOfWeekEnum.TUESDAY,
            google_ads_client.enums.DayOfWeekEnum.WEDNESDAY,
            google_ads_client.enums.DayOfWeekEnum.THURSDAY,
            google_ads_client.enums.DayOfWeekEnum.FRIDAY,
            google_ads_client.enums.DayOfWeekEnum.SATURDAY,
            google_ads_client.enums.DayOfWeekEnum.SUNDAY,
        ]
        
        for day_of_week in days_of_week:
            operation = google_ads_client.get_type("CampaignCriterionOperation")
            criterion = operation.create
            criterion.campaign = campaign_resource_name
            criterion.status = google_ads_client.enums.CampaignCriterionStatusEnum.ENABLED
            
            # Set ad schedule
            ad_schedule = criterion.ad_schedule
            ad_schedule.start_hour = start_hour
            ad_schedule.start_minute = google_ads_client.enums.MinuteOfHourEnum.ZERO
            ad_schedule.end_hour = end_hour
            ad_schedule.end_minute = google_ads_client.enums.MinuteOfHourEnum.ZERO
            ad_schedule.day_of_week = day_of_week
            
            operations.append(operation)
    
    return operations


def get_location_ids_for_countries(
    google_ads_client: GoogleAdsClient, customer_id: str, countries: list[str]
) -> dict[str, int]:
    """
    Get location criterion IDs for a list of country names in a single query.
    Returns a dict mapping country name to location ID.
    """
    if not countries:
        return {}
    
    ga_service = google_ads_client.get_service("GoogleAdsService")
    
    # Build IN clause with all country names
    country_names = "', '".join(countries)
    
    query = f"""
        SELECT
            geo_target_constant.id,
            geo_target_constant.name,
            geo_target_constant.country_code,
            geo_target_constant.target_type
        FROM geo_target_constant
        WHERE geo_target_constant.name IN ('{country_names}')
        AND geo_target_constant.target_type = 'Country'
    """
    
    response = ga_service.search(customer_id=customer_id, query=query)
    
    location_map = {row.geo_target_constant.name : row.geo_target_constant.id for row in response}
    
    # Check if all countries were found
    missing = set(countries) - set(location_map.keys())
    if missing:
        raise ValueError(f"Locations not found for countries: {missing}")
    
    return location_map


def create_location_operations(
    google_ads_client: GoogleAdsClient,
    campaign_resource_name: str,
    countries: list[str],
    location_map: dict[str, int],
) -> list[CampaignCriterionOperation]:
    """
    Create location targeting operations for a campaign using pre-fetched location IDs.
    """
    operations = []
    
    for country in countries:
        location_id = location_map[country]
        
        operation = google_ads_client.get_type("CampaignCriterionOperation")
        criterion = operation.create
        criterion.campaign = campaign_resource_name
        criterion.status = google_ads_client.enums.CampaignCriterionStatusEnum.ENABLED
        criterion.location.geo_target_constant = google_ads_client.get_service("GeoTargetConstantService").geo_target_constant_path(location_id)
        
        operations.append(operation)
    
    return operations


def create_campaigns_for_course(
    google_ads_client: GoogleAdsClient, customer_id: str, course: str, execute: bool
) -> list[CampaignSpec]:
    """Create all campaigns and ad groups for a given course."""
    course_config = COURSE_CONFIG.get(course)
    if not course_config:
        print(f"Error: Course '{course}' not found in config")
        sys.exit(1)

    course_title = course_config.get("course_title_base", course.replace("_", " ").title())
    regions = course_config.get("regions", {})
    match_types = course_config.get("match_types", ["Exact", "Phrase", "Broad"])
    default_budget = course_config.get("default_daily_budget_micros", 1_000_000)

    # Collect all unique countries across all regions, deduplicate in case of manual errors in config
    all_countries = set()
    for countries in regions.values():
        all_countries.update(countries)
    
    # Fetch all location IDs in a single query
    print(f"\n{'='*60}")
    print(f"Fetching location IDs for {len(all_countries)} countries...")
    print(f"{'='*60}")
    location_map = get_location_ids_for_countries(google_ads_client, customer_id, all_countries)
    print(f"✓ Retrieved {len(location_map)} location IDs")

    # Prepare all campaign specifications
    campaign_specs = []
    for region_label, countries in regions.items():
        for match_type in match_types:
            campaign_name = f"Course - {course_title} - {region_label} - {match_type}"
            ad_group_name = f"{course_title} - {region_label} - {match_type}"
            budget_name = f"Budget - {course_title} - {region_label} - {match_type}"

            spec = CampaignSpec(
                campaign_name=campaign_name,
                ad_group_name=ad_group_name,
                budget_name=budget_name,
                default_budget=default_budget,
                region_label=region_label,
                countries=countries,
            )
            campaign_specs.append(spec)

            print(f"Planned: {campaign_name}")

    if not execute:
        print(f"\n[DRY RUN] Would create {len(campaign_specs)} campaigns with ad groups")
        return []

    # Batch create all budgets
    print(f"\n{'='*60}")
    print(f"Creating {len(campaign_specs)} campaign budgets...")
    print(f"{'='*60}")

    campaign_budget_service = google_ads_client.get_service("CampaignBudgetService")
    budget_operations = [
        create_campaign_budget_operation(google_ads_client, spec.budget_name, spec.default_budget)
        for spec in campaign_specs
    ]

    try:
        # Create request with response_content_type to get full resource data back
        request = google_ads_client.get_type("MutateCampaignBudgetsRequest")
        request.customer_id = customer_id
        request.operations = budget_operations
        # MUTABLE_RESOURCE is required for things like result.campaign_budget.name to be populated in the response.
        request.response_content_type = (
            google_ads_client.enums.ResponseContentTypeEnum.MUTABLE_RESOURCE
        )

        budget_response = campaign_budget_service.mutate_campaign_budgets(request=request)
        print(f"✓ Created {len(budget_response.results)} budgets")

        # Map each result back to the corresponding spec using the budget name from response
        for result in budget_response.results:
            budget_name = result.campaign_budget.name
            spec = find_spec_by_name(campaign_specs, budget_name, "budget_name")
            if spec:
                spec.budget_resource_name = result.resource_name
    except Exception as e:
        print(f"✗ Error creating budgets: {e}")
        sys.exit(1)

    # Batch create all campaigns
    print(f"\n{'='*60}")
    print(f"Creating {len(campaign_specs)} campaigns...")
    print(f"{'='*60}")

    campaign_service = google_ads_client.get_service("CampaignService")
    campaign_operations = [
        create_campaign_operation(google_ads_client, spec.campaign_name, spec.budget_resource_name)
        for spec in campaign_specs
    ]

    try:
        request = google_ads_client.get_type("MutateCampaignsRequest")
        request.customer_id = customer_id
        request.operations = campaign_operations
        request.response_content_type = (
            google_ads_client.enums.ResponseContentTypeEnum.MUTABLE_RESOURCE
        )

        campaign_response = campaign_service.mutate_campaigns(request=request)
        print(f"✓ Created {len(campaign_response.results)} campaigns")

        for result in campaign_response.results:
            campaign_name = result.campaign.name
            spec = find_spec_by_name(campaign_specs, campaign_name, "campaign_name")
            if spec:
                spec.campaign_resource_name = result.resource_name
    except Exception as e:
        print(f"✗ Error creating campaigns: {e}")
        sys.exit(1)

    # Batch create all ad groups
    print(f"\n{'='*60}")
    print(f"Creating {len(campaign_specs)} ad groups...")
    print(f"{'='*60}")

    ad_group_service = google_ads_client.get_service("AdGroupService")
    ad_group_operations = [
        create_ad_group_operation(
            google_ads_client, spec.ad_group_name, spec.campaign_resource_name
        )
        for spec in campaign_specs
    ]

    try:
        request = google_ads_client.get_type("MutateAdGroupsRequest")
        request.customer_id = customer_id
        request.operations = ad_group_operations
        request.response_content_type = (
            google_ads_client.enums.ResponseContentTypeEnum.MUTABLE_RESOURCE
        )

        ad_group_response = ad_group_service.mutate_ad_groups(request=request)
        print(f"✓ Created {len(ad_group_response.results)} ad groups")

        for result in ad_group_response.results:
            ad_group_name = result.ad_group.name
            spec = find_spec_by_name(campaign_specs, ad_group_name, "ad_group_name")
            if spec:
                spec.ad_group_resource_name = result.resource_name
    except Exception as e:
        print(f"✗ Error creating ad groups: {e}")
        sys.exit(1)

    # Batch create ad schedules for all campaigns
    print(f"\n{'='*60}")
    print(f"Creating ad schedules for {len(campaign_specs)} campaigns...")
    print(f"{'='*60}")

    # Ad schedules and location targeting could be batched together, but we'll keep them seperate for now
    # It's not much less efficient, and this lets us have more granular logging and error handling for each type of criterion if we need it.
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    all_ad_schedule_operations = []
    
    for spec in campaign_specs:
        ad_schedule_ops = create_ad_schedule_operations(google_ads_client, spec.campaign_resource_name)
        all_ad_schedule_operations.extend(ad_schedule_ops)
    
    try:
        request = google_ads_client.get_type("MutateCampaignCriteriaRequest")
        request.customer_id = customer_id
        request.operations = all_ad_schedule_operations
        
        ad_schedule_response = campaign_criterion_service.mutate_campaign_criteria(request=request)
        print(f"✓ Created {len(ad_schedule_response.results)} ad schedule criteria")
    except Exception as e:
        print(f"✗ Error creating ad schedules: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Creating location targeting for {len(campaign_specs)} campaigns...")
    print(f"{'='*60}")

    all_location_operations = []
    
    for spec in campaign_specs:
        location_ops = create_location_operations(google_ads_client, spec.campaign_resource_name, spec.countries, location_map)
        all_location_operations.extend(location_ops)
    
    try:
        request = google_ads_client.get_type("MutateCampaignCriteriaRequest")
        request.customer_id = customer_id
        request.operations = all_location_operations
        
        location_response = campaign_criterion_service.mutate_campaign_criteria(request=request)
        print(f"✓ Created {len(location_response.results)} location targeting criteria")
    except Exception as e:
        print(f"✗ Error creating location targeting: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Summary: Successfully created {len(campaign_specs)} campaigns with ad schedules and location targeting")
    print(f"{'='*60}")

    return campaign_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Google Ads campaigns and ad groups for a given course. Be advised that nothing in this script is transactional and it doesn't attempt to avoid collisions. Use with caution."
    )
    parser.add_argument(
        "--course",
        type=str,
        choices=["gen_ai", "ml", "sys_eng", "sys_think"],
        required=True,
        help="The course to create campaigns for",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Execute the campaign creation (without this flag, runs in dry-run mode)",
    )

    args = parser.parse_args()

    # Validate environment variables
    yaml_path = os.getenv("GOOGLE_ADS_YAML_PATH")
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID")

    if not yaml_path:
        print("Error: GOOGLE_ADS_YAML_PATH environment variable not set")
        sys.exit(1)

    if not customer_id:
        print("Error: GOOGLE_ADS_CUSTOMER_ID environment variable not set")
        sys.exit(1)

    # Initialize Google Ads client
    google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)

    # Create campaigns
    create_campaigns_for_course(google_ads_client, customer_id, args.course, args.execute)

    if not args.execute:
        print("\n⚠️  DRY RUN MODE - No changes were made")
        print("Run with --execute to create campaigns")


if __name__ == "__main__":
    main()
