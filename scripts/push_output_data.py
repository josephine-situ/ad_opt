#!/usr/bin/env python3
"""
Script to output to Google Ads. Can set overall budget, kw level max cpc and bid adjustments as output by pipeline.
"""

import argparse
import csv
import decimal
import os
import sys
from decimal import Decimal
from pathlib import Path

from google.ads.googleads.client import GoogleAdsClient
from google.api_core import protobuf_helpers

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.gaql_queries import (
    GET_CAMPAIGN_BUDGET_FOR_CAMPAIGN_NAME,
    SELECT_KEYWORD_CRITERION_IN_AD_GROUP,
    SELECT_AD_GROUPS_FOR_CAMPAIGNS,
)
from config import COURSE_CONFIG

BATCH_SIZE = 5000  # Google Ads API limit

BUDGET = "budget"
CPC = "cpc"
BID_ADJ = "bid_adj"
VALID_DATASETS = {BUDGET, CPC, BID_ADJ}

# Map match type strings to enum values
MATCH_TYPE_MAP = {"Exact match": "EXACT", "Phrase match": "PHRASE", "Broad match": "BROAD"}

from google.ads.googleads.v23.enums.types import AgeRangeTypeEnum

# Map CSV age ranges to Google Ads age range types
AGE_RANGE_MAP = {
    "18 - 24": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_18_24,
    "25 - 34": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34,
    "35 - 44": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_35_44,
    "45 - 54": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_45_54,
    "55 - 64": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_55_64,
    "65+": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_65_UP,
    "Unknown": AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_UNDETERMINED,
}

def validate_environment_variables(datasets):
    """Validate that required environment variables are set for the given dataset."""

    missing_vars = [
        var for var in ["GOOGLE_ADS_CUSTOMER_ID", "GOOGLE_ADS_YAML_PATH"] if not os.getenv(var)
    ]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return True

def normalize_bid_adjustment(bid_adj):
    """Normalize bid adjustment to be within Google Ads limits (0.1 to 10.0, or exactly 0 for device criteria)."""
    bid_adj_decimal = Decimal(bid_adj)
    if bid_adj_decimal < 0:
        return Decimal(1.0) - bid_adj_decimal
    elif bid_adj_decimal > 0:
        return bid_adj_decimal

    return bid_adj_decimal

def should_skip_campaign(campaign_name, check_course=None, check_region=None, check_match_type=None):
    """Determine whether to skip a campaign based on its name and the specified region, match_type or course name."""
    if check_course and check_course not in campaign_name:
        return True
    if check_region and check_region not in campaign_name:
        return True
    if check_match_type and check_match_type not in campaign_name:
        return True
    return False

def construct_campaign_name_for_args(course, match_type, region):
    """Construct campaign name based on course, match type and region."""
    return f"Course - {COURSE_CONFIG[course]['course_title_base']} - {region} - {match_type}"


def get_course_campaign_budget_resource_name(google_ads_client, customer_id, campaign_name):
    """Get campaign budget resource name for a given campaign name."""

    ga_service = google_ads_client.get_service("GoogleAdsService")
    query = GET_CAMPAIGN_BUDGET_FOR_CAMPAIGN_NAME.format(campaign_name=campaign_name)

    response = ga_service.search(customer_id=customer_id, query=query)
    results = list(response)
    if not results:
        print(f"Error: No campaign found with name {campaign_name}")
        return None
    return results[0].campaign.campaign_budget


def push_budget(google_ads_client, customer_id, output_course):
    """Push overall budget to Google Ads."""
    budget_output_dir = Path(f"opt_results/{output_course}/bids")
    daily_budget_filepath = budget_output_dir / "daily_budget.csv"

    campaign_budget_service = google_ads_client.get_service("CampaignBudgetService")
    operations = []

    with open(daily_budget_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row["Region"]
            match_type = row["Match type"]
            daily_budget = float(row["Daily Budget"])

            campaign_name = construct_campaign_name_for_args(output_course, match_type, region)
            campaign_budget_resource_name = get_course_campaign_budget_resource_name(
                google_ads_client, customer_id, campaign_name
            )

            if not campaign_budget_resource_name:
                print(f"Skipping budget update for {campaign_name} - campaign not found")
                continue

            # Create update operation
            operation = google_ads_client.get_type("CampaignBudgetOperation")
            campaign_budget = operation.update
            campaign_budget.resource_name = campaign_budget_resource_name
            campaign_budget.amount_micros = int(round(Decimal(daily_budget), 2) * 1_000_000)

            # See https://developers.google.com/google-ads/api/docs/client-libs/python/field-masks
            # This is the preferred way to use field masks with the Google Ads API client.
            field_mask = protobuf_helpers.field_mask(None, campaign_budget._pb)
            google_ads_client.copy_from(operation.update_mask, field_mask)

            operations.append(operation)
            print(f"Prepared budget update: {campaign_name} -> ${daily_budget:.2f}/day")

    # Execute all budget updates
    if operations:
        response = campaign_budget_service.mutate_campaign_budgets(
            customer_id=customer_id, operations=operations
        )
        print(f"Successfully updated {len(response.results)} campaign budgets")


def push_cpc(google_ads_client, customer_id, output_course):
    """
    Push keyword-level max CPC to Google Ads.
    Note that this will only be visible if the associated campaign is using a manual CPC bidding strategy.
    """
    budget_output_dir = Path(f"opt_results/{output_course}/bids")
    optimized_costs_filepath = budget_output_dir / "optimized_costs.csv"

    google_ads_service = google_ads_client.get_service("GoogleAdsService")
    ad_group_criterion_service = google_ads_client.get_service("AdGroupCriterionService")

    # Read all rows to determine unique campaigns needed
    rows = []
    campaign_names = set()

    with open(optimized_costs_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
            campaign_name = construct_campaign_name_for_args(
                output_course, row["Match type"], row["Region"]
            )
            campaign_names.add(campaign_name)

    # Bulk query: Get all ad groups for the campaigns we need
    print(f"Fetching ad groups for {len(campaign_names)} campaigns...")
    campaign_list = "', '".join(campaign_names)
    query = SELECT_AD_GROUPS_FOR_CAMPAIGNS.format(campaign_list=campaign_list)

    response = google_ads_service.search(customer_id=customer_id, query=query)
    campaign_to_ad_group = {}
    ad_group_ids = set()

    for result in response:
        campaign_name = result.campaign.name
        ad_group_id = result.ad_group.id
        # Take first ad group per campaign
        if campaign_name not in campaign_to_ad_group:
            campaign_to_ad_group[campaign_name] = ad_group_id
            ad_group_ids.add(ad_group_id)

    print(f"Found {len(campaign_to_ad_group)} ad groups")

    # Get all keyword criteria for all ad groups
    print(f"Fetching keywords for {len(ad_group_ids)} ad groups...")
    ad_group_list = "', '".join(
        f"customers/{customer_id}/adGroups/{ag_id}" for ag_id in ad_group_ids
    )
    query = SELECT_KEYWORD_CRITERION_IN_AD_GROUP.format(ad_group_list=ad_group_list)
    response = google_ads_service.search(customer_id=customer_id, query=query)

    # Build lookup: (ad_group_id, keyword_text, match_type) -> (criterion_id, status)
    # This represents the keywords in google ads for the specified campaigns.
    gaql_keyword_lookup = {}
    for result in response:
        ad_group_id = result.ad_group.id
        criterion_id = result.ad_group_criterion.criterion_id
        keyword_text = result.ad_group_criterion.keyword.text
        match_type = result.ad_group_criterion.keyword.match_type.name
        status = result.ad_group_criterion.status

        key = (ad_group_id, keyword_text.lower(), match_type)
        gaql_keyword_lookup[key] = (criterion_id, status)

    print(f"Found {len(gaql_keyword_lookup)} keywords")

    # Process each row and create operations
    operations = []

    for row in rows:
        keyword = row["Keyword"]
        region = row["Region"]
        match_type = row["Match type"]
        bid = float(row["Bid"])
        status = row["Status"]

        # Get campaign and corresponding ad group based on row parameters
        campaign_name = construct_campaign_name_for_args(output_course, match_type, region)

        if campaign_name not in campaign_to_ad_group:
            print(
                f"Warning: No ad group found for campaign '{campaign_name}', skipping keyword '{keyword}'"
            )
            continue

        ad_group_id = campaign_to_ad_group[campaign_name]
        match_type_enum = MATCH_TYPE_MAP[match_type]

        # Look up keyword criterion
        key = (ad_group_id, keyword.lower(), match_type_enum)
        # TODO: We may want to conditionally create missing keywords in the future.
        # For now, we'll assume that manual ad groups have been set up and just need statuses to be flipped.
        if key not in gaql_keyword_lookup:
            print(
                f"Warning: Keyword '{keyword}' ({match_type}) not found in ad group {ad_group_id}, skipping"
            )
            continue

        criterion_id, current_status = gaql_keyword_lookup[key]

        # Create update operation
        operation = google_ads_client.get_type("AdGroupCriterionOperation")
        criterion = operation.update
        criterion.resource_name = ad_group_criterion_service.ad_group_criterion_path(
            customer_id, ad_group_id, criterion_id
        )

        # Enable if status is ENABLED. If enabling, also set the bid.
        if status == "PAUSED":
            if current_status != google_ads_client.enums.AdGroupCriterionStatusEnum.PAUSED:
                criterion.status = google_ads_client.enums.AdGroupCriterionStatusEnum.PAUSED
        else:
            # Enable keyword if currently disabled
            if current_status == google_ads_client.enums.AdGroupCriterionStatusEnum.PAUSED:
                criterion.status = google_ads_client.enums.AdGroupCriterionStatusEnum.ENABLED

            criterion.cpc_bid_micros = int(round(Decimal(bid), 2) * 1_000_000)

        print(
            f'Updating keyword "{keyword}" ({match_type}) in ad group {ad_group_id}: optimal cost ${bid:.2f}, status {status}'
        )

        # Generate field mask from the updated criterion
        field_mask = protobuf_helpers.field_mask(None, criterion._pb)
        google_ads_client.copy_from(operation.update_mask, field_mask)

        operations.append(operation)

    # Execute all keyword updates in batches as the input files can be quite large
    if operations:
        print(f"Updating {len(operations)} keywords...")

        for i in range(0, len(operations), BATCH_SIZE):
            batch = operations[i : i + BATCH_SIZE]
            response = ad_group_criterion_service.mutate_ad_group_criteria(
                customer_id=customer_id, operations=batch
            )
            print(f"Updated {len(response.results)} keywords (batch {i//BATCH_SIZE + 1})")

        print(f"Successfully updated {len(operations)} total keywords")
    else:
        print("No keyword updates to perform")


def get_campaigns_for_course(google_ads_service, customer_id, output_course):
    """Get all campaign IDs for a given course."""
    # TODO: Move this to the gaql file
    course_title = COURSE_CONFIG[output_course]['course_title_base']
    
    query = f"""
        SELECT campaign.id, campaign.name
        FROM campaign
        WHERE campaign.name LIKE 'Course - {course_title}%'
        AND campaign.status != 'REMOVED'
    """
    
    response = google_ads_service.search(customer_id=customer_id, query=query)
    return {row.campaign.name: row.campaign.id for row in response}


def get_existing_campaign_criteria(google_ads_service, customer_id, campaign_ids):
    """Get all existing campaign criteria for the specified campaigns."""
    # TODO: Move this to the gaql file.
    # Build query to get all campaign criteria
    campaign_id_list = "', '".join([f"customers/{customer_id}/campaigns/{cid}" for cid in campaign_ids])
    
    query = f"""
        SELECT
            campaign_criterion.campaign,
            campaign_criterion.criterion_id,
            campaign_criterion.bid_modifier,
            campaign_criterion.type,
            campaign_criterion.device.type,
            campaign_criterion.ad_schedule.day_of_week,
            campaign_criterion.ad_schedule.start_hour,
            campaign_criterion.ad_schedule.end_hour,
            campaign_criterion.location.geo_target_constant
        FROM campaign_criterion
        WHERE campaign_criterion.campaign IN ('{campaign_id_list}')
        AND campaign_criterion.status != 'REMOVED'
    """
    
    response = google_ads_service.search(customer_id=customer_id, query=query)
    
    # Organize criteria by type and campaign
    criteria = {
        'device': {},   # (campaign_id, device_type) -> criterion_id
        'schedule': {}, # (campaign_id, day, start_hour, end_hour) -> criterion_id
        'location': {}, # (campaign_id, geo_target) -> criterion_id
    }
    
    for row in response:
        campaign_path = row.campaign_criterion.campaign
        campaign_id = campaign_path.split('/')[-1]
        criterion_id = row.campaign_criterion.criterion_id
        criterion_type = row.campaign_criterion.type_

        if criterion_type.name == 'DEVICE':
            device_type = row.campaign_criterion.device.type_
            criteria['device'][(campaign_id, device_type)] = criterion_id
        elif criterion_type.name == 'AD_SCHEDULE':
            day = row.campaign_criterion.ad_schedule.day_of_week
            start_hour = row.campaign_criterion.ad_schedule.start_hour
            end_hour = row.campaign_criterion.ad_schedule.end_hour
            criteria['schedule'][(campaign_id, day, start_hour, end_hour)] = criterion_id
        elif criterion_type.name == 'LOCATION':
            geo_target = row.campaign_criterion.location.geo_target_constant
            criteria['location'][(campaign_id, geo_target)] = criterion_id
    
    return criteria


def push_age_bid_adjustments(google_ads_client, customer_id, campaigns, existing_criteria, adj_age_filepath):
    """Push age bid adjustments to Google Ads (applied at ad group level)."""
    ad_group_criterion_service = google_ads_client.get_service("AdGroupCriterionService")
    operations = []
    
    with open(adj_age_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row["Region"]
            age = row["Age"]
            bid_adj_decimal = row.get("BidAdjustment", "")
            
            # Skip if no bid adjustment calculated
            if not bid_adj_decimal:
                continue

            # bid_modifier expects to be 0.1 - 10.0. It can be set to 0 only for device criteria
            bid_adjustment = normalize_bid_adjustment(bid_adj_decimal)
            age_type = AGE_RANGE_MAP.get(age)
            
            if not age_type:
                print(f"Warning: Unknown age range '{age}', skipping")
                continue
            
            # Apply to all campaigns for this region
            for campaign_name, campaign_id in campaigns.items():
                if should_skip_campaign(campaign_name, check_region=region):
                    continue
                
                # Check if criterion exists (age criteria are stored as (campaign_id, age_type) -> (ad_group_id, criterion_id))
                key = (str(campaign_id), age_type)
                if key in existing_criteria:
                    ad_group_id, criterion_id = existing_criteria[key]

                    operation = google_ads_client.get_type("AdGroupCriterionOperation")
                    criterion = operation.update
                    criterion.resource_name = ad_group_criterion_service.ad_group_criterion_path(
                        customer_id, ad_group_id, criterion_id
                    )
                    criterion.bid_modifier = bid_adjustment

                    field_mask = protobuf_helpers.field_mask(None, criterion._pb)
                    google_ads_client.copy_from(operation.update_mask, field_mask)
                    
                    operations.append(operation)
                    print(f"Prepared age adjustment: {campaign_name}, Age {age} -> {bid_adjustment:.2%}")
                else:
                    print(f"Warning: Age criterion not found for {campaign_name}, Age {age} - skipping")
    
    return operations


def push_device_bid_adjustments(google_ads_client, customer_id, campaigns, existing_criteria, adj_device_filepath):
    """Push device bid adjustments to Google Ads."""
    from google.ads.googleads.v23.enums.types import DeviceEnum
    
    # Map CSV device names to Google Ads device types
    device_map = {
        "Mobile phones": DeviceEnum.Device.MOBILE,
        "Tablets": DeviceEnum.Device.TABLET,
        "Computers": DeviceEnum.Device.DESKTOP,
    }
    
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    operations = []
    
    with open(adj_device_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row["Region"]
            device = row["Device"]
            bid_adj_decimal = row.get("BidAdjustment", "")
            
            # Skip if no bid adjustment calculated
            if not bid_adj_decimal:
                continue

            bid_adjustment = normalize_bid_adjustment(bid_adj_decimal)
            device_type = device_map.get(device)
            
            if device_type is None:
                print(f"Warning: Unknown device '{device}', skipping")
                continue
            
            # Apply to all campaigns for this region
            for campaign_name, campaign_id in campaigns.items():
                if should_skip_campaign(campaign_name, check_region=region):
                    continue
                
                # Check if criterion exists
                key = (str(campaign_id), device_type)
                if key in existing_criteria['device']:
                    criterion_id = existing_criteria['device'][key]
                    
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
                    print(f"Prepared device adjustment: {campaign_name}, {device} -> {bid_adjustment:.2%}")
                else:
                    print(f"Warning: Device criterion not found for {campaign_name}, {device} - skipping")
    
    return operations


def push_hour_of_day_bid_adjustments(google_ads_client, customer_id, campaigns, existing_criteria, adj_hour_of_day_filepath):
    """Push hour-of-day (ad schedule) bid adjustments to Google Ads."""
    from google.ads.googleads.v23.enums.types import DayOfWeekEnum
    
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    operations = []
    
    # All days of the week (Monday through Sunday)
    all_days = [
        DayOfWeekEnum.DayOfWeek.MONDAY,
        DayOfWeekEnum.DayOfWeek.TUESDAY,
        DayOfWeekEnum.DayOfWeek.WEDNESDAY,
        DayOfWeekEnum.DayOfWeek.THURSDAY,
        DayOfWeekEnum.DayOfWeek.FRIDAY,
        DayOfWeekEnum.DayOfWeek.SATURDAY,
        DayOfWeekEnum.DayOfWeek.SUNDAY,
    ]
    
    with open(adj_hour_of_day_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row["Region"]
            hour_group = row["Hour Group"]
            bid_adj_decimal = row.get("BidAdjustment", "")
            
            # Skip if no bid adjustment calculated
            if not bid_adj_decimal:
                continue
            
            bid_adjustment = normalize_bid_adjustment(bid_adj_decimal)
            
            # Parse hour range (e.g., "15 - 17" -> start_hour=15, end_hour=17)
            hour_parts = hour_group.split(" - ")
            if len(hour_parts) != 2:
                print(f"Warning: Invalid hour group format '{hour_group}', skipping")
                continue
            
            start_hour = int(hour_parts[0])
            end_hour = int(hour_parts[1])
            
            # Apply to all campaigns for this region
            for campaign_name, campaign_id in campaigns.items():
                if should_skip_campaign(campaign_name, check_region=region):
                    continue
                
                # Update ad schedule criterion for each day of the week
                for day in all_days:
                    key = (str(campaign_id), day, start_hour, end_hour)
                    if key in existing_criteria['schedule']:
                        criterion_id = existing_criteria['schedule'][key]
                        
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
                    else:
                        print(f"Warning: Ad schedule criterion not found for {campaign_name}, Hours {hour_group}, Day {day} - skipping")
                
                if operations:
                    print(f"Prepared hour-of-day adjustment: {campaign_name}, Hours {hour_group} (all days) -> {bid_adjustment:.2%}")
    
    return operations


def push_location_bid_adjustments(google_ads_client, customer_id, campaigns, existing_criteria, adj_location_filepath):
    """Push location bid adjustments to Google Ads."""
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    geo_target_service = google_ads_client.get_service("GeoTargetConstantService")
    operations = []
    
    # Cache for geo target lookups to avoid repeated API calls
    geo_target_cache = {}
    
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
                        print(f"Warning: Could not find geo target for location '{location}', skipping")
                        geo_target_cache[location] = None
                        continue
                    
                    # Use the first (best) suggestion
                    geo_target_constant = suggestions.geo_target_constant_suggestions[0].geo_target_constant.resource_name
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
                if key in existing_criteria['location']:
                    criterion_id = existing_criteria['location'][key]

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
                    print(f"Prepared location adjustment: {campaign_name}, {location} -> {bid_adjustment:.2%}")
                else:
                    print(f"Warning: Location criterion not found for {campaign_name}, {location} - skipping")

    
    return operations

def get_existing_ad_group_age_for_campaigns(google_ads_service, customer_id, campaign_ids):
    """Get existing ad group age criteria for the specified campaigns.
    
    Returns:
        dict: Map of (campaign_id, age_range_type) -> (ad_group_id, criterion_id)
    """
    # Build query to get ad group age criteria
    campaign_id_list = "', '".join([f"customers/{customer_id}/campaigns/{cid}" for cid in campaign_ids])

    query = f"""
        SELECT
            campaign.id,
            ad_group.id,
            ad_group_criterion.criterion_id,
            ad_group_criterion.age_range.type
        FROM ad_group_criterion
        WHERE campaign.id IN ({', '.join(map(str, campaign_ids))})
        AND ad_group_criterion.type = 'AGE_RANGE'
        AND ad_group_criterion.status != 'REMOVED'
    """

    response = google_ads_service.search(customer_id=customer_id, query=query)

    # Organize criteria by (campaign_id, age_range_type) -> (ad_group_id, criterion_id)
    criteria = {}

    for row in response:
        campaign_id = str(row.campaign.id)
        ad_group_id = str(row.ad_group.id)
        criterion_id = row.ad_group_criterion.criterion_id
        age_range_type = row.ad_group_criterion.age_range.type_

        key = (campaign_id, age_range_type)
        criteria[key] = (ad_group_id, criterion_id)

    return criteria


def push_bid_adjustments(google_ads_client, customer_id, output_course):
    """Push bid adjustments to Google Ads."""
    budget_output_dir = Path(f"opt_results/{output_course}/bid_adjustments")
    adj_age_filepath = budget_output_dir / "bid_adj_age.csv"
    adj_device_filepath = budget_output_dir / "bid_adj_device.csv"
    adj_hour_of_day_filepath = budget_output_dir / "bid_adj_hour_of_day.csv"
    adj_location_filepath = budget_output_dir / "bid_adj_location.csv"
    
    # Get all campaigns for this course
    ga_service = google_ads_client.get_service("GoogleAdsService")
    campaigns = get_campaigns_for_course(ga_service, customer_id, output_course)
    
    if not campaigns:
        print(f"Warning: No campaigns found for course {output_course}")
        return
    
    print(f"Found {len(campaigns)} campaigns for {output_course}")
    
    # Get existing campaign criteria
    print("Fetching existing campaign criteria...")
    existing_campaign_criteria = get_existing_campaign_criteria(ga_service, customer_id, campaigns.values())
    print(f"Found {len(existing_campaign_criteria['device'])} device criteria, "
          f"{len(existing_campaign_criteria['schedule'])} schedule criteria, {len(existing_campaign_criteria['location'])} location criteria")

    print("Fetching existing ad group age criteria...")
    existing_ad_group_age_criteria = get_existing_ad_group_age_for_campaigns(ga_service, customer_id, campaigns.values())

    
    all_operations = []
    # Age bid adjustments are the only one which are done on the ad-group level.
    age_ops = []
    
    # Process each type of bid adjustment
    if adj_age_filepath.exists():
        print("\n--- Processing Age Bid Adjustments ---")
        age_ops.extend(push_age_bid_adjustments(google_ads_client, customer_id, campaigns, existing_ad_group_age_criteria, adj_age_filepath))
    else:
        print(f"Age adjustment file not found: {adj_age_filepath}")
    
    if adj_device_filepath.exists():
        print("\n--- Processing Device Bid Adjustments ---")
        device_ops = push_device_bid_adjustments(google_ads_client, customer_id, campaigns, existing_campaign_criteria, adj_device_filepath)
        all_operations.extend(device_ops)
    else:
        print(f"Device adjustment file not found: {adj_device_filepath}")

    if adj_hour_of_day_filepath.exists():
        print("\n--- Processing Hour-of-Day Bid Adjustments ---")
        hour_ops = push_hour_of_day_bid_adjustments(google_ads_client, customer_id, campaigns, existing_campaign_criteria, adj_hour_of_day_filepath)
        all_operations.extend(hour_ops)
    else:
        print(f"Hour-of-day adjustment file not found: {adj_hour_of_day_filepath}")

    if adj_location_filepath.exists():
        print("\n--- Processing Location Bid Adjustments ---")
        location_ops = push_location_bid_adjustments(google_ads_client, customer_id, campaigns, existing_campaign_criteria, adj_location_filepath)
        all_operations.extend(location_ops)
    else:
        print(f"Location adjustment file not found: {adj_location_filepath}")
    
    # Execute all bid adjustment operations
    # Age operations use AdGroupCriterionService, others use CampaignCriterionService
    if age_ops:
        print(f"\n--- Executing {len(age_ops)} age bid adjustment operations ---")
        ad_group_criterion_service = google_ads_client.get_service("AdGroupCriterionService")
        
        # Process in batches
        for i in range(0, len(age_ops), BATCH_SIZE):
            batch = age_ops[i : i + BATCH_SIZE]
            try:
                response = ad_group_criterion_service.mutate_ad_group_criteria(
                    customer_id=customer_id, operations=batch
                )
                print(f"Successfully applied {len(response.results)} age bid adjustments (batch {i//BATCH_SIZE + 1})")
            except Exception as e:
                print(f"Error applying age bid adjustments in batch {i//BATCH_SIZE + 1}: {e}")
        
        print(f"Successfully processed {len(age_ops)} total age bid adjustments")
    else:
        print("No age bid adjustments to apply")

    if all_operations:
        print(f"\n--- Executing {len(all_operations)} device/schedule/location bid adjustment operations ---")
        campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")

        # Process in batches
        for i in range(0, len(all_operations), BATCH_SIZE):
            batch = all_operations[i : i + BATCH_SIZE]
            try:
                response = campaign_criterion_service.mutate_campaign_criteria(
                    customer_id=customer_id, operations=batch
                )
                print(f"Successfully applied {len(response.results)} device/schedule/location bid adjustments (batch {i//BATCH_SIZE + 1})")
            except Exception as e:
                print(f"Error applying device/schedule/location bid adjustments in batch {i//BATCH_SIZE + 1}: {e}")

        print(f"Successfully processed {len(all_operations)} total device/schedule/location bid adjustments")
    else:
        print("No device/schedule/location bid adjustments to apply")




def main():
    parser = argparse.ArgumentParser(description="Pull input data from various sources")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of datasets to push (choices: budget, cpc, bid_adj)",
    )
    parser.add_argument(
        "--output-course",
        type=str,
        default="",
        choices=["gen_ai", "ml", "sys_eng", "sys_think"],
        required=True,
        help="The course to push data for, determines the location of the file inputs.",
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
    # TODO: Map customer ID to course, if we have one manager account over all course accounts.
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID")
    output_course = args.output_course

    if BUDGET in requested_datasets:
        push_budget(google_ads_client, customer_id, output_course)
        print(f"Successfully pushed budget")

    if CPC in requested_datasets:
        push_cpc(google_ads_client, customer_id, output_course)
        print(f"Successfully pushed max cpc data")

    if BID_ADJ in requested_datasets:
        push_bid_adjustments(google_ads_client, customer_id, output_course)
        print(f"Successfully pushed bid adjustments")

    print(f"All requested datasets pushed successfully")


if __name__ == "__main__":
    main()
