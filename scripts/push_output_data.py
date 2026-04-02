#!/usr/bin/env python3
"""
Script to output to Google Ads. Can set overall budget, kw level max cpc and bid adjustments as output by pipeline.
"""

import argparse
import csv
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from google.ads.googleads.client import GoogleAdsClient
from google.api_core import protobuf_helpers

from utils.name_generation import construct_campaign_name_for_args

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.bid_adjustments import (
    get_device_bid_adjustments,
    get_age_bid_adjustments,
    get_hour_of_day_bid_adjustments,
)
from utils.google_ads_api import (
    get_existing_campaign_criteria,
    get_enabled_campaigns_for_course,
    get_location_bid_adjustments,
    get_existing_ad_group_age_for_campaigns, get_campaign_budget_info, get_ad_groups_for_enabled_campaigns,
)

from utils.gaql_queries import (
    SELECT_KEYWORD_CRITERION_IN_AD_GROUP,
)
from config import COURSE_CONFIG

BATCH_SIZE = 5000  # Google Ads API limit

BUDGET = "budget"
CPC = "cpc"
BID_ADJ = "bid_adj"
VALID_DATASETS = {BUDGET, CPC, BID_ADJ}

# Map match type strings to enum values
MATCH_TYPE_MAP = {"Exact match": "EXACT", "Phrase match": "PHRASE", "Broad match": "BROAD"}

def warn_on_large_cpc_changes(
    new_cpc_bids: dict[tuple[int, str, str], float],
    current_cpc_lookup: dict[tuple[int, str, str], tuple[int, Any, int]],
    threshold: float,
) -> None:
    # Compare with current CPC and warn if change is too large
    for key, new_bid in new_cpc_bids.items():
        if key not in current_cpc_lookup:
            continue
        ad_group_id, keyword_text, match_type = key
        _, _, current_cpc_micros = current_cpc_lookup[key]
        current_bid = current_cpc_micros / 1_000_000
        if current_bid > 0:
            pct_change = abs(new_bid - current_bid) / current_bid
            if pct_change > threshold:
                print(f"WARNING: Large CPC change detected for keyword '{keyword_text}' ({match_type}) in ad group {ad_group_id}:")
                print(f"  Current: ${current_bid:.2f}, New: ${new_bid:.2f}")
                print(f"  Change: {pct_change * 100:.1f}% (threshold: {threshold * 100:.1f}%)")


def warn_on_large_budget_changes(
    new_budgets: dict[str, float],
    current_budgets: dict[str, dict[str, Any]],
    threshold: float,
) -> None:
    # Compare with current budget and warn if change is too large
    for campaign_name, new_budget in new_budgets.items():
        current_budget = current_budgets[campaign_name]["current_budget_amount"]
        if current_budget > 0:
            pct_change = abs(new_budget - current_budget) / current_budget
            if pct_change > threshold:
                print(f"WARNING: Large budget change detected for {campaign_name}:")
                print(f"  Current: ${current_budget:.2f}/day, New: ${new_budget:.2f}/day")
                print(f"  Change: {pct_change * 100:.1f}% (threshold: {threshold * 100:.1f}%)")


def push_budget(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    execute: bool,
) -> None:
    """Push overall budget to Google Ads."""
    budget_output_dir = Path(f"opt_results/{output_course}/bids")
    daily_budget_filepath = budget_output_dir / "daily_budget.csv"

    campaign_budget_service = google_ads_client.get_service("CampaignBudgetService")
    google_ads_service = google_ads_client.get_service("GoogleAdsService")

    # First pass: collect campaign names and new budgets
    campaign_data = {}
    with open(daily_budget_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            region = row["Region"]
            match_type = row["Match type"]
            daily_budget = float(row["Daily Budget"])
            campaign_name = construct_campaign_name_for_args(output_course, match_type, region)
            campaign_data[campaign_name] = daily_budget

    if not campaign_data:
        print("No budget data to push")
        return

    # Fetch current budgets for all campaigns
    current_campaign_budget_resources = get_campaign_budget_info(
        google_ads_service, customer_id, campaign_data.keys()
    )

    # Get budget change threshold from config
    budget_change_threshold = COURSE_CONFIG[output_course]["budget_change_threshold"]

    # Iterate through existing campaign budgets and warn if there are any which are above the configured threshold
    warn_on_large_budget_changes(
        campaign_data, current_campaign_budget_resources, budget_change_threshold
    )

    operations = []
    for campaign_name, new_budget in campaign_data.items():

        if campaign_name not in current_campaign_budget_resources:
            print(f"Skipping budget update for {campaign_name} - campaign not found")
            continue

        # Create update operation
        operation = google_ads_client.get_type("CampaignBudgetOperation")
        campaign_budget = operation.update
        campaign_budget.resource_name = current_campaign_budget_resources[campaign_name][
            "budget_resource_id"
        ]
        campaign_budget.amount_micros = int(round(Decimal(new_budget), 2) * 1_000_000)

        # See https://developers.google.com/google-ads/api/docs/client-libs/python/field-masks
        # This is the preferred way to use field masks with the Google Ads API client.
        field_mask = protobuf_helpers.field_mask(None, campaign_budget._pb)
        google_ads_client.copy_from(operation.update_mask, field_mask)

        operations.append(operation)
        print(f"Prepared budget update: {campaign_name} -> ${new_budget:.2f}/day")

    # Execute all budget updates
    if operations and execute:
        response = campaign_budget_service.mutate_campaign_budgets(
            customer_id=customer_id, operations=operations
        )
        print(f"Successfully updated {len(response.results)} campaign budgets")


def push_cpc(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    execute: bool,
) -> None:
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
    campaign_to_ad_group = get_ad_groups_for_enabled_campaigns(google_ads_service, customer_id, campaign_names)
    ad_group_ids = set(campaign_to_ad_group.values())

    print(f"Found {len(campaign_to_ad_group)} ad groups")

    # Get all keyword criteria for all ad groups
    # TODO: Pull this into a function in google_ads_api.py, clean up the returned data structure in a subsequent PR
    print(f"Fetching keywords for {len(ad_group_ids)} ad groups...")
    ad_group_list = "', '".join(
        f"customers/{customer_id}/adGroups/{ag_id}" for ag_id in ad_group_ids
    )
    query = SELECT_KEYWORD_CRITERION_IN_AD_GROUP.format(ad_group_list=ad_group_list)
    response = google_ads_service.search(customer_id=customer_id, query=query)

    # Build lookup: (ad_group_id, keyword_text, match_type) -> (criterion_id, status, cpc_bid_micros)
    # This represents the keywords in google ads for the specified campaigns.
    gaql_keyword_lookup = {}
    for result in response:
        ad_group_id = result.ad_group.id
        criterion_id = result.ad_group_criterion.criterion_id
        keyword_text = result.ad_group_criterion.keyword.text
        match_type = result.ad_group_criterion.keyword.match_type.name
        # The result from the API is the enum value, so we'll convert it to the name to match our inputs
        status = google_ads_client.enums.AdGroupCriterionStatusEnum(result.ad_group_criterion.status).name

        cpc_bid_micros = result.ad_group_criterion.cpc_bid_micros
        key = (ad_group_id, keyword_text.lower(), match_type)
        gaql_keyword_lookup[key] = (criterion_id, status, cpc_bid_micros)

    print(f"Found {len(gaql_keyword_lookup)} keywords")

    # Warn if any keyword CPC would change by more than the configured threshold
    cpc_change_threshold = COURSE_CONFIG[output_course]["cpc_change_threshold"]
    new_cpc_bids = {}
    for row in rows:
        if row["Status"] == "PAUSED":
            continue
        campaign_name = construct_campaign_name_for_args(output_course, row["Match type"], row["Region"])
        if campaign_name not in campaign_to_ad_group:
            continue
        ad_group_id = campaign_to_ad_group[campaign_name]
        match_type_enum = MATCH_TYPE_MAP[row["Match type"]]
        key = (ad_group_id, row["Keyword"].lower(), match_type_enum)
        new_cpc_bids[key] = float(row["Bid"])

    # Note that this won't catch increases if you've never set a bid on a keyword or it's currently set to 0.
    warn_on_large_cpc_changes(new_cpc_bids, gaql_keyword_lookup, cpc_change_threshold)

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
        # For now, we'll assume that manual ad groups have been set up and just need statuses to be flipped.
        if key not in gaql_keyword_lookup:
            print(
                f"Warning: Keyword '{keyword}' ({match_type}) not found in ad group {ad_group_id}, skipping"
            )
            continue

        criterion_id, current_status, current_cpc_bid_micros = gaql_keyword_lookup[key]

        new_cpc_bid_micros = int(round(Decimal(bid), 2) * 1_000_000)

        status_changed = current_status != status
        cpc_changed = new_cpc_bid_micros != current_cpc_bid_micros

        # If neither the bid nor the status is changing, skip to avoid unnecessary API calls
        # The overwhelming majority of keywords are paused in any given run so this should
        # substantially reduce the number of API calls we make each run
        if not status_changed and not cpc_changed:
            print(f'Skipping update for keyword "{keyword}" ({match_type}) in ad group {ad_group_id} - no changes detected')
            continue

        # Create update operation
        operation = google_ads_client.get_type("AdGroupCriterionOperation")
        criterion = operation.update
        criterion.resource_name = ad_group_criterion_service.ad_group_criterion_path(
            customer_id, ad_group_id, criterion_id
        )

        criterion.cpc_bid_micros = new_cpc_bid_micros

        if status == "PAUSED":
            criterion.status = google_ads_client.enums.AdGroupCriterionStatusEnum.PAUSED
        else:
            criterion.status = google_ads_client.enums.AdGroupCriterionStatusEnum.ENABLED

        print(
            f'Updating keyword "{keyword}" ({match_type}) in ad group {ad_group_id}: optimal cost ${bid:.2f}, status {status}'
        )

        # Generate field mask from the updated criterion
        field_mask = protobuf_helpers.field_mask(None, criterion._pb)
        google_ads_client.copy_from(operation.update_mask, field_mask)

        operations.append(operation)

    # Execute all keyword updates in batches as the input files can be quite large
    if operations and execute:
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


def push_bid_adjustments(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    output_course: str,
    execute: bool,
) -> None:
    """Push bid adjustments to Google Ads."""
    budget_output_dir = Path(f"opt_results/{output_course}/bid_adjustments")
    adj_age_filepath = budget_output_dir / "bid_adj_age.csv"
    adj_device_filepath = budget_output_dir / "bid_adj_device.csv"
    adj_hour_of_day_filepath = budget_output_dir / "bid_adj_hour_of_day.csv"
    adj_location_filepath = budget_output_dir / "bid_adj_location.csv"

    # Get all campaigns for this course
    ga_service = google_ads_client.get_service("GoogleAdsService")
    campaigns = get_enabled_campaigns_for_course(ga_service, customer_id, output_course)

    if not campaigns:
        print(f"Warning: No campaigns found for course {output_course}")
        return

    print(f"Found {len(campaigns)} campaigns for {output_course}")

    # Get existing campaign criteria
    print("Fetching existing campaign criteria...")
    existing_campaign_criteria = get_existing_campaign_criteria(
        ga_service, customer_id, campaigns.values()
    )
    print(
        f"Found {len(existing_campaign_criteria['device'])} device criteria, "
        f"{len(existing_campaign_criteria['schedule'])} schedule criteria, {len(existing_campaign_criteria['location'])} location criteria"
    )

    print("Fetching existing ad group age criteria...")
    existing_ad_group_age_criteria = get_existing_ad_group_age_for_campaigns(
        ga_service, customer_id, campaigns.values()
    )

    all_operations = []
    # Age bid adjustments are the only one which are done on the ad-group level.
    age_ops = []

    # Process each type of bid adjustment
    if adj_age_filepath.exists():
        print("\n--- Processing Age Bid Adjustments ---")
        age_ops.extend(
            get_age_bid_adjustments(
                google_ads_client,
                customer_id,
                campaigns,
                existing_ad_group_age_criteria,
                adj_age_filepath,
            )
        )
    else:
        print(f"Age adjustment file not found: {adj_age_filepath}")

    if adj_device_filepath.exists():
        print("\n--- Processing Device Bid Adjustments ---")
        device_ops = get_device_bid_adjustments(
            google_ads_client,
            customer_id,
            campaigns,
            existing_campaign_criteria,
            adj_device_filepath,
        )
        all_operations.extend(device_ops)
    else:
        print(f"Device adjustment file not found: {adj_device_filepath}")

    if adj_hour_of_day_filepath.exists():
        print("\n--- Processing Hour-of-Day Bid Adjustments ---")
        hour_ops = get_hour_of_day_bid_adjustments(
            google_ads_client,
            customer_id,
            campaigns,
            existing_campaign_criteria,
            adj_hour_of_day_filepath,
        )
        all_operations.extend(hour_ops)
    else:
        print(f"Hour-of-day adjustment file not found: {adj_hour_of_day_filepath}")

    if adj_location_filepath.exists():
        print("\n--- Processing Location Bid Adjustments ---")
        location_ops = get_location_bid_adjustments(
            google_ads_client,
            customer_id,
            campaigns,
            existing_campaign_criteria,
            adj_location_filepath,
        )
        all_operations.extend(location_ops)
    else:
        print(f"Location adjustment file not found: {adj_location_filepath}")

    # Execute all bid adjustment operations
    # Age operations use AdGroupCriterionService, others use CampaignCriterionService
    if age_ops and execute:
        print(f"\n--- Executing {len(age_ops)} age bid adjustment operations ---")
        ad_group_criterion_service = google_ads_client.get_service("AdGroupCriterionService")

        # Process in batches
        for i in range(0, len(age_ops), BATCH_SIZE):
            batch = age_ops[i : i + BATCH_SIZE]
            try:
                response = ad_group_criterion_service.mutate_ad_group_criteria(
                    customer_id=customer_id, operations=batch
                )
                print(
                    f"Successfully applied {len(response.results)} age bid adjustments (batch {i//BATCH_SIZE + 1})"
                )
            except Exception as e:
                print(f"Error applying age bid adjustments in batch {i//BATCH_SIZE + 1}: {e}")

        print(f"Successfully processed {len(age_ops)} total age bid adjustments")
    else:
        print("No age bid adjustments to apply")

    if all_operations and execute:
        print(
            f"\n--- Executing {len(all_operations)} device/schedule/location bid adjustment operations ---"
        )
        campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")

        # Process in batches
        for i in range(0, len(all_operations), BATCH_SIZE):
            batch = all_operations[i : i + BATCH_SIZE]
            try:
                response = campaign_criterion_service.mutate_campaign_criteria(
                    customer_id=customer_id, operations=batch
                )
                print(
                    f"Successfully applied {len(response.results)} device/schedule/location bid adjustments (batch {i//BATCH_SIZE + 1})"
                )
            except Exception as e:
                print(
                    f"Error applying device/schedule/location bid adjustments in batch {i//BATCH_SIZE + 1}: {e}"
                )

        print(
            f"Successfully processed {len(all_operations)} total device/schedule/location bid adjustments"
        )
    else:
        print("No device/schedule/location bid adjustments to apply")


def main() -> None:
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
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="The course to push data for, determines the location of the file inputs.",
    )
    parser.add_argument(
        "--google-ads-yaml",
        type=str,
        required=True,
        help="Path to Google Ads YAML configuration file",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        required=True,
        help="Google Ads customer ID",
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

    yaml_path = args.google_ads_yaml
    google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)
    customer_id = args.customer_id
    output_course = args.output_course
    execute = args.execute

    if BUDGET in requested_datasets:
        push_budget(google_ads_client, customer_id, output_course, execute)
        print(f"Successfully pushed budget")

    if CPC in requested_datasets:
        push_cpc(google_ads_client, customer_id, output_course, execute)
        print(f"Successfully pushed max cpc data")

    if BID_ADJ in requested_datasets:
        push_bid_adjustments(google_ads_client, customer_id, output_course, execute)
        print(f"Successfully pushed bid adjustments")

    print(f"All requested datasets pushed successfully")


if __name__ == "__main__":
    main()
