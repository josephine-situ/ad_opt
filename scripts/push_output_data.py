#!/usr/bin/env python3
"""
Script to output to Google Ads. Can set overall budget, kw level max cpc and bid adjustments as output by pipeline.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from google.ads.googleads.client import GoogleAdsClient
from google.api_core import protobuf_helpers

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.gaql_queries import (
    GET_CAMPAIGN_BUDGET_FOR_CAMPAIGN_NAME,
)
from config import COURSE_CONFIG

BUDGET = "budget"
CPC = "cpc"
BID_ADJ = "bid_adj"
VALID_DATASETS = {BUDGET, CPC, BID_ADJ}


def validate_environment_variables(datasets):
    """Validate that required environment variables are set for the given dataset."""

    missing_vars = [
        var for var in ["GOOGLE_ADS_CUSTOMER_ID", "GOOGLE_ADS_YAML_PATH"] if not os.getenv(var)
    ]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    return True


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
            # Convert daily budget to micros (Google Ads uses micros: 1 unit = 1,000,000 micros)
            campaign_budget.amount_micros = int(round(daily_budget, 2) * 1_000_000)

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
    """Push kw level max cpc to Google Ads."""
    budget_output_dir = Path(f"opt_results/{output_course}/bids")
    daily_budget_filepath = budget_output_dir / "optimized_costs.csv"
    with open(daily_budget_filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)
    raise NotImplementedError


def push_bid_adjustments(google_ads_client, customer_id, output_course):
    """Push bid adjustments to Google Ads."""
    budget_output_dir = Path(f"opt_results/{output_course}/bid_adjustments")
    adj_age_filepath = budget_output_dir / "bid_adj_age.csv"
    adj_device_filepath = budget_output_dir / "bid_adj_device.csv"
    adj_hour_of_day_filepath = budget_output_dir / "bid_adj_hour_of_day.csv"
    adj_location_filepath = budget_output_dir / "bid_adj_location.csv"
    raise NotImplementedError


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
