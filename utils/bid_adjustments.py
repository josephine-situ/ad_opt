import csv
from pathlib import Path
from typing import Any

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.v23.enums import AgeRangeTypeEnum, DayOfWeekEnum, DeviceEnum
from google.api_core import protobuf_helpers

from utils.google_ads_api import normalize_bid_adjustment, should_skip_campaign

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

AGE_ENUM_TO_RANGE = {v: k for k, v in AGE_RANGE_MAP.items()}

ALL_DAYS = [
        DayOfWeekEnum.DayOfWeek.MONDAY,
        DayOfWeekEnum.DayOfWeek.TUESDAY,
        DayOfWeekEnum.DayOfWeek.WEDNESDAY,
        DayOfWeekEnum.DayOfWeek.THURSDAY,
        DayOfWeekEnum.DayOfWeek.FRIDAY,
        DayOfWeekEnum.DayOfWeek.SATURDAY,
        DayOfWeekEnum.DayOfWeek.SUNDAY,
    ]

# Map CSV device names to Google Ads device types
DEVICE_MAP = {
    "Mobile phones": DeviceEnum.Device.MOBILE,
    "Tablets": DeviceEnum.Device.TABLET,
    "Computers": DeviceEnum.Device.DESKTOP,
    "Connected TV": DeviceEnum.Device.CONNECTED_TV,
    "Other": DeviceEnum.Device.OTHER
}

DEVICE_ENUM_TO_NAME = {v: k for k, v in DEVICE_MAP.items()}

def get_device_bid_adjustments(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    campaigns: dict[str, int],
    existing_criteria: dict[str, dict[Any, int]],
    adj_device_filepath: str | Path,
) -> list[Any]:
    """Push device bid adjustments to Google Ads."""

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
            device_type = DEVICE_MAP.get(device)

            if not DEVICE_MAP:
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


def get_age_bid_adjustments(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    campaigns: dict[str, int],
    existing_criteria: dict[tuple[str, Any], tuple[str, int]],
    adj_age_filepath: str | Path,
) -> list[Any]:
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
                    print('Skipping campaign due to region filter: ', campaign_name, region)
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


def get_hour_of_day_bid_adjustments(
    google_ads_client: GoogleAdsClient,
    customer_id: str,
    campaigns: dict[str, int],
    existing_criteria: dict[str, dict[Any, int]],
    adj_hour_of_day_filepath: str | Path,
) -> list[Any]:
    """Push hour-of-day (ad schedule) bid adjustments to Google Ads."""
    campaign_criterion_service = google_ads_client.get_service("CampaignCriterionService")
    operations = []

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
                for day in ALL_DAYS:
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
                        print(
                            f"Warning: Ad schedule criterion not found for {campaign_name}, Hours {hour_group}, Day {day} - skipping")

                if operations:
                    print(
                        f"Prepared hour-of-day adjustment: {campaign_name}, Hours {hour_group} (all days) -> {bid_adjustment:.2%}")

    return operations