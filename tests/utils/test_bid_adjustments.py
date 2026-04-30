import csv
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from google.ads.googleads.v23.enums import AgeRangeTypeEnum, DeviceEnum

from utils.bid_adjustments import (
    ALL_DAYS,
    get_age_bid_adjustments,
    get_device_bid_adjustments,
    get_hour_of_day_bid_adjustments,
)

CUSTOMER_ID = "1234567890"
CAMPAIGN_ID = 9876543210
CAMPAIGN_NAME = "Course - Python 101 - US - Exact - Experiment"
CRITERION_ID = 111222333


def _write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# get_device_bid_adjustments
# ---------------------------------------------------------------------------

class TestGetDeviceBidAdjustments:
    EXISTING_CRITERIA = {"device": {(str(CAMPAIGN_ID), DeviceEnum.Device.MOBILE): CRITERION_ID}, "schedule": {}}

    def test_returns_operation_for_existing_criterion(self, tmp_path):
        csv_file = tmp_path / "device.csv"
        _write_csv(csv_file, ["Region", "Device", "BidAdjustment"], [
            {"Region": "US", "Device": "Mobile phones", "BidAdjustment": "0.05"},
        ])
        client = MagicMock()
        campaigns = {CAMPAIGN_NAME: CAMPAIGN_ID}

        operations = get_device_bid_adjustments(client, CUSTOMER_ID, campaigns, self.EXISTING_CRITERIA, csv_file)

        assert len(operations) == 1
        service = client.get_service.return_value
        service.campaign_criterion_path.assert_called_once_with(CUSTOMER_ID, CAMPAIGN_ID, CRITERION_ID)
        operation = client.get_type.return_value
        assert operation.update.resource_name == service.campaign_criterion_path.return_value
        assert operation.update.bid_modifier == Decimal("1.05")

    def test_skips_row_with_empty_bid_adjustment(self, tmp_path):
        csv_file = tmp_path / "device.csv"
        _write_csv(csv_file, ["Region", "Device", "BidAdjustment"], [
            {"Region": "US", "Device": "Mobile phones", "BidAdjustment": ""},
        ])

        operations = get_device_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID},  self.EXISTING_CRITERIA, csv_file
        )

        assert operations == []

    def test_skips_campaign_with_wrong_region(self, tmp_path):
        csv_file = tmp_path / "device.csv"
        _write_csv(csv_file, ["Region", "Device", "BidAdjustment"], [
            {"Region": "A", "Device": "Mobile phones", "BidAdjustment": "0.05"},
        ])

        operations = get_device_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, self.EXISTING_CRITERIA, csv_file
        )

        assert operations == []

    def test_skips_missing_criterion(self, tmp_path):
        csv_file = tmp_path / "device.csv"
        _write_csv(csv_file, ["Region", "Device", "BidAdjustment"], [
            {"Region": "US", "Device": "Mobile phones", "BidAdjustment": "0.05"},
        ])

        operations = get_device_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID},
            {"device": {}, "schedule": {}}, csv_file
        )

        assert operations == []


# ---------------------------------------------------------------------------
# get_age_bid_adjustments
# ---------------------------------------------------------------------------

class TestGetAgeBidAdjustments:
    AD_GROUP_ID = 444555666
    EXISTING_CRITERIA = {(str(CAMPAIGN_ID), AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34): (AD_GROUP_ID, CRITERION_ID)}

    def test_returns_operation_for_existing_criterion(self, tmp_path):
        csv_file = tmp_path / "age.csv"
        _write_csv(csv_file, ["Region", "Age", "BidAdjustment"], [
            {"Region": "US", "Age": "25 - 34", "BidAdjustment": "0.10"},
        ])
        client = MagicMock()

        operations = get_age_bid_adjustments(
            client, CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, self.EXISTING_CRITERIA, csv_file
        )

        assert len(operations) == 1
        service = client.get_service.return_value
        service.ad_group_criterion_path.assert_called_once_with(
            CUSTOMER_ID, self.AD_GROUP_ID, CRITERION_ID
        )
        operation = client.get_type.return_value
        assert operation.update.resource_name == service.ad_group_criterion_path.return_value
        assert operation.update.bid_modifier == Decimal("1.10")

    def test_skips_row_with_empty_bid_adjustment(self, tmp_path):
        csv_file = tmp_path / "age.csv"
        _write_csv(csv_file, ["Region", "Age", "BidAdjustment"], [
            {"Region": "US", "Age": "25 - 34", "BidAdjustment": ""},
        ])

        operations = get_age_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, {}, csv_file
        )

        assert operations == []

    def test_skips_campaign_with_wrong_region(self, tmp_path):
        csv_file = tmp_path / "age.csv"
        _write_csv(csv_file, ["Region", "Age", "BidAdjustment"], [
            {"Region": "A", "Age": "25 - 34", "BidAdjustment": "0.10"},
        ])

        operations = get_age_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, self.EXISTING_CRITERIA, csv_file
        )

        assert operations == []

    def test_skips_unknown_age_range(self, tmp_path):
        csv_file = tmp_path / "age.csv"
        _write_csv(csv_file, ["Region", "Age", "BidAdjustment"], [
            {"Region": "US", "Age": "100+", "BidAdjustment": "0.10"},
        ])

        operations = get_age_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, {}, csv_file
        )

        assert operations == []

    def test_skips_missing_criterion(self, tmp_path):
        csv_file = tmp_path / "age.csv"
        _write_csv(csv_file, ["Region", "Age", "BidAdjustment"], [
            {"Region": "US", "Age": "25 - 34", "BidAdjustment": "0.10"},
        ])

        operations = get_age_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, {}, csv_file
        )

        assert operations == []


# ---------------------------------------------------------------------------
# get_hour_of_day_bid_adjustments
# ---------------------------------------------------------------------------

class TestGetHourOfDayBidAdjustments:
    EXISTING_CRITERIA = {
            "schedule": {
                (str(CAMPAIGN_ID), day, 9, 12): CRITERION_ID
                for day in ALL_DAYS
            }
        }

    def test_returns_operation_for_each_day_of_week(self, tmp_path):
        csv_file = tmp_path / "hod.csv"
        _write_csv(csv_file, ["Region", "Hour Group", "BidAdjustment"], [
            {"Region": "US", "Hour Group": "9 - 12", "BidAdjustment": "0.15"},
        ])
        client = MagicMock()

        operations = get_hour_of_day_bid_adjustments(
            client, CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, self.EXISTING_CRITERIA, csv_file
        )

        assert len(operations) == len(ALL_DAYS)
        assert client.get_type.call_count == len(ALL_DAYS)
        service = client.get_service.return_value
        assert service.campaign_criterion_path.call_count == len(ALL_DAYS)
        operation = client.get_type.return_value
        assert operation.update.bid_modifier == Decimal("1.15")

    def test_skips_row_with_empty_bid_adjustment(self, tmp_path):
        csv_file = tmp_path / "hod.csv"
        _write_csv(csv_file, ["Region", "Hour Group", "BidAdjustment"], [
            {"Region": "US", "Hour Group": "9 - 12", "BidAdjustment": ""},
        ])

        operations = get_hour_of_day_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, {"schedule": {}}, csv_file
        )

        assert operations == []

    def test_skips_campaign_with_wrong_region(self, tmp_path):
        csv_file = tmp_path / "hod.csv"
        _write_csv(csv_file, ["Region", "Hour Group", "BidAdjustment"], [
            {"Region": "A", "Hour Group": "9 - 12", "BidAdjustment": "0.15"},
        ])

        operations = get_hour_of_day_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, self.EXISTING_CRITERIA, csv_file
        )

        assert operations == []

    def test_skips_invalid_hour_group_format(self, tmp_path):
        csv_file = tmp_path / "hod.csv"
        _write_csv(csv_file, ["Region", "Hour Group", "BidAdjustment"], [
            {"Region": "US", "Hour Group": "morning", "BidAdjustment": "0.15"},
        ])

        operations = get_hour_of_day_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, {"schedule": {}}, csv_file
        )

        assert operations == []

    def test_skips_missing_criterion(self, tmp_path):
        csv_file = tmp_path / "hod.csv"
        _write_csv(csv_file, ["Region", "Hour Group", "BidAdjustment"], [
            {"Region": "US", "Hour Group": "9 - 12", "BidAdjustment": "0.15"},
        ])

        operations = get_hour_of_day_bid_adjustments(
            MagicMock(), CUSTOMER_ID, {CAMPAIGN_NAME: CAMPAIGN_ID}, {"schedule": {}}, csv_file
        )

        assert operations == []
