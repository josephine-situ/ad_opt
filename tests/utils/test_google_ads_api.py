from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from google.ads.googleads.v23.enums.types.age_range_type import AgeRangeTypeEnum

from utils.google_ads_api import (
    build_location_cache,
    get_ad_groups_for_enabled_campaigns,
    get_campaign_budget_info,
    get_enabled_campaigns_for_course,
    get_existing_ad_group_age_for_campaigns,
    get_existing_campaign_criteria,
    get_location_resource_names_for_countries,
    normalize_bid_adjustment,
    should_skip_campaign,
    LOCATION_CACHE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_stream_service(*rows):
    """Return a mock ads service whose search_stream yields one batch of rows."""
    batch = MagicMock()
    batch.results = list(rows)
    service = MagicMock()
    service.search_stream.return_value = [batch]
    return service


# ---------------------------------------------------------------------------
# normalize_bid_adjustment
# ---------------------------------------------------------------------------

class TestNormalizeBidAdjustment:
    def test_positive_string(self):
        assert normalize_bid_adjustment("0.5") == Decimal("1.5")

    def test_negative_string(self):
        assert normalize_bid_adjustment("-0.2") == Decimal("0.8")

    def test_zero(self):
        assert normalize_bid_adjustment(0) == Decimal("1.0")

    def test_float_input(self):
        assert normalize_bid_adjustment(Decimal("0.25")) == Decimal("1.25")

    def test_decimal_input(self):
        assert normalize_bid_adjustment(Decimal("-1.0")) == Decimal("0.0")


# ---------------------------------------------------------------------------
# should_skip_campaign
# ---------------------------------------------------------------------------

VALID_COURSE = "Course - Python 101 - US - Exact - Experiment"
VALID_PROGRAM = "Program - Data Science - A - Broad - Experiment"
CUSTOMER_ID = "1234567890"
CAMPAIGN_ID = "9876543210"

class TestShouldSkipCampaign:
    def test_invalid_format_is_skipped(self):
        assert should_skip_campaign("Some Random Campaign") is True

    def test_valid_course_campaign_not_skipped(self):
        assert should_skip_campaign(VALID_COURSE) is False

    def test_valid_program_campaign_not_skipped(self):
        assert should_skip_campaign(VALID_PROGRAM) is False

    def test_wrong_region_is_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_region="A") is True

    def test_matching_region_not_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_region="US") is False

    def test_wrong_match_type_is_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_match_type="Broad") is True

    def test_matching_match_type_not_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_match_type="Exact") is False

    def test_wrong_course_is_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_course="Java") is True

    def test_matching_course_substring_not_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_course="Python") is False

    def test_all_filters_match_not_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_course="Python", check_region="US", check_match_type="Exact") is False

    def test_one_filter_mismatch_is_skipped(self):
        assert should_skip_campaign(VALID_COURSE, check_course="Python", check_region="A", check_match_type="Exact") is True

    def test_missing_experiment_suffix_is_skipped(self):
        assert should_skip_campaign("Course - Python 101 - US - Exact") is True

# ---------------------------------------------------------------------------
# get_from_location_cache / build_location_cache
# ---------------------------------------------------------------------------

class TestLocationCache:

    def test_build_location_cache_populates_cache(self):
        row = MagicMock()
        row.geo_target_constant.id = 10
        row.geo_target_constant.canonical_name = "Canada"

        ads_service = MagicMock()
        ads_service.search.return_value = [row]

        client = MagicMock()
        client.get_service.return_value = ads_service

        with patch.dict("utils.google_ads_api.LOCATION_CACHE", {}, clear=True):
            with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
                build_location_cache(client, CUSTOMER_ID, [10])
            assert LOCATION_CACHE[10] == "Canada"
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search", 1)

    def test_build_location_cache_skips_already_cached_ids(self):
        client = MagicMock()
        with patch.dict("utils.google_ads_api.LOCATION_CACHE", {5: "Existing"}, clear=True):
            build_location_cache(client, CUSTOMER_ID, [5])
        client.get_service.assert_not_called()

    def test_build_location_cache_uses_fallback_on_error(self):
        client = MagicMock()
        client.get_service.side_effect = Exception("API error")

        with patch.dict("utils.google_ads_api.LOCATION_CACHE", {}, clear=True):
            build_location_cache(client, CUSTOMER_ID, [99])
            assert LOCATION_CACHE[99] == "Location 99"


# ---------------------------------------------------------------------------
# get_location_resource_names_for_countries
# ---------------------------------------------------------------------------

class TestGetLocationResourceNamesForCountries:

    def test_returns_resource_name_map(self):
        suggestion = MagicMock()
        suggestion.search_term = "United States"
        suggestion.geo_target_constant.resource_name = "geoTargetConstants/2840"

        geo_service = MagicMock()
        geo_service.suggest_geo_target_constants.return_value.geo_target_constant_suggestions = [suggestion]

        client = MagicMock()
        client.get_service.return_value = geo_service

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_location_resource_names_for_countries(client, ["United States"])

        assert result == {"United States": "geoTargetConstants/2840"}
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("suggest_geo_target_constants", 1)

    def test_raises_for_unresolved_countries(self):
        suggestion = MagicMock()
        suggestion.search_term = "United States"
        suggestion.geo_target_constant.resource_name = "geoTargetConstants/2840"

        geo_service = MagicMock()
        geo_service.suggest_geo_target_constants.return_value.geo_target_constant_suggestions = [suggestion]

        client = MagicMock()
        client.get_service.return_value = geo_service

        with patch("utils.google_ads_api.google_ads_metrics_client"):
            with pytest.raises(ValueError, match="Locations not found"):
                get_location_resource_names_for_countries(client, ["United States", "Canada"])

    def test_duplicate_countries_deduplicated(self):
        suggestion = MagicMock()
        suggestion.search_term = "Canada"
        suggestion.geo_target_constant.resource_name = "geoTargetConstants/2124"

        geo_service = MagicMock()
        geo_service.suggest_geo_target_constants.return_value.geo_target_constant_suggestions = [suggestion]

        client = MagicMock()
        client.get_service.return_value = geo_service

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_location_resource_names_for_countries(client, ["Canada", "Canada"])

        assert result == {"Canada": "geoTargetConstants/2124"}
        assert geo_service.suggest_geo_target_constants.call_count == 1
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("suggest_geo_target_constants", 1)


# ---------------------------------------------------------------------------
# get_enabled_campaigns_for_course
# ---------------------------------------------------------------------------

class TestGetEnabledCampaignsForCourse:
    def test_returns_campaign_id_map(self):
        row = MagicMock()
        row.campaign.name = VALID_COURSE
        row.campaign.id = 42

        service = _make_search_stream_service(row)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_enabled_campaigns_for_course(service, CUSTOMER_ID, [VALID_COURSE])

        assert result == {VALID_COURSE: 42}
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)

    def test_empty_stream_returns_empty_dict(self):
        service = _make_search_stream_service()

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_enabled_campaigns_for_course(service, CUSTOMER_ID, [])

        assert result == {}
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)


def _make_device_criterion_row(campaign_id: str, device_type: str, criterion_id: int = 1) -> MagicMock:
    row = MagicMock()
    row.campaign_criterion.campaign = f"customers/{CUSTOMER_ID}/campaigns/{campaign_id}"
    row.campaign_criterion.criterion_id = criterion_id
    row.campaign_criterion.type_.name = "DEVICE"
    row.campaign_criterion.device.type_ = device_type
    return row


def _make_schedule_criterion_row(
    campaign_id: str,
    day: str,
    start_hour: int,
    end_hour: int,
    criterion_id: int = 1,
) -> MagicMock:
    row = MagicMock()
    row.campaign_criterion.campaign = f"customers/{CUSTOMER_ID}/campaigns/{campaign_id}"
    row.campaign_criterion.criterion_id = criterion_id
    row.campaign_criterion.type_.name = "AD_SCHEDULE"
    row.campaign_criterion.ad_schedule.day_of_week = day
    row.campaign_criterion.ad_schedule.start_hour = start_hour
    row.campaign_criterion.ad_schedule.end_hour = end_hour
    return row


def _make_location_criterion_row(
    campaign_id: str, geo_target: str, criterion_id: int = 1
) -> MagicMock:
    row = MagicMock()
    row.campaign_criterion.campaign = f"customers/{CUSTOMER_ID}/campaigns/{campaign_id}"
    row.campaign_criterion.criterion_id = criterion_id
    row.campaign_criterion.type_.name = "LOCATION"
    row.campaign_criterion.location.geo_target_constant = geo_target
    return row


# ---------------------------------------------------------------------------
# get_existing_campaign_criteria
# ---------------------------------------------------------------------------

class TestGetExistingCampaignCriteria:
    def test_device_criterion_indexed_correctly(self):
        row = _make_device_criterion_row(CAMPAIGN_ID, device_type="MOBILE", criterion_id=7)
        service = _make_search_stream_service(row)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_existing_campaign_criteria(service, CUSTOMER_ID, [CAMPAIGN_ID])

        assert result["device"][(CAMPAIGN_ID, "MOBILE")] == 7
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)

    def test_schedule_criterion_indexed_correctly(self):
        row = _make_schedule_criterion_row(
            CAMPAIGN_ID, day="MONDAY", start_hour=9, end_hour=17, criterion_id=8
        )
        service = _make_search_stream_service(row)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_existing_campaign_criteria(service, CUSTOMER_ID, [CAMPAIGN_ID])

        assert result["schedule"][(CAMPAIGN_ID, "MONDAY", 9, 17)] == 8
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)

    def test_location_criterion_indexed_correctly(self):
        row = _make_location_criterion_row(
            CAMPAIGN_ID, geo_target="geoTargetConstants/2840", criterion_id=9
        )
        service = _make_search_stream_service(row)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_existing_campaign_criteria(service, CUSTOMER_ID, [CAMPAIGN_ID])

        assert result["location"][(CAMPAIGN_ID, "geoTargetConstants/2840")] == 9
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)


# ---------------------------------------------------------------------------
# get_existing_ad_group_age_for_campaigns
# ---------------------------------------------------------------------------

class TestGetExistingAdGroupAgeForCampaigns:
    def test_returns_age_criteria_map(self):
        row = MagicMock()
        row.campaign.id = CAMPAIGN_ID
        row.ad_group.id = 20
        row.ad_group_criterion.criterion_id = 30
        row.ad_group_criterion.age_range.type_ = AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_18_24

        service = _make_search_stream_service(row)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_existing_ad_group_age_for_campaigns(service, CUSTOMER_ID, [10])

        assert result[(CAMPAIGN_ID, AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_18_24)] == ("20", 30)
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)


# ---------------------------------------------------------------------------
# get_campaign_budget_info
# ---------------------------------------------------------------------------

class TestGetCampaignBudgetInfo:
    def test_returns_budget_map(self):
        row = MagicMock()
        row.campaign.name = VALID_COURSE
        row.campaign_budget.amount_micros = 5_000_000
        row.campaign.campaign_budget = f"customers/{CUSTOMER_ID}/campaignBudgets/99"

        service = _make_search_stream_service(row)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_campaign_budget_info(service, CUSTOMER_ID, [VALID_COURSE])

        assert result[VALID_COURSE]["current_budget_amount"] == 5.0
        assert result[VALID_COURSE]["budget_resource_id"] == f"customers/{CUSTOMER_ID}/campaignBudgets/99"
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)


# ---------------------------------------------------------------------------
# get_ad_groups_for_enabled_campaigns
# ---------------------------------------------------------------------------

class TestGetAdGroupsForEnabledCampaigns:
    def test_returns_first_ad_group_per_campaign(self):
        row1 = MagicMock()
        row1.campaign.name = VALID_COURSE
        row1.ad_group.id = 101

        # second ad group — should be ignored. In practice, we only ever expect a single ad group per campaign
        row2 = MagicMock()
        row2.campaign.name = VALID_COURSE
        row2.ad_group.id = 102

        service = _make_search_stream_service(row1, row2)

        with patch("utils.google_ads_api.google_ads_metrics_client") as mock_metrics:
            result = get_ad_groups_for_enabled_campaigns(service, CUSTOMER_ID, [VALID_COURSE])

        assert result[VALID_COURSE] == 101
        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)
