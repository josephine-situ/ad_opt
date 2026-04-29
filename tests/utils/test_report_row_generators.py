import pytest
from unittest.mock import patch

from google.ads.googleads.v23.enums import AgeRangeTypeEnum, DeviceEnum

from tests.utils.conftest import make_ads_client, make_row, make_stream
from utils.report_row_generators import (
    generate_age_clicks_rows,
    generate_age_conversions_rows,
    generate_device_clicks_rows,
    generate_device_conversions_rows,
    generate_hod_clicks_rows,
    generate_hod_conversions_rows,
    generate_loc_clicks_rows,
    generate_loc_conversions_rows,
    generate_purchase_report_rows,
    generate_search_keyword_rows,
    generate_search_terms_row,
)

CAMPAIGN = "Course - MLx - USA - Exact - Experiment"
CUSTOMER_ID = "1234567890"
CRITERION_ID = 2840


# ---------------------------------------------------------------------------
# Simple stateless generators — tested together
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn, make_row, expected", [
    pytest.param(
        generate_search_keyword_rows,
        lambda: make_row({
            "segments.date": "2025-01-15",
            "ad_group_criterion.keyword.text": "machine learning",
            "ad_group_criterion.keyword.match_type.name": "EXACT",
            "campaign.name": CAMPAIGN,
            "metrics.clicks": 100,
            "metrics.all_conversions_value": 500.0,
            "customer.currency_code": "USD",
            "metrics.cost_micros": 1_500_000,
        }),
        {
            "Day": "2025-01-15",
            "Search keyword": "machine learning",
            "Search keyword match type": "Exact",
            "Campaign": CAMPAIGN,
            "Clicks": 100,
            "Conv. value": "500.00",
            "Currency code": "USD",
            "Cost": "1.50",
        },
        id="search_keyword",
    ),
    pytest.param(
        generate_search_terms_row,
        lambda: make_row({
            "segments.keyword.info.text": "machine learning",
            "segments.keyword.info.match_type.name": "PHRASE",
            "search_term_view.search_term": "ml course online",
            "segments.conversion_action_name": "Purchase",
            "metrics.all_conversions": 3.0,
        }),
        {
            "Search keyword": "machine learning",
            "Search keyword match type": "Phrase",
            "Search term": "ml course online",
            "Conversion action": "Purchase",
            "Conversions": "3.00",
        },
        id="search_terms",
    ),
    pytest.param(
        generate_purchase_report_rows,
        lambda: make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "metrics.all_conversions": 5.0,
        }),
        {"Campaign": CAMPAIGN, "Conversion action": "Purchase", "All conv.": "5.00"},
        id="purchase",
    ),
    pytest.param(
        generate_hod_clicks_rows,
        lambda: make_row({
            "campaign.name": CAMPAIGN,
            "segments.hour": 9,
            "metrics.clicks": 50,
        }),
        {"Campaign": CAMPAIGN, "Hour of the day": 9, "Clicks": 50},
        id="hod_clicks",
    ),
    pytest.param(
        generate_hod_conversions_rows,
        lambda: make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "segments.hour": 14,
            "metrics.all_conversions": 2.0,
        }),
        {"Campaign": CAMPAIGN, "Conversion action": "Purchase", "Hour of the day": 14, "All conv.": "2.00"},
        id="hod_conversions",
    ),
    pytest.param(
        generate_device_clicks_rows,
        lambda: make_row({
            "campaign.name": CAMPAIGN,
            "segments.device": DeviceEnum.Device.MOBILE,
            "metrics.clicks": 200,
        }),
        {"Campaign": CAMPAIGN, "Device": "Mobile phones", "Clicks": 200},
        id="device_clicks",
    ),
    pytest.param(
        generate_device_conversions_rows,
        lambda: make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "segments.device": DeviceEnum.Device.MOBILE,
            "metrics.all_conversions": 6.0,
        }),
        {"Campaign": CAMPAIGN, "Conversion action": "Purchase", "Device": "Mobile phones", "All conv.": "6.00"},
        id="device_conversions",
    ),
])
def test_simple_generators_yield_correct_row(fn, make_row, expected):
    assert list(fn(make_stream([make_row()]))) == [expected]


# ---------------------------------------------------------------------------
# Age generators — aggregate across rows
# ---------------------------------------------------------------------------

class TestAgeClicksRows:
    def test_aggregates_clicks_by_campaign_and_age(self):
        age_enum = AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34
        row1 = make_row({"campaign.name": CAMPAIGN, "ad_group_criterion.age_range.type_": age_enum, "metrics.clicks": 30})
        row2 = make_row({"campaign.name": CAMPAIGN, "ad_group_criterion.age_range.type_": age_enum, "metrics.clicks": 20})

        result = list(generate_age_clicks_rows(make_stream([row1, row2])))

        assert result == [{"Campaign": CAMPAIGN, "Age": "25 - 34", "Clicks": 50}]

    def test_output_is_sorted_by_campaign_then_age(self):
        age_18 = AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_18_24
        age_25 = AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34
        row_b = make_row({"campaign.name": "B Campaign", "ad_group_criterion.age_range.type_": age_18, "metrics.clicks": 5})
        row_a = make_row({"campaign.name": "A Campaign", "ad_group_criterion.age_range.type_": age_25, "metrics.clicks": 10})

        result = list(generate_age_clicks_rows(make_stream([row_b, row_a])))

        assert [r["Campaign"] for r in result] == ["A Campaign", "B Campaign"]


class TestAgeConversionsRows:
    def test_aggregates_conversions_by_campaign_action_and_age(self):
        age_enum = AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_35_44
        row1 = make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "ad_group_criterion.age_range.type": age_enum,
            "metrics.all_conversions": 1.5,
        })
        row2 = make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "ad_group_criterion.age_range.type": age_enum,
            "metrics.all_conversions": 2.5,
        })

        result = list(generate_age_conversions_rows(make_stream([row1, row2])))

        assert result == [{
            "Campaign": CAMPAIGN,
            "Conversion action": "Purchase",
            "Age": "35 - 44",
            "All conv.": "4.00",
        }]


# ---------------------------------------------------------------------------
# Location generators — two-pass with cache lookup
# ---------------------------------------------------------------------------

class TestLocClicksRows:
    def test_yields_row_with_resolved_location_name(self):
        row = make_row({
            "campaign.name": CAMPAIGN,
            "geographic_view.country_criterion_id": CRITERION_ID,
            "metrics.clicks": 30,
        })
        client, _ = make_ads_client()

        with patch("utils.report_row_generators.build_location_cache"), \
             patch.dict("utils.google_ads_api.LOCATION_CACHE", {CRITERION_ID: "United States"}):
            result = list(generate_loc_clicks_rows(make_stream([row]), client, CUSTOMER_ID))

        assert result == [{"Campaign": CAMPAIGN, "Targeted location": "United States", "Clicks": 30}]

    def test_calls_build_location_cache_with_collected_criterion_ids(self):
        row = make_row({
            "campaign.name": CAMPAIGN,
            "geographic_view.country_criterion_id": CRITERION_ID,
            "metrics.clicks": 10,
        })
        client, _ = make_ads_client()

        with patch("utils.report_row_generators.build_location_cache") as mock_build, \
             patch.dict("utils.google_ads_api.LOCATION_CACHE", {CRITERION_ID: "United States"}):
            list(generate_loc_clicks_rows(make_stream([row]), client, CUSTOMER_ID))

        mock_build.assert_called_once_with(client, CUSTOMER_ID, {CRITERION_ID})


class TestLocConversionsRows:
    def test_yields_row_with_resolved_location_name(self):
        row = make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "geographic_view.country_criterion_id": CRITERION_ID,
            "metrics.all_conversions": 1.5,
        })
        client, _ = make_ads_client()

        with patch("utils.report_row_generators.build_location_cache"), \
             patch.dict("utils.google_ads_api.LOCATION_CACHE", {CRITERION_ID: "United States"}):
            result = list(generate_loc_conversions_rows(make_stream([row]), client, CUSTOMER_ID))

        assert result == [{
            "Campaign": CAMPAIGN,
            "Conversion action": "Purchase",
            "Targeted location": "United States",
            "All conv.": "1.50",
        }]

    def test_calls_build_location_cache_with_collected_criterion_ids(self):
        row = make_row({
            "campaign.name": CAMPAIGN,
            "segments.conversion_action_name": "Purchase",
            "geographic_view.country_criterion_id": CRITERION_ID,
            "metrics.all_conversions": 1.0,
        })
        client, _ = make_ads_client()

        with patch("utils.report_row_generators.build_location_cache") as mock_build, \
             patch.dict("utils.google_ads_api.LOCATION_CACHE", {CRITERION_ID: "United States"}):
            list(generate_loc_conversions_rows(make_stream([row]), client, CUSTOMER_ID))

        mock_build.assert_called_once_with(client, CUSTOMER_ID, {CRITERION_ID})
