import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.ads.googleads.v23.enums import AgeRangeTypeEnum, DeviceEnum

from tests.utils.conftest import make_ads_client
from utils.ads_reporting import (
    generate_age_clicks_and_conversion_report,
    generate_device_clicks_and_conversion_report,
    generate_hod_clicks_and_conversion_report,
    generate_loc_clicks_and_conversion_report,
    generate_purchase_report,
    generate_search_keyword_report,
    generate_search_terms_report,
    write_to_file,
)

CUSTOMER_ID = "1234567890"
OUTPUT_COURSE = "ml"
START_DATE = "2025-01-01"
END_DATE = "2025-01-31"


# ---------------------------------------------------------------------------
# write_to_file
# ---------------------------------------------------------------------------

class TestWriteToFile:
    def test_writes_header_and_rows(self, tmp_path):
        output_file = tmp_path / "out.csv"
        rows = [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}]
        write_to_file(["A", "B"], iter(rows), output_file, delimiter=",")

        with open(output_file) as f:
            reader = list(csv.DictReader(f))

        assert reader == rows

    def test_uses_restval_for_missing_fields(self, tmp_path):
        output_file = tmp_path / "out.csv"
        rows = [{"A": "1"}]  # missing "B"
        write_to_file(["A", "B"], iter(rows), output_file, delimiter=",", restval="0")

        with open(output_file) as f:
            reader = list(csv.DictReader(f))

        assert reader[0]["B"] == "0"

    def test_uses_tab_delimiter_by_default(self, tmp_path):
        output_file = tmp_path / "out.tsv"
        rows = [{"X": "hello", "Y": "world"}]
        write_to_file(["X", "Y"], iter(rows), output_file)

        raw = output_file.read_text()
        assert "\t" in raw

    def test_empty_row_generator_writes_only_header(self, tmp_path):
        output_file = tmp_path / "out.csv"
        write_to_file(["A", "B"], iter([]), output_file, delimiter=",")

        with open(output_file) as f:
            content = f.read()

        assert content.strip() == "A,B"


# ---------------------------------------------------------------------------
# generate_search_keyword_report
# ---------------------------------------------------------------------------

class TestGenerateSearchKeywordReport:
    def test_queries_service_and_writes_file(self, tmp_path):
        row = MagicMock()
        row.segments.date = "2025-01-15"
        row.ad_group_criterion.keyword.text = "machine learning"
        row.ad_group_criterion.keyword.match_type.name = "EXACT"
        row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        row.metrics.clicks = 100
        row.metrics.all_conversions_value = 500.0
        row.customer.currency_code = "USD"
        row.metrics.cost_micros = 1_500_000

        client, _ = make_ads_client([row])
        output_file = tmp_path / "output.csv"

        with patch("utils.ads_reporting.Path", return_value=output_file), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics:
            generate_search_keyword_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)
        with open(output_file) as f:
            rows = list(csv.DictReader(f))
        assert rows == [{
            "Day": "2025-01-15",
            "Search keyword": "machine learning",
            "Search keyword match type": "Exact",
            "Campaign": "Course - MLx - USA - Exact - Experiment",
            "Clicks": "100",
            "Conv. value": "500.00",
            "Currency code": "USD",
            "Cost": "1.50",
        }]


# ---------------------------------------------------------------------------
# generate_search_terms_report
# ---------------------------------------------------------------------------

class TestGenerateSearchTermsReport:
    def test_queries_service_and_writes_file(self, tmp_path):
        row = MagicMock()
        row.segments.keyword.info.text = "machine learning course"
        row.segments.keyword.info.match_type.name = "PHRASE"
        row.search_term_view.search_term = "ml course online"
        row.segments.conversion_action_name = "Purchase"
        row.metrics.all_conversions = 3.0

        client, _ = make_ads_client([row])
        output_file = tmp_path / "output.csv"

        with patch("utils.ads_reporting.Path", return_value=output_file), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics:
            generate_search_terms_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)
        with open(output_file) as f:
            rows = list(csv.DictReader(f))
        assert rows == [{
            "Search keyword": "machine learning course",
            "Search keyword match type": "Phrase",
            "Search term": "ml course online",
            "Conversion action": "Purchase",
            "Conversions": "3.00",
        }]


# ---------------------------------------------------------------------------
# generate_purchase_report
# ---------------------------------------------------------------------------

class TestGeneratePurchaseReport:
    def test_queries_service_and_writes_file(self, tmp_path):
        row = MagicMock()
        row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        row.segments.conversion_action_name = "Purchase"
        row.metrics.all_conversions = 5.0

        client, _ = make_ads_client([row])
        output_file = tmp_path / "output.csv"

        with patch("utils.ads_reporting.Path", return_value=output_file), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics:
            generate_purchase_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 1)
        with open(output_file) as f:
            rows = list(csv.DictReader(f))
        assert rows == [{
            "Campaign": "Course - MLx - USA - Exact - Experiment",
            "Conversion action": "Purchase",
            "All conv.": "5.00",
        }]


# ---------------------------------------------------------------------------
# generate_hod_clicks_and_conversion_report
# ---------------------------------------------------------------------------

class TestGenerateHodClicksAndConversionReport:
    def test_writes_correct_content_to_both_files(self, tmp_path):
        clicks_row = MagicMock()
        clicks_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        clicks_row.segments.hour = 9
        clicks_row.metrics.clicks = 50

        conv_row = MagicMock()
        conv_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        conv_row.segments.conversion_action_name = "Purchase"
        conv_row.segments.hour = 9
        conv_row.metrics.all_conversions = 2.0

        client, ads_service = make_ads_client()
        clicks_batch, conv_batch = MagicMock(), MagicMock()
        clicks_batch.results = [clicks_row]
        conv_batch.results = [conv_row]
        ads_service.search_stream.side_effect = [[clicks_batch], [conv_batch]]

        clicks_file = tmp_path / "hod_clicks.csv"
        conv_file = tmp_path / "hod_conv.csv"

        with patch("utils.ads_reporting.Path", side_effect=[clicks_file, conv_file]), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics:
            generate_hod_clicks_and_conversion_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 2)

        with open(clicks_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Hour of the day": "9",
                "Clicks": "50",
            }]

        with open(conv_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Conversion action": "Purchase",
                "Hour of the day": "9",
                "All conv.": "2.00",
            }]


# ---------------------------------------------------------------------------
# generate_age_clicks_and_conversion_report
# ---------------------------------------------------------------------------

class TestGenerateAgeClicksAndConversionReport:
    def test_writes_correct_content_to_both_files(self, tmp_path):
        age_enum = AgeRangeTypeEnum.AgeRangeType.AGE_RANGE_25_34

        clicks_row = MagicMock()
        clicks_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        clicks_row.ad_group_criterion.age_range.type_ = age_enum
        clicks_row.metrics.clicks = 120

        conv_row = MagicMock()
        conv_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        conv_row.segments.conversion_action_name = "Purchase"
        conv_row.ad_group_criterion.age_range.type = age_enum
        conv_row.metrics.all_conversions = 4.0

        client, ads_service = make_ads_client()
        clicks_batch, conv_batch = MagicMock(), MagicMock()
        clicks_batch.results = [clicks_row]
        conv_batch.results = [conv_row]
        ads_service.search_stream.side_effect = [[clicks_batch], [conv_batch]]

        clicks_file = tmp_path / "age_clicks.csv"
        conv_file = tmp_path / "age_conv.csv"

        with patch("utils.ads_reporting.Path", side_effect=[clicks_file, conv_file]), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics:
            generate_age_clicks_and_conversion_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 2)

        with open(clicks_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Age": "25 - 34",
                "Clicks": "120",
            }]

        with open(conv_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Conversion action": "Purchase",
                "Age": "25 - 34",
                "All conv.": "4.00",
            }]


# ---------------------------------------------------------------------------
# generate_device_clicks_and_conversion_report
# ---------------------------------------------------------------------------

class TestGenerateDeviceClicksAndConversionReport:
    def test_writes_correct_content_to_both_files(self, tmp_path):
        device_enum = DeviceEnum.Device.MOBILE

        clicks_row = MagicMock()
        clicks_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        clicks_row.segments.device = device_enum
        clicks_row.metrics.clicks = 200

        conv_row = MagicMock()
        conv_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        conv_row.segments.conversion_action_name = "Purchase"
        conv_row.segments.device = device_enum
        conv_row.metrics.all_conversions = 6.0

        client, ads_service = make_ads_client()
        clicks_batch, conv_batch = MagicMock(), MagicMock()
        clicks_batch.results = [clicks_row]
        conv_batch.results = [conv_row]
        ads_service.search_stream.side_effect = [[clicks_batch], [conv_batch]]

        clicks_file = tmp_path / "device_clicks.csv"
        conv_file = tmp_path / "device_conv.csv"

        with patch("utils.ads_reporting.Path", side_effect=[clicks_file, conv_file]), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics:
            generate_device_clicks_and_conversion_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 2)

        with open(clicks_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Device": "Mobile phones",
                "Clicks": "200",
            }]

        with open(conv_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Conversion action": "Purchase",
                "Device": "Mobile phones",
                "All conv.": "6.00",
            }]


# ---------------------------------------------------------------------------
# generate_loc_clicks_and_conversion_report
# ---------------------------------------------------------------------------

class TestGenerateLocClicksAndConversionReport:
    def test_writes_correct_content_to_both_files(self, tmp_path):
        CRITERION_ID = 2840

        clicks_row = MagicMock()
        clicks_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        clicks_row.geographic_view.country_criterion_id = CRITERION_ID
        clicks_row.metrics.clicks = 30

        conv_row = MagicMock()
        conv_row.campaign.name = "Course - MLx - USA - Exact - Experiment"
        conv_row.segments.conversion_action_name = "Purchase"
        conv_row.geographic_view.country_criterion_id = CRITERION_ID
        conv_row.metrics.all_conversions = 1.5

        client, ads_service = make_ads_client()
        clicks_batch, conv_batch = MagicMock(), MagicMock()
        clicks_batch.results = [clicks_row]
        conv_batch.results = [conv_row]
        ads_service.search_stream.side_effect = [[clicks_batch], [conv_batch]]

        clicks_file = tmp_path / "loc_clicks.csv"
        conv_file = tmp_path / "loc_conv.csv"

        with patch("utils.ads_reporting.Path", side_effect=[clicks_file, conv_file]), \
             patch("utils.ads_reporting.google_ads_metrics_client") as mock_metrics, \
             patch("utils.ads_reporting.get_location_resource_names_for_countries",
                   return_value={"United States": f"geoTargetConstants/{CRITERION_ID}"}), \
             patch("utils.report_row_generators.build_location_cache"), \
             patch.dict("utils.google_ads_api.LOCATION_CACHE", {CRITERION_ID: "United States"}):
            generate_loc_clicks_and_conversion_report(client, CUSTOMER_ID, OUTPUT_COURSE, START_DATE, END_DATE)

        mock_metrics.track_google_ads_operation_count.assert_called_once_with("search_stream", 2)

        with open(clicks_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Targeted location": "United States",
                "Clicks": "30",
            }]

        with open(conv_file) as f:
            assert list(csv.DictReader(f)) == [{
                "Campaign": "Course - MLx - USA - Exact - Experiment",
                "Conversion action": "Purchase",
                "Targeted location": "United States",
                "All conv.": "1.50",
            }]
