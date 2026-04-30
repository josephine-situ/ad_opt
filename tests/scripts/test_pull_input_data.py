import csv
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from scripts.pull_input_data import (
    _gkp_month_header_sort_key,
    generate_rows_from_gkp_response,
    pull_semrush,
    validate_environment_variables,
)


def _make_monthly_volume(month_name, year, searches):
    v = MagicMock()
    v.month.name = month_name
    v.year = year
    v.monthly_searches = searches
    return v


def _make_gkp_result(keyword, avg_monthly=100, competition="HIGH", competition_index=80,
                     low_bid_micros=500_000, high_bid_micros=1_000_000, monthly_volumes=None):
    result = MagicMock()
    result.text = keyword
    m = result.keyword_metrics
    m.avg_monthly_searches = avg_monthly
    m.competition.name = competition
    m.competition_index = competition_index
    m.low_top_of_page_bid_micros = low_bid_micros
    m.high_top_of_page_bid_micros = high_bid_micros
    m.monthly_search_volumes = monthly_volumes or []
    return result


# ---------------------------------------------------------------------------
# _gkp_month_header_sort_key
# ---------------------------------------------------------------------------

class TestGkpMonthHeaderSortKey:
    def test_parses_month_and_year(self):
        assert _gkp_month_header_sort_key("Searches: Jan 2024") == (2024, 1)

    def test_december_sorts_last_in_year(self):
        assert _gkp_month_header_sort_key("Searches: Dec 2023") == (2023, 12)

    def test_sorts_chronologically(self):
        headers = ["Searches: Mar 2024", "Searches: Jan 2023", "Searches: Nov 2023"]
        assert sorted(headers, key=_gkp_month_header_sort_key) == [
            "Searches: Jan 2023",
            "Searches: Nov 2023",
            "Searches: Mar 2024",
        ]


# ---------------------------------------------------------------------------
# validate_environment_variables
# ---------------------------------------------------------------------------

class TestValidateEnvironmentVariables:
    def test_returns_true_when_semrush_key_is_set(self):
        with patch.dict(os.environ, {"SEMRUSH_API_KEY": "abc123"}):
            assert validate_environment_variables(["semrush"]) is True

    def test_exits_when_semrush_key_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                validate_environment_variables(["semrush"])

    def test_returns_true_for_non_semrush_datasets_without_env_vars(self):
        assert validate_environment_variables(["ads_reports"]) is True


# ---------------------------------------------------------------------------
# generate_rows_from_gkp_response
# ---------------------------------------------------------------------------

class TestGenerateRowsFromGkpResponse:
    def test_maps_result_fields_to_row(self):
        response = MagicMock()
        response.results = [_make_gkp_result("machine learning", avg_monthly=500,
                                              low_bid_micros=250_000, high_bid_micros=750_000)]

        rows, headers = generate_rows_from_gkp_response(response)

        assert len(rows) == 1
        assert rows[0]["Keyword"] == "machine learning"
        assert rows[0]["Avg. monthly searches"] == 500
        assert rows[0]["Competition"] == "High"
        assert rows[0]["Top of page bid (low range)"] == Decimal("0.25")
        assert rows[0]["Top of page bid (high range)"] == Decimal("0.75")

    def test_empty_fields_when_metrics_are_falsy(self):
        result = _make_gkp_result("ml", avg_monthly=0, competition_index=0,
                                   low_bid_micros=0, high_bid_micros=0)
        result.keyword_metrics.competition = None
        response = MagicMock()
        response.results = [result]

        rows, _ = generate_rows_from_gkp_response(response)

        assert rows[0]["Avg. monthly searches"] == ""
        assert rows[0]["Competition"] == ""
        assert rows[0]["Competition (indexed value)"] == ""
        assert rows[0]["Top of page bid (low range)"] == ""
        assert rows[0]["Top of page bid (high range)"] == ""

    def test_adds_monthly_volume_columns(self):
        response = MagicMock()
        response.results = [_make_gkp_result("ml", monthly_volumes=[
            _make_monthly_volume("JANUARY", 2024, 1200),
        ])]

        rows, headers = generate_rows_from_gkp_response(response)

        assert "Searches: Jan 2024" in headers
        assert rows[0]["Searches: Jan 2024"] == 1200

    def test_monthly_headers_are_sorted_chronologically(self):
        response = MagicMock()
        response.results = [_make_gkp_result("ml", monthly_volumes=[
            _make_monthly_volume("MARCH", 2024, 300),
            _make_monthly_volume("JANUARY", 2023, 100),
            _make_monthly_volume("NOVEMBER", 2023, 200),
        ])]

        _, headers = generate_rows_from_gkp_response(response)

        assert headers == ["Searches: Jan 2023", "Searches: Nov 2023", "Searches: Mar 2024"]


# ---------------------------------------------------------------------------
# pull_semrush
# ---------------------------------------------------------------------------

class TestPullSemrush:
    def test_skips_header_row_and_writes_remaining_keywords(self, tmp_path):
        mock_response = MagicMock()
        mock_response.text = "Keyword\nmachine learning\nml online"

        with patch("scripts.pull_input_data.requests.get", return_value=mock_response), \
             patch("scripts.pull_input_data.Path", return_value=tmp_path), \
             patch.dict(os.environ, {"SEMRUSH_API_KEY": "test-key"}):
            pull_semrush("ml", num_keywords=2)

        with open(tmp_path / "semrush_new_kws.csv") as f:
            rows = list(csv.DictReader(f))

        assert [r["Keyword"] for r in rows] == ["machine learning", "ml online"]
