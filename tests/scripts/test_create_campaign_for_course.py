import csv
from unittest.mock import MagicMock, patch

import pytest

from scripts.create_campaign_for_course import (
    CampaignSpec,
    _get_keywords_csv_path,
    create_ad_group_keyword_operations,
    create_ad_schedule_operations,
    create_age_range_criteria_operations,
    create_location_operations,
    find_spec_by_name,
    get_keywords_to_create,
)
from utils.bid_adjustments import AGE_RANGE_MAP

def _make_spec(
    campaign_name="Campaign A",
    ad_group_name="Ad Group A",
    budget_name="Budget A",
    default_budget=1_000_000,
    region_label="USA",
    match_type="EXACT",
    countries=None,
) -> CampaignSpec:
    return CampaignSpec(
        campaign_name=campaign_name,
        ad_group_name=ad_group_name,
        budget_name=budget_name,
        default_budget=default_budget,
        region_label=region_label,
        match_type=match_type,
        countries=countries if countries is not None else ["United States"],
    )


# ---------------------------------------------------------------------------
# find_spec_by_name
# ---------------------------------------------------------------------------

class TestFindSpecByName:
    @pytest.mark.parametrize("spec,search_name,spec_field", [
        (_make_spec(campaign_name="My Campaign"), "My Campaign", "campaign_name"),
        (_make_spec(ad_group_name="My Ad Group"), "My Ad Group", "ad_group_name"),
        (_make_spec(budget_name="My Budget"),     "My Budget",   "budget_name"),
    ])
    def test_finds_by_field(self, spec, search_name, spec_field):
        assert find_spec_by_name([spec], search_name, spec_field) is spec

    def test_returns_none_when_not_found(self):
        assert find_spec_by_name([_make_spec()], "Nonexistent", "campaign_name") is None


# ---------------------------------------------------------------------------
# get_keywords_to_create
# ---------------------------------------------------------------------------

class TestGetKeywordsToCreate:
    def test_reads_keywords_and_groups_by_region_and_match_type(self, tmp_path):
        csv_file = tmp_path / "optimized_costs.csv"
        csv_file.write_text(
            "Region,Match type,Keyword\n"
            "USA,Exact match,machine learning\n"
            "USA,Exact match,ml online\n"
            "USA,Phrase match,learn ml\n"
        )

        with patch("scripts.create_campaign_for_course._get_keywords_csv_path", return_value=csv_file):
            result = get_keywords_to_create("ml")

        assert result[("USA", "EXACT")] == ["machine learning", "ml online"]
        assert result[("USA", "PHRASE")] == ["learn ml"]

    def test_returns_empty_dict_when_file_missing(self):
        result = get_keywords_to_create("__nonexistent_course__")
        assert result == {}


# ---------------------------------------------------------------------------
# create_ad_schedule_operations
# ---------------------------------------------------------------------------

class TestCreateAdScheduleOperations:

    # Tests a few things:
    # - Start hours are aligned on 4-hour windows (0, 4, 8, 12, 16, 20)
    # - Each window is 4 hours long
    # - There are 42 total operations (6 windows × 7 days)
    # - Each operation is associated with the correct campaign resource name
    def test_ad_schedule_ops(self):
        client = MagicMock()
        client.get_type.side_effect = lambda t: MagicMock()
        ops = create_ad_schedule_operations(client, "campaigns/123")
        start_hours = {op.create.ad_schedule.start_hour for op in ops}
        assert start_hours == {0, 4, 8, 12, 16, 20}
        for op in ops:
            sched = op.create.ad_schedule
            assert sched.end_hour == sched.start_hour + 4
        assert len(ops) == 42  # 6 windows × 7 days
        assert all(op.create.campaign == "campaigns/123" for op in ops)



# ---------------------------------------------------------------------------
# create_age_range_criteria_operations
# ---------------------------------------------------------------------------

class TestCreateAgeRangeCriteriaOperations:
    def test_creates_one_operation_per_age_range(self):
        client = MagicMock()
        client.get_type.side_effect = lambda t: MagicMock()
        ops = create_age_range_criteria_operations(client, "customers/1/adGroups/2")
        assert len(ops) == len(AGE_RANGE_MAP)
        assert all(op.create.ad_group == "customers/1/adGroups/2" for op in ops)
        set_types = {op.create.age_range.type_ for op in ops}
        assert set_types == set(AGE_RANGE_MAP.values())


# ---------------------------------------------------------------------------
# create_location_operations
# ---------------------------------------------------------------------------

class TestCreateLocationOperations:
    def test_creates_one_operation_per_country(self):
        client = MagicMock()
        client.get_type.side_effect = lambda t: MagicMock()
        location_map = {"United States": "geoTargetConstants/2840", "Canada": "geoTargetConstants/2124"}
        ops = create_location_operations(client, "campaigns/123", ["United States", "Canada"], location_map)
        assert len(ops) == 2
        assert ops[0].create.location.geo_target_constant == "geoTargetConstants/2840"
        assert ops[0].create.campaign == "campaigns/123"



# ---------------------------------------------------------------------------
# create_ad_group_keyword_operations
# ---------------------------------------------------------------------------

class TestCreateAdGroupKeywordOperations:
    def test_creates_one_operation_per_keyword(self):
        client = MagicMock()
        client.get_type.side_effect = lambda t: MagicMock()
        ops = create_ad_group_keyword_operations(
            client, "customers/1/adGroups/2", ["machine learning", "ml course"], "EXACT"
        )
        assert len(ops) == 2
        assert ops[0].create.keyword.text == "machine learning"
        assert ops[0].create.keyword.match_type == "EXACT"
        assert ops[0].create.ad_group == "customers/1/adGroups/2"

