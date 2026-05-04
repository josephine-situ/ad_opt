import io
from datetime import date
from unittest.mock import patch

import pytest

from config import COURSE_CONFIG
from scripts.push_output_data import (
    compute_daily_spend,
    generate_campaign_names_for_configured_course,
    warn_on_large_budget_changes,
    warn_on_large_cpc_changes,
)
from utils.name_generation import construct_campaign_name_for_args


# ---------------------------------------------------------------------------
# warn_on_large_cpc_changes
# ---------------------------------------------------------------------------

class TestWarnOnLargeCpcChanges:
    def test_prints_warning_when_change_exceeds_threshold(self, capsys):
        # current=$1.00, new=$2.00 => abs change=$1.00, threshold=$0.50 => warn
        new_bids = {(1, "machine learning", "EXACT"): 2.00}
        current_lookup = {(1, "machine learning", "EXACT"): (1, "machine learning", 1_000_000)}

        warn_on_large_cpc_changes(new_bids, current_lookup, threshold=0.50)

        assert "WARNING: Large CPC change detected for keyword 'machine learning' (EXACT) in ad group 1:" in capsys.readouterr().out

    def test_no_warning_when_change_is_within_threshold(self, capsys):
        # current=$1.00, new=$1.10 => abs change=$0.10, threshold=$0.50 => no warn
        new_bids = {(1, "machine learning", "EXACT"): 1.10}
        current_lookup = {(1, "machine learning", "EXACT"): (1, "machine learning", 1_000_000)}

        warn_on_large_cpc_changes(new_bids, current_lookup, threshold=0.50)

        assert capsys.readouterr().out == ""

    def test_skips_keywords_not_in_current_lookup(self, capsys):
        warn_on_large_cpc_changes(
            {(1, "new keyword", "BROAD"): 5.00},
            {},
            threshold=0.1,
        )
        assert capsys.readouterr().out == ""

    def test_skips_keywords_with_zero_current_bid(self, capsys):
        warn_on_large_cpc_changes(
            {(1, "kw", "EXACT"): 2.00},
            {(1, "kw", "EXACT"): (1, "kw", 0)},
            threshold=0.1,
        )
        assert capsys.readouterr().out == ""

    def test_warning_message_shows_absolute_dollar_threshold(self, capsys):
        # current=$1.00, new=$3.00 => abs change=$2.00, threshold=$1.50
        new_bids = {(1, "kw", "EXACT"): 3.00}
        current_lookup = {(1, "kw", "EXACT"): (1, "kw", 1_000_000)}

        warn_on_large_cpc_changes(new_bids, current_lookup, threshold=1.50)

        out = capsys.readouterr().out
        assert "$2.00" in out
        assert "$1.50" in out


# ---------------------------------------------------------------------------
# warn_on_large_budget_changes
# ---------------------------------------------------------------------------

class TestWarnOnLargeBudgetChanges:
    def test_prints_warning_when_change_exceeds_threshold(self, capsys):
        # current=$100, new=$200 => abs change=$100, threshold=$50 => warn
        warn_on_large_budget_changes(
            {"Campaign A": 200.00},
            {"Campaign A": {"current_budget_amount": 100.00}},
            threshold=50.00,
        )
        assert "WARNING" in capsys.readouterr().out

    def test_no_warning_when_change_is_within_threshold(self, capsys):
        # current=$100, new=$110 => abs change=$10, threshold=$50 => no warn
        warn_on_large_budget_changes(
            {"Campaign A": 110.00},
            {"Campaign A": {"current_budget_amount": 100.00}},
            threshold=50.00,
        )
        assert capsys.readouterr().out == ""

    def test_skips_campaigns_with_zero_current_budget(self, capsys):
        warn_on_large_budget_changes(
            {"Campaign A": 50.00},
            {"Campaign A": {"current_budget_amount": 0}},
            threshold=1.00,
        )
        assert capsys.readouterr().out == ""

    def test_warning_message_shows_absolute_dollar_threshold(self, capsys):
        # current=$100, new=$250 => abs change=$150, threshold=$75
        warn_on_large_budget_changes(
            {"Campaign A": 250.00},
            {"Campaign A": {"current_budget_amount": 100.00}},
            threshold=75.00,
        )
        out = capsys.readouterr().out
        assert "$150.00" in out
        assert "$75.00" in out


# ---------------------------------------------------------------------------
# generate_campaign_names_for_configured_course
# ---------------------------------------------------------------------------

class TestGenerateCampaignNamesForConfiguredCourse:
    def test_returns_one_name_per_region_and_match_type(self):
        course = "ml"
        config = COURSE_CONFIG[course]
        expected_count = len(config["regions"]) * len(config["match_types"])

        names = generate_campaign_names_for_configured_course(course)

        assert len(names) == expected_count

    def test_names_match_construct_campaign_name_helper(self):
        course = "ml"
        config = COURSE_CONFIG[course]
        expected = {
            construct_campaign_name_for_args(course, match_type, region)
            for region in config["regions"]
            for match_type in config["match_types"]
        }

        assert generate_campaign_names_for_configured_course(course) == expected
