from config import COURSE_CONFIG


def construct_campaign_name_for_args(course, match_type, region):
    """Construct campaign name based on course, match type and region."""
    return f"{COURSE_CONFIG[course]['course_title_base']} - {region} - {match_type.split()[0]} - Experiment"

def construct_ad_group_name_for_args(course, match_type, region):
    """Construct ad group name based on course, match type and region."""
    return f"Ad Group - {COURSE_CONFIG[course]['course_title_base']} - {region} - {match_type.split()[0]} - Experiment"

def construct_budget_name_for_args(course, match_type, region):
    """Construct budget name based on course, match type and region."""
    return f"Budget - {COURSE_CONFIG[course]['course_title_base']} - {region} - {match_type.split()[0]} - Experiment"