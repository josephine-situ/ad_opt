from typing import TypedDict

# Configuration for AD Optimization Project


class CourseConfig(TypedDict):
    start_dates: list[str]
    min_date: str
    campaign_budget: float
    current_campaign_start_date: str
    current_campaign_end_date: str
    course_title_base: str
    regions: dict[str, list[str]]
    conversion_actions: list[str]
    purchase_actions: list[str]
    match_types: list[str]
    default_daily_budget_micros: int
    budget_change_threshold: float
    cpc_change_threshold: float


REGION_CONFIG = {
    "ml": {
        "USA": ["United States"],
        "A": [
            "Japan",
            "Italy",
            "United Kingdom",
            "Belgium",
            "Canada",
            "Germany",
            "Switzerland",
            "France",
            "Denmark",
            "Spain",
            "Netherlands",
            "Sweden",
            "South Korea",
            "Israel",
            "Australia",
            "Norway",
            "Mexico",
            "Luxembourg",
        ],
        "B": [
            "Bolivia",
            "Portugal",
            "Argentina",
            "Saudi Arabia",
            "Austria",
            "Singapore",
            "Poland",
            "Morocco",
            "Ghana",
            "Russia",
            "China",
            "Chile",
            "Brazil",
            "Slovakia",
            "Iceland",
            "Peru",
            "Greece",
            "Colombia",
            "Thailand",
            "Hong Kong",
            "South Africa",
            "Qatar",
            "Croatia",
            "United Arab Emirates",
            "Taiwan",
            "Indonesia",
            "Ecuador",
        ],
    },
    "sys_think": {
        "USA": ["United States"],
        "A": [
            "France",
            "Switzerland",
            "Sweden",
            "Canada",
            "New Zealand",
            "Netherlands",
            "United Kingdom",
            "Japan",
            "Spain",
            "Denmark",
            "Australia",
            "Ireland",
            "Germany",
            "Norway",
            "Belgium",
            "Mexico",
            "Italy",
        ],
        "B": [
            "Indonesia",
            "Philippines",
            "Uganda",
            "Morocco",
            "Tanzania",
            "Zimbabwe",
            "Tunisia",
            "Sri Lanka",
            "Liberia",
            "Thailand",
            "Turkiye",
            "Cote d'Ivoire",
            "Peru",
            "Jordan",
            "Argentina",
            "Senegal",
            "Saudi Arabia",
            "Malaysia",
            "Singapore",
            "Colombia",
            "Chile",
            "Armenia",
            "China",
            "Georgia",
            "Hong Kong",
            "Israel",
            "Romania",
            "Bulgaria",
            "Trinidad and Tobago",
            "Serbia",
            "Poland",
            "Lithuania",
            "Greece",
            "Qatar",
            "Bolivia",
            "Portugal",
            "Bahrain",
            "Paraguay",
            "Austria",
            "Hungary",
            "Moldova",
            "United Arab Emirates",
            "Czechia",
            "South Korea",
            "Taiwan",
            "Croatia",
            "Estonia",
            "Iceland",
            "Slovakia",
            "Finland",
            "Luxembourg",
            "Monaco",
        ],
    },
}


COURSE_CONFIG: dict[str, CourseConfig] = {
    "gen_ai": {
        "start_dates": ["2024-10-15", "2025-02-10", "2025-09-29", "2026-02-09"],
        "min_date": "2024-11-03",
        "campaign_budget": 10000.0,  # TODO: update campaign config with actuals
        "current_campaign_start_date": "2026-02-09",
        "current_campaign_end_date": "2026-04-15",
        # Used to derive course names when pushing to Google Ads.
        "course_title_base": "Course - Generative AI",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        "regions": REGION_CONFIG["ml"],
        "conversion_actions": [
            "Purchase - Gen AI",
            "Add to Cart - Gen AI",
            "idimension - account create",
        ],
        "purchase_actions": ["Purchase - Gen AI"],
        "match_types": ["Exact", "Phrase", "Broad"],
        "default_daily_budget_micros": 1_000_000,  # Can we use "budgets" or does this need to be separate?
        "budget_change_threshold": 0.5,  # Warn if budget changes by more than 50%
        "cpc_change_threshold": 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    "ml": {
        "start_dates": [
            "2021-04-12",
            "2021-09-20",
            "2022-01-24",
            "2022-04-12",
            "2024-03-29",
            "2024-09-16",
            "2025-02-03",
            "2025-04-28",
            "2025-09-29",
            "2026-02-02",
            "2026-04-27",
        ],
        "min_date": "2022-07-01",
        "campaign_budget": 15000.0,  # TODO: update campaign config with actuals
        "current_campaign_start_date": "2026-02-03",
        "current_campaign_end_date": "2026-04-27",
        "course_title_base": "Program - MLx",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        "regions": REGION_CONFIG["ml"],
        # TODO: We need to add these for all other accounts too.
        "conversion_actions": [
            "Purchase",
            "Add to cart - MLx - iDimension",
            "idimension - account create",
        ],
        "purchase_actions": ["Purchase"],
        "match_types": ["Exact", "Phrase", "Broad"],
        "default_daily_budget_micros": 1_000_000,  # Can we use "budgets" or does this need to be separate?
        "budget_change_threshold": 0.5,  # Warn if budget changes by more than 50%
        "cpc_change_threshold": 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    "sys_eng": {
        "start_dates": [
            "2022-01-24",
            "2022-04-05",
            "2022-09-26",
            "2023-02-06",
            "2023-04-10",
            "2023-09-25",
            "2024-02-05",
            "2024-04-08",
            "2024-09-23",
            "2024-02-03",
            "2025-04-07",
            "2025-09-29",
            "2026-02-02",
            "2026-04-06",
        ],
        "min_date": "2022-07-01",  # Some campaigns name without a region early on
        "campaign_budget": 20000.0,  # TODO: update campaign config with actuals
        "current_campaign_start_date": "2026-02-02",
        "current_campaign_end_date": "2026-04-15",
        "course_title_base": "Program - SysEng",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        "regions": REGION_CONFIG["ml"],
        "conversion_actions": [
            "Purchase",
            "SysEng - Add to cart - iDimension",
            "idimension - account create",
        ],
        "purchase_actions": ["Purchase"],
        "match_types": ["Exact", "Phrase", "Broad"],
        "default_daily_budget_micros": 1_000_000,  # Can we use "budgets" or does this need to be separate?
        "budget_change_threshold": 0.5,  # Warn if budget changes by more than 50%
        "cpc_change_threshold": 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    "sys_think": {
        "start_dates": [
            "2021-01-25",
            "2021-04-05",
            "2021-10-04",
            "2022-01-31",
            "2022-04-25",
            "2022-10-03",
            "2023-01-30",
            "2023-04-10",
            "2023-10-02",
            "2024-02-05",
            "2024-04-08",
            "2024-10-07",
            "2025-02-10",
            "2025-04-14",
            "2025-10-06",
            "2026-02-02",
            "2026-04-06",
            "2026-06-15",
        ],
        "min_date": "2022-07-01",  # Start of search history
        "campaign_budget": 40000.0,
        "current_campaign_start_date": "2026-04-28",
        "current_campaign_end_date": "2026-06-15",
        "course_title_base": "Course - System Thinking",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        "regions": REGION_CONFIG["sys_think"],
        "conversion_actions": [
            "Purchase",
            "System Thinking - Add to cart",
            "idimension - account create",
            "Add to cart - iDimension",
        ],
        "purchase_actions": ["Purchase"],
        "match_types": ["Exact", "Phrase", "Broad"],
        "default_daily_budget_micros": 1_000_000,  # Can we use "budgets" or does this need to be separate?
        "budget_change_threshold": 0.5,  # Warn if budget changes by more than 50%
        "cpc_change_threshold": 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    "quant_comp": {
        "start_dates": [],  # TODO: add start dates
        "min_date": "2022-07-01",  # TODO: configure min date
        "campaign_budget": 10000.0,  # TODO: update campaign config with actuals
        "current_campaign_start_date": "2026-02-02",
        "current_campaign_end_date": "2026-04-15",
        "course_title_base": "Course - Quantum Computing",
        "regions": REGION_CONFIG["ml"],
        "conversion_actions": [
            "idimension - Purchase",
            "idimension - QCX = Add to cart",
            "idimension - account create",
        ],
        "purchase_actions": ["idimension - Purchase"],
        "match_types": ["Exact", "Phrase", "Broad"],
        "default_daily_budget_micros": 1_000_000,
        "budget_change_threshold": 0.5,  # Warn if budget changes by more than 50%
        "cpc_change_threshold": 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    "dai": {
        "start_dates": [],  # TODO: add start dates
        "min_date": "2022-07-01",  # TODO: configure min date
        "campaign_budget": 10000.0,  # TODO: update campaign config with actuals
        "current_campaign_start_date": "2026-02-02",
        "current_campaign_end_date": "2026-04-15",
        "course_title_base": "Course - Deploying AI",
        "conversion_actions": [
            "Purchase",
            "Add to Cart",
            "idimension - account create",
            "Account Creation",
        ],
        "purchase_actions": ["Purchase"],
        "regions": REGION_CONFIG["ml"],
        "match_types": ["Exact", "Phrase", "Broad"],
        "default_daily_budget_micros": 1_000_000,
        "budget_change_threshold": 0.5,  # Warn if budget changes by more than 50%
        "cpc_change_threshold": 0.5,  # Warn if keyword CPC changes by more than 50%
    },
}
