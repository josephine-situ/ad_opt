
from typing import TypedDict


# Configuration for AD Optimization Project

class CourseConfig(TypedDict):
    start_dates: list[str]
    min_date: str
    budgets: list[float]
    course_title_base: str
    regions: dict[str, list[str]]
    conversion_actions: list[str]
    purchase_actions: list[str]
    match_types: list[str]
    default_daily_budget_micros: int
    budget_change_threshold: float
    cpc_change_threshold: float


COURSE_CONFIG: dict[str, CourseConfig] = {
    'gen_ai': {
        'start_dates': [
            '2024-10-15', '2025-02-10', '2025-09-29', '2026-02-09'
        ],
        'min_date': '2024-11-03',
        'budgets': [362.91],
        # Used to derive course names when pushing to Google Ads.
        'course_title_base': "Course - Generative AI",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        'regions': {
            "USA": ["United States"],
            "A": ["Canada", "United Kingdom", "Australia"],
            "B": ["Germany", "France", "Spain", "Italy"]
        },
        'conversion_actions': [],
        'purchase_actions': [],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,  # Can we use "budgets" or does this need to be separate?
        'budget_change_threshold': 0.5,  # Warn if budget changes by more than 50%
        'cpc_change_threshold': 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    'ml': {
        'start_dates': [
            '2021-04-12', '2021-09-20', '2022-01-24', '2022-04-12', 
            '2024-03-29', '2024-09-16', '2025-02-03', '2025-04-28', 
            '2025-09-29', '2026-02-02', '2026-04-27'
        ],
        'min_date': '2022-01-01',
        'budgets': [353.99],
        'course_title_base': "Program - MLx",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        'regions': {
            "USA": ["United States"],
            "A": ["Canada", "United Kingdom", "Australia"],
            "B": ["Germany", "France", "Spain", "Italy"]
        },
        # TODO: We need to add these for all other accounts too.
        'conversion_actions': ['Purchase', 'Add to cart - MLx - iDimension', 'idimension - account create'],
        'purchase_actions': ['Purchase'],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,  # Can we use "budgets" or does this need to be separate?
        'budget_change_threshold': 0.5,  # Warn if budget changes by more than 50%
        'cpc_change_threshold': 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    'sys_eng': {
        'start_dates': [
            '2022-01-24', '2022-04-05', '2022-09-26', '2023-02-06', 
            '2023-04-10', '2023-09-25', '2024-02-05', '2024-04-08', 
            '2024-09-23', '2024-02-03', '2025-04-07', '2025-09-29', 
            '2026-02-02', '2026-04-06'
        ],
        'min_date': '2022-06-01', # Some campaigns name without a region early on
        'budgets': [847.46],
        'course_title_base': "Program - SysEng",
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        'regions': {
            "USA": ["United States"],
            "A": ["Canada", "United Kingdom", "Australia"],
            "B": ["Germany", "France", "Spain", "Italy"]
        },
        'conversion_actions': [],
        'purchase_actions': [],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,   # Can we use "budgets" or does this need to be separate?
        'budget_change_threshold': 0.5,  # Warn if budget changes by more than 50%
        'cpc_change_threshold': 0.5,  # Warn if keyword CPC changes by more than 50%
    },
    'sys_think': {
        'start_dates': [
            '2021-01-25', '2021-04-05', '2021-10-04', '2022-01-31',
            '2022-04-25', '2022-10-03', '2023-01-30', '2023-04-10',
            '2023-10-02', '2024-02-05', '2024-04-08', '2024-10-07',
            '2025-02-10', '2025-04-14', '2025-10-06', '2026-02-02',
            '2026-04-06'
        ],
        'min_date': '2022-06-01', # Start of search history
        'budgets': [357.14],
        'course_title_base': "Course - System Thinking",
        'conversion_actions': ['Purchase', 'System Thinking - Add to cart', 'idimension - account create', 'Add to cart - iDimension'],
        'purchase_actions': ['Purchase'],
        # Regions is not reflective of production data and is only used for create_campaign_for_course.py
        'regions': {
            "USA": ["United States"],
            "A": ["Canada", "United Kingdom", "Australia"],
            "B": ["Germany", "France", "Spain", "Italy"]
        },
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,   # Can we use "budgets" or does this need to be separate?
        'budget_change_threshold': 0.5,  # Warn if budget changes by more than 50%
        'cpc_change_threshold': 0.5,  # Warn if keyword CPC changes by more than 50%
    }
}