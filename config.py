
# Configuration for AD Optimization Project

COURSE_CONFIG = {
    'gen_ai': {
        'start_dates': [
            '2024-10-15', '2025-02-10', '2025-09-29', '2026-02-09'
        ],
        'min_date': '2024-11-03',
        'budgets': [362.91],
        # Used to derive course names when pushing to Google Ads.
        # TODO: Identify the others as well and add to config.
        'course_title_base': "Generative AI",
        'regions': ["USA", "A", "B"],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 100_000_000,  # Can we use "budgets" or does this need to be separate?
    },
    'ml': {
        'start_dates': [
            '2021-04-12', '2021-09-20', '2022-01-24', '2022-04-12', 
            '2024-03-29', '2024-09-16', '2025-02-03', '2025-04-28', 
            '2025-09-29', '2026-02-02', '2026-04-27'
        ],
        'min_date': '2022-01-01',
        'budgets': [353.99],
        'regions': ["USA", "A", "B"],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,  # Can we use "budgets" or does this need to be separate?
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
        'regions': ["USA", "A", "B"],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,   # Can we use "budgets" or does this need to be separate?
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
        'regions': ["USA", "A", "B"],
        'match_types': ["Exact", "Phrase", "Broad"],
        'default_daily_budget_micros': 1_000_000,   # Can we use "budgets" or does this need to be separate?
    }
}
