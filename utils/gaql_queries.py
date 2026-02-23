RAW_INPUT_TO_MODELS_QUERY = """
    SELECT
        segments.date,
        search_term_view.search_term,
        segments.search_term_match_type,
        campaign.name,
        metrics.clicks,
        metrics.conversions_value,
        customer.currency_code,
        metrics.cost_micros
    FROM search_term_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY segments.date
"""

GET_CAMPAIGNS_IN_ACCOUNT = """
SELECT campaign.id, campaign.name
FROM campaign
WHERE campaign.status != 'REMOVED'
"""

GET_CAMPAIGN_BUDGET_FOR_CAMPAIGN_NAME = """
    SELECT
        campaign.campaign_budget
    FROM campaign
    WHERE campaign.name = '{campaign_name}'
    AND campaign.status != 'REMOVED'
"""