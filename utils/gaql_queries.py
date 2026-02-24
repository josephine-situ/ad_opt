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

GET_AD_GROUP_FOR_CAMPAIGN = """
    SELECT
        ad_group.id,
        ad_group.name
    FROM ad_group
    WHERE campaign.name = '{campaign_name}'
    AND ad_group.status != 'REMOVED'
    LIMIT 1
"""

GET_KEYWORD_CRITERION_IN_AD_GROUP = """
    SELECT
        ad_group_criterion.criterion_id,
        ad_group_criterion.keyword.text,
        ad_group_criterion.keyword.match_type,
        ad_group_criterion.status
    FROM ad_group_criterion
    WHERE ad_group_criterion.ad_group = 'customers/{customer_id}/adGroups/{ad_group_id}'
    AND ad_group_criterion.type = 'KEYWORD'
    AND ad_group_criterion.keyword.text = '{keyword_text}'
    AND ad_group_criterion.keyword.match_type = {match_type_enum}
"""

SELECT_KEYWORD_CRITERION_IN_AD_GROUP = """
        SELECT
            ad_group.id,
            ad_group_criterion.criterion_id,
            ad_group_criterion.keyword.text,
            ad_group_criterion.keyword.match_type,
            ad_group_criterion.status
        FROM ad_group_criterion
        WHERE ad_group_criterion.ad_group IN ('{ad_group_list}')
        AND ad_group_criterion.type = 'KEYWORD'
        AND ad_group_criterion.status != 'REMOVED'
    """
SELECT_AD_GROUPS_FOR_CAMPAIGNS = """
        SELECT
            campaign.name,
            ad_group.id,
            ad_group.name
        FROM ad_group
        WHERE campaign.name IN ('{campaign_list}')
        AND ad_group.status != 'REMOVED'
    """
