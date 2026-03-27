GET_CAMPAIGNS_IN_ACCOUNT = """
    SELECT campaign.id, campaign.name
    FROM campaign
    WHERE campaign.advertising_channel_type = 'SEARCH'
"""

GET_CAMPAIGN_BUDGETS_BY_NAMES = """
    SELECT
        campaign.name,
        campaign.campaign_budget,
        campaign_budget.amount_micros
    FROM campaign
    WHERE campaign.name IN ('{campaign_names}')
    AND campaign.advertising_channel_type = 'SEARCH'
"""

GET_AD_GROUP_FOR_CAMPAIGN = """
    SELECT
        ad_group.id,
        ad_group.name
    FROM ad_group
    WHERE campaign.name = '{campaign_name}'
    AND campaign.advertising_channel_type = 'SEARCH'
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
            ad_group_criterion.status,
            ad_group_criterion.cpc_bid_micros
        FROM ad_group_criterion
        WHERE ad_group_criterion.ad_group IN ('{ad_group_list}')
        AND ad_group_criterion.type = 'KEYWORD'
    """

SELECT_AD_GROUPS_FOR_ENABLED_CAMPAIGNS = """
        SELECT
            campaign.name,
            ad_group.id,
            ad_group.name
        FROM ad_group
        WHERE campaign.name IN ('{campaign_list}')
        AND campaign.advertising_channel_type = 'SEARCH'
        AND campaign.status = 'ENABLED'
    """

GET_CRITERIA_FOR_CAMPAIGNS = """
        SELECT
            campaign_criterion.campaign,
            campaign_criterion.criterion_id,
            campaign_criterion.bid_modifier,
            campaign_criterion.type,
            campaign_criterion.device.type,
            campaign_criterion.ad_schedule.day_of_week,
            campaign_criterion.ad_schedule.start_hour,
            campaign_criterion.ad_schedule.end_hour,
            campaign_criterion.location.geo_target_constant
        FROM campaign_criterion
        WHERE campaign_criterion.campaign IN ('{campaign_id_list}')
        AND campaign.advertising_channel_type = 'SEARCH'
    """

GET_ENABLED_CAMPAIGNS_FOR_COURSE = """
        SELECT campaign.id, campaign.name
        FROM campaign
        WHERE campaign.name NOT LIKE 'EXCLUDE%'
        AND campaign.advertising_channel_type = 'SEARCH'
        AND campaign.status = 'ENABLED'
    """

# TODO: It only appears that you have age range criteria provisioned if you've set a bid adjustment for age ranges once.
# As soon as I set it, even to 0, they show up, but before that they don't appear to exist at all.
# We'll need to address this somehow.
GET_AGE_CRITERIA_FOR_CAMPAIGNS = """
        SELECT
            campaign.id,
            ad_group.id,
            ad_group_criterion.criterion_id,
            ad_group_criterion.age_range.type
        FROM ad_group_criterion
        WHERE campaign.id IN ({campaign_ids})
        AND ad_group_criterion.type = 'AGE_RANGE'
        AND campaign.advertising_channel_type = 'SEARCH'
    """

# Report generation queries
SEARCH_KEYWORD_REPORT_QUERY = """
    SELECT
        segments.date,
        ad_group_criterion.keyword.text,
        ad_group_criterion.keyword.match_type,
        campaign.name,
        metrics.clicks,
        metrics.all_conversions_value,
        customer.currency_code,
        metrics.cost_micros
    FROM keyword_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND ad_group_criterion.keyword.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    AND metrics.clicks > 0
    ORDER BY segments.date
"""

PURCHASE_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND metrics.all_conversions > 0
    AND campaign.advertising_channel_type = 'SEARCH'
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name
"""

HOD_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.hour,
        metrics.clicks
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, segments.hour
"""

AGE_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        ad_group_criterion.age_range.type,
        metrics.clicks
    FROM age_range_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, ad_group_criterion.age_range.type
"""

DEVICE_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.device,
        metrics.clicks
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, segments.device
"""

LOC_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        geographic_view.location_type,
        geographic_view.country_criterion_id,
        metrics.clicks,
        campaign.advertising_channel_type
    FROM geographic_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, geographic_view.location_type
"""

HOD_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        segments.hour,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, segments.hour
"""

AGE_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        ad_group_criterion.age_range.type,
        metrics.all_conversions
    FROM age_range_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, ad_group_criterion.age_range.type
"""

DEVICE_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        segments.device,
        metrics.all_conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, segments.device
"""

LOC_CONVERSIONS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        geographic_view.location_type,
        geographic_view.country_criterion_id,
        metrics.all_conversions,
        campaign.advertising_channel_type
    FROM geographic_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name NOT LIKE 'EXCLUDE%'
    AND campaign.advertising_channel_type = 'SEARCH'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{purchase_action_list}')
    ORDER BY campaign.name, segments.conversion_action_name, geographic_view.location_type
"""

SEARCH_TERM_REPORT_QUERY = """
    SELECT
        search_term_view.search_term,
        segments.keyword.info.text,
        segments.keyword.info.match_type,
        segments.conversion_action_name,
        metrics.all_conversions
    FROM search_term_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND metrics.all_conversions > 0
    AND segments.conversion_action_name IN ('{conversion_action_list}')
    AND segments.keyword.info.match_type IN ('EXACT', 'PHRASE', 'BROAD')
    ORDER BY segments.keyword.info.text
"""
