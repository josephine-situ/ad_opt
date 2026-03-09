GET_CAMPAIGNS_IN_ACCOUNT = """
    SELECT campaign.id, campaign.name
    FROM campaign
    WHERE campaign.advertising_channel_type = 'SEARCH'
"""

GET_CAMPAIGN_BUDGET_FOR_CAMPAIGN_NAME = """
    SELECT
        campaign.campaign_budget
    FROM campaign
    WHERE campaign.name = '{campaign_name}'
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
            ad_group_criterion.status
        FROM ad_group_criterion
        WHERE ad_group_criterion.ad_group IN ('{ad_group_list}')
        AND ad_group_criterion.type = 'KEYWORD'
    """
SELECT_AD_GROUPS_FOR_CAMPAIGNS = """
        SELECT
            campaign.name,
            ad_group.id,
            ad_group.name
        FROM ad_group
        WHERE campaign.name IN ('{campaign_list}')
        AND campaign.advertising_channel_type = 'SEARCH'
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

GET_CAMPAIGNS_FOR_COURSE = """
        SELECT campaign.id, campaign.name
        FROM campaign
        WHERE campaign.name LIKE 'Course - {course_title}%'
        AND campaign.advertising_channel_type = 'SEARCH'
    """

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
        search_term_view.search_term,
        segments.search_term_match_type,
        campaign.name,
        metrics.clicks,
        metrics.conversions_value,
        customer.currency_code,
        metrics.cost_micros
    FROM search_term_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name LIKE 'Course - {course_title}%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY segments.date
"""

PURCHASE_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.conversion_action_name,
        metrics.conversions
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name LIKE 'Course - {course_title}%'
    AND metrics.conversions > 0
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, segments.conversion_action_name
"""

LOCATION_REPORT_QUERY = """
    SELECT
        geographic_view.location_type,
        geographic_view.country_criterion_id,
        campaign.name,
        metrics.clicks,
        customer.currency_code,
        metrics.cost_micros,
        metrics.conversions,
        metrics.conversions_value,
        campaign.advertising_channel_type
    FROM geographic_view
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name LIKE 'Course - {course_title}%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, geographic_view.location_type
"""

HOD_CLICKS_REPORT_QUERY = """
    SELECT
        campaign.name,
        segments.hour,
        metrics.clicks
    FROM campaign
    WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
    AND campaign.name LIKE 'Course - {course_title}%'
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
    AND campaign.name LIKE 'Course - {course_title}%'
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
    AND campaign.name LIKE 'Course - {course_title}%'
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
    AND campaign.name LIKE 'Course - {course_title}%'
    AND campaign.advertising_channel_type = 'SEARCH'
    ORDER BY campaign.name, geographic_view.location_type
"""