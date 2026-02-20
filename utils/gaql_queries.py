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