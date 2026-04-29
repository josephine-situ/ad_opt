#!/usr/bin/env python3
"""
Script to query and optionally delete campaign budgets from Google Ads.
Useful for cleaning up test budgets or removing old campaigns.
"""

import argparse
import os
import sys
from pathlib import Path

from google.ads.googleads.client import GoogleAdsClient

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_all_campaign_budgets(google_ads_client, customer_id, name_filter=None):
    """
    Query all campaign budgets, optionally filtering by name pattern.
    
    Args:
        google_ads_client: The Google Ads API client
        customer_id: The customer ID
        name_filter: Optional substring to filter budget names (e.g., "Budget - ")
    
    Returns:
        List of tuples: (resource_name, budget_name, amount_micros, status)
    """
    google_ads_service = google_ads_client.get_service("GoogleAdsService")
    
    query = """
        SELECT 
            campaign_budget.resource_name,
            campaign_budget.name,
            campaign_budget.amount_micros,
            campaign_budget.status
        FROM campaign_budget
        WHERE campaign_budget.status != 'REMOVED'
    """
    
    if name_filter:
        # Escape single quotes in the filter
        escaped_filter = name_filter.replace("'", "\\'")
        query += f" AND campaign_budget.name LIKE '%{escaped_filter}%'"
    
    query += " ORDER BY campaign_budget.name"
    
    response = google_ads_service.search(customer_id=customer_id, query=query)
    
    budgets = []
    for row in response:
        budgets.append((
            row.campaign_budget.resource_name,
            row.campaign_budget.name,
            row.campaign_budget.amount_micros,
            row.campaign_budget.status.name
        ))
    
    return budgets


def delete_campaign_budgets(google_ads_client, customer_id, resource_names):
    """
    Delete campaign budgets by resource name.
    
    Args:
        google_ads_client: The Google Ads API client
        customer_id: The customer ID
        resource_names: List of budget resource names to delete
    
    Returns:
        Number of budgets successfully deleted
    """
    campaign_budget_service = google_ads_client.get_service("CampaignBudgetService")
    
    operations = []
    for resource_name in resource_names:
        operation = google_ads_client.get_type("CampaignBudgetOperation")
        operation.remove = resource_name
        operations.append(operation)
    
    try:
        response = campaign_budget_service.mutate_campaign_budgets(
            customer_id=customer_id, operations=operations
        )
        return len(response.results)
    except Exception as e:
        print(f"Error deleting budgets: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Query and optionally delete campaign budgets from Google Ads"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter budgets by name (substring match, e.g., 'Budget - Generative AI')",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Delete the matching budgets (without this flag, only lists them)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt when deleting",
    )
    
    args = parser.parse_args()
    
    # Validate environment variables
    yaml_path = os.getenv("GOOGLE_ADS_YAML_PATH")
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID")
    
    if not yaml_path:
        print("Error: GOOGLE_ADS_YAML_PATH environment variable not set")
        sys.exit(1)
    
    if not customer_id:
        print("Error: GOOGLE_ADS_CUSTOMER_ID environment variable not set")
        sys.exit(1)
    
    # Initialize Google Ads client
    google_ads_client = GoogleAdsClient.load_from_storage(yaml_path)
    
    # Query budgets
    print("Querying campaign budgets...")
    if args.filter:
        print(f"Filtering by name containing: '{args.filter}'")
    
    budgets = get_all_campaign_budgets(google_ads_client, customer_id, args.filter)
    
    if not budgets:
        print("No budgets found matching the criteria.")
        return
    
    # Display results
    print(f"\nFound {len(budgets)} budget(s):")
    print(f"\n{'Budget Name':<60} {'Amount (USD)':<15} {'Status':<10}")
    print("=" * 85)
    
    for resource_name, name, amount_micros, status in budgets:
        amount_usd = amount_micros / 1_000_000
        print(f"{name:<60} ${amount_usd:<14.2f} {status:<10}")
    
    # Delete if requested
    if args.delete:
        print(f"\n{'='*85}")
        
        if not args.yes:
            confirm = input(f"\nAre you sure you want to delete {len(budgets)} budget(s)? (yes/no): ")
            if confirm.lower() != "yes":
                print("Deletion cancelled.")
                return
        
        print(f"\nDeleting {len(budgets)} budget(s)...")
        resource_names = [b[0] for b in budgets]
        deleted_count = delete_campaign_budgets(google_ads_client, customer_id, resource_names)
        
        if deleted_count > 0:
            print(f"✓ Successfully deleted {deleted_count} budget(s)")
        else:
            print("✗ No budgets were deleted")
    else:
        print("\n💡 To delete these budgets, run with --delete flag")


if __name__ == "__main__":
    main()
