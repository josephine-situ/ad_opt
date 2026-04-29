from unittest.mock import MagicMock


def make_ads_client(rows=None):
    """Return a (client, ads_service) pair whose search_stream yields one batch containing rows."""
    if rows is None:
        rows = []
    batch = MagicMock()
    batch.results = rows
    ads_service = MagicMock()
    ads_service.search_stream.return_value = [batch]
    client = MagicMock()
    client.get_service.return_value = ads_service
    return client, ads_service
