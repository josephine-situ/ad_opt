from unittest.mock import MagicMock


def make_row(attrs: dict) -> MagicMock:
    """Create a MagicMock row with dotted-path attributes pre-set.

    Example: make_row({"campaign.name": "Foo", "metrics.clicks": 10})
    yields a mock where mock.campaign.name == "Foo" and mock.metrics.clicks == 10.
    """
    row = MagicMock()
    for path, value in attrs.items():
        obj = row
        parts = path.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return row


def make_stream(rows=None):
    """Return a stream (list of one batch) containing the given rows."""
    if rows is None:
        rows = []
    batch = MagicMock()
    batch.results = rows
    return [batch]


def make_ads_client(rows=None):
    """Return a (client, ads_service) pair whose search_stream yields one batch containing rows."""
    ads_service = MagicMock()
    ads_service.search_stream.return_value = make_stream(rows)
    client = MagicMock()
    client.get_service.return_value = ads_service
    return client, ads_service
