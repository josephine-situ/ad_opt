import pytest
from unittest.mock import MagicMock, patch

from utils.metrics import GoogleAdsMetricsClient, MetricsClient

METRICS_URL = "http://metrics.example.com"
METRICS_USERNAME = "user"
METRICS_TOKEN = "tok"


# ---------------------------------------------------------------------------
# MetricsClient.emit_metric
# ---------------------------------------------------------------------------

class TestEmitMetric:
    def test_posts_correctly_formatted_payload(self):
        client = MetricsClient(url=METRICS_URL, username=METRICS_USERNAME, token=METRICS_TOKEN)
        mock_response = MagicMock()

        with patch("utils.metrics.requests.post", return_value=mock_response) as mock_post:
            client.emit_metric("my_metric", 42.0, {"env": "prod", "region": "us"})

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["data"] == "google_ads_optimization,env=prod,region=us my_metric=42.0"
        assert kwargs["auth"] == (METRICS_USERNAME, METRICS_TOKEN)
        mock_response.raise_for_status.assert_called_once()

    def test_uses_custom_metric_prefix_when_provided(self):
        client = MetricsClient(url=METRICS_URL, username=METRICS_USERNAME, token=METRICS_TOKEN)
        mock_response = MagicMock()

        with patch("utils.metrics.requests.post", return_value=mock_response) as mock_post:
            client.emit_metric("my_metric", 1.0, {"env": "test"}, metric_prefix="custom_prefix")

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["data"] == "custom_prefix,env=test my_metric=1.0"
        assert kwargs["auth"] == (METRICS_USERNAME, METRICS_TOKEN)
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.parametrize("url,username,token", [
        ("",          METRICS_USERNAME, METRICS_TOKEN),
        (METRICS_URL, "",               METRICS_TOKEN),
        (METRICS_URL, METRICS_USERNAME, ""),
    ])
    def test_skips_post_when_any_credential_is_missing(self, url, username, token):
        client = MetricsClient(url=url, username=username, token=token)

        with patch("utils.metrics.requests.post") as mock_post:
            client.emit_metric("my_metric", 1.0, {})

        mock_post.assert_not_called()

    def test_swallows_exception_from_failed_request(self):
        client = MetricsClient(url=METRICS_URL, username=METRICS_USERNAME, token=METRICS_TOKEN)

        with patch("utils.metrics.requests.post", side_effect=Exception("network error")):
            client.emit_metric("my_metric", 1.0, {})  # should not raise


# ---------------------------------------------------------------------------
# GoogleAdsMetricsClient.track_google_ads_operation_count
# ---------------------------------------------------------------------------

class TestTrackGoogleAdsOperationCount:
    def test_emits_api_operation_count_with_operation_label(self):
        client = GoogleAdsMetricsClient(url=METRICS_URL, username=METRICS_USERNAME, token=METRICS_TOKEN)

        with patch.object(client, "emit_metric") as mock_emit:
            client.track_google_ads_operation_count("search_stream", 3)

        mock_emit.assert_called_once_with("api_operation_count", 3.0, {"operation": "search_stream"})
