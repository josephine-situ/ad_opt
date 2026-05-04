import traceback
from typing import Dict

import requests

REQUEST_HEADERS = {'Content-Type': 'text/plain',}

GRAFANA_METRIC_URL = ""

# Barebones client for emitting metrics to a metrics server
# Currently, this emits Influx line format to our grafana cloud prometheus endpoint for simplicity
# It may make sense to change this in the future or rely on a more fully featured client,
# but this can be shimmed or removed as needed.
# See https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-influxdb/push-from-telegraf/#examples
class MetricsClient():

    def __init__(self, url: str, username: str, token: str) -> None:
        self.url = url
        self.username = username
        self.token = token
        self.metric_prefix = "google_ads_optimization"

    # Emits a single metric of the following format:
    def emit_metric(self, metric_name:str, value:float, labels: Dict[str, str], metric_prefix: str | None = None, ) -> None:

        if not all([self.url, self.username, self.token]):
            print(f"Credentials not set, skipping metric {metric_name} with value {value} and labels {labels}")
            return

        try:
            if not metric_prefix:
                metric_prefix = self.metric_prefix
            labels_string = ",".join([f'{key}={value}' for key, value in labels.items()])
            payload = f'{metric_prefix},{labels_string} {metric_name}={value}'
            response = requests.post(self.url,
                                     headers=REQUEST_HEADERS,
                                     data=payload,
                                     auth=(self.username, self.token)
                                     )
            response.raise_for_status()
        except Exception:
            # We don't want metrics emission to cause pipeline failures, so we catch and log any exceptions that occur here.
            print(f"Error emitting metric {metric_name} with value {value} and labels {labels}")
            print(traceback.format_exc())

# Thin wrapper class to inject specific tags or other formatting for Google Ads related metrics
# These seem approximately correct when we use sum_over_time in to query.
class GoogleAdsMetricsClient(MetricsClient):
    def __init__(self, url: str, username: str, token: str) -> None:
        super().__init__(url, username, token)

    def track_google_ads_operation_count(self, operation_type: str, value:int) -> None:
        labels: Dict[str, str] = {"operation": operation_type}
        self.emit_metric( "api_operation_count", float(value), labels)

def get_metrics_client() -> GoogleAdsMetricsClient:
    import os
    url = os.getenv("GRAFANA_URL")
    username = os.getenv("GRAFANA_USERNAME")
    token = os.getenv("GRAFANA_TOKEN")
    return GoogleAdsMetricsClient(url, username, token)

google_ads_metrics_client = get_metrics_client()