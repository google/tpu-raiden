# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU Raiden Prometheus telemetry integration for Python serving (Mode A).

Integrates Raiden C++ telemetry metrics with Python prometheus_client.REGISTRY
to enable automatic metric scraping on serving endpoints (e.g., vLLM
:8000/metrics)
without modifying vLLM core.
"""

from typing import Iterable, Optional

from absl import logging
import prometheus_client
from prometheus_client import parser
import prometheus_client.core
import prometheus_client.registry

from tpu_raiden.telemetry import _raiden_telemetry


class RaidenPrometheusCollector(prometheus_client.registry.Collector):
  """Custom Prometheus collector yielding TPU Raiden C++ metrics."""

  def collect(self) -> Iterable[prometheus_client.core.Metric]:
    """Collects Raiden C++ Prometheus metrics and yields MetricFamily objects."""
    try:
      text_snapshot = _raiden_telemetry.get_raiden_metrics_prometheus_text()
      if not text_snapshot:
        return
      for metric_family in parser.text_string_to_metric_families(text_snapshot):
        yield metric_family
    except Exception as e:
      # Strict exception wrapping: collection failures must never crash the serving endpoint.
      logging.exception(
          "Failed to collect TPU Raiden C++ Prometheus metrics: %s", e
      )


_GLOBAL_COLLECTOR: Optional[RaidenPrometheusCollector] = None


def init_raiden_telemetry(
    registry: Optional[prometheus_client.CollectorRegistry] = None,
    force_reinit: bool = False,
) -> RaidenPrometheusCollector:
  """Registers RaidenPrometheusCollector with the Prometheus client registry.

  Args:
    registry: Target CollectorRegistry. If None, defaults to
      prometheus_client.REGISTRY.
    force_reinit: If True, forces re-initialization and re-registration even if
      already initialized.

  Returns:
    The registered RaidenPrometheusCollector instance.
  """
  global _GLOBAL_COLLECTOR
  if registry is None:
    registry = prometheus_client.REGISTRY

  if _GLOBAL_COLLECTOR is not None and not force_reinit:
    return _GLOBAL_COLLECTOR

  # Ensure the C++ Prometheus backend is initialized in RaidenMetricStore
  _raiden_telemetry.init_prometheus_backend()

  collector = RaidenPrometheusCollector()
  try:
    registry.register(collector)
  except ValueError:
    # Collector already registered in this registry
    pass

  _GLOBAL_COLLECTOR = collector
  return collector
