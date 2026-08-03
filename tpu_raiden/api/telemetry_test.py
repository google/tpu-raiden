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

"""Tests for TPU Raiden Python Prometheus collector and integration."""

from unittest import mock

from absl.testing import absltest
import prometheus_client

from tpu_raiden.api import telemetry
from tpu_raiden.telemetry import _raiden_telemetry


class RaidenTelemetryTest(absltest.TestCase):

  def test_collector_yields_metrics(self):
    _raiden_telemetry.init_prometheus_backend()
    collector = telemetry.RaidenPrometheusCollector()
    metrics = list(collector.collect())
    self.assertIsInstance(metrics, list)

  def test_init_raiden_telemetry_registers_with_registry(self):
    test_registry = prometheus_client.CollectorRegistry()
    collector = telemetry.init_raiden_telemetry(
        registry=test_registry, force_reinit=True
    )
    self.assertIn(collector, test_registry._collector_to_names)

  @mock.patch.object(
      _raiden_telemetry,
      "get_raiden_metrics_prometheus_text",
      side_effect=RuntimeError("Simulated FFI failure"),
  )
  def test_collector_exception_wrapping(self, mock_get_text):
    collector = telemetry.RaidenPrometheusCollector()
    # Should catch exception, log it, and yield no metrics without raising
    metrics = list(collector.collect())
    self.assertEmpty(metrics)
    mock_get_text.assert_called_once()


if __name__ == "__main__":
  absltest.main()
