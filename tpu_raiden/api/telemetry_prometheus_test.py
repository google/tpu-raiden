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

"""Tests for TPU Raiden Python Prometheus collector and backend utilities."""

import os
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
import prometheus_client

from tpu_raiden.api import telemetry_prometheus


_PROMETHEUS = telemetry_prometheus.PROMETHEUS

_SAMPLE_METRIC_TEXT = (
    "# HELP tpu_raiden_sent_bytes_total Total bytes sent\n"
    "# TYPE tpu_raiden_sent_bytes_total counter\n"
    'tpu_raiden_sent_bytes_total{interface="ICI"} 4096.0\n'
)


class RaidenPrometheusCollectorTest(absltest.TestCase):

  def test_collector_default_snapshot_yields_empty(self):
    collector = telemetry_prometheus.RaidenPrometheusCollector()
    metrics = list(collector.collect())
    self.assertEmpty(metrics)

  def test_collector_empty_snapshot_yields_empty(self):
    collector = telemetry_prometheus.RaidenPrometheusCollector(
        text_snapshot_fn=lambda: ""
    )
    metrics = list(collector.collect())
    self.assertEmpty(metrics)

  def test_collector_parses_metrics_from_snapshot(self):
    collector = telemetry_prometheus.RaidenPrometheusCollector(
        text_snapshot_fn=lambda: _SAMPLE_METRIC_TEXT
    )
    metrics = list(collector.collect())
    self.assertLen(metrics, 1)
    metric = metrics[0]
    self.assertTrue(metric.name.startswith("tpu_raiden_sent_bytes"))
    self.assertEqual(metric.type, "counter")
    self.assertEqual(metric.documentation, "Total bytes sent")
    self.assertLen(metric.samples, 1)
    sample = metric.samples[0]
    self.assertEqual(sample.name, "tpu_raiden_sent_bytes_total")
    self.assertEqual(sample.labels, {"interface": "ICI"})
    self.assertEqual(sample.value, 4096.0)

  def test_collector_multiprocess_merges_duplicate_metric_families(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      collector = telemetry_prometheus.RaidenPrometheusCollector(
          multiproc_dir=tmpdir,
          text_snapshot_fn=lambda: _SAMPLE_METRIC_TEXT,
      )

      # Simulate another worker dumping its metrics into multiproc directory
      other_worker_file = os.path.join(tmpdir, "raiden_metrics_pid_999.prom")
      with open(other_worker_file, "w", encoding="utf-8") as f:
        f.write(
            "# HELP tpu_raiden_sent_bytes_total Total bytes sent\n"
            "# TYPE tpu_raiden_sent_bytes_total counter\n"
            'tpu_raiden_sent_bytes_total{interface="HBM"} 8192.0\n'
        )

      metrics = list(collector.collect())
      # Crucial: verify that samples from both processes are merged into ONE MetricFamily
      self.assertLen(metrics, 1)
      metric_family = metrics[0]
      self.assertEqual(metric_family.name, "tpu_raiden_sent_bytes")
      self.assertLen(metric_family.samples, 2)

      # Verify local process snapshot file was written
      local_file = os.path.join(
          tmpdir, f"raiden_metrics_pid_{os.getpid()}.prom"
      )
      self.assertTrue(os.path.exists(local_file))

      # Verify prometheus_client.generate_latest succeeds without duplicate family errors
      test_registry = prometheus_client.CollectorRegistry()
      collector.register_with(test_registry)
      output = prometheus_client.generate_latest(test_registry).decode("utf-8")
      self.assertIn('tpu_raiden_sent_bytes_total{interface="ICI"} 4096.0', output)
      self.assertIn('tpu_raiden_sent_bytes_total{interface="HBM"} 8192.0', output)

  def test_collector_empty_snapshot_cleans_existing_pid_file(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      local_file = os.path.join(
          tmpdir, f"raiden_metrics_pid_{os.getpid()}.prom"
      )
      # Create pre-existing file
      with open(local_file, "w", encoding="utf-8") as f:
        f.write(_SAMPLE_METRIC_TEXT)
      self.assertTrue(os.path.exists(local_file))

      # Flush with empty snapshot should delete the stale PID file
      collector = telemetry_prometheus.RaidenPrometheusCollector(
          multiproc_dir=tmpdir,
          text_snapshot_fn=lambda: "",
      )
      metrics = list(collector.collect())
      self.assertEmpty(metrics)
      self.assertFalse(os.path.exists(local_file))

  def test_collector_handles_snapshot_fn_exception_gracefully(self):
    def _failing_snapshot_fn():
      raise RuntimeError("Simulated FFI failure")

    collector = telemetry_prometheus.RaidenPrometheusCollector(
        text_snapshot_fn=_failing_snapshot_fn
    )
    metrics = list(collector.collect())
    self.assertEmpty(metrics)

  def test_collector_handles_flush_io_error_gracefully(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      collector = telemetry_prometheus.RaidenPrometheusCollector(
          multiproc_dir=tmpdir,
          text_snapshot_fn=lambda: _SAMPLE_METRIC_TEXT,
      )
      with mock.patch("builtins.open", side_effect=OSError("Read-only FS")):
        # Should not throw exception despite file write error
        metrics = list(collector.collect())
        self.assertEmpty(metrics)


class PrometheusBackendLifecycleTest(absltest.TestCase):

  def test_init_and_disable_prometheus_backend(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles: dict[str, Any] = {}

    collector1 = telemetry_prometheus.init_prometheus_backend(
        registry=test_registry,
        active_handles=handles,
        text_snapshot_fn=lambda: _SAMPLE_METRIC_TEXT,
    )
    self.assertIsNotNone(collector1)
    self.assertIn(collector1, test_registry._collector_to_names)
    self.assertIs(handles.get(_PROMETHEUS), collector1)

    # Idempotent re-initialization with same active_handles
    collector2 = telemetry_prometheus.init_prometheus_backend(
        registry=test_registry,
        active_handles=handles,
        text_snapshot_fn=lambda: _SAMPLE_METRIC_TEXT,
    )
    self.assertIs(collector1, collector2)

    # Disable backend
    telemetry_prometheus.disable_prometheus_backend(
        collector1, registry=test_registry
    )
    self.assertNotIn(collector1, test_registry._collector_to_names)

  def test_disable_cleans_all_tracked_registries(self):
    reg1 = prometheus_client.CollectorRegistry()
    reg2 = prometheus_client.CollectorRegistry()
    collector = telemetry_prometheus.RaidenPrometheusCollector()
    collector.register_with(reg1)
    collector.register_with(reg2)
    self.assertIn(collector, reg1._collector_to_names)
    self.assertIn(collector, reg2._collector_to_names)

    # Unregister with registry=None cleans all tracked registries
    collector.unregister_from(registry=None)
    self.assertNotIn(collector, reg1._collector_to_names)
    self.assertNotIn(collector, reg2._collector_to_names)

  def test_disable_none_or_unregistered_collector_safe(self):
    test_registry = prometheus_client.CollectorRegistry()
    # Disabling None should not throw
    telemetry_prometheus.disable_prometheus_backend(
        None, registry=test_registry
    )

    # Disabling a collector that is not registered should not throw
    collector = telemetry_prometheus.RaidenPrometheusCollector()
    telemetry_prometheus.disable_prometheus_backend(
        collector, registry=test_registry
    )


if __name__ == "__main__":
  absltest.main()
