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

"""Tests for TPU Raiden Python telemetry integration and lifecycle."""

import concurrent.futures
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import prometheus_client

from tpu_raiden.api import telemetry
from tpu_raiden.telemetry.python import _telemetry_binding_test_ext as _raiden_telemetry


_PROMETHEUS = telemetry.PROMETHEUS


class RaidenTelemetryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    telemetry._CACHED_TELEMETRY_MODULE = _raiden_telemetry
    self._mock_get_mod = mock.patch.object(
        telemetry, "_get_telemetry_module", return_value=_raiden_telemetry
    )
    self._mock_get_mod.start()

  def tearDown(self):
    super().tearDown()
    _raiden_telemetry.configure_telemetry(set())
    telemetry._active_backend_handles.clear()
    telemetry._CACHED_TELEMETRY_MODULE = None
    self._mock_get_mod.stop()

  def test_no_module_fallback_returns_empty(self):
    with mock.patch.object(
        telemetry, "_get_telemetry_module", return_value=None
    ):
      handles = telemetry.init_raiden_telemetry(backends=[_PROMETHEUS])
      collector = handles.get(_PROMETHEUS)
      self.assertIsNotNone(collector)
      self.assertEmpty(list(collector.collect()))

  def test_collector_yields_metrics(self):
    handles = telemetry.init_raiden_telemetry(backends=[_PROMETHEUS])
    collector = handles.get(_PROMETHEUS)
    self.assertIsNotNone(collector)
    metrics = list(collector.collect())
    self.assertNotEmpty(metrics)
    for metric in metrics:
      self.assertTrue(
          metric.name.startswith("tpu_raiden_"),
          f"Unexpected metric family: {metric.name}",
      )

  def test_collector_multiprocess_directory_aggregation(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      # Simulate Process A (Worker 0) metric dump
      proc_a_file = os.path.join(tmpdir, "raiden_metrics_pid_101.prom")
      with open(proc_a_file, "w", encoding="utf-8") as f:
        f.write(
            "# HELP tpu_raiden_sent_bytes_total Total bytes sent\n"
            "# TYPE tpu_raiden_sent_bytes_total counter\n"
            "tpu_raiden_sent_bytes_total 1000.0\n"
        )
      # Simulate Process B (Worker 1) metric dump
      proc_b_file = os.path.join(tmpdir, "raiden_metrics_pid_102.prom")
      with open(proc_b_file, "w", encoding="utf-8") as f:
        f.write(
            "# HELP tpu_raiden_sent_bytes_total Total bytes sent\n"
            "# TYPE tpu_raiden_sent_bytes_total counter\n"
            "tpu_raiden_sent_bytes_total 2500.0\n"
        )

      with mock.patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": tmpdir}):
        handles = telemetry.init_raiden_telemetry(backends=[_PROMETHEUS])
        collector = handles.get(_PROMETHEUS)
        metrics = list(collector.collect())
        self.assertLen(metrics, 1)
        metric_family = metrics[0]
        sent_total_samples = [s.value for s in metric_family.samples if s.name == "tpu_raiden_sent_bytes_total"]
        self.assertIn(1000.0, sent_total_samples)
        self.assertIn(2500.0, sent_total_samples)

  def test_init_raiden_telemetry_registers_with_registry(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles1 = telemetry.init_raiden_telemetry(
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector1 = handles1.get(_PROMETHEUS)
    self.assertIsNotNone(collector1)
    self.assertIn(collector1, test_registry._collector_to_names)

    handles2 = telemetry.init_raiden_telemetry(
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector2 = handles2.get(_PROMETHEUS)
    self.assertIs(collector1, collector2)
    self.assertIn(collector2, test_registry._collector_to_names)

  def test_init_raiden_telemetry_multiple_registries_and_cleanup(self):
    registry1 = prometheus_client.CollectorRegistry()
    registry2 = prometheus_client.CollectorRegistry()
    handles1 = telemetry.init_raiden_telemetry(
        backends=[_PROMETHEUS],
        registry=registry1,
    )
    handles2 = telemetry.init_raiden_telemetry(
        backends=[_PROMETHEUS],
        registry=registry2,
    )
    collector1 = handles1.get(_PROMETHEUS)
    collector2 = handles2.get(_PROMETHEUS)
    self.assertIs(collector1, collector2)
    self.assertIn(collector1, registry1._collector_to_names)
    self.assertIn(collector2, registry2._collector_to_names)

    # Disabling telemetry cleans up from ALL registered registries
    telemetry.init_raiden_telemetry(backends=[])
    self.assertNotIn(collector1, registry1._collector_to_names)
    self.assertNotIn(collector2, registry2._collector_to_names)

  def test_init_raiden_telemetry_unknown_backend_raises_value_error(self):
    test_registry = prometheus_client.CollectorRegistry()
    with self.assertRaises(ValueError):
      telemetry.init_raiden_telemetry(
          backends=["custom_backend"],
          registry=test_registry,
      )

    # Verify ValueError is also raised when C++ module is None
    with mock.patch.object(
        telemetry, "_get_telemetry_module", return_value=None
    ):
      with self.assertRaises(ValueError):
        telemetry.init_raiden_telemetry(
            backends=["custom_backend"],
            registry=test_registry,
        )

  @mock.patch.object(
      _raiden_telemetry,
      "get_raiden_metrics_prometheus_text",
      side_effect=RuntimeError("Simulated FFI failure"),
  )
  def test_collector_exception_wrapping(self, mock_get_text):
    handles = telemetry.init_raiden_telemetry(backends=[_PROMETHEUS])
    collector = handles.get(_PROMETHEUS)
    metrics = list(collector.collect())
    self.assertEmpty(metrics)
    self.assertTrue(mock_get_text.called)

  def test_init_raiden_telemetry_default_disabled(self):
    test_registry = prometheus_client.CollectorRegistry()
    result = telemetry.init_raiden_telemetry(registry=test_registry)
    self.assertEqual(result, {})
    self.assertEmpty(test_registry._collector_to_names)

  def test_init_raiden_telemetry_empty_backends_disabled(self):
    test_registry = prometheus_client.CollectorRegistry()
    result = telemetry.init_raiden_telemetry(
        backends=[], registry=test_registry
    )
    self.assertEqual(result, {})
    self.assertEmpty(test_registry._collector_to_names)

  def test_init_raiden_telemetry_disable_clears_backends_and_unregisters(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles = telemetry.init_raiden_telemetry(
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector = handles.get(_PROMETHEUS)
    self.assertIsNotNone(collector)
    self.assertIn(collector, test_registry._collector_to_names)
    self.assertIn(_PROMETHEUS, telemetry._active_backend_handles)

    # Disable telemetry with empty backends and verify unregistration & cleanup
    disabled_handles = telemetry.init_raiden_telemetry(
        backends=[],
        registry=test_registry,
    )
    self.assertEqual(disabled_handles, {})
    self.assertNotIn(collector, test_registry._collector_to_names)
    self.assertNotIn(_PROMETHEUS, telemetry._active_backend_handles)

  def test_backend_reconfiguration_unregisters_removed_backends(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles1 = telemetry.init_raiden_telemetry(
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector1 = handles1.get(_PROMETHEUS)
    self.assertIn(collector1, test_registry._collector_to_names)

    # Reconfigure with no backends
    telemetry.init_raiden_telemetry(
        backends=[],
        registry=test_registry,
    )
    self.assertNotIn(collector1, test_registry._collector_to_names)
    self.assertNotIn(_PROMETHEUS, telemetry._active_backend_handles)

  def test_get_telemetry_module_no_cache_on_none(self):
    self._mock_get_mod.stop()
    try:
      telemetry._CACHED_TELEMETRY_MODULE = None
      with mock.patch.object(
          telemetry, "_iter_candidate_modules", return_value=[]
      ):
        # When candidate modules yield nothing, returns None and does not cache
        self.assertIsNone(telemetry._get_telemetry_module())
        self.assertIsNone(telemetry._CACHED_TELEMETRY_MODULE)

      # Once a module becomes available, discovery finds it and caches it
      with mock.patch.object(
          telemetry, "_iter_candidate_modules", return_value=[_raiden_telemetry]
      ):
        mod = telemetry._get_telemetry_module()
        self.assertIs(mod, _raiden_telemetry)
        self.assertIs(telemetry._CACHED_TELEMETRY_MODULE, _raiden_telemetry)
    finally:
      self._mock_get_mod.start()

  def test_init_raiden_telemetry_thread_safety(self):
    test_registry = prometheus_client.CollectorRegistry()

    def _init_call():
      return telemetry.init_raiden_telemetry(
          backends=[_PROMETHEUS],
          registry=test_registry,
      )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(_init_call) for _ in range(20)]
      results = [f.result() for f in futures]

    first_handles = results[0]
    self.assertIn(_PROMETHEUS, first_handles)
    first_collector = first_handles[_PROMETHEUS]
    self.assertIsNotNone(first_collector)
    for res in results:
      self.assertIs(res.get(_PROMETHEUS), first_collector)
    self.assertIn(first_collector, test_registry._collector_to_names)


if __name__ == "__main__":
  absltest.main()
