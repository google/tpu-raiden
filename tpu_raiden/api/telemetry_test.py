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

import concurrent.futures
from unittest import mock

from absl.testing import absltest
import prometheus_client

from tpu_raiden.api import telemetry
from tpu_raiden.telemetry.python import _telemetry_py_test_ext as _raiden_telemetry

_PROMETHEUS = telemetry.PROMETHEUS


class RaidenTelemetryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._mock_get_mod = mock.patch.object(
        telemetry, "_get_telemetry_module", return_value=_raiden_telemetry
    )
    self._mock_get_mod.start()

  def tearDown(self):
    super().tearDown()
    _raiden_telemetry.configure_telemetry([])
    telemetry._active_backend_handles.clear()
    self._mock_get_mod.stop()

  def test_no_module_fallback_returns_empty(self):
    with mock.patch.object(
        telemetry, "_get_telemetry_module", return_value=None
    ):
      self.assertEqual(telemetry.get_raiden_metrics_prometheus_text(), "")
      collector = telemetry.RaidenPrometheusCollector()
      self.assertEmpty(list(collector.collect()))

  def test_collector_yields_metrics(self):
    _raiden_telemetry.configure_telemetry([_PROMETHEUS])
    collector = telemetry.RaidenPrometheusCollector()
    metrics = list(collector.collect())
    self.assertNotEmpty(metrics)
    for metric in metrics:
      self.assertTrue(
          metric.name.startswith("tpu_raiden_"),
          f"Unexpected metric family: {metric.name}",
      )

  def test_init_raiden_telemetry_registers_with_registry(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles1 = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector1 = handles1.get(_PROMETHEUS)
    self.assertIsNotNone(collector1)
    self.assertIn(collector1, test_registry._collector_to_names)

    handles2 = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector2 = handles2.get(_PROMETHEUS)
    self.assertIs(collector1, collector2)
    self.assertIn(collector2, test_registry._collector_to_names)

  def test_init_raiden_telemetry_multiple_registries(self):
    registry1 = prometheus_client.CollectorRegistry()
    registry2 = prometheus_client.CollectorRegistry()
    handles1 = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS],
        registry=registry1,
    )
    handles2 = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS],
        registry=registry2,
    )
    collector1 = handles1.get(_PROMETHEUS)
    collector2 = handles2.get(_PROMETHEUS)
    self.assertIs(collector1, collector2)
    self.assertIn(collector1, registry1._collector_to_names)
    self.assertIn(collector2, registry2._collector_to_names)

  def test_init_raiden_telemetry_multi_backends(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS, "custom_backend"],
        registry=test_registry,
    )
    self.assertIn(_PROMETHEUS, handles)
    self.assertIn("custom_backend", handles)
    self.assertIsNotNone(handles[_PROMETHEUS])
    self.assertIn(handles[_PROMETHEUS], test_registry._collector_to_names)
    self.assertIsNone(handles["custom_backend"])

  @mock.patch.object(
      _raiden_telemetry,
      "get_raiden_metrics_prometheus_text",
      side_effect=RuntimeError("Simulated FFI failure"),
  )
  def test_collector_exception_wrapping(self, mock_get_text):
    collector = telemetry.RaidenPrometheusCollector()
    metrics = list(collector.collect())
    self.assertEmpty(metrics)
    mock_get_text.assert_called_once()

  def test_init_raiden_telemetry_default_disabled(self):
    test_registry = prometheus_client.CollectorRegistry()
    result = telemetry.init_raiden_telemetry(registry=test_registry)
    self.assertEqual(result, {})
    self.assertEmpty(test_registry._collector_to_names)

  def test_init_raiden_telemetry_backends_ignored_when_disabled(self):
    test_registry = prometheus_client.CollectorRegistry()
    result = telemetry.init_raiden_telemetry(
        enabled=False,
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    self.assertEqual(result, {})
    self.assertEmpty(test_registry._collector_to_names)

  def test_init_raiden_telemetry_empty_backends(self):
    test_registry = prometheus_client.CollectorRegistry()
    result = telemetry.init_raiden_telemetry(
        enabled=True, backends=[], registry=test_registry
    )
    self.assertEqual(result, {})
    self.assertEmpty(test_registry._collector_to_names)

  def test_init_raiden_telemetry_disable_clears_backends_and_unregisters(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector = handles.get(_PROMETHEUS)
    self.assertIsNotNone(collector)
    self.assertIn(collector, test_registry._collector_to_names)
    self.assertIn(_PROMETHEUS, telemetry._active_backend_handles)

    # Disable telemetry and verify unregistration & cleanup
    disabled_handles = telemetry.init_raiden_telemetry(
        enabled=False,
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    self.assertEqual(disabled_handles, {})
    self.assertNotIn(collector, test_registry._collector_to_names)
    self.assertNotIn(_PROMETHEUS, telemetry._active_backend_handles)
    self.assertEqual(telemetry.get_raiden_metrics_prometheus_text(), "")

  def test_init_raiden_telemetry_empty_backends_returns_empty(self):
    test_registry = prometheus_client.CollectorRegistry()
    handles = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[_PROMETHEUS],
        registry=test_registry,
    )
    collector = handles.get(_PROMETHEUS)
    self.assertIn(collector, test_registry._collector_to_names)

    # Update with empty backends list
    empty_handles = telemetry.init_raiden_telemetry(
        enabled=True,
        backends=[],
        registry=test_registry,
    )
    self.assertEqual(empty_handles, {})

  def test_get_raiden_metrics_prometheus_text(self):
    _raiden_telemetry.configure_telemetry([_PROMETHEUS])
    text = telemetry.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(text, str)
    self.assertIn("tpu_raiden_sent_bytes_total", text)

  def test_init_raiden_telemetry_thread_safety(self):
    test_registry = prometheus_client.CollectorRegistry()

    def _init_call():
      return telemetry.init_raiden_telemetry(
          enabled=True,
          backends=[_PROMETHEUS, "custom_backend"],
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

  def test_get_telemetry_module_thread_safety(self):
    self._mock_get_mod.stop()
    try:
      telemetry._get_telemetry_module.cache_clear()
      with mock.patch.object(
          telemetry,
          "_iter_candidate_modules",
          return_value=[_raiden_telemetry],
      ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
          futures = [
              executor.submit(telemetry._get_telemetry_module)
              for _ in range(20)
          ]
          results = [f.result() for f in futures]

        for res in results:
          self.assertIs(res, _raiden_telemetry)
    finally:
      self._mock_get_mod.start()


if __name__ == "__main__":
  absltest.main()
