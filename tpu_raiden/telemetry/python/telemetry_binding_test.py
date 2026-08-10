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

"""Tests for TPU Raiden Python telemetry bindings."""

from absl.testing import absltest
from tpu_raiden.telemetry.python import _telemetry_binding_test_ext as telemetry_ext


class TelemetryBindingTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    telemetry_ext.configure_telemetry([])

  def test_configure_telemetry_enable_prometheus(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(snapshot, str)

  def test_configure_telemetry_case_insensitive(self):
    telemetry_ext.configure_telemetry(["Prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(snapshot, str)

  def test_configure_telemetry_empty_backends_clears_backends(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    telemetry_ext.configure_telemetry([])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertEqual(snapshot, "")

  def test_configure_telemetry_unknown_backend_ignored_and_logged(self):
    telemetry_ext.configure_telemetry(["unknown_backend"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertEqual(snapshot, "")


if __name__ == "__main__":
  absltest.main()
