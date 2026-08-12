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
    self.assertIn(
        "# HELP tpu_raiden_sent_bytes_total Total count of bytes sent over TPU"
        " Raiden interfaces.",
        snapshot,
    )
    self.assertIn("# TYPE tpu_raiden_sent_bytes_total counter", snapshot)

  def test_configure_telemetry_case_insensitive(self):
    telemetry_ext.configure_telemetry(["Prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertIn(
        "# HELP tpu_raiden_sent_bytes_total Total count of bytes sent over TPU"
        " Raiden interfaces.",
        snapshot,
    )
    self.assertIn("# TYPE tpu_raiden_sent_bytes_total counter", snapshot)

  def test_configure_telemetry_duplicate_backends_deduplicated(self):
    telemetry_ext.configure_telemetry(["prometheus", "prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertEqual(
        snapshot.count(
            "# HELP tpu_raiden_sent_bytes_total Total count of bytes sent over"
            " TPU Raiden interfaces."
        ),
        1,
    )
    self.assertEqual(
        snapshot.count("# TYPE tpu_raiden_sent_bytes_total counter"), 1
    )

  def test_configure_telemetry_empty_backends_clears_backends(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    telemetry_ext.configure_telemetry([])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertEqual(snapshot, "")

  def test_configure_telemetry_unknown_backend_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown telemetry backend: unknown_backend"
    ):
      telemetry_ext.configure_telemetry(["unknown_backend"])

  def test_configure_telemetry_invalid_argument_type_raises_type_error(self):
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry(123)
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry([123])
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry(None)


if __name__ == "__main__":
  absltest.main()
