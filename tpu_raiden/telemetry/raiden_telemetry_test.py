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

"""Tests for _raiden_telemetry C++/Python FFI bridge module."""

from absl.testing import absltest

try:
  from tpu_raiden.telemetry import _raiden_telemetry
except ImportError:
  from tpu_raiden.telemetry import _raiden_telemetry


class RaidenTelemetryTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    # Reset telemetry backends after each test
    _raiden_telemetry.configure_telemetry(False, [])

  def test_get_prometheus_text_snapshot(self):
    # Call before backend initialization (should return empty string)
    _raiden_telemetry.configure_telemetry(False, [])
    snapshot = _raiden_telemetry.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(snapshot, str)
    self.assertEqual(snapshot, "")

  def test_configure_telemetry_enable_and_disable(self):
    # Test enabling Prometheus backend
    _raiden_telemetry.configure_telemetry(True, ["prometheus"])
    self.assertTrue(_raiden_telemetry.is_telemetry_enabled())
    snapshot = _raiden_telemetry.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(snapshot, str)

    # Test disabling telemetry
    _raiden_telemetry.configure_telemetry(False, [])
    self.assertFalse(_raiden_telemetry.is_telemetry_enabled())
    self.assertEqual(_raiden_telemetry.get_raiden_metrics_prometheus_text(), "")

  def test_configure_telemetry_case_insensitive(self):
    # Test case insensitivity for backend names
    _raiden_telemetry.configure_telemetry(True, ["PROMETHEUS"])
    self.assertTrue(_raiden_telemetry.is_telemetry_enabled())

  def test_configure_telemetry_unsupported_backend(self):
    # Test unsupported backend name
    _raiden_telemetry.configure_telemetry(True, ["unsupported_sink"])
    self.assertFalse(_raiden_telemetry.is_telemetry_enabled())

  def test_configure_telemetry_idempotent(self):
    # Multiple configuration calls should be safe and idempotent
    _raiden_telemetry.configure_telemetry(True, ["prometheus"])
    _raiden_telemetry.configure_telemetry(True, ["prometheus"])
    self.assertTrue(_raiden_telemetry.is_telemetry_enabled())


if __name__ == "__main__":
  absltest.main()
