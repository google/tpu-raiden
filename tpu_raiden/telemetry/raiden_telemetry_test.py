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
from tpu_raiden.telemetry import _raiden_telemetry


class RaidenTelemetryTest(absltest.TestCase):

  def test_get_prometheus_text_snapshot(self):
    # Call before backend initialization (should return empty string or valid text)
    snapshot = _raiden_telemetry.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(snapshot, str)

  def test_init_prometheus_backend(self):
    # Initialize the C++ Prometheus backend
    _raiden_telemetry.init_prometheus_backend()
    snapshot = _raiden_telemetry.get_raiden_metrics_prometheus_text()
    self.assertIsInstance(snapshot, str)


if __name__ == "__main__":
  absltest.main()
