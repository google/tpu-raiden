// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>

#include "absl/base/call_once.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include "tpu_raiden/telemetry/metrics_api.h"
#include "tpu_raiden/telemetry/prometheus_exporter.h"

namespace nb = nanobind;

NB_MODULE(_raiden_telemetry, m) {
  m.doc() = "Python C++ bridge for TPU Raiden telemetry text snapshots.";

  m.def(
      "init_prometheus_backend",
      []() {
        static absl::once_flag prometheus_init_flag;
        absl::call_once(prometheus_init_flag, []() {
          tpu_raiden::telemetry::RaidenMetricStore::GetGlobalMetricStore()
              .AddBackend(std::make_unique<
                          tpu_raiden::telemetry::PrometheusExporter>());
        });
      },
      "Initializes and registers the C++ 3P Prometheus backend with the "
      "global RaidenMetricStore if not already registered.");

  m.def(
      "get_raiden_metrics_prometheus_text",
      []() -> std::string {
        nb::gil_scoped_release release;
        return tpu_raiden::telemetry::RaidenMetricStore::GetGlobalMetricStore()
            .GetTextSnapshot();
      },
      "Exports the Prometheus text snapshot of TPU Raiden metrics without "
      "holding the Python GIL.");
}
