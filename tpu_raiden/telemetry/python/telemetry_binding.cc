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

#include "tpu_raiden/telemetry/python/telemetry_binding.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_raiden/telemetry/metrics_api.h"
#include "tpu_raiden/telemetry/prometheus_exporter.h"

namespace nb = nanobind;

namespace tpu_raiden::telemetry {

void BindTelemetryApi(nb::module_& m) {
  m.def(
      "configure_telemetry",
      [](const std::vector<std::string>& backends) {
        if (backends.empty()) {
          RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
          return;
        }
        std::vector<std::unique_ptr<MetricsBackend>> new_backends;
        for (const std::string& backend_name : backends) {
          if (absl::EqualsIgnoreCase(backend_name, kPrometheus)) {
            new_backends.push_back(std::make_unique<PrometheusExporter>());
          } else {
            LOG(WARNING) << "Unknown telemetry backend: " << backend_name;
          }
        }
        RaidenMetricStore::GetGlobalMetricStore().SetBackends(
            std::move(new_backends));
      },
      nb::arg("backends"), nb::call_guard<nb::gil_scoped_release>(),
      "Configures active C++ telemetry backends.");

  m.def(
      "get_raiden_metrics_prometheus_text",
      []() -> std::string {
        return RaidenMetricStore::GetGlobalMetricStore().GetTextSnapshot();
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Exports Prometheus text snapshot of TPU Raiden metrics.");
}

}  // namespace tpu_raiden::telemetry
