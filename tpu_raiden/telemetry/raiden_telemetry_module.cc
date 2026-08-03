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
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>  // IWYU pragma: keep
#include <nanobind/stl/vector.h>  // IWYU pragma: keep
#include "tpu_raiden/telemetry/metrics_api.h"
#include "tpu_raiden/telemetry/prometheus_exporter.h"

namespace nb = nanobind;

using ::tpu_raiden::telemetry::MetricsBackend;
using ::tpu_raiden::telemetry::PrometheusExporter;
using ::tpu_raiden::telemetry::RaidenMetricStore;

namespace {

// Factory helper to instantiate telemetry backends by identifier.
std::unique_ptr<MetricsBackend> CreateBackend(absl::string_view backend_name) {
  if (absl::EqualsIgnoreCase(backend_name, "prometheus")) {
    return std::make_unique<PrometheusExporter>();
  }
  // Additional telemetry backends can be added here.
  return nullptr;
}

}  // namespace

NB_MODULE(_raiden_telemetry, m) {
  m.doc() = "Python C++ bridge for TPU Raiden telemetry text snapshots.";

  m.def(
      "configure_telemetry",
      [](bool enabled, const std::vector<std::string>& backends) {
        auto& store = RaidenMetricStore::GetGlobalMetricStore();
        store.ClearBackends();
        if (!enabled) {
          return;
        }
        for (const auto& backend_name : backends) {
          auto backend = CreateBackend(backend_name);
          if (backend != nullptr) {
            store.AddBackend(std::move(backend));
          }
        }
      },
      nb::arg("enabled"),
      nb::arg("backends"),
      "Configures active C++ telemetry backends.");

  m.def(
      "is_telemetry_enabled",
      []() -> bool {
        return RaidenMetricStore::GetGlobalMetricStore().HasBackends();
      },
      "Returns true if C++ telemetry backends are actively enabled.");

  m.def(
      "get_raiden_metrics_prometheus_text",
      []() -> std::string {
        return RaidenMetricStore::GetGlobalMetricStore().GetTextSnapshot();
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Exports the Prometheus text snapshot of TPU Raiden metrics without "
      "holding the Python GIL.");
}
