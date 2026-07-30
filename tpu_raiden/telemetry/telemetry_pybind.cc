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
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/stl.h"
#include "tpu_raiden/telemetry/metrics_api.h"
#include "tpu_raiden/telemetry/prometheus_exporter.h"

namespace py = pybind11;

namespace tpu_raiden::telemetry {

void InitPrometheusExporterPybind() {
  if (!RaidenMetricStore::GetGlobalMetricStore().HasBackends()) {
    auto exporter = GetGlobalPrometheusExporter();
    RaidenMetricStore::GetGlobalMetricStore().AddBackend(exporter);
  }
}

std::string GetRaidenMetricsPrometheusText() {
  return RaidenMetricStore::GetGlobalMetricStore().GetPrometheusTextSnapshot();
}

PYBIND11_MODULE(telemetry_pybind, m) {
  m.doc() = "Pybind11 bindings for TPU Raiden Telemetry Engine";

  m.def("init_prometheus_exporter", &InitPrometheusExporterPybind,
        "Initialize and register the 3P Prometheus Exporter.");

  m.def("get_raiden_metrics_prometheus_text", &GetRaidenMetricsPrometheusText,
        "Get snapshot of Raiden metrics formatted as Prometheus exposition "
        "text.");

  m.def(
      "reset_prometheus_exporter",
      []() { GetGlobalPrometheusExporter()->Reset(); },
      "Reset data stored in global Prometheus exporter.");

  m.def(
      "clear_backends",
      []() { RaidenMetricStore::GetGlobalMetricStore().ClearBackends(); },
      "Clear all registered telemetry backends.");

  m.def(
      "record_counter",
      [](absl::string_view name, double value,
         const std::vector<std::pair<std::string, std::string>>& labels) {
        std::vector<std::pair<absl::string_view, absl::string_view>>
            label_views;
        label_views.reserve(labels.size());
        for (const auto& kv : labels) {
          label_views.push_back({kv.first, kv.second});
        }
        RaidenMetricStore::GetGlobalMetricStore().IncrementCounter(
            name, static_cast<uint64_t>(value), label_views);
      },
      py::arg("name"), py::arg("value"),
      py::arg("labels") = std::vector<std::pair<std::string, std::string>>{});

  m.def(
      "record_gauge",
      [](absl::string_view name, double value,
         const std::vector<std::pair<std::string, std::string>>& labels) {
        std::vector<std::pair<absl::string_view, absl::string_view>>
            label_views;
        label_views.reserve(labels.size());
        for (const auto& kv : labels) {
          label_views.push_back({kv.first, kv.second});
        }
        RaidenMetricStore::GetGlobalMetricStore().SetGauge(
            name, static_cast<int64_t>(value), label_views);
      },
      py::arg("name"), py::arg("value"),
      py::arg("labels") = std::vector<std::pair<std::string, std::string>>{});

  m.def(
      "record_histogram",
      [](absl::string_view name, double value,
         const std::vector<std::pair<std::string, std::string>>& labels) {
        std::vector<std::pair<absl::string_view, absl::string_view>>
            label_views;
        label_views.reserve(labels.size());
        for (const auto& kv : labels) {
          label_views.push_back({kv.first, kv.second});
        }
        RaidenMetricStore::GetGlobalMetricStore().ObserveHistogram(name, value,
                                                                   label_views);
      },
      py::arg("name"), py::arg("value"),

      py::arg("labels") = std::vector<std::pair<std::string, std::string>>{});
}

}  // namespace tpu_raiden::telemetry
