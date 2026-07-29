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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

// Prometheus Exporter for 3P Prometheus metric collection.
class PrometheusExporter : public TelemetryExporter {
 public:
  explicit PrometheusExporter(
      std::vector<double> custom_buckets = {0.005, 0.01, 0.025, 0.05, 0.1,
                                           0.25,  0.5,  1.0,   2.5,  5.0,  10.0});

  ~PrometheusExporter() override = default;

  void RecordCounter(absl::string_view name, double value,
                     LabelSpan labels = {}) noexcept override;

  void RecordGauge(absl::string_view name, double value,
                   LabelSpan labels = {}) noexcept override;

  void RecordHistogram(absl::string_view name, double value,
                       LabelSpan labels = {}) noexcept override;

  std::string ExportPrometheusText() noexcept override;
  std::string GetPrometheusText() const noexcept {
    return const_cast<PrometheusExporter*>(this)->ExportPrometheusText();
  }

  void Reset() noexcept;

 private:
  std::string FormatLabels(LabelSpan labels) const;

  struct MetricKey {
    std::string name;
    std::string formatted_labels;

    bool operator==(const MetricKey& other) const {
      return name == other.name && formatted_labels == other.formatted_labels;
    }

    template <typename H>
    friend H AbslHashValue(H h, const MetricKey& k) {
      return H::combine(std::move(h), k.name, k.formatted_labels);
    }
  };

  struct HistogramValue {
    double sum = 0.0;
    int64_t count = 0;
    std::map<double, int64_t> bucket_counts;
  };

  mutable absl::Mutex mutex_;
  std::vector<double> default_buckets_;

  absl::flat_hash_map<MetricKey, double> counters_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<MetricKey, double> gauges_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<MetricKey, HistogramValue> histograms_
      ABSL_GUARDED_BY(mutex_);
};

std::shared_ptr<PrometheusExporter> GetGlobalPrometheusExporter();

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_
