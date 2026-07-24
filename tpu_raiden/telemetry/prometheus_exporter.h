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

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/telemetry_exporter.h"

namespace tpu_raiden::telemetry {

struct PrometheusExporterOptions {
  std::string metric_prefix = "";
  bool include_help_text = true;
  bool include_type_declaration = true;
};

class PrometheusExporter : public TelemetryExporter {
 public:
  explicit PrometheusExporter(
      PrometheusExporterOptions options = PrometheusExporterOptions());
  ~PrometheusExporter() override = default;

  absl::Status Export(const MetricValue& metric) override;
  absl::Status Flush() override;

  bool IsEnabled() const override;
  void SetEnabled(bool enabled) override;

  // Generates the Prometheus text exposition format of recorded metrics.
  std::string CollectExpositionFormat() const;

  // Clears all recorded metrics.
  void Clear();

  // Returns current recorded value for a given metric name and label map.
  double GetMetricValue(
      const std::string& name,
      const absl::flat_hash_map<std::string, std::string>& labels = {}) const;

  // Returns total number of distinct metric entries recorded.
  size_t GetMetricCount() const;

  // Sanitizes metric names to conform to Prometheus naming conventions ([a-zA-Z0-9_:]).
  static std::string SanitizeMetricName(const std::string& name);

 private:
  struct MetricKey {
    std::string name;
    MetricType type;
    std::map<std::string, std::string> sorted_labels;

    bool operator==(const MetricKey& other) const {
      return name == other.name && type == other.type &&
             sorted_labels == other.sorted_labels;
    }

    template <typename H>
    friend H AbslHashValue(H h, const MetricKey& k) {
      return H::combine(std::move(h), k.name, static_cast<int>(k.type),
                        k.sorted_labels);
    }
  };

  struct HistogramData {
    uint64_t count = 0;
    double sum = 0.0;
    std::map<double, uint64_t> bucket_counts;
  };

  PrometheusExporterOptions options_;
  mutable absl::Mutex mutex_;
  bool enabled_ ABSL_GUARDED_BY(mutex_) = true;
  absl::flat_hash_map<MetricKey, double> metric_values_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<MetricKey, HistogramData> histograms_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, std::string> metric_descriptions_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_
