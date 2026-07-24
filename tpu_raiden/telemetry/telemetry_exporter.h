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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_TELEMETRY_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_TELEMETRY_EXPORTER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace tpu_raiden::telemetry {

// Enum representing the metric types supported by TelemetryExporter.
enum class MetricType {
  kCounter,
  kGauge,
  kHistogram,
};

std::string MetricTypeToString(MetricType type);

// Data structure representing a single metric measurement.
struct MetricValue {
  std::string name;
  std::string description;
  MetricType type = MetricType::kCounter;
  double value = 0.0;
  absl::flat_hash_map<std::string, std::string> labels;
  absl::Time timestamp = absl::UnixEpoch();
};

// Data structure representing a batch of metric measurements from a source.
struct MetricBatch {
  std::string source_id;
  std::vector<MetricValue> metrics;
  absl::Time timestamp = absl::UnixEpoch();
};

// Abstract interface for telemetry exporters.
class TelemetryExporter {
 public:
  virtual ~TelemetryExporter() = default;

  // Exports a single metric observation.
  virtual absl::Status Export(const MetricValue& metric) = 0;

  // Exports a batch of metric observations.
  virtual absl::Status ExportBatch(absl::Span<const MetricValue> metrics) {
    for (const auto& metric : metrics) {
      absl::Status status = Export(metric);
      if (!status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  }

  // Overload for exporting a structured MetricBatch.
  virtual absl::Status ExportBatch(MetricBatch batch) {
    if (batch.source_id.empty() && batch.timestamp == absl::UnixEpoch()) {
      return ExportBatch(absl::MakeConstSpan(batch.metrics));
    }
    for (auto& metric : batch.metrics) {
      if (!batch.source_id.empty()) {
        metric.labels.try_emplace("source_id", batch.source_id);
      }
      if (metric.timestamp == absl::UnixEpoch() &&
          batch.timestamp != absl::UnixEpoch()) {
        metric.timestamp = batch.timestamp;
      }
    }
    return ExportBatch(absl::MakeConstSpan(batch.metrics));
  }

  // Flushes any buffered metric data.
  virtual absl::Status Flush() { return absl::OkStatus(); }

  // Returns whether the exporter is enabled.
  virtual bool IsEnabled() const = 0;

  // Enables or disables the exporter.
  virtual void SetEnabled(bool enabled) = 0;

  // Convenience helper methods for recording metrics.
  absl::Status RecordCounter(
      const std::string& name, double delta,
      const absl::flat_hash_map<std::string, std::string>& labels = {},
      const std::string& description = "") {
    MetricValue mv;
    mv.name = name;
    mv.description = description;
    mv.type = MetricType::kCounter;
    mv.value = delta;
    mv.labels = labels;
    mv.timestamp = absl::Now();
    return Export(mv);
  }

  absl::Status RecordGauge(
      const std::string& name, double value,
      const absl::flat_hash_map<std::string, std::string>& labels = {},
      const std::string& description = "") {
    MetricValue mv;
    mv.name = name;
    mv.description = description;
    mv.type = MetricType::kGauge;
    mv.value = value;
    mv.labels = labels;
    mv.timestamp = absl::Now();
    return Export(mv);
  }

  absl::Status RecordHistogram(
      const std::string& name, double value,
      const absl::flat_hash_map<std::string, std::string>& labels = {},
      const std::string& description = "") {
    MetricValue mv;
    mv.name = name;
    mv.description = description;
    mv.type = MetricType::kHistogram;
    mv.value = value;
    mv.labels = labels;
    mv.timestamp = absl::Now();
    return Export(mv);
  }
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_TELEMETRY_EXPORTER_H_
