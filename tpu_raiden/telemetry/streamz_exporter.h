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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_STREAMZ_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_STREAMZ_EXPORTER_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tpu_raiden/telemetry/telemetry_exporter.h"

namespace tpu_raiden::telemetry {

struct StreamzExporterOptions {
  std::string root_prefix = "/tpu_raiden";
};

// Structure holding snapshot data for a Streamz metric cell.
struct StreamzCellSnapshot {
  std::string metric_path;
  MetricType type;
  std::map<std::string, std::string> labels;
  double value = 0.0;
  uint64_t sample_count = 0;
  absl::Time last_updated = absl::Now();
};

class StreamzExporter : public TelemetryExporter {
 public:
  explicit StreamzExporter(
      StreamzExporterOptions options = StreamzExporterOptions());
  explicit StreamzExporter(std::string root_prefix);
  ~StreamzExporter() override = default;

  absl::Status Export(const MetricValue& metric) override;
  absl::Status Flush() override;

  bool IsEnabled() const override;
  void SetEnabled(bool enabled) override;

  // Returns all metric cell snapshots currently tracked by the Streamz exporter.
  std::vector<StreamzCellSnapshot> GetSnapshot() const;

  // Returns the value for a specific cell identified by metric name and label map.
  double GetCellValue(
      const std::string& name,
      const absl::flat_hash_map<std::string, std::string>& labels = {}) const;

  // Returns total number of active Streamz metric cells.
  size_t GetRecordCount() const;

  // Formats recorded Streamz metrics into a diagnostic summary text string.
  std::string ExportStreamzSummary() const;

  // Clears all recorded cells.
  void Clear();

 private:
  struct CellKey {
    std::string full_path;
    MetricType type;
    std::map<std::string, std::string> sorted_labels;

    bool operator==(const CellKey& other) const {
      return full_path == other.full_path && type == other.type &&
             sorted_labels == other.sorted_labels;
    }

    template <typename H>
    friend H AbslHashValue(H h, const CellKey& k) {
      return H::combine(std::move(h), k.full_path, static_cast<int>(k.type),
                        k.sorted_labels);
    }
  };

  StreamzExporterOptions options_;
  mutable absl::Mutex mutex_;
  bool enabled_ ABSL_GUARDED_BY(mutex_) = true;
  absl::flat_hash_map<CellKey, StreamzCellSnapshot> cells_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_STREAMZ_EXPORTER_H_
