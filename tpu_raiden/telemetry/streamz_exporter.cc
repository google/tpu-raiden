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

#include "tpu_raiden/telemetry/streamz_exporter.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tpu_raiden::telemetry {

StreamzExporter::StreamzExporter(StreamzExporterOptions options)
    : options_(std::move(options)) {}

StreamzExporter::StreamzExporter(std::string root_prefix) {
  options_.root_prefix = std::move(root_prefix);
}

absl::Status StreamzExporter::Export(const MetricValue& metric) {
  if (metric.name.empty()) {
    return absl::InvalidArgumentError("Metric name cannot be empty.");
  }

  std::string full_path = metric.name;
  if (!absl::StartsWith(metric.name, "/")) {
    std::string prefix = options_.root_prefix;
    if (absl::EndsWith(prefix, "/")) {
      prefix.pop_back();
    }
    full_path = absl::StrCat(prefix, "/", metric.name);
  }

  std::map<std::string, std::string> sorted_labels(metric.labels.begin(),
                                                    metric.labels.end());
  CellKey key{full_path, metric.type, sorted_labels};

  absl::MutexLock lock(&mutex_);
  if (!enabled_) {
    return absl::OkStatus();
  }

  auto& cell = cells_[key];
  cell.metric_path = full_path;
  cell.type = metric.type;
  cell.labels = sorted_labels;
  cell.sample_count++;
  cell.last_updated = metric.timestamp;

  if (metric.type == MetricType::kCounter) {
    cell.value += metric.value;
  } else if (metric.type == MetricType::kGauge) {
    cell.value = metric.value;
  } else if (metric.type == MetricType::kHistogram) {
    cell.value += metric.value;
  }

  return absl::OkStatus();
}

absl::Status StreamzExporter::Flush() {
  return absl::OkStatus();
}

bool StreamzExporter::IsEnabled() const {
  absl::MutexLock lock(&mutex_);
  return enabled_;
}

void StreamzExporter::SetEnabled(bool enabled) {
  absl::MutexLock lock(&mutex_);
  enabled_ = enabled;
}

std::vector<StreamzCellSnapshot> StreamzExporter::GetSnapshot() const {
  absl::MutexLock lock(&mutex_);
  std::vector<StreamzCellSnapshot> result;
  result.reserve(cells_.size());
  for (const auto& [key, cell] : cells_) {
    result.push_back(cell);
  }
  return result;
}

double StreamzExporter::GetCellValue(
    const std::string& name,
    const absl::flat_hash_map<std::string, std::string>& labels) const {
  std::string full_path = name;
  if (!absl::StartsWith(name, "/")) {
    std::string prefix = options_.root_prefix;
    if (absl::EndsWith(prefix, "/")) {
      prefix.pop_back();
    }
    full_path = absl::StrCat(prefix, "/", name);
  }

  std::map<std::string, std::string> sorted_labels(labels.begin(), labels.end());
  absl::MutexLock lock(&mutex_);

  for (MetricType type :
       {MetricType::kCounter, MetricType::kGauge, MetricType::kHistogram}) {
    CellKey key{full_path, type, sorted_labels};
    auto it = cells_.find(key);
    if (it != cells_.end()) {
      return it->second.value;
    }
  }
  return 0.0;
}

size_t StreamzExporter::GetRecordCount() const {
  absl::MutexLock lock(&mutex_);
  return cells_.size();
}

std::string StreamzExporter::ExportStreamzSummary() const {
  absl::MutexLock lock(&mutex_);
  std::string summary = "Streamz Telemetry Summary:\n";

  for (const auto& [key, cell] : cells_) {
    std::string labels_str;
    if (!cell.labels.empty()) {
      std::vector<std::string> label_pairs;
      for (const auto& [k, v] : cell.labels) {
        label_pairs.push_back(absl::StrFormat("%s=\"%s\"", k, v));
      }
      labels_str = absl::StrCat(" {", absl::StrJoin(label_pairs, ", "), "}");
    }

    absl::StrAppend(
        &summary,
        absl::StrFormat("  Cell: %s [%s] Value: %g Samples: %v%s\n",
                        cell.metric_path, MetricTypeToString(cell.type),
                        cell.value, cell.sample_count, labels_str));
  }

  return summary;
}

void StreamzExporter::Clear() {
  absl::MutexLock lock(&mutex_);
  cells_.clear();
}

}  // namespace tpu_raiden::telemetry
