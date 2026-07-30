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

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

namespace {

std::string EscapeLabelValue(absl::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '"') {
      escaped.push_back('\\');
    } else if (c == '\n') {
      escaped.push_back('\\');
      escaped.push_back('n');
      continue;
    }
    escaped.push_back(c);
  }
  return escaped;
}

// Maps canonical 3P snake_case metric names to 1P Streamz slash paths (LLD
// Section 2.1.3 & Section 4)
std::string MapTo1PStreamzPath(absl::string_view name) {
  static const auto* kPathMap =
      new absl::flat_hash_map<absl::string_view, absl::string_view>{
          {kMetricSentBytesTotal, "/tpu_raiden/kv_cache/sent_bytes_total"},
          {kMetricReceivedBytesTotal,
           "/tpu_raiden/kv_cache/received_bytes_total"},
          {kMetricTransferDurationSeconds,
           "/tpu_raiden/kv_cache/transfer_duration_seconds"},
          {kMetricStageLatencySeconds,
           "/tpu_raiden/kv_cache/stage_latency_seconds"},
          {kMetricActiveTransfers, "/tpu_raiden/kv_cache/active_transfers"},
          {kMetricBufferOccupancyBytes,
           "/tpu_raiden/kv_cache/buffer_occupancy_bytes"},
          {kMetricTransferFailuresTotal,
           "/tpu_raiden/kv_cache/transfer_failures_total"},
      };

  auto it = kPathMap->find(name);
  if (it != kPathMap->end()) {
    return std::string(it->second);
  }
  return absl::StrCat("/tpu_raiden/", name);
}

}  // namespace

std::string StreamzExporter::FormatLabels(LabelSpan labels) const {
  if (labels.empty()) return "";
  std::vector<std::pair<absl::string_view, absl::string_view>> sorted_labels(
      labels.begin(), labels.end());
  std::sort(sorted_labels.begin(), sorted_labels.end());

  std::string result = "{";
  for (size_t i = 0; i < sorted_labels.size(); ++i) {
    if (i > 0) absl::StrAppend(&result, ",");
    absl::StrAppend(&result, sorted_labels[i].first, "=\"",
                    EscapeLabelValue(sorted_labels[i].second), "\"");
  }
  absl::StrAppend(&result, "}");
  return result;
}

void StreamzExporter::IncrementCounter(absl::string_view name, uint64_t val,
                                       LabelSpan labels) noexcept {
  try {
    std::string key =
        absl::StrCat(MapTo1PStreamzPath(name), FormatLabels(labels));
    absl::WriterMutexLock lock(&mutex_);
    counters_[key] += val;
  } catch (...) {
    // noexcept safety
  }
}

void StreamzExporter::SetGauge(absl::string_view name, int64_t val,
                               LabelSpan labels) noexcept {
  try {
    std::string key =
        absl::StrCat(MapTo1PStreamzPath(name), FormatLabels(labels));
    absl::WriterMutexLock lock(&mutex_);
    gauges_[key] = val;
  } catch (...) {
    // noexcept safety
  }
}

void StreamzExporter::ObserveHistogram(absl::string_view name, double val,
                                       LabelSpan labels) noexcept {
  try {
    std::string key =
        absl::StrCat(MapTo1PStreamzPath(name), FormatLabels(labels));
    absl::WriterMutexLock lock(&mutex_);
    auto& dist = histograms_[key];
    dist.sum += val;
    dist.count += 1;
  } catch (...) {
    // noexcept safety
  }
}

void Initialize1PBackend(bool enable_1p_telemetry) noexcept {
  try {
    if (enable_1p_telemetry) {
      auto exporter = std::make_shared<Streamz1PBackend>();
      RaidenMetricStore::GetGlobalMetricStore().AddBackend(exporter);
    }
  } catch (...) {
    // noexcept safety
  }
}

}  // namespace tpu_raiden::telemetry
