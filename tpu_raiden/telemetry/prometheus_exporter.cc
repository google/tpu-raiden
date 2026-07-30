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

#include "tpu_raiden/telemetry/prometheus_exporter.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

PrometheusExporter::PrometheusExporter(std::vector<double> custom_buckets)
    : default_buckets_(std::move(custom_buckets)) {
  std::sort(default_buckets_.begin(), default_buckets_.end());
}

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

}  // namespace

std::string PrometheusExporter::FormatLabels(LabelSpan labels) const {
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

void PrometheusExporter::IncrementCounter(absl::string_view name, uint64_t val,
                                          LabelSpan labels) noexcept {
  try {
    MetricKey key{std::string(name), FormatLabels(labels)};
    absl::WriterMutexLock lock(&mutex_);
    counters_[key] += static_cast<double>(val);
  } catch (...) {
    // noexcept safety guarantee
  }
}

void PrometheusExporter::SetGauge(absl::string_view name, int64_t val,
                                  LabelSpan labels) noexcept {
  try {
    MetricKey key{std::string(name), FormatLabels(labels)};
    absl::WriterMutexLock lock(&mutex_);
    gauges_[key] = static_cast<double>(val);
  } catch (...) {
    // noexcept safety guarantee
  }
}

void PrometheusExporter::ObserveHistogram(absl::string_view name, double val,
                                          LabelSpan labels) noexcept {
  try {
    MetricKey key{std::string(name), FormatLabels(labels)};
    absl::WriterMutexLock lock(&mutex_);
    auto& hist = histograms_[key];
    if (hist.bucket_counts.empty()) {
      for (double b : default_buckets_) {
        hist.bucket_counts[b] = 0;
      }
    }
    hist.sum += val;
    hist.count += 1;
    for (auto it = hist.bucket_counts.lower_bound(val);
         it != hist.bucket_counts.end(); ++it) {
      it->second += 1;
    }

  } catch (...) {
    // noexcept safety guarantee
  }
}

void PrometheusExporter::Reset() noexcept {
  try {
    absl::WriterMutexLock lock(&mutex_);
    counters_.clear();
    gauges_.clear();
    histograms_.clear();
  } catch (...) {
    // noexcept safety guarantee
  }
}

std::string PrometheusExporter::GetPrometheusTextSnapshot() {
  try {
    absl::ReaderMutexLock lock(&mutex_);
    std::string text;

    // Group and export counter metrics
    absl::flat_hash_map<std::string, std::vector<std::pair<MetricKey, double>>>
        grouped_counters;
    for (const auto& [key, val] : counters_) {
      grouped_counters[key.name].push_back({key, val});
    }
    for (const auto& [name, items] : grouped_counters) {
      absl::StrAppend(&text, "# HELP ", name, " TPU Raiden counter metric\n");
      absl::StrAppend(&text, "# TYPE ", name, " counter\n");
      for (const auto& [key, val] : items) {
        absl::StrAppend(&text, name, key.formatted_labels, " ", val, "\n");
      }
    }

    // Group and export gauge metrics
    absl::flat_hash_map<std::string, std::vector<std::pair<MetricKey, double>>>
        grouped_gauges;
    for (const auto& [key, val] : gauges_) {
      grouped_gauges[key.name].push_back({key, val});
    }
    for (const auto& [name, items] : grouped_gauges) {
      absl::StrAppend(&text, "# HELP ", name, " TPU Raiden gauge metric\n");
      absl::StrAppend(&text, "# TYPE ", name, " gauge\n");
      for (const auto& [key, val] : items) {
        absl::StrAppend(&text, name, key.formatted_labels, " ", val, "\n");
      }
    }

    // Group and export histogram metrics
    absl::flat_hash_map<std::string,
                        std::vector<std::pair<MetricKey, HistogramValue>>>
        grouped_histograms;
    for (const auto& [key, val] : histograms_) {
      grouped_histograms[key.name].push_back({key, val});
    }
    for (const auto& [name, items] : grouped_histograms) {
      absl::StrAppend(&text, "# HELP ", name, " TPU Raiden histogram metric\n");
      absl::StrAppend(&text, "# TYPE ", name, " histogram\n");
      for (const auto& [key, hist] : items) {
        std::string base_labels = key.formatted_labels;
        for (const auto& [b_upper, count] : hist.bucket_counts) {
          std::string b_str = absl::StrCat(b_upper);

          if (base_labels.empty()) {
            absl::StrAppend(&text, name, "_bucket{le=\"", b_str, "\"} ", count,
                            "\n");
          } else {
            std::string l = base_labels.substr(0, base_labels.size() - 1);
            absl::StrAppend(&text, name, "_bucket", l, ",le=\"", b_str, "\"} ",
                            count, "\n");
          }
        }
        if (base_labels.empty()) {
          absl::StrAppend(&text, name, "_bucket{le=\"+Inf\"} ", hist.count,
                          "\n");
        } else {
          std::string l = base_labels.substr(0, base_labels.size() - 1);
          absl::StrAppend(&text, name, "_bucket", l, ",le=\"+Inf\"} ",
                          hist.count, "\n");
        }
        absl::StrAppend(&text, name, "_sum", base_labels, " ", hist.sum, "\n");
        absl::StrAppend(&text, name, "_count", base_labels, " ", hist.count,
                        "\n");
      }
    }

    return text;
  } catch (...) {
    return "";
  }
}

std::shared_ptr<PrometheusExporter> GetGlobalPrometheusExporter() {
  static auto global_exporter = std::make_shared<PrometheusExporter>();
  return global_exporter;
}

}  // namespace tpu_raiden::telemetry
