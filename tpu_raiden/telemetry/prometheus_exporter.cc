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
#include <cctype>
#include <cstddef>
#include <cstdint>
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

namespace tpu_raiden::telemetry {

namespace {

const std::vector<double>& DefaultHistogramBuckets() {
  static const auto* buckets = new std::vector<double>{
      0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0};
  return *buckets;
}

std::string FormatLabels(const std::map<std::string, std::string>& sorted_labels,
                         const std::string& extra_label = "") {
  std::vector<std::string> parts;
  for (const auto& [k, v] : sorted_labels) {
    parts.push_back(absl::StrFormat("%s=\"%s\"", k, v));
  }
  if (!extra_label.empty()) {
    parts.push_back(extra_label);
  }
  if (parts.empty()) {
    return "";
  }
  return absl::StrCat("{", absl::StrJoin(parts, ","), "}");
}

}  // namespace

PrometheusExporter::PrometheusExporter(PrometheusExporterOptions options)
    : options_(std::move(options)) {}

std::string PrometheusExporter::SanitizeMetricName(const std::string& name) {
  std::string result;
  result.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == ':') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }
  return result;
}

absl::Status PrometheusExporter::Export(const MetricValue& metric) {
  if (metric.name.empty()) {
    return absl::InvalidArgumentError("Metric name cannot be empty.");
  }

  std::string sanitized_name = SanitizeMetricName(metric.name);
  if (!options_.metric_prefix.empty() &&
      !absl::StartsWith(sanitized_name, options_.metric_prefix)) {
    sanitized_name = absl::StrCat(options_.metric_prefix, sanitized_name);
  }

  std::map<std::string, std::string> sorted_labels(metric.labels.begin(),
                                                    metric.labels.end());
  MetricKey key{sanitized_name, metric.type, sorted_labels};

  absl::MutexLock lock(&mutex_);
  if (!enabled_) {
    return absl::OkStatus();
  }

  if (!metric.description.empty()) {
    metric_descriptions_[sanitized_name] = metric.description;
  }

  if (metric.type == MetricType::kCounter) {
    metric_values_[key] += metric.value;
  } else if (metric.type == MetricType::kGauge) {
    metric_values_[key] = metric.value;
  } else if (metric.type == MetricType::kHistogram) {
    auto& hist = histograms_[key];
    hist.count++;
    hist.sum += metric.value;
    for (double bucket : DefaultHistogramBuckets()) {
      if (metric.value <= bucket) {
        hist.bucket_counts[bucket]++;
      }
    }
  }

  return absl::OkStatus();
}

absl::Status PrometheusExporter::Flush() {
  return absl::OkStatus();
}

bool PrometheusExporter::IsEnabled() const {
  absl::MutexLock lock(&mutex_);
  return enabled_;
}

void PrometheusExporter::SetEnabled(bool enabled) {
  absl::MutexLock lock(&mutex_);
  enabled_ = enabled;
}

std::string PrometheusExporter::CollectExpositionFormat() const {
  absl::MutexLock lock(&mutex_);
  std::string output;

  absl::flat_hash_map<std::string, std::string> exported_types;

  for (const auto& [key, val] : metric_values_) {
    if (exported_types.find(key.name) == exported_types.end()) {
      std::string type_str = MetricTypeToString(key.type);
      if (options_.include_help_text) {
        auto desc_it = metric_descriptions_.find(key.name);
        std::string desc = (desc_it != metric_descriptions_.end())
                               ? desc_it->second
                               : absl::StrCat("Metric ", key.name);
        absl::StrAppend(&output, absl::StrFormat("# HELP %s %s\n", key.name, desc));
      }
      if (options_.include_type_declaration) {
        absl::StrAppend(&output, absl::StrFormat("# TYPE %s %s\n", key.name, type_str));
      }
      exported_types[key.name] = type_str;
    }
    std::string label_str = FormatLabels(key.sorted_labels);
    absl::StrAppend(&output, absl::StrFormat("%s%s %g\n", key.name, label_str, val));
  }

  for (const auto& [key, hist] : histograms_) {
    if (exported_types.find(key.name) == exported_types.end()) {
      if (options_.include_help_text) {
        auto desc_it = metric_descriptions_.find(key.name);
        std::string desc = (desc_it != metric_descriptions_.end())
                               ? desc_it->second
                               : absl::StrCat("Metric ", key.name);
        absl::StrAppend(&output, absl::StrFormat("# HELP %s %s\n", key.name, desc));
      }
      if (options_.include_type_declaration) {
        absl::StrAppend(&output, absl::StrFormat("# TYPE %s histogram\n", key.name));
      }
      exported_types[key.name] = "histogram";
    }

    for (double bucket : DefaultHistogramBuckets()) {
      uint64_t bucket_count = 0;
      auto it = hist.bucket_counts.find(bucket);
      if (it != hist.bucket_counts.end()) {
        bucket_count = it->second;
      }
      std::string extra_label = absl::StrFormat("le=\"%g\"", bucket);
      std::string label_str = FormatLabels(key.sorted_labels, extra_label);
      absl::StrAppend(&output, absl::StrFormat("%s_bucket%s %v\n", key.name, label_str, bucket_count));
    }
    std::string inf_label = FormatLabels(key.sorted_labels, "le=\"+Inf\"");
    absl::StrAppend(&output, absl::StrFormat("%s_bucket%s %v\n", key.name, inf_label, hist.count));
    std::string base_label = FormatLabels(key.sorted_labels);
    absl::StrAppend(&output, absl::StrFormat("%s_sum%s %g\n", key.name, base_label, hist.sum));
    absl::StrAppend(&output, absl::StrFormat("%s_count%s %v\n", key.name, base_label, hist.count));
  }

  return output;
}

void PrometheusExporter::Clear() {
  absl::MutexLock lock(&mutex_);
  metric_values_.clear();
  histograms_.clear();
  metric_descriptions_.clear();
}

double PrometheusExporter::GetMetricValue(
    const std::string& name,
    const absl::flat_hash_map<std::string, std::string>& labels) const {
  std::string sanitized_name = SanitizeMetricName(name);
  if (!options_.metric_prefix.empty() &&
      !absl::StartsWith(sanitized_name, options_.metric_prefix)) {
    sanitized_name = absl::StrCat(options_.metric_prefix, sanitized_name);
  }

  std::map<std::string, std::string> sorted_labels(labels.begin(), labels.end());
  absl::MutexLock lock(&mutex_);

  for (MetricType type : {MetricType::kCounter, MetricType::kGauge}) {
    MetricKey key{sanitized_name, type, sorted_labels};
    auto it = metric_values_.find(key);
    if (it != metric_values_.end()) {
      return it->second;
    }
  }
  return 0.0;
}

size_t PrometheusExporter::GetMetricCount() const {
  absl::MutexLock lock(&mutex_);
  return metric_values_.size() + histograms_.size();
}

}  // namespace tpu_raiden::telemetry
