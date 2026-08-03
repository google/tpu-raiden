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

#include <cstdint>
#include <map>  // NOLINT: Required by prometheus-cpp client API.
#include <memory>
#include <string>
#include <utility>

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

namespace {

std::map<std::string, std::string> ConvertLabels(LabelSpan labels) {
  if (labels.empty()) {
    return {};
  }
  std::map<std::string, std::string> result;
  for (const auto& [key, value] : labels) {
    result.emplace(key, value);
  }
  return result;
}

}  // namespace

void PrometheusExporter::RegisterKnownFamilies() {
  for (const auto& meta : metric_metadata::kAllMetrics) {
    switch (meta.type) {
      case MetricType::kCounter: {
        auto* family = &prometheus::BuildCounter()
                            .Name(std::string(meta.prometheus_name))
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        counter_families_.emplace(std::string(meta.name), family);
        counter_families_.emplace(std::string(meta.prometheus_name), family);
        counter_families_.emplace(std::string(meta.streamz_name), family);
        break;
      }
      case MetricType::kGauge: {
        auto* family = &prometheus::BuildGauge()
                            .Name(std::string(meta.prometheus_name))
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        gauge_families_.emplace(std::string(meta.name), family);
        gauge_families_.emplace(std::string(meta.prometheus_name), family);
        gauge_families_.emplace(std::string(meta.streamz_name), family);
        break;
      }
      case MetricType::kHistogram: {
        auto* family = &prometheus::BuildHistogram()
                            .Name(std::string(meta.prometheus_name))
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        histogram_families_.emplace(std::string(meta.name), family);
        histogram_families_.emplace(std::string(meta.prometheus_name), family);
        histogram_families_.emplace(std::string(meta.streamz_name), family);
        break;
      }
    }
  }
}

PrometheusExporter::PrometheusExporter(
    prometheus::Histogram::BucketBoundaries custom_buckets)
    : registry_(std::make_shared<prometheus::Registry>()),
      default_buckets_(std::move(custom_buckets)) {
  RegisterKnownFamilies();
}

prometheus::Family<prometheus::Counter>* PrometheusExporter::GetCounterFamily(
    absl::string_view name) const {
  auto it = counter_families_.find(name);
  if (it == counter_families_.end()) {
    return nullptr;
  }
  return it->second;
}

prometheus::Family<prometheus::Gauge>* PrometheusExporter::GetGaugeFamily(
    absl::string_view name) const {
  auto it = gauge_families_.find(name);
  if (it == gauge_families_.end()) {
    return nullptr;
  }
  return it->second;
}

prometheus::Family<prometheus::Histogram>*
PrometheusExporter::GetHistogramFamily(absl::string_view name) const {
  auto it = histogram_families_.find(name);
  if (it == histogram_families_.end()) {
    return nullptr;
  }
  return it->second;
}

void PrometheusExporter::IncrementCounter(absl::string_view name,
                                          LabelSpan labels,
                                          uint64_t val) const {
  prometheus::Family<prometheus::Counter>* family = nullptr;
  family = GetCounterFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Counter& counter = family->Add(ConvertLabels(labels));
  counter.Increment(static_cast<double>(val));
}

void PrometheusExporter::SetGauge(absl::string_view name, LabelSpan labels,
                                  double val) const {
  prometheus::Family<prometheus::Gauge>* family = nullptr;
  family = GetGaugeFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Gauge& gauge = family->Add(ConvertLabels(labels));
  gauge.Set(val);
}

void PrometheusExporter::ObserveHistogram(absl::string_view name,
                                          LabelSpan labels, double val) const {
  prometheus::Family<prometheus::Histogram>* family = nullptr;
  family = GetHistogramFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Histogram& histogram =
      family->Add(ConvertLabels(labels), default_buckets_);
  histogram.Observe(val);
}

std::string PrometheusExporter::GetTextSnapshot() const {
  prometheus::TextSerializer serializer;
  return serializer.Serialize(registry_->Collect());
}

}  // namespace tpu_raiden::telemetry
