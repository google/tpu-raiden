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
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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

namespace prometheus_names {

inline constexpr absl::string_view kSentBytesTotal =
    "tpu_raiden_sent_bytes_total";
inline constexpr absl::string_view kReceivedBytesTotal =
    "tpu_raiden_received_bytes_total";
inline constexpr absl::string_view kTransferDurationSeconds =
    "tpu_raiden_transfer_duration_seconds";
inline constexpr absl::string_view kStageLatencySeconds =
    "tpu_raiden_stage_latency_seconds";
inline constexpr absl::string_view kActiveTransfers =
    "tpu_raiden_active_transfers";
inline constexpr absl::string_view kBufferOccupancyBytes =
    "tpu_raiden_buffer_occupancy_bytes";
inline constexpr absl::string_view kTransferFailuresTotal =
    "tpu_raiden_transfer_failures_total";

}  // namespace prometheus_names

PrometheusMetricInfo GetPrometheusMetricInfo(absl::string_view name) {
  absl::string_view prom_name = name;
  absl::string_view desc = GetMetricDescription(name);
  if (name == metric_names::kSentBytesTotal ||
      name == prometheus_names::kSentBytesTotal) {
    prom_name = prometheus_names::kSentBytesTotal;
    desc = GetMetricDescription(metric_names::kSentBytesTotal);
  } else if (name == metric_names::kReceivedBytesTotal ||
             name == prometheus_names::kReceivedBytesTotal) {
    prom_name = prometheus_names::kReceivedBytesTotal;
    desc = GetMetricDescription(metric_names::kReceivedBytesTotal);
  } else if (name == metric_names::kTransferDurationSeconds ||
             name == prometheus_names::kTransferDurationSeconds) {
    prom_name = prometheus_names::kTransferDurationSeconds;
    desc = GetMetricDescription(metric_names::kTransferDurationSeconds);
  } else if (name == metric_names::kStageLatencySeconds ||
             name == prometheus_names::kStageLatencySeconds) {
    prom_name = prometheus_names::kStageLatencySeconds;
    desc = GetMetricDescription(metric_names::kStageLatencySeconds);
  } else if (name == metric_names::kActiveTransfers ||
             name == prometheus_names::kActiveTransfers) {
    prom_name = prometheus_names::kActiveTransfers;
    desc = GetMetricDescription(metric_names::kActiveTransfers);
  } else if (name == metric_names::kBufferOccupancyBytes ||
             name == prometheus_names::kBufferOccupancyBytes) {
    prom_name = prometheus_names::kBufferOccupancyBytes;
    desc = GetMetricDescription(metric_names::kBufferOccupancyBytes);
  } else if (name == metric_names::kTransferFailuresTotal ||
             name == prometheus_names::kTransferFailuresTotal) {
    prom_name = prometheus_names::kTransferFailuresTotal;
    desc = GetMetricDescription(metric_names::kTransferFailuresTotal);
  }
  return PrometheusMetricInfo{.prometheus_name = prom_name, .help_text = desc};
}

void PrometheusExporter::RegisterKnownFamilies() {
  sent_bytes_family_ =
      &prometheus::BuildCounter()
           .Name(std::string(prometheus_names::kSentBytesTotal))
           .Help(std::string(
               GetMetricDescription(metric_names::kSentBytesTotal)))
           .Register(*registry_);
  received_bytes_family_ =
      &prometheus::BuildCounter()
           .Name(std::string(prometheus_names::kReceivedBytesTotal))
           .Help(std::string(
               GetMetricDescription(metric_names::kReceivedBytesTotal)))
           .Register(*registry_);
  transfer_failures_family_ =
      &prometheus::BuildCounter()
           .Name(std::string(prometheus_names::kTransferFailuresTotal))
           .Help(std::string(
               GetMetricDescription(metric_names::kTransferFailuresTotal)))
           .Register(*registry_);

  active_transfers_family_ =
      &prometheus::BuildGauge()
           .Name(std::string(prometheus_names::kActiveTransfers))
           .Help(std::string(
               GetMetricDescription(metric_names::kActiveTransfers)))
           .Register(*registry_);
  buffer_occupancy_family_ =
      &prometheus::BuildGauge()
           .Name(std::string(prometheus_names::kBufferOccupancyBytes))
           .Help(std::string(
               GetMetricDescription(metric_names::kBufferOccupancyBytes)))
           .Register(*registry_);

  transfer_duration_family_ =
      &prometheus::BuildHistogram()
           .Name(std::string(prometheus_names::kTransferDurationSeconds))
           .Help(std::string(GetMetricDescription(
               metric_names::kTransferDurationSeconds)))
           .Register(*registry_);
  stage_latency_family_ =
      &prometheus::BuildHistogram()
           .Name(std::string(prometheus_names::kStageLatencySeconds))
           .Help(std::string(
               GetMetricDescription(metric_names::kStageLatencySeconds)))
           .Register(*registry_);
}

PrometheusExporter::PrometheusExporter(
    prometheus::Histogram::BucketBoundaries custom_buckets)
    : registry_(std::make_shared<prometheus::Registry>()),
      default_buckets_(std::move(custom_buckets)) {
  absl::MutexLock lock(&mutex_);
  RegisterKnownFamilies();
}

prometheus::Family<prometheus::Counter>* PrometheusExporter::GetCounterFamily(
    absl::string_view name) const {
  if (name == metric_names::kSentBytesTotal ||
      name == prometheus_names::kSentBytesTotal) {
    return sent_bytes_family_;
  }
  if (name == metric_names::kReceivedBytesTotal ||
      name == prometheus_names::kReceivedBytesTotal) {
    return received_bytes_family_;
  }
  if (name == metric_names::kTransferFailuresTotal ||
      name == prometheus_names::kTransferFailuresTotal) {
    return transfer_failures_family_;
  }
  PrometheusMetricInfo info = GetPrometheusMetricInfo(name);
  auto it = counter_families_.find(info.prometheus_name);
  if (it != counter_families_.end()) {
    return it->second;
  }
  auto* family = &prometheus::BuildCounter()
                      .Name(std::string(info.prometheus_name))
                      .Help(std::string(info.help_text))
                      .Register(*registry_);
  counter_families_.emplace(std::string(info.prometheus_name), family);
  return family;
}

prometheus::Family<prometheus::Gauge>* PrometheusExporter::GetGaugeFamily(
    absl::string_view name) const {
  if (name == metric_names::kActiveTransfers ||
      name == prometheus_names::kActiveTransfers) {
    return active_transfers_family_;
  }
  if (name == metric_names::kBufferOccupancyBytes ||
      name == prometheus_names::kBufferOccupancyBytes) {
    return buffer_occupancy_family_;
  }
  PrometheusMetricInfo info = GetPrometheusMetricInfo(name);
  auto it = gauge_families_.find(info.prometheus_name);
  if (it != gauge_families_.end()) {
    return it->second;
  }
  auto* family = &prometheus::BuildGauge()
                      .Name(std::string(info.prometheus_name))
                      .Help(std::string(info.help_text))
                      .Register(*registry_);
  gauge_families_.emplace(std::string(info.prometheus_name), family);
  return family;
}

prometheus::Family<prometheus::Histogram>*
PrometheusExporter::GetHistogramFamily(absl::string_view name) const {
  if (name == metric_names::kTransferDurationSeconds ||
      name == prometheus_names::kTransferDurationSeconds) {
    return transfer_duration_family_;
  }
  if (name == metric_names::kStageLatencySeconds ||
      name == prometheus_names::kStageLatencySeconds) {
    return stage_latency_family_;
  }
  PrometheusMetricInfo info = GetPrometheusMetricInfo(name);
  auto it = histogram_families_.find(info.prometheus_name);
  if (it != histogram_families_.end()) {
    return it->second;
  }
  auto* family = &prometheus::BuildHistogram()
                      .Name(std::string(info.prometheus_name))
                      .Help(std::string(info.help_text))
                      .Register(*registry_);
  histogram_families_.emplace(std::string(info.prometheus_name), family);
  return family;
}

void PrometheusExporter::IncrementCounter(absl::string_view name,
                                          LabelSpan labels,
                                          uint64_t val) const {
  std::shared_ptr<prometheus::Registry> current_registry;
  prometheus::Family<prometheus::Counter>* family = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    current_registry = registry_;
    family = GetCounterFamily(name);
  }
  prometheus::Counter& counter = family->Add(ConvertLabels(labels));
  counter.Increment(static_cast<double>(val));
}

void PrometheusExporter::SetGauge(absl::string_view name, LabelSpan labels,
                                  double val) const {
  std::shared_ptr<prometheus::Registry> current_registry;
  prometheus::Family<prometheus::Gauge>* family = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    current_registry = registry_;
    family = GetGaugeFamily(name);
  }
  prometheus::Gauge& gauge = family->Add(ConvertLabels(labels));
  gauge.Set(val);
}

void PrometheusExporter::ObserveHistogram(absl::string_view name,
                                          LabelSpan labels, double val) const {
  std::shared_ptr<prometheus::Registry> current_registry;
  prometheus::Family<prometheus::Histogram>* family = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    current_registry = registry_;
    family = GetHistogramFamily(name);
  }
  prometheus::Histogram& histogram =
      family->Add(ConvertLabels(labels), default_buckets_);
  histogram.Observe(val);
}

void PrometheusExporter::Reset() {
  absl::MutexLock lock(&mutex_);
  registry_ = std::make_shared<prometheus::Registry>();
  counter_families_.clear();
  gauge_families_.clear();
  histogram_families_.clear();
  RegisterKnownFamilies();
}

std::string PrometheusExporter::GetTextSnapshot() const {
  std::shared_ptr<prometheus::Registry> reg;
  {
    absl::MutexLock lock(&mutex_);
    reg = registry_;
  }
  prometheus::TextSerializer serializer;
  return serializer.Serialize(reg->Collect());
}

PrometheusExporter& GetGlobalPrometheusExporter() {
  static absl::NoDestructor<PrometheusExporter> global_exporter;
  return *global_exporter;
}

}  // namespace tpu_raiden::telemetry
