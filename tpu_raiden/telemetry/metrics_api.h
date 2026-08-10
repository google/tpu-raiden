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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_METRICS_API_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_METRICS_API_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace tpu_raiden::telemetry {

inline constexpr absl::string_view kPrometheus = "prometheus";

enum class MetricType {
  kCounter,
  kGauge,
  kHistogram,
};

// Structure defining centralized metadata for a Raiden metric.
struct MetricMetadata {
  absl::string_view name;
  absl::string_view description;
  MetricType type;
};

namespace metric_names {

inline constexpr absl::string_view kSentBytesTotal = "sent_bytes_total";
inline constexpr absl::string_view kReceivedBytesTotal = "received_bytes_total";
inline constexpr absl::string_view kTransferDurationSeconds =
    "transfer_duration_seconds";
inline constexpr absl::string_view kStageLatencySeconds =
    "stage_latency_seconds";
inline constexpr absl::string_view kActiveTransfers = "active_transfers";
inline constexpr absl::string_view kBufferOccupancyBytes =
    "buffer_occupancy_bytes";
inline constexpr absl::string_view kTransferFailuresTotal =
    "transfer_failures_total";

}  // namespace metric_names

namespace metric_descriptions {

inline constexpr absl::string_view kSentBytesTotal =
    "Total count of bytes sent over TPU Raiden interfaces.";
inline constexpr absl::string_view kReceivedBytesTotal =
    "Total count of bytes received over TPU Raiden interfaces.";
inline constexpr absl::string_view kTransferDurationSeconds =
    "Total end-to-end duration of KV cache transfers in seconds.";
inline constexpr absl::string_view kStageLatencySeconds =
    "Latency breakdown across KV cache transfer execution stages in seconds.";
inline constexpr absl::string_view kActiveTransfers =
    "Current number of active KV cache transfers in progress.";
inline constexpr absl::string_view kBufferOccupancyBytes =
    "Current memory occupancy in bytes of TPU Raiden staging buffers.";
inline constexpr absl::string_view kTransferFailuresTotal =
    "Total count of failed KV cache transfers over TPU Raiden.";

}  // namespace metric_descriptions

namespace metric_metadata {

inline constexpr MetricMetadata kSentBytesTotal{
    .name = metric_names::kSentBytesTotal,
    .description = metric_descriptions::kSentBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kReceivedBytesTotal{
    .name = metric_names::kReceivedBytesTotal,
    .description = metric_descriptions::kReceivedBytesTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kTransferDurationSeconds{
    .name = metric_names::kTransferDurationSeconds,
    .description = metric_descriptions::kTransferDurationSeconds,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kStageLatencySeconds{
    .name = metric_names::kStageLatencySeconds,
    .description = metric_descriptions::kStageLatencySeconds,
    .type = MetricType::kHistogram};

inline constexpr MetricMetadata kActiveTransfers{
    .name = metric_names::kActiveTransfers,
    .description = metric_descriptions::kActiveTransfers,
    .type = MetricType::kGauge};

inline constexpr MetricMetadata kBufferOccupancyBytes{
    .name = metric_names::kBufferOccupancyBytes,
    .description = metric_descriptions::kBufferOccupancyBytes,
    .type = MetricType::kGauge};

inline constexpr MetricMetadata kTransferFailuresTotal{
    .name = metric_names::kTransferFailuresTotal,
    .description = metric_descriptions::kTransferFailuresTotal,
    .type = MetricType::kCounter};

inline constexpr MetricMetadata kAllMetrics[] = {
    kSentBytesTotal,        kReceivedBytesTotal, kTransferDurationSeconds,
    kStageLatencySeconds,   kActiveTransfers,    kBufferOccupancyBytes,
    kTransferFailuresTotal,
};

}  // namespace metric_metadata

// Structure defining a metric key-value label pair.
struct MetricLabel {
  absl::string_view key;
  absl::string_view value;
};

// Allocation-free label view span type definition
using LabelSpan = absl::Span<const MetricLabel>;

// Abstract Dual-Backend Interface
class MetricsBackend {
 public:
  MetricsBackend() = default;
  MetricsBackend(const MetricsBackend&) = delete;
  MetricsBackend& operator=(const MetricsBackend&) = delete;
  MetricsBackend(MetricsBackend&&) = delete;
  MetricsBackend& operator=(MetricsBackend&&) = delete;

  virtual ~MetricsBackend() = default;

  virtual void IncrementCounter(absl::string_view name, LabelSpan labels,
                                uint64_t val) const = 0;

  virtual void SetGauge(absl::string_view name, LabelSpan labels,
                        double val) const = 0;

  virtual void ObserveHistogram(absl::string_view name, LabelSpan labels,
                                double val) const = 0;

  virtual std::string GetTextSnapshot() const = 0;
};

// Central Telemetry Facade for managing metrics across registered backends.
// This class is thread-safe for all concurrent operations.
class RaidenMetricStore {
 public:
  static RaidenMetricStore& GetGlobalMetricStore();

  RaidenMetricStore() = default;
  ~RaidenMetricStore() = default;

  RaidenMetricStore(const RaidenMetricStore&) = delete;
  RaidenMetricStore& operator=(const RaidenMetricStore&) = delete;
  RaidenMetricStore(RaidenMetricStore&&) = delete;
  RaidenMetricStore& operator=(RaidenMetricStore&&) = delete;

  void SetBackends(std::vector<std::unique_ptr<MetricsBackend>> backends);
  bool HasBackends() const;

  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val = 1) const;

  void SetGauge(absl::string_view name, LabelSpan labels, double val) const;

  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const;

  std::string GetTextSnapshot() const;

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::unique_ptr<MetricsBackend>> backends_
      ABSL_GUARDED_BY(mutex_);
  std::atomic<bool> has_backends_{false};
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_METRICS_API_H_
