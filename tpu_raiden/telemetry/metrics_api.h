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
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace tpu_raiden::telemetry {

// --- Phase 0 Primary Metrics Specification Constants (LLD Section 2.1.3 &
// Section 4) ---
inline constexpr absl::string_view kMetricSentBytesTotal =
    "tpu_raiden_sent_bytes_total";
inline constexpr absl::string_view kMetricReceivedBytesTotal =
    "tpu_raiden_received_bytes_total";
inline constexpr absl::string_view kMetricTransferDurationSeconds =
    "tpu_raiden_transfer_duration_seconds";
inline constexpr absl::string_view kMetricStageLatencySeconds =
    "tpu_raiden_stage_latency_seconds";
inline constexpr absl::string_view kMetricActiveTransfers =
    "tpu_raiden_active_transfers";
inline constexpr absl::string_view kMetricBufferOccupancyBytes =
    "tpu_raiden_buffer_occupancy_bytes";
inline constexpr absl::string_view kMetricTransferFailuresTotal =
    "tpu_raiden_transfer_failures_total";

// Allocation-free label view span type and label set definition
using LabelSet = std::vector<std::pair<absl::string_view, absl::string_view>>;
using LabelSpan =
    absl::Span<const std::pair<absl::string_view, absl::string_view>>;

// Abstract Dual-Backend Interface (PrometheusCppBackend in 3P, Streamz1PBackend
// in 1P)
class MetricsBackend {
 public:
  virtual ~MetricsBackend() = default;

  virtual void IncrementCounter(absl::string_view name, uint64_t val = 1,
                                LabelSpan labels = {}) noexcept = 0;

  virtual void SetGauge(absl::string_view name, int64_t val,
                        LabelSpan labels = {}) noexcept = 0;

  virtual void ObserveHistogram(absl::string_view name, double val,
                                LabelSpan labels = {}) noexcept = 0;

  virtual std::string GetPrometheusTextSnapshot() = 0;
};

// Central Telemetry Facade with Multi-Backend Support & Fast-Path Exit
class RaidenMetricStore {
 public:
  static RaidenMetricStore& GetGlobalMetricStore();

  RaidenMetricStore() = default;
  ~RaidenMetricStore() = default;

  RaidenMetricStore(const RaidenMetricStore&) = delete;
  RaidenMetricStore& operator=(const RaidenMetricStore&) = delete;

  // Register one or more backends
  void AddBackend(std::shared_ptr<MetricsBackend> backend) noexcept;

  // Clear backends to disable telemetry instantly
  void ClearBackends() noexcept;

  // Returns whether any backends are registered
  bool HasBackends() const noexcept;

  // LLD-aligned telemetry API methods with sub-nanosecond fast-path exit when
  // disabled
  void IncrementCounter(absl::string_view name, uint64_t val = 1,
                        LabelSpan labels = {}) noexcept;

  void SetGauge(absl::string_view name, int64_t val,
                LabelSpan labels = {}) noexcept;

  void ObserveHistogram(absl::string_view name, double val,
                        LabelSpan labels = {}) noexcept;

  std::string GetPrometheusTextSnapshot();

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::shared_ptr<MetricsBackend>> backends_
      ABSL_GUARDED_BY(mutex_);
  std::atomic<bool> has_backends_{false};
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_METRICS_API_H_
