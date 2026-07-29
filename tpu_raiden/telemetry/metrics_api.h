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

// --- Phase 0 Telemetry Metric Constants ---
inline constexpr absl::string_view kMetricKVCacheTransferBytes =
    "tpu_raiden_kv_cache_transfer_bytes";
inline constexpr absl::string_view kMetricKVCacheTransferLatencyMs =
    "tpu_raiden_kv_cache_transfer_latency_ms";
inline constexpr absl::string_view kMetricKVCacheHitRatio =
    "tpu_raiden_kv_cache_hit_ratio";
inline constexpr absl::string_view kMetricRPCRequestCount =
    "tpu_raiden_rpc_request_count";
inline constexpr absl::string_view kMetricRPCErrorCount =
    "tpu_raiden_rpc_error_count";
inline constexpr absl::string_view kMetricRPCLatencyMs =
    "tpu_raiden_rpc_latency_ms";
inline constexpr absl::string_view kMetricActiveTransfers =
    "tpu_raiden_active_transfers";

// Allocation-free label view span type
using LabelSpan =
    absl::Span<const std::pair<absl::string_view, absl::string_view>>;

// Abstract base class for telemetry exporters (3P Prometheus, 1P Streamz).
class TelemetryExporter {
 public:
  virtual ~TelemetryExporter() = default;

  virtual void RecordCounter(absl::string_view name, double value,
                             LabelSpan labels = {}) noexcept = 0;
  virtual void RecordGauge(absl::string_view name, double value,
                            LabelSpan labels = {}) noexcept = 0;
  virtual void RecordHistogram(absl::string_view name, double value,
                               LabelSpan labels = {}) noexcept = 0;

  virtual std::string ExportPrometheusText() noexcept { return ""; }
};

// Facade metric store managing active backends.
class RaidenMetricStore {
 public:
  static RaidenMetricStore& GetGlobalMetricStore();
  static RaidenMetricStore& GetInstance() { return GetGlobalMetricStore(); }

  RaidenMetricStore() = default;
  ~RaidenMetricStore() = default;

  RaidenMetricStore(const RaidenMetricStore&) = delete;
  RaidenMetricStore& operator=(const RaidenMetricStore&) = delete;

  // Add a backend exporter.
  void AddBackend(std::shared_ptr<TelemetryExporter> backend) noexcept;

  // Clear all registered backends.
  void ClearBackends() noexcept;

  // Returns whether any backends are registered.
  bool HasBackends() const noexcept;

  // Telemetry recording facade methods with lock-free fast-path exit.
  void RecordCounter(absl::string_view name, double value = 1.0,
                     LabelSpan labels = {}) noexcept;

  void RecordGauge(absl::string_view name, double value = 0.0,
                   LabelSpan labels = {}) noexcept;

  void RecordHistogram(absl::string_view name, double value = 0.0,
                       LabelSpan labels = {}) noexcept;

  // Aggregate Prometheus exposition text across registered backends.
  std::string ExportPrometheusText() noexcept;

 private:
  mutable absl::Mutex mutex_;
  std::vector<std::shared_ptr<TelemetryExporter>> backends_
      ABSL_GUARDED_BY(mutex_);
  std::atomic<bool> has_backends_{false};
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_METRICS_API_H_
