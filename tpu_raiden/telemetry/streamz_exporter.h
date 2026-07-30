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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

// 1P Google Streamz Exporter for internal Google monitoring (Streamz1PBackend
// in LLD 2.1.3 & 2.3.1)
class StreamzExporter : public MetricsBackend {
 public:
  StreamzExporter() = default;
  ~StreamzExporter() override = default;

  void IncrementCounter(absl::string_view name, uint64_t val = 1,
                        LabelSpan labels = {}) noexcept override;

  void SetGauge(absl::string_view name, int64_t val,
                LabelSpan labels = {}) noexcept override;

  void ObserveHistogram(absl::string_view name, double val,
                        LabelSpan labels = {}) noexcept override;

  std::string GetPrometheusTextSnapshot() override { return ""; }

 private:
  std::string FormatLabels(LabelSpan labels) const;

  struct DistributionValue {
    double sum = 0.0;
    int64_t count = 0;
  };

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, uint64_t> counters_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, int64_t> gauges_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<std::string, DistributionValue> histograms_
      ABSL_GUARDED_BY(mutex_);
};

using Streamz1PBackend = StreamzExporter;

// Global initializer helper to enable/disable 1P Streamz backend.
void Initialize1PBackend(bool enable_1p_telemetry) noexcept;

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_STREAMZ_EXPORTER_H_
