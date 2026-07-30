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

#include "tpu_raiden/telemetry/metrics_api.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden::telemetry {

RaidenMetricStore& RaidenMetricStore::GetGlobalMetricStore() {
  static RaidenMetricStore* global_store = new RaidenMetricStore();
  return *global_store;
}

void RaidenMetricStore::AddBackend(
    std::shared_ptr<MetricsBackend> backend) noexcept {
  if (!backend) return;
  absl::WriterMutexLock lock(&mutex_);
  backends_.push_back(std::move(backend));
  has_backends_.store(!backends_.empty(), std::memory_order_release);
}

void RaidenMetricStore::ClearBackends() noexcept {
  absl::WriterMutexLock lock(&mutex_);
  backends_.clear();
  has_backends_.store(false, std::memory_order_release);
}

bool RaidenMetricStore::HasBackends() const noexcept {
  return has_backends_.load(std::memory_order_relaxed);
}

void RaidenMetricStore::IncrementCounter(absl::string_view name, uint64_t val,
                                         LabelSpan labels) noexcept {
  if (ABSL_PREDICT_TRUE(!has_backends_.load(std::memory_order_relaxed))) return;
  absl::ReaderMutexLock lock(&mutex_);
  if (backends_.empty()) return;
  for (const auto& backend : backends_) {
    if (backend) {
      backend->IncrementCounter(name, val, labels);
    }
  }
}

void RaidenMetricStore::SetGauge(absl::string_view name, int64_t val,
                                 LabelSpan labels) noexcept {
  if (ABSL_PREDICT_TRUE(!has_backends_.load(std::memory_order_relaxed))) return;
  absl::ReaderMutexLock lock(&mutex_);
  if (backends_.empty()) return;
  for (const auto& backend : backends_) {
    if (backend) {
      backend->SetGauge(name, val, labels);
    }
  }
}

void RaidenMetricStore::ObserveHistogram(absl::string_view name, double val,
                                         LabelSpan labels) noexcept {
  if (ABSL_PREDICT_TRUE(!has_backends_.load(std::memory_order_relaxed))) return;
  absl::ReaderMutexLock lock(&mutex_);
  if (backends_.empty()) return;
  for (const auto& backend : backends_) {
    if (backend) {
      backend->ObserveHistogram(name, val, labels);
    }
  }
}

std::string RaidenMetricStore::GetPrometheusTextSnapshot() {
  if (ABSL_PREDICT_TRUE(!has_backends_.load(std::memory_order_relaxed)))
    return "";
  absl::ReaderMutexLock lock(&mutex_);
  std::string result;
  for (const auto& backend : backends_) {
    if (backend) {
      absl::StrAppend(&result, backend->GetPrometheusTextSnapshot());
    }
  }
  return result;
}

}  // namespace tpu_raiden::telemetry
