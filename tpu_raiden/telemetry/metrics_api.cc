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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden::telemetry {

RaidenMetricStore& RaidenMetricStore::GetGlobalMetricStore() {
  static absl::NoDestructor<RaidenMetricStore> global_store;
  return *global_store;
}

void RaidenMetricStore::AddBackend(std::unique_ptr<MetricsBackend> backend) {
  if (!backend) return;
  absl::MutexLock lock(mutex_);
  backends_.push_back(std::move(backend));
  has_backends_.store(true, std::memory_order_release);
}

void RaidenMetricStore::ClearBackends() {
  absl::MutexLock lock(mutex_);
  backends_.clear();
  has_backends_.store(false, std::memory_order_release);
}

bool RaidenMetricStore::HasBackends() const {
  return has_backends_.load(std::memory_order_acquire);
}

void RaidenMetricStore::IncrementCounter(absl::string_view name,
                                         LabelSpan labels, uint64_t val) const {
  if (!HasBackends()) return;
  // TODO: Explore RCU optimization for lock-free reads.
  absl::ReaderMutexLock lock(mutex_);
  for (const auto& backend : backends_) {
    backend->IncrementCounter(name, labels, val);
  }
}

void RaidenMetricStore::SetGauge(absl::string_view name, LabelSpan labels,
                                 double val) const {
  if (!HasBackends()) return;
  absl::ReaderMutexLock lock(mutex_);
  for (const auto& backend : backends_) {
    backend->SetGauge(name, labels, val);
  }
}

void RaidenMetricStore::ObserveHistogram(absl::string_view name,
                                         LabelSpan labels, double val) const {
  if (!HasBackends()) return;
  absl::ReaderMutexLock lock(mutex_);
  for (const auto& backend : backends_) {
    backend->ObserveHistogram(name, labels, val);
  }
}

std::string RaidenMetricStore::GetTextSnapshot() const {
  if (!HasBackends()) return "";
  absl::ReaderMutexLock lock(mutex_);
  std::string result;
  for (const auto& backend : backends_) {
    absl::StrAppend(&result, backend->GetTextSnapshot());
  }
  return result;
}

}  // namespace tpu_raiden::telemetry
