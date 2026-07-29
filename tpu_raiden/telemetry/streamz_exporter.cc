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

#include "tpu_raiden/telemetry/streamz_exporter.h"

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

void StreamzExporter::RecordCounter(absl::string_view name, double value,
                                     LabelSpan labels) noexcept {
  try {
    absl::WriterMutexLock lock(&mutex_);
    counters_[std::string(name)] += value;
  } catch (...) {
    // noexcept safety
  }
}

void StreamzExporter::RecordGauge(absl::string_view name, double value,
                                   LabelSpan labels) noexcept {
  try {
    absl::WriterMutexLock lock(&mutex_);
    gauges_[std::string(name)] = value;
  } catch (...) {
    // noexcept safety
  }
}

void StreamzExporter::RecordHistogram(absl::string_view name, double value,
                                       LabelSpan labels) noexcept {
  try {
    absl::WriterMutexLock lock(&mutex_);
    histograms_[std::string(name)] += value;
  } catch (...) {
    // noexcept safety
  }
}

void Initialize1PBackend(bool enable_1p_telemetry) noexcept {
  try {
    if (enable_1p_telemetry) {
      auto exporter = std::make_shared<StreamzExporter>();
      RaidenMetricStore::GetGlobalMetricStore().AddBackend(exporter);
    }
  } catch (...) {
    // noexcept safety
  }
}

}  // namespace tpu_raiden::telemetry
