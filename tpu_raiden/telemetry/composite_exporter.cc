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

#include "tpu_raiden/telemetry/composite_exporter.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace tpu_raiden::telemetry {

CompositeExporter::CompositeExporter(
    std::vector<std::shared_ptr<TelemetryExporter>> exporters)
    : exporters_(std::move(exporters)) {}

void CompositeExporter::AddExporter(
    std::shared_ptr<TelemetryExporter> exporter) {
  if (exporter == nullptr) return;
  absl::MutexLock lock(&mutex_);
  exporters_.push_back(std::move(exporter));
}

absl::Status CompositeExporter::Export(const MetricValue& metric) {
  absl::MutexLock lock(&mutex_);
  if (!enabled_) {
    return absl::OkStatus();
  }
  for (auto& exporter : exporters_) {
    if (exporter && exporter->IsEnabled()) {
      absl::Status status = exporter->Export(metric);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CompositeExporter::ExportBatch(
    absl::Span<const MetricValue> metrics) {
  absl::MutexLock lock(&mutex_);
  if (!enabled_) {
    return absl::OkStatus();
  }
  for (auto& exporter : exporters_) {
    if (exporter && exporter->IsEnabled()) {
      absl::Status status = exporter->ExportBatch(metrics);
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CompositeExporter::ExportBatch(const MetricBatch& batch) {
  return ExportBatch(absl::MakeConstSpan(batch.metrics));
}

absl::Status CompositeExporter::Flush() {
  absl::MutexLock lock(&mutex_);
  if (!enabled_) {
    return absl::OkStatus();
  }
  for (auto& exporter : exporters_) {
    if (exporter && exporter->IsEnabled()) {
      absl::Status status = exporter->Flush();
      if (!status.ok()) {
        return status;
      }
    }
  }
  return absl::OkStatus();
}

bool CompositeExporter::IsEnabled() const {
  absl::MutexLock lock(&mutex_);
  return enabled_;
}

void CompositeExporter::SetEnabled(bool enabled) {
  absl::MutexLock lock(&mutex_);
  enabled_ = enabled;
}

size_t CompositeExporter::size() const {
  absl::MutexLock lock(&mutex_);
  return exporters_.size();
}

}  // namespace tpu_raiden::telemetry
