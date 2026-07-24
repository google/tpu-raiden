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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_COMPOSITE_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_COMPOSITE_EXPORTER_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tpu_raiden/telemetry/telemetry_exporter.h"

namespace tpu_raiden::telemetry {

class CompositeExporter : public TelemetryExporter {
 public:
  CompositeExporter() = default;
  explicit CompositeExporter(
      std::vector<std::shared_ptr<TelemetryExporter>> exporters);
  ~CompositeExporter() override = default;

  void AddExporter(std::shared_ptr<TelemetryExporter> exporter);

  absl::Status Export(const MetricValue& metric) override;
  absl::Status ExportBatch(absl::Span<const MetricValue> metrics) override;
  absl::Status ExportBatch(const MetricBatch& batch) override;
  absl::Status Flush() override;

  bool IsEnabled() const override;
  void SetEnabled(bool enabled) override;

  size_t size() const;

 private:
  mutable absl::Mutex mutex_;
  bool enabled_ ABSL_GUARDED_BY(mutex_) = true;
  std::vector<std::shared_ptr<TelemetryExporter>> exporters_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_COMPOSITE_EXPORTER_H_
