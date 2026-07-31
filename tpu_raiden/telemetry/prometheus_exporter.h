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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

// Mapping structure for 3P Prometheus metric name and help text.
// The help_text uses the common metric description from GetMetricDescription.
struct PrometheusMetricInfo {
  absl::string_view prometheus_name;
  absl::string_view help_text;
};

// Returns PrometheusMetricInfo for a given general metric name/key,
// using GetMetricDescription for common human-readable help text.
PrometheusMetricInfo GetPrometheusMetricInfo(absl::string_view name);

inline const prometheus::Histogram::BucketBoundaries&
DefaultHistogramBuckets() {
  static const absl::NoDestructor<prometheus::Histogram::BucketBoundaries>
      kBuckets(prometheus::Histogram::BucketBoundaries{
          0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0});
  return *kBuckets;
}

class PrometheusExporter : public MetricsBackend {
 public:
  explicit PrometheusExporter(prometheus::Histogram::BucketBoundaries
                                  custom_buckets = DefaultHistogramBuckets());

  ~PrometheusExporter() override = default;

  PrometheusExporter(const PrometheusExporter&) = delete;
  PrometheusExporter& operator=(const PrometheusExporter&) = delete;
  PrometheusExporter(PrometheusExporter&&) = delete;
  PrometheusExporter& operator=(PrometheusExporter&&) = delete;

  void IncrementCounter(absl::string_view name, LabelSpan labels,
                        uint64_t val) const override;

  void SetGauge(absl::string_view name, LabelSpan labels,
                double val) const override;

  void ObserveHistogram(absl::string_view name, LabelSpan labels,
                        double val) const override;

  std::string GetTextSnapshot() const override;

  void Reset();

  std::shared_ptr<prometheus::Registry> GetRegistry() const {
    absl::ReaderMutexLock lock(mutex_);
    return registry_;
  }

 private:
  void RegisterKnownFamilies() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  prometheus::Family<prometheus::Counter>* GetCounterFamily(
      absl::string_view name) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  prometheus::Family<prometheus::Gauge>* GetGaugeFamily(absl::string_view name)
      const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  prometheus::Family<prometheus::Histogram>* GetHistogramFamily(
      absl::string_view name) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  std::shared_ptr<prometheus::Registry> registry_ ABSL_GUARDED_BY(mutex_);
  prometheus::Histogram::BucketBoundaries default_buckets_;

  prometheus::Family<prometheus::Counter>* sent_bytes_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;
  prometheus::Family<prometheus::Counter>* received_bytes_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;
  prometheus::Family<prometheus::Counter>* transfer_failures_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;
  prometheus::Family<prometheus::Gauge>* active_transfers_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;
  prometheus::Family<prometheus::Gauge>* buffer_occupancy_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;
  prometheus::Family<prometheus::Histogram>* transfer_duration_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;
  prometheus::Family<prometheus::Histogram>* stage_latency_family_
      ABSL_GUARDED_BY(mutex_) = nullptr;

  mutable absl::flat_hash_map<std::string,
                              prometheus::Family<prometheus::Counter>*>
      counter_families_ ABSL_GUARDED_BY(mutex_);
  mutable absl::flat_hash_map<std::string,
                              prometheus::Family<prometheus::Gauge>*>
      gauge_families_ ABSL_GUARDED_BY(mutex_);
  mutable absl::flat_hash_map<std::string,
                              prometheus::Family<prometheus::Histogram>*>
      histogram_families_ ABSL_GUARDED_BY(mutex_);
};

PrometheusExporter& GetGlobalPrometheusExporter();

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_TELEMETRY_PROMETHEUS_EXPORTER_H_
