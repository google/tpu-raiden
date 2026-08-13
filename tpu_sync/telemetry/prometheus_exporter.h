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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_EXPORTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_EXPORTER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tpu_sync/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

inline const prometheus::Histogram::BucketBoundaries&
DefaultHistogramBuckets() {
  static const absl::NoDestructor<prometheus::Histogram::BucketBoundaries>
      kBuckets(prometheus::Histogram::BucketBoundaries{
          0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0});
  return *kBuckets;
}

// Custom MetricsBackend that formats and exports TPU Raiden metrics to
// prometheus-cpp.
//
// Thread Safety:
// Concurrent calls to IncrementCounter(), SetGauge(), and ObserveHistogram()
// are thread-safe post-construction.
//
// Ownership:
// Manages a std::shared_ptr<prometheus::Registry> which holds registered
// metric families.
class PrometheusExporter : public MetricsBackend {
 public:
  explicit PrometheusExporter(const prometheus::Histogram::BucketBoundaries&
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

  const std::shared_ptr<prometheus::Registry>& GetRegistry() const {
    return registry_;
  }

 private:
  void RegisterKnownFamilies();

  prometheus::Family<prometheus::Counter>* GetCounterFamily(
      absl::string_view name) const;
  prometheus::Family<prometheus::Gauge>* GetGaugeFamily(
      absl::string_view name) const;
  prometheus::Family<prometheus::Histogram>* GetHistogramFamily(
      absl::string_view name) const;
  std::shared_ptr<prometheus::Registry> registry_;
  prometheus::Histogram::BucketBoundaries default_buckets_;

  absl::flat_hash_map<absl::string_view,
                      prometheus::Family<prometheus::Counter>*>
      counter_families_;
  absl::flat_hash_map<absl::string_view, prometheus::Family<prometheus::Gauge>*>
      gauge_families_;
  absl::flat_hash_map<absl::string_view,
                      prometheus::Family<prometheus::Histogram>*>
      histogram_families_;
};

}  // namespace tpu_raiden::telemetry

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TELEMETRY_PROMETHEUS_EXPORTER_H_
