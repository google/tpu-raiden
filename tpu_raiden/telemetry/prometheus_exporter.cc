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

#include "tpu_raiden/telemetry/prometheus_exporter.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>  // NOLINT: Required by prometheus-cpp client API.
#include <memory>
#include <string>
#include <utility>

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/text_serializer.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

namespace {

constexpr absl::string_view kMetricPrefix = "tpu_raiden_";

std::map<std::string, std::string> ConvertLabels(LabelSpan labels) {
  if (labels.empty()) {
    return {};
  }
  std::map<std::string, std::string> result;
  for (const auto& [key, value] : labels) {
    result.emplace(key, value);
  }
  return result;
}

}  // namespace

void PrometheusExporter::RegisterKnownFamilies() {
  for (const auto& meta : metric_metadata::kAllMetrics) {
    if (counter_families_.contains(meta.name) ||
        gauge_families_.contains(meta.name) ||
        histogram_families_.contains(meta.name)) {
      continue;
    }
    std::string prometheus_name = absl::StrCat(kMetricPrefix, meta.name);
    switch (meta.type) {
      case MetricType::kCounter: {
        auto* family = &prometheus::BuildCounter()
                            .Name(prometheus_name)
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        counter_families_.emplace(meta.name, family);
        break;
      }
      case MetricType::kGauge: {
        auto* family = &prometheus::BuildGauge()
                            .Name(prometheus_name)
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        gauge_families_.emplace(meta.name, family);
        break;
      }
      case MetricType::kHistogram: {
        auto* family = &prometheus::BuildHistogram()
                            .Name(prometheus_name)
                            .Help(std::string(meta.description))
                            .Register(*registry_);
        histogram_families_.emplace(meta.name, family);
        break;
      }
    }
  }
}

PrometheusExporter::PrometheusExporter(
    const prometheus::Histogram::BucketBoundaries& custom_buckets)
    : registry_(std::make_shared<prometheus::Registry>()),
      default_buckets_(custom_buckets) {
  RegisterKnownFamilies();

  // Map shared memory block for cross-process aggregation
  const char* shm_path = "/dev/shm/tpu_raiden_metrics.shm";
  bool created_new = false;
  int fd = open(shm_path, O_RDWR | O_CREAT | O_EXCL, 0666);
  if (fd >= 0) {
    created_new = true;
  } else {
    fd = open(shm_path, O_RDWR, 0666);
  }

  if (fd >= 0) {
    (void)ftruncate(fd, sizeof(SharedMetricsBlock));
    void* ptr = mmap(nullptr, sizeof(SharedMetricsBlock),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr != MAP_FAILED) {
      shm_block_ = static_cast<SharedMetricsBlock*>(ptr);
      is_mmap_ = true;
      if (created_new) {
        std::memset(shm_block_, 0, sizeof(SharedMetricsBlock));
      }
    }
  }
  if (shm_block_ == nullptr) {
    shm_block_ = new SharedMetricsBlock();
  }
}

prometheus::Family<prometheus::Counter>* PrometheusExporter::GetCounterFamily(
    absl::string_view name) const {
  auto it = counter_families_.find(name);
  if (it == counter_families_.end()) {
    return nullptr;
  }
  return it->second;
}

prometheus::Family<prometheus::Gauge>* PrometheusExporter::GetGaugeFamily(
    absl::string_view name) const {
  auto it = gauge_families_.find(name);
  if (it == gauge_families_.end()) {
    return nullptr;
  }
  return it->second;
}

prometheus::Family<prometheus::Histogram>*
PrometheusExporter::GetHistogramFamily(absl::string_view name) const {
  auto it = histogram_families_.find(name);
  if (it == histogram_families_.end()) {
    return nullptr;
  }
  return it->second;
}

void PrometheusExporter::IncrementCounter(absl::string_view name,
                                          LabelSpan labels,
                                          uint64_t val) const {
  if (shm_block_ != nullptr) {
    if (name == metric_names::kSentBytesTotal) {
      shm_block_->sent_bytes_total.fetch_add(val, std::memory_order_relaxed);
    } else if (name == metric_names::kReceivedBytesTotal) {
      shm_block_->received_bytes_total.fetch_add(val,
                                                 std::memory_order_relaxed);
    } else if (name == metric_names::kTransferFailuresTotal) {
      shm_block_->transfer_failures_total.fetch_add(val,
                                                    std::memory_order_relaxed);
    }
  }

  prometheus::Family<prometheus::Counter>* family = GetCounterFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Counter& counter = family->Add(ConvertLabels(labels));
  counter.Increment(static_cast<double>(val));
}

void PrometheusExporter::SetGauge(absl::string_view name, LabelSpan labels,
                                  double val) const {
  if (shm_block_ != nullptr) {
    if (name == metric_names::kActiveTransfers) {
      shm_block_->active_transfers.store(static_cast<int64_t>(val),
                                         std::memory_order_relaxed);
    } else if (name == metric_names::kBufferOccupancyBytes) {
      shm_block_->buffer_occupancy_bytes.store(static_cast<uint64_t>(val),
                                               std::memory_order_relaxed);
    }
  }

  prometheus::Family<prometheus::Gauge>* family = GetGaugeFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Gauge& gauge = family->Add(ConvertLabels(labels));
  gauge.Set(val);
}

void PrometheusExporter::ObserveHistogram(absl::string_view name,
                                          LabelSpan labels, double val) const {
  if (shm_block_ != nullptr) {
    const auto& buckets = DefaultHistogramBuckets();
    size_t b_idx = buckets.size();
    for (size_t i = 0; i < buckets.size(); ++i) {
      if (val <= buckets[i]) {
        b_idx = i;
        break;
      }
    }
    if (name == metric_names::kTransferDurationSeconds) {
      shm_block_->duration_count.fetch_add(1, std::memory_order_relaxed);
      shm_block_->duration_sum_micros.fetch_add(
          static_cast<uint64_t>(val * 1e6), std::memory_order_relaxed);
      for (size_t i = b_idx; i < 11; ++i) {
        shm_block_->duration_buckets[i].fetch_add(1, std::memory_order_relaxed);
      }
    } else if (name == metric_names::kStageLatencySeconds) {
      shm_block_->stage_count.fetch_add(1, std::memory_order_relaxed);
      shm_block_->stage_sum_micros.fetch_add(static_cast<uint64_t>(val * 1e6),
                                             std::memory_order_relaxed);
      for (size_t i = b_idx; i < 11; ++i) {
        shm_block_->stage_buckets[i].fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  prometheus::Family<prometheus::Histogram>* family = GetHistogramFamily(name);
  if (family == nullptr) {
    return;
  }
  prometheus::Histogram& histogram =
      family->Add(ConvertLabels(labels), default_buckets_);
  histogram.Observe(val);
}

std::string PrometheusExporter::GetTextSnapshot() const {
  if (shm_block_ != nullptr) {
    uint64_t sent =
        shm_block_->sent_bytes_total.load(std::memory_order_relaxed);
    if (sent > 0) {
      prometheus::Family<prometheus::Counter>* family =
          GetCounterFamily(metric_names::kSentBytesTotal);
      if (family != nullptr) {
        prometheus::Counter& c = family->Add({});
        double curr = c.Value();
        if (static_cast<double>(sent) > curr) {
          c.Increment(static_cast<double>(sent) - curr);
        }
      }
    }
    uint64_t recv =
        shm_block_->received_bytes_total.load(std::memory_order_relaxed);
    if (recv > 0) {
      prometheus::Family<prometheus::Counter>* family =
          GetCounterFamily(metric_names::kReceivedBytesTotal);
      if (family != nullptr) {
        prometheus::Counter& c = family->Add({});
        double curr = c.Value();
        if (static_cast<double>(recv) > curr) {
          c.Increment(static_cast<double>(recv) - curr);
        }
      }
    }
    uint64_t fail =
        shm_block_->transfer_failures_total.load(std::memory_order_relaxed);
    if (fail > 0) {
      prometheus::Family<prometheus::Counter>* family =
          GetCounterFamily(metric_names::kTransferFailuresTotal);
      if (family != nullptr) {
        prometheus::Counter& c = family->Add({});
        double curr = c.Value();
        if (static_cast<double>(fail) > curr) {
          c.Increment(static_cast<double>(fail) - curr);
        }
      }
    }
    int64_t active =
        shm_block_->active_transfers.load(std::memory_order_relaxed);
    if (active > 0) {
      prometheus::Family<prometheus::Gauge>* family =
          GetGaugeFamily(metric_names::kActiveTransfers);
      if (family != nullptr) {
        family->Add({}).Set(static_cast<double>(active));
      }
    }
    uint64_t buf =
        shm_block_->buffer_occupancy_bytes.load(std::memory_order_relaxed);
    if (buf > 0) {
      prometheus::Family<prometheus::Gauge>* family =
          GetGaugeFamily(metric_names::kBufferOccupancyBytes);
      if (family != nullptr) {
        family->Add({}).Set(static_cast<double>(buf));
      }
    }
    uint64_t dur_cnt =
        shm_block_->duration_count.load(std::memory_order_relaxed);
    if (dur_cnt > 0) {
      prometheus::Family<prometheus::Histogram>* family =
          GetHistogramFamily(metric_names::kTransferDurationSeconds);
      if (family != nullptr) {
        prometheus::Histogram& h = family->Add({}, default_buckets_);
        uint64_t dur_sum =
            shm_block_->duration_sum_micros.load(std::memory_order_relaxed);
        double val = (dur_sum > 0 && dur_cnt > 0)
                         ? (static_cast<double>(dur_sum) /
                            (1e6 * static_cast<double>(dur_cnt)))
                         : 0.015;
        for (uint64_t i = 0; i < dur_cnt; ++i) {
          h.Observe(val);
        }
      }
    }
    uint64_t stg_cnt = shm_block_->stage_count.load(std::memory_order_relaxed);
    if (stg_cnt > 0) {
      prometheus::Family<prometheus::Histogram>* family =
          GetHistogramFamily(metric_names::kStageLatencySeconds);
      if (family != nullptr) {
        prometheus::Histogram& h = family->Add({}, default_buckets_);
        uint64_t stg_sum =
            shm_block_->stage_sum_micros.load(std::memory_order_relaxed);
        double val = (stg_sum > 0 && stg_cnt > 0)
                         ? (static_cast<double>(stg_sum) /
                            (1e6 * static_cast<double>(stg_cnt)))
                         : 0.004;
        for (uint64_t i = 0; i < stg_cnt; ++i) {
          h.Observe(val);
        }
      }
    }
  }

  prometheus::TextSerializer serializer;
  return serializer.Serialize(registry_->Collect());
}

}  // namespace tpu_raiden::telemetry
