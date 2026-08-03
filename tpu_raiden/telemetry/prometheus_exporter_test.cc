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

#include "tpu_raiden/telemetry/prometheus_exporter.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tpu_raiden/telemetry/metrics_api.h"

namespace tpu_raiden::telemetry {

using ::testing::HasSubstr;

TEST(PrometheusExporterTest, RecordAndExportFormat) {
  auto exporter = std::make_unique<PrometheusExporter>();
  RaidenMetricStore store;
  store.AddBackend(std::move(exporter));

  MetricLabel label1{"interface", "ICI"};

  store.IncrementCounter(metric_names::kSentBytesTotal, {label1}, 1024);

  std::string output = store.GetTextSnapshot();

  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE tpu_raiden_sent_bytes_total counter"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_sent_bytes_total{interface=\"ICI\"} 1024"));
}

TEST(PrometheusExporterTest, MetricMetadataConstantsMapped) {
  EXPECT_EQ(metric_metadata::kSentBytesTotal.prometheus_name,
            "tpu_raiden_sent_bytes_total");
  EXPECT_EQ(metric_metadata::kSentBytesTotal.streamz_name,
            "/tpu_raiden/sent_bytes_total");
  EXPECT_THAT(
      metric_metadata::kSentBytesTotal.description,
      HasSubstr("Total count of bytes sent over TPU Raiden interfaces."));
}

TEST(PrometheusExporterTest, UnmappedMetricIgnored) {
  PrometheusExporter exporter;
  exporter.IncrementCounter("custom_unmapped_counter", {}, 42);
  exporter.SetGauge("custom_unmapped_gauge", {}, 99);
  exporter.ObserveHistogram("custom_unmapped_histogram", {}, 1.23);

  std::string output = exporter.GetTextSnapshot();

  EXPECT_FALSE(absl::StrContains(output, "custom_unmapped_counter"));
  EXPECT_FALSE(absl::StrContains(output, "custom_unmapped_gauge"));
  EXPECT_FALSE(absl::StrContains(output, "custom_unmapped_histogram"));
}

TEST(PrometheusExporterTest, ConstReferenceAccess) {
  PrometheusExporter exporter;
  const PrometheusExporter& const_exporter = exporter;
  const MetricsBackend& const_backend = exporter;

  const_exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 500);

  std::string snapshot = const_backend.GetTextSnapshot();
  EXPECT_TRUE(absl::StrContains(snapshot, "tpu_raiden_sent_bytes_total 500"));
}

TEST(PrometheusExporterTest, ConcurrentMetricUpdates) {
  PrometheusExporter exporter;
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&exporter]() {
      for (int j = 0; j < kIterations; ++j) {
        exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 1);
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  std::string snapshot = exporter.GetTextSnapshot();
  EXPECT_TRUE(absl::StrContains(
      snapshot,
      absl::StrCat("tpu_raiden_sent_bytes_total ", kNumThreads * kIterations)));
}

}  // namespace tpu_raiden::telemetry
