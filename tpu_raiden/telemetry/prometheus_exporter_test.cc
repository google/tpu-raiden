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
namespace {

using ::testing::HasSubstr;

TEST(PrometheusExporterTest, RecordAndExportFormat) {
  auto exporter = std::make_unique<PrometheusExporter>();
  RaidenMetricStore store;
  store.AddBackend(std::move(exporter));

  MetricLabel label1{"interface", "ICI"};
  MetricLabel label2{"direction", "OUTBOUND"};
  MetricLabel label3{"status", "OK"};

  store.IncrementCounter(metric_names::kSentBytesTotal, {label1}, 1024);
  store.SetGauge(metric_names::kActiveTransfers, {label2}, 5);
  store.ObserveHistogram(metric_names::kTransferDurationSeconds, {label3},
                         0.0055);

  std::string output = store.GetTextSnapshot();

  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE tpu_raiden_sent_bytes_total counter"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_sent_bytes_total{interface=\"ICI\"} 1024"));
  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE tpu_raiden_active_transfers gauge"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_active_transfers{direction=\"OUTBOUND\"} 5"));
  EXPECT_TRUE(absl::StrContains(
      output, "# TYPE tpu_raiden_transfer_duration_seconds histogram"));
  EXPECT_TRUE(absl::StrContains(
      output,
      "tpu_raiden_transfer_duration_seconds_sum{status=\"OK\"} 0.0055"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_transfer_duration_seconds_count{status=\"OK\"} 1"));
}

TEST(PrometheusExporterTest, GetPrometheusMetricInfoMapped) {
  PrometheusMetricInfo info1 =
      GetPrometheusMetricInfo(metric_names::kSentBytesTotal);
  EXPECT_EQ(info1.prometheus_name, "tpu_raiden_sent_bytes_total");
  EXPECT_THAT(
      info1.help_text,
      HasSubstr("Total count of bytes sent over TPU Raiden interfaces."));

  PrometheusMetricInfo info2 =
      GetPrometheusMetricInfo(metric_names::kReceivedBytesTotal);
  EXPECT_EQ(info2.prometheus_name, "tpu_raiden_received_bytes_total");

  PrometheusMetricInfo info3 =
      GetPrometheusMetricInfo(metric_names::kTransferDurationSeconds);
  EXPECT_EQ(info3.prometheus_name, "tpu_raiden_transfer_duration_seconds");

  PrometheusMetricInfo info4 =
      GetPrometheusMetricInfo(metric_names::kStageLatencySeconds);
  EXPECT_EQ(info4.prometheus_name, "tpu_raiden_stage_latency_seconds");

  PrometheusMetricInfo info5 =
      GetPrometheusMetricInfo(metric_names::kActiveTransfers);
  EXPECT_EQ(info5.prometheus_name, "tpu_raiden_active_transfers");

  PrometheusMetricInfo info6 =
      GetPrometheusMetricInfo(metric_names::kBufferOccupancyBytes);
  EXPECT_EQ(info6.prometheus_name, "tpu_raiden_buffer_occupancy_bytes");

  PrometheusMetricInfo info7 =
      GetPrometheusMetricInfo(metric_names::kTransferFailuresTotal);
  EXPECT_EQ(info7.prometheus_name, "tpu_raiden_transfer_failures_total");

  // Also verify alias lookup via 3P string
  PrometheusMetricInfo info_alias =
      GetPrometheusMetricInfo("tpu_raiden_sent_bytes_total");
  EXPECT_EQ(info_alias.prometheus_name, "tpu_raiden_sent_bytes_total");
  EXPECT_EQ(info_alias.help_text, info1.help_text);
}

TEST(PrometheusExporterTest, AllPhase0MetricsMapped) {
  PrometheusExporter exporter;

  exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 100);
  exporter.IncrementCounter(metric_names::kReceivedBytesTotal, {}, 200);
  exporter.ObserveHistogram(metric_names::kTransferDurationSeconds, {}, 0.5);
  exporter.ObserveHistogram(metric_names::kStageLatencySeconds, {}, 0.1);
  exporter.SetGauge(metric_names::kActiveTransfers, {}, 10);
  exporter.SetGauge(metric_names::kBufferOccupancyBytes, {}, 2048);
  exporter.IncrementCounter(metric_names::kTransferFailuresTotal, {}, 1);

  std::string output = exporter.GetTextSnapshot();

  EXPECT_TRUE(
      absl::StrContains(output,
                        "# HELP tpu_raiden_sent_bytes_total Total count of "
                        "bytes sent over TPU Raiden interfaces."));
  EXPECT_TRUE(
      absl::StrContains(output,
                        "# HELP tpu_raiden_received_bytes_total Total count of "
                        "bytes received over TPU Raiden interfaces."));
  EXPECT_TRUE(
      absl::StrContains(output,
                        "# HELP tpu_raiden_transfer_duration_seconds Histogram "
                        "of TPU Raiden transfer durations in seconds."));
  EXPECT_TRUE(
      absl::StrContains(output,
                        "# HELP tpu_raiden_stage_latency_seconds Histogram of "
                        "TPU Raiden pipeline stage latencies in seconds."));
  EXPECT_TRUE(absl::StrContains(output,
                                "# HELP tpu_raiden_active_transfers Number of "
                                "currently active TPU Raiden data transfers."));
  EXPECT_TRUE(
      absl::StrContains(output,
                        "# HELP tpu_raiden_buffer_occupancy_bytes Gauge of TPU "
                        "Raiden buffer occupancy in bytes."));
  EXPECT_TRUE(
      absl::StrContains(output,
                        "# HELP tpu_raiden_transfer_failures_total Total count "
                        "of failed TPU Raiden data transfers."));

  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_sent_bytes_total 100"));
  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_received_bytes_total 200"));
  EXPECT_TRUE(absl::StrContains(
      output, "tpu_raiden_transfer_duration_seconds_sum 0.5"));
  EXPECT_TRUE(
      absl::StrContains(output, "tpu_raiden_stage_latency_seconds_sum 0.1"));
  EXPECT_TRUE(absl::StrContains(output, "tpu_raiden_active_transfers 10"));
  EXPECT_TRUE(
      absl::StrContains(output, "tpu_raiden_buffer_occupancy_bytes 2048"));
  EXPECT_TRUE(
      absl::StrContains(output, "tpu_raiden_transfer_failures_total 1"));
}

TEST(PrometheusExporterTest, UnmappedCustomMetricFallback) {
  PrometheusMetricInfo custom_info =
      GetPrometheusMetricInfo("custom_unmapped_metric");
  EXPECT_EQ(custom_info.prometheus_name, "custom_unmapped_metric");
  EXPECT_EQ(custom_info.help_text, "custom_unmapped_metric");

  PrometheusExporter exporter;
  exporter.IncrementCounter("custom_unmapped_counter", {}, 42);
  exporter.SetGauge("custom_unmapped_gauge", {}, 99);
  exporter.ObserveHistogram("custom_unmapped_histogram", {}, 1.23);

  std::string output = exporter.GetTextSnapshot();

  EXPECT_TRUE(absl::StrContains(
      output, "# HELP custom_unmapped_counter custom_unmapped_counter"));
  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE custom_unmapped_counter counter"));
  EXPECT_TRUE(absl::StrContains(output, "custom_unmapped_counter 42"));

  EXPECT_TRUE(absl::StrContains(
      output, "# HELP custom_unmapped_gauge custom_unmapped_gauge"));
  EXPECT_TRUE(absl::StrContains(output, "# TYPE custom_unmapped_gauge gauge"));
  EXPECT_TRUE(absl::StrContains(output, "custom_unmapped_gauge 99"));

  EXPECT_TRUE(absl::StrContains(
      output, "# HELP custom_unmapped_histogram custom_unmapped_histogram"));
  EXPECT_TRUE(
      absl::StrContains(output, "# TYPE custom_unmapped_histogram histogram"));
  EXPECT_TRUE(absl::StrContains(output, "custom_unmapped_histogram_sum 1.23"));
}

TEST(PrometheusExporterTest, ResetAndGlobalExporter) {
  auto& global_exporter = GetGlobalPrometheusExporter();
  global_exporter.Reset();
  global_exporter.IncrementCounter(metric_names::kReceivedBytesTotal, {}, 3);

  std::string text = global_exporter.GetTextSnapshot();
  EXPECT_THAT(text, HasSubstr("tpu_raiden_received_bytes_total 3"));

  global_exporter.Reset();
  EXPECT_THAT(global_exporter.GetTextSnapshot(),
              Not(HasSubstr("tpu_raiden_received_bytes_total 3")));
  EXPECT_THAT(
      global_exporter.GetTextSnapshot(),
      HasSubstr("# HELP tpu_raiden_received_bytes_total Total count of bytes "
                "received over TPU Raiden interfaces."));
}

TEST(PrometheusExporterTest, ConstReferenceAccess) {
  PrometheusExporter exporter;
  const PrometheusExporter& const_exporter = exporter;
  const MetricsBackend& const_backend = exporter;

  const_exporter.IncrementCounter(metric_names::kSentBytesTotal, {}, 500);
  const_backend.SetGauge(metric_names::kActiveTransfers, {}, 2);
  const_backend.ObserveHistogram(metric_names::kTransferDurationSeconds, {},
                                 0.01);

  std::string snapshot = const_backend.GetTextSnapshot();
  EXPECT_TRUE(absl::StrContains(snapshot, "tpu_raiden_sent_bytes_total 500"));
  EXPECT_TRUE(absl::StrContains(snapshot, "tpu_raiden_active_transfers 2"));
  EXPECT_TRUE(absl::StrContains(
      snapshot, "tpu_raiden_transfer_duration_seconds_sum 0.01"));
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
        exporter.SetGauge(metric_names::kActiveTransfers, {}, j);
        exporter.ObserveHistogram(metric_names::kTransferDurationSeconds, {},
                                  0.001);
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

}  // namespace
}  // namespace tpu_raiden::telemetry
