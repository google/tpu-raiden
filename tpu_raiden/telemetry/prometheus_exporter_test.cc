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

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "tpu_raiden/telemetry/telemetry_exporter.h"

namespace tpu_raiden::telemetry {
namespace {

TEST(PrometheusExporterTest, MetricNameSanitization) {
  EXPECT_EQ(PrometheusExporter::SanitizeMetricName("valid_name_123:foo"),
            "valid_name_123:foo");
  EXPECT_EQ(PrometheusExporter::SanitizeMetricName("invalid-metric.name#1"),
            "invalid_metric_name_1");
}

TEST(PrometheusExporterTest, CounterAccumulationAndExpositionFormat) {
  PrometheusExporter exporter;

  EXPECT_TRUE(exporter.RecordCounter("http_requests", 5, {{"method", "GET"}},
                                     "Total HTTP Requests").ok());
  EXPECT_TRUE(exporter.RecordCounter("http_requests", 3, {{"method", "GET"}}).ok());

  EXPECT_EQ(exporter.GetMetricValue("http_requests", {{"method", "GET"}}), 8.0);

  std::string text = exporter.CollectExpositionFormat();
  EXPECT_TRUE(absl::StrContains(text, "# HELP http_requests Total HTTP Requests"));
  EXPECT_TRUE(absl::StrContains(text, "# TYPE http_requests counter"));
  EXPECT_TRUE(absl::StrContains(text, "http_requests{method=\"GET\"} 8"));
}

TEST(PrometheusExporterTest, GaugeValueOverwriting) {
  PrometheusExporter exporter;

  EXPECT_TRUE(exporter.RecordGauge("queue_size", 10.0).ok());
  EXPECT_EQ(exporter.GetMetricValue("queue_size"), 10.0);

  EXPECT_TRUE(exporter.RecordGauge("queue_size", 3.0).ok());
  EXPECT_EQ(exporter.GetMetricValue("queue_size"), 3.0);
}

TEST(PrometheusExporterTest, HistogramBucketingAndExpositionFormat) {
  PrometheusExporter exporter;

  EXPECT_TRUE(exporter.RecordHistogram("latency", 0.003, {}, "Latency histogram").ok());
  EXPECT_TRUE(exporter.RecordHistogram("latency", 0.5, {}, "Latency histogram").ok());

  std::string text = exporter.CollectExpositionFormat();
  EXPECT_TRUE(absl::StrContains(text, "# TYPE latency histogram"));
  EXPECT_TRUE(absl::StrContains(text, "latency_bucket{le=\"0.005\"} 1"));
  EXPECT_TRUE(absl::StrContains(text, "latency_bucket{le=\"1\"} 2"));
  EXPECT_TRUE(absl::StrContains(text, "latency_bucket{le=\"+Inf\"} 2"));
  EXPECT_TRUE(absl::StrContains(text, "latency_sum 0.503"));
  EXPECT_TRUE(absl::StrContains(text, "latency_count 2"));
}

TEST(PrometheusExporterTest, OptionPrefixStartsWithHandling) {
  PrometheusExporterOptions options;
  options.metric_prefix = "tpu_raiden_";
  PrometheusExporter exporter(options);

  // Metric containing substring tpu_raiden_ but not starting with it
  EXPECT_TRUE(exporter.RecordCounter("gpu_to_tpu_raiden_bytes", 100).ok());

  std::string text = exporter.CollectExpositionFormat();
  // Should properly prepend prefix because it did not START with tpu_raiden_
  EXPECT_TRUE(absl::StrContains(text, "tpu_raiden_gpu_to_tpu_raiden_bytes 100"));
}

TEST(PrometheusExporterTest, EnablementToggle) {
  PrometheusExporter exporter;
  exporter.SetEnabled(false);
  EXPECT_FALSE(exporter.IsEnabled());

  EXPECT_TRUE(exporter.RecordCounter("counter_while_disabled", 10).ok());
  EXPECT_EQ(exporter.GetMetricCount(), 0);

  exporter.SetEnabled(true);
  EXPECT_TRUE(exporter.RecordCounter("counter_while_enabled", 10).ok());
  EXPECT_EQ(exporter.GetMetricCount(), 1);
}

}  // namespace
}  // namespace tpu_raiden::telemetry
