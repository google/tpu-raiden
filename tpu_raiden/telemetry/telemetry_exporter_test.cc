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

#include "tpu_raiden/telemetry/telemetry_exporter.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace tpu_raiden::telemetry {
namespace {

class DummyExporter : public TelemetryExporter {
 public:
  absl::Status Export(const MetricValue& metric) override {
    exported_metrics_.push_back(metric);
    return absl::OkStatus();
  }

  bool IsEnabled() const override { return enabled_; }
  void SetEnabled(bool enabled) override { enabled_ = enabled; }

  const std::vector<MetricValue>& exported_metrics() const {
    return exported_metrics_;
  }

 private:
  bool enabled_ = true;
  std::vector<MetricValue> exported_metrics_;
};

TEST(TelemetryExporterTest, MetricTypeToStringConversion) {
  EXPECT_EQ(MetricTypeToString(MetricType::kCounter), "counter");
  EXPECT_EQ(MetricTypeToString(MetricType::kGauge), "gauge");
  EXPECT_EQ(MetricTypeToString(MetricType::kHistogram), "histogram");
}

TEST(TelemetryExporterTest, ConvenienceHelpersRecordMetrics) {
  DummyExporter exporter;

  EXPECT_TRUE(exporter.RecordCounter("test_counter", 10.0, {{"job", "worker"}}, "Help text").ok());
  EXPECT_TRUE(exporter.RecordGauge("test_gauge", 42.5, {{"env", "prod"}}).ok());
  EXPECT_TRUE(exporter.RecordHistogram("test_hist", 0.123).ok());

  const auto& metrics = exporter.exported_metrics();
  ASSERT_EQ(metrics.size(), 3);

  EXPECT_EQ(metrics[0].name, "test_counter");
  EXPECT_EQ(metrics[0].type, MetricType::kCounter);
  EXPECT_EQ(metrics[0].value, 10.0);
  EXPECT_EQ(metrics[0].description, "Help text");
  EXPECT_EQ(metrics[0].labels.at("job"), "worker");

  EXPECT_EQ(metrics[1].name, "test_gauge");
  EXPECT_EQ(metrics[1].type, MetricType::kGauge);
  EXPECT_EQ(metrics[1].value, 42.5);

  EXPECT_EQ(metrics[2].name, "test_hist");
  EXPECT_EQ(metrics[2].type, MetricType::kHistogram);
  EXPECT_EQ(metrics[2].value, 0.123);
}

TEST(TelemetryExporterTest, BatchExportAndStructOverload) {
  DummyExporter exporter;

  MetricValue m1;
  m1.name = "batch_m1";
  m1.value = 1.0;

  MetricValue m2;
  m2.name = "batch_m2";
  m2.value = 2.0;

  std::vector<MetricValue> list = {m1, m2};
  EXPECT_TRUE(exporter.ExportBatch(list).ok());
  EXPECT_EQ(exporter.exported_metrics().size(), 2);

  MetricBatch batch;
  batch.source_id = "src1";
  batch.metrics = list;
  EXPECT_TRUE(exporter.ExportBatch(batch).ok());
  ASSERT_EQ(exporter.exported_metrics().size(), 4);
  EXPECT_EQ(exporter.exported_metrics()[2].labels.at("source_id"), "src1");
  EXPECT_EQ(exporter.exported_metrics()[3].labels.at("source_id"), "src1");

  MetricBatch move_batch;
  move_batch.source_id = "src2";
  move_batch.metrics = list;
  EXPECT_TRUE(exporter.ExportBatch(std::move(move_batch)).ok());
  ASSERT_EQ(exporter.exported_metrics().size(), 6);
  EXPECT_EQ(exporter.exported_metrics()[4].labels.at("source_id"), "src2");
  EXPECT_EQ(exporter.exported_metrics()[5].labels.at("source_id"), "src2");
}

}  // namespace
}  // namespace tpu_raiden::telemetry
