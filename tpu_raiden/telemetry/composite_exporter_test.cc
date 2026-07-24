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

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tpu_raiden/telemetry/prometheus_exporter.h"
#include "tpu_raiden/telemetry/streamz_exporter.h"

namespace tpu_raiden::telemetry {
namespace {

TEST(CompositeExporterTest, MultiExporterFanOut) {
  auto prom = std::make_shared<PrometheusExporter>();
  auto streamz = std::make_shared<StreamzExporter>();

  CompositeExporter composite({prom, streamz});
  EXPECT_EQ(composite.size(), 2);

  EXPECT_TRUE(composite.RecordCounter("fanout_metric", 10.0, {{"app", "test"}}).ok());

  EXPECT_EQ(prom->GetMetricValue("fanout_metric", {{"app", "test"}}), 10.0);
  EXPECT_EQ(streamz->GetCellValue("fanout_metric", {{"app", "test"}}), 10.0);
}

TEST(CompositeExporterTest, BatchFanOutAndFlush) {
  auto prom = std::make_shared<PrometheusExporter>();
  auto streamz = std::make_shared<StreamzExporter>();

  CompositeExporter composite;
  composite.AddExporter(prom);
  composite.AddExporter(streamz);

  MetricValue m1;
  m1.name = "batch_fanout_1";
  m1.type = MetricType::kGauge;
  m1.value = 123.0;

  MetricValue m2;
  m2.name = "batch_fanout_2";
  m2.type = MetricType::kGauge;
  m2.value = 456.0;

  std::vector<MetricValue> batch = {m1, m2};
  EXPECT_TRUE(composite.ExportBatch(batch).ok());
  EXPECT_TRUE(composite.Flush().ok());

  EXPECT_EQ(prom->GetMetricValue("batch_fanout_1"), 123.0);
  EXPECT_EQ(prom->GetMetricValue("batch_fanout_2"), 456.0);
  EXPECT_EQ(streamz->GetCellValue("batch_fanout_1"), 123.0);
  EXPECT_EQ(streamz->GetCellValue("batch_fanout_2"), 456.0);
}

TEST(CompositeExporterTest, EnablementHierarchy) {
  auto prom = std::make_shared<PrometheusExporter>();
  auto streamz = std::make_shared<StreamzExporter>();

  CompositeExporter composite({prom, streamz});

  // Disabling composite prevents export to all child exporters
  composite.SetEnabled(false);
  EXPECT_TRUE(composite.RecordCounter("ignored_metric", 5.0).ok());
  EXPECT_EQ(prom->GetMetricCount(), 0);
  EXPECT_EQ(streamz->GetRecordCount(), 0);

  // Re-enable composite, but disable streamz child
  composite.SetEnabled(true);
  streamz->SetEnabled(false);

  EXPECT_TRUE(composite.RecordCounter("selective_metric", 5.0).ok());
  EXPECT_EQ(prom->GetMetricValue("selective_metric"), 5.0);
  EXPECT_EQ(streamz->GetRecordCount(), 0);
}

}  // namespace
}  // namespace tpu_raiden::telemetry
