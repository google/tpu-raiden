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

#include "tpu_raiden/telemetry/metrics_api.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tpu_raiden::telemetry {
namespace {

class MockExporter : public TelemetryExporter {
 public:
  struct Record {
    std::string type;
    std::string name;
    double value;
  };

  void RecordCounter(absl::string_view name, double value,
                     LabelSpan labels) noexcept override {
    records.push_back({"counter", std::string(name), value});
  }

  void RecordGauge(absl::string_view name, double value,
                   LabelSpan labels) noexcept override {
    records.push_back({"gauge", std::string(name), value});
  }

  void RecordHistogram(absl::string_view name, double value,
                       LabelSpan labels) noexcept override {
    records.push_back({"histogram", std::string(name), value});
  }

  std::vector<Record> records;
};

class MetricsApiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RaidenMetricStore::GetGlobalMetricStore().ClearBackends();
  }

  void TearDown() override {
    RaidenMetricStore::GetGlobalMetricStore().ClearBackends();
  }
};

TEST_F(MetricsApiTest, FastPathExitWhenNoBackends) {
  auto& store = RaidenMetricStore::GetInstance();
  EXPECT_FALSE(store.HasBackends());

  store.RecordCounter(kMetricKVCacheTransferBytes, 1024.0);
  store.RecordGauge(kMetricKVCacheHitRatio, 0.95);
  store.RecordHistogram(kMetricKVCacheTransferLatencyMs, 12.5);
}

TEST_F(MetricsApiTest, DispatchesToRegisteredBackend) {
  auto& store = RaidenMetricStore::GetGlobalMetricStore();
  auto mock_exporter = std::make_shared<MockExporter>();

  store.AddBackend(mock_exporter);
  EXPECT_TRUE(store.HasBackends());

  store.RecordCounter(kMetricKVCacheTransferBytes, 2048.0);
  ASSERT_EQ(mock_exporter->records.size(), 1);
  EXPECT_EQ(mock_exporter->records[0].name, kMetricKVCacheTransferBytes);
  EXPECT_DOUBLE_EQ(mock_exporter->records[0].value, 2048.0);

  store.RecordGauge(kMetricKVCacheHitRatio, 0.99);
  ASSERT_EQ(mock_exporter->records.size(), 2);
  EXPECT_EQ(mock_exporter->records[1].name, kMetricKVCacheHitRatio);
  EXPECT_DOUBLE_EQ(mock_exporter->records[1].value, 0.99);

  store.RecordHistogram(kMetricKVCacheTransferLatencyMs, 5.0);
  ASSERT_EQ(mock_exporter->records.size(), 3);
  EXPECT_EQ(mock_exporter->records[2].name, kMetricKVCacheTransferLatencyMs);
  EXPECT_DOUBLE_EQ(mock_exporter->records[2].value, 5.0);
}

TEST_F(MetricsApiTest, ClearBackendsResetsFastPath) {
  auto& store = RaidenMetricStore::GetGlobalMetricStore();
  auto mock_exporter = std::make_shared<MockExporter>();

  store.AddBackend(mock_exporter);
  EXPECT_TRUE(store.HasBackends());

  store.ClearBackends();
  EXPECT_FALSE(store.HasBackends());

  store.RecordCounter(kMetricRPCRequestCount, 1.0);
  EXPECT_EQ(mock_exporter->records.size(), 0);
}

}  // namespace
}  // namespace tpu_raiden::telemetry
