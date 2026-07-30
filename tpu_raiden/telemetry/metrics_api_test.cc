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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tpu_raiden::telemetry {
namespace {

class MockBackend : public MetricsBackend {
 public:
  struct Record {
    std::string type;
    std::string name;
    double value;
  };

  void IncrementCounter(absl::string_view name, uint64_t val = 1,
                        LabelSpan labels = {}) noexcept override {
    records.push_back({"counter", std::string(name), static_cast<double>(val)});
  }

  void SetGauge(absl::string_view name, int64_t val,
                LabelSpan labels = {}) noexcept override {
    records.push_back({"gauge", std::string(name), static_cast<double>(val)});
  }

  void ObserveHistogram(absl::string_view name, double val,
                        LabelSpan labels = {}) noexcept override {
    records.push_back({"histogram", std::string(name), val});
  }

  std::string GetPrometheusTextSnapshot() override { return "# HELP mock\n"; }

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
  auto& store = RaidenMetricStore::GetGlobalMetricStore();

  EXPECT_FALSE(store.HasBackends());

  store.IncrementCounter(kMetricSentBytesTotal, 1024);
  store.SetGauge(kMetricActiveTransfers, 5);
  store.ObserveHistogram(kMetricTransferDurationSeconds, 0.0125);
}

TEST_F(MetricsApiTest, DispatchesToRegisteredBackend) {
  auto& store = RaidenMetricStore::GetGlobalMetricStore();
  auto mock_backend = std::make_shared<MockBackend>();

  store.AddBackend(mock_backend);
  EXPECT_TRUE(store.HasBackends());

  store.IncrementCounter(kMetricSentBytesTotal, 2048);
  ASSERT_EQ(mock_backend->records.size(), 1);
  EXPECT_EQ(mock_backend->records[0].name, kMetricSentBytesTotal);
  EXPECT_DOUBLE_EQ(mock_backend->records[0].value, 2048.0);

  store.SetGauge(kMetricActiveTransfers, 3);
  ASSERT_EQ(mock_backend->records.size(), 2);
  EXPECT_EQ(mock_backend->records[1].name, kMetricActiveTransfers);
  EXPECT_DOUBLE_EQ(mock_backend->records[1].value, 3.0);

  store.ObserveHistogram(kMetricTransferDurationSeconds, 0.005);
  ASSERT_EQ(mock_backend->records.size(), 3);
  EXPECT_EQ(mock_backend->records[2].name, kMetricTransferDurationSeconds);
  EXPECT_DOUBLE_EQ(mock_backend->records[2].value, 0.005);
}

TEST_F(MetricsApiTest, ClearBackendsResetsFastPath) {
  auto& store = RaidenMetricStore::GetGlobalMetricStore();
  auto mock_backend = std::make_shared<MockBackend>();

  store.AddBackend(mock_backend);
  EXPECT_TRUE(store.HasBackends());

  store.ClearBackends();
  EXPECT_FALSE(store.HasBackends());

  store.IncrementCounter(kMetricReceivedBytesTotal, 1);
  EXPECT_EQ(mock_backend->records.size(), 0);
}

}  // namespace
}  // namespace tpu_raiden::telemetry
