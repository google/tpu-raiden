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
#include <stdexcept>
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
                        LabelSpan labels = {}) override {
    records.push_back({"counter", std::string(name), static_cast<double>(val)});
  }

  void SetGauge(absl::string_view name, int64_t val,
                LabelSpan labels = {}) override {
    records.push_back({"gauge", std::string(name), static_cast<double>(val)});
  }

  void ObserveHistogram(absl::string_view name, double val,
                        LabelSpan labels = {}) override {
    records.push_back({"histogram", std::string(name), val});
  }

  std::string GetTextSnapshot() override { return "# HELP mock\n"; }

  std::vector<Record> records;
};

class ThrowingBackend : public MetricsBackend {
 public:
  void IncrementCounter(absl::string_view name, uint64_t val = 1,
                        LabelSpan labels = {}) override {
    throw std::runtime_error("Simulated counter backend failure");
  }

  void SetGauge(absl::string_view name, int64_t val,
                LabelSpan labels = {}) override {
    throw std::runtime_error("Simulated gauge backend failure");
  }

  void ObserveHistogram(absl::string_view name, double val,
                        LabelSpan labels = {}) override {
    throw std::runtime_error("Simulated histogram backend failure");
  }

  std::string GetTextSnapshot() override {
    throw std::runtime_error("Simulated text snapshot failure");
  }
};

class MetricsApiTest : public ::testing::Test {
 protected:
  RaidenMetricStore store_;
};

TEST_F(MetricsApiTest, GlobalMetricStoreSingleton) {
  auto& global1 = RaidenMetricStore::GetGlobalMetricStore();
  auto& global2 = RaidenMetricStore::GetGlobalMetricStore();
  EXPECT_EQ(&global1, &global2);
}

TEST_F(MetricsApiTest, FastPathExitWhenNoBackends) {
  EXPECT_FALSE(store_.HasBackends());

  store_.IncrementCounter(metrics::kMetricSentBytesTotal, 1024);
  store_.SetGauge(metrics::kMetricActiveTransfers, 5);
  store_.ObserveHistogram(metrics::kMetricTransferDurationSeconds, 0.0125);
}

TEST_F(MetricsApiTest, DispatchesToRegisteredBackend) {
  auto mock_backend = std::make_shared<MockBackend>();

  store_.AddBackend(mock_backend);
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metrics::kMetricSentBytesTotal, 2048);
  ASSERT_EQ(mock_backend->records.size(), 1);
  EXPECT_EQ(mock_backend->records[0].name, metrics::kMetricSentBytesTotal);
  EXPECT_DOUBLE_EQ(mock_backend->records[0].value, 2048.0);

  store_.SetGauge(metrics::kMetricActiveTransfers, 3);
  ASSERT_EQ(mock_backend->records.size(), 2);
  EXPECT_EQ(mock_backend->records[1].name, metrics::kMetricActiveTransfers);
  EXPECT_DOUBLE_EQ(mock_backend->records[1].value, 3.0);

  store_.ObserveHistogram(metrics::kMetricTransferDurationSeconds, 0.005);
  ASSERT_EQ(mock_backend->records.size(), 3);
  EXPECT_EQ(mock_backend->records[2].name,
            metrics::kMetricTransferDurationSeconds);
  EXPECT_DOUBLE_EQ(mock_backend->records[2].value, 0.005);
}

TEST_F(MetricsApiTest, ClearBackendsResetsFastPath) {
  auto mock_backend = std::make_shared<MockBackend>();

  store_.AddBackend(mock_backend);
  EXPECT_TRUE(store_.HasBackends());

  store_.ClearBackends();
  EXPECT_FALSE(store_.HasBackends());

  store_.IncrementCounter(metrics::kMetricReceivedBytesTotal, 1);
  EXPECT_EQ(mock_backend->records.size(), 0);
}

TEST_F(MetricsApiTest, SwallowsBackendExceptions) {
  auto throwing_backend = std::make_shared<ThrowingBackend>();
  store_.AddBackend(throwing_backend);

  EXPECT_NO_THROW(
      store_.IncrementCounter(metrics::kMetricSentBytesTotal, 1024));
  EXPECT_NO_THROW(store_.SetGauge(metrics::kMetricActiveTransfers, 5));
  EXPECT_NO_THROW(
      store_.ObserveHistogram(metrics::kMetricTransferDurationSeconds, 0.05));
  EXPECT_NO_THROW(store_.GetTextSnapshot());
}

}  // namespace
}  // namespace tpu_raiden::telemetry
