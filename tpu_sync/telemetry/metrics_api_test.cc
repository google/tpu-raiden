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

#include "tpu_sync/telemetry/metrics_api.h"

#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tpu_raiden::telemetry {
namespace {

static_assert(!std::is_copy_constructible_v<MetricsBackend>,
              "MetricsBackend must not be copy constructible");
static_assert(!std::is_copy_assignable_v<MetricsBackend>,
              "MetricsBackend must not be copy assignable");
static_assert(!std::is_move_constructible_v<MetricsBackend>,
              "MetricsBackend must not be move constructible");
static_assert(!std::is_move_assignable_v<MetricsBackend>,
              "MetricsBackend must not be move assignable");

using ::testing::_;
using ::testing::Eq;
using ::testing::Return;

class MockMetricsBackend : public MetricsBackend {
 public:
  MOCK_METHOD(void, IncrementCounter,
              (absl::string_view name, LabelSpan labels, uint64_t val),
              (override, const));
  MOCK_METHOD(void, SetGauge,
              (absl::string_view name, LabelSpan labels, double val),
              (override, const));
  MOCK_METHOD(void, ObserveHistogram,
              (absl::string_view name, LabelSpan labels, double val),
              (override, const));
  MOCK_METHOD(std::string, GetTextSnapshot, (), (override, const));
};

class MetricsApiTest : public testing::Test {
 protected:
  void SetUp() override {
    RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  }

  void TearDown() override {
    RaidenMetricStore::GetGlobalMetricStore().SetBackends({});
  }

  RaidenMetricStore store_;
};

TEST_F(MetricsApiTest, GlobalMetricStoreSingleton) {
  RaidenMetricStore& global1 = RaidenMetricStore::GetGlobalMetricStore();
  RaidenMetricStore& global2 = RaidenMetricStore::GetGlobalMetricStore();
  EXPECT_EQ(&global1, &global2);
}

TEST_F(MetricsApiTest, MetricMetadataConstants) {
  EXPECT_EQ(metric_metadata::kSentBytesTotal.description,
            metric_descriptions::kSentBytesTotal);
  EXPECT_EQ(metric_descriptions::kSentBytesTotal,
            "Total count of bytes sent over TPU Raiden interfaces.");
}

TEST_F(MetricsApiTest, FastPathExitWhenNoBackends) {
  EXPECT_FALSE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 1024);
}

TEST_F(MetricsApiTest, DispatchesToRegisteredBackend) {
  auto mock_backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_mock = mock_backend.get();

  EXPECT_CALL(*raw_mock,
              IncrementCounter(Eq(metric_names::kSentBytesTotal), _, 2048))
      .Times(1);
  EXPECT_CALL(*raw_mock, GetTextSnapshot()).WillOnce(Return("# HELP mock\n"));
  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(mock_backend));
  store_.SetBackends(std::move(backends));
  EXPECT_TRUE(store_.HasBackends());

  store_.IncrementCounter(metric_names::kSentBytesTotal, {}, 2048);
  EXPECT_EQ(store_.GetTextSnapshot(), "# HELP mock\n");
}

TEST_F(MetricsApiTest, SetBackendsReplacesExistingBackends) {
  auto backend1 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend1 = backend1.get();
  std::vector<std::unique_ptr<MetricsBackend>> initial_backends;
  initial_backends.push_back(std::move(backend1));
  store_.SetBackends(std::move(initial_backends));

  auto backend2 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend2 = backend2.get();

  EXPECT_CALL(*raw_backend1, IncrementCounter(_, _, _)).Times(0);
  EXPECT_CALL(*raw_backend2,
              IncrementCounter(Eq("tpu_raiden_sent_bytes_total"), _, 4096))
      .Times(1);

  std::vector<std::unique_ptr<MetricsBackend>> new_backends;
  new_backends.push_back(std::move(backend2));
  store_.SetBackends(std::move(new_backends));

  EXPECT_TRUE(store_.HasBackends());
  store_.IncrementCounter("tpu_raiden_sent_bytes_total", {}, 4096);
}

TEST_F(MetricsApiTest, SetBackendsEmptyClearsBackends) {
  auto backend = std::make_unique<MockMetricsBackend>();
  std::vector<std::unique_ptr<MetricsBackend>> initial_backends;
  initial_backends.push_back(std::move(backend));
  store_.SetBackends(std::move(initial_backends));
  EXPECT_TRUE(store_.HasBackends());

  store_.SetBackends({});
  EXPECT_FALSE(store_.HasBackends());

  // Verify metric emission safely no-ops on empty backends.
  store_.IncrementCounter("tpu_raiden_sent_bytes_total", {}, 1024);
  EXPECT_EQ(store_.GetTextSnapshot(), "");
}

TEST_F(MetricsApiTest, SetBackendsMultipleBackends) {
  auto backend1 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend1 = backend1.get();
  auto backend2 = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend2 = backend2.get();

  EXPECT_CALL(*raw_backend1, IncrementCounter(Eq("counter"), _, 100)).Times(1);
  EXPECT_CALL(*raw_backend2, IncrementCounter(Eq("counter"), _, 100)).Times(1);
  EXPECT_CALL(*raw_backend1, GetTextSnapshot()).WillOnce(Return("snap1\n"));
  EXPECT_CALL(*raw_backend2, GetTextSnapshot()).WillOnce(Return("snap2\n"));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend1));
  backends.push_back(std::move(backend2));
  store_.SetBackends(std::move(backends));

  EXPECT_TRUE(store_.HasBackends());
  store_.IncrementCounter("counter", {}, 100);
  EXPECT_EQ(store_.GetTextSnapshot(), "snap1\nsnap2\n");
}

TEST_F(MetricsApiTest, ConcurrentTelemetryEmissions) {
  auto backend = std::make_unique<MockMetricsBackend>();
  MockMetricsBackend* raw_backend = backend.get();

  static constexpr int kNumThreads = 8;
  static constexpr int kIterations = 100;
  static constexpr int kTotalCalls = kNumThreads * kIterations;

  EXPECT_CALL(*raw_backend, IncrementCounter(Eq("counter"), _, 1))
      .Times(kTotalCalls);
  EXPECT_CALL(*raw_backend, SetGauge(Eq("gauge"), _, 42)).Times(kTotalCalls);
  EXPECT_CALL(*raw_backend, ObserveHistogram(Eq("histogram"), _, 3.14))
      .Times(kTotalCalls);
  EXPECT_CALL(*raw_backend, GetTextSnapshot())
      .Times(kTotalCalls)
      .WillRepeatedly(Return("snapshot\n"));

  std::vector<std::unique_ptr<MetricsBackend>> backends;
  backends.push_back(std::move(backend));
  store_.SetBackends(std::move(backends));

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this] {
      for (int j = 0; j < kIterations; ++j) {
        store_.IncrementCounter("counter", {}, 1);
        store_.SetGauge("gauge", {}, 42);
        store_.ObserveHistogram("histogram", {}, 3.14);
        (void)store_.GetTextSnapshot();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(MetricsApiTest, ConstMetricsBackendReference) {
  MockMetricsBackend backend;
  const MetricsBackend& const_ref = backend;

  EXPECT_CALL(backend, IncrementCounter(Eq("counter"), _, 5)).Times(1);
  EXPECT_CALL(backend, SetGauge(Eq("gauge"), _, 10)).Times(1);
  EXPECT_CALL(backend, ObserveHistogram(Eq("histogram"), _, 1.23)).Times(1);
  EXPECT_CALL(backend, GetTextSnapshot()).WillOnce(Return("snapshot\n"));

  const_ref.IncrementCounter("counter", {}, 5);
  const_ref.SetGauge("gauge", {}, 10);
  const_ref.ObserveHistogram("histogram", {}, 1.23);
  EXPECT_EQ(const_ref.GetTextSnapshot(), "snapshot\n");
}

}  // namespace
}  // namespace tpu_raiden::telemetry
