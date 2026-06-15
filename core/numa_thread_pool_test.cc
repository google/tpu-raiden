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

#include "core/numa_thread_pool.h"

#include <future>  // NOLINT(build/c++11)
#include <optional>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "core/tpu_utils.h"

namespace tpu_raiden {
namespace {

TEST(NumaThreadPoolTest, BasicExecution) {
  NumaThreadPool pool(2);
  auto future = pool.Schedule([]() { return 42; });
  EXPECT_EQ(future.get(), 42);
}

TEST(NumaThreadPoolTest, CompositeRouting) {
  // Initialize a composite pool with 2 threads.
  // It will spawn sub-pools for each NUMA node.
  NumaThreadPool pool(2, std::nullopt, /*multi_numa=*/true);

  auto f1 = pool.Schedule(0, []() { return std::this_thread::get_id(); });
  auto f2 = pool.Schedule(1, []() { return std::this_thread::get_id(); });

  std::thread::id tid1 = f1.get();
  std::thread::id tid2 = f2.get();

  EXPECT_NE(tid1, std::thread::id());
  EXPECT_NE(tid2, std::thread::id());

  // If the system has at least 2 NUMA nodes, these should be different threads
  // because they are routed to different sub-pools.
  if (tpu_raiden::GetNumaNodeCount() >= 2) {
    EXPECT_NE(tid1, tid2);
  }
}

TEST(NumaThreadPoolTest, StaticPinning) {
  // Initialize a pinned pool with 1 thread on node 0.
  NumaThreadPool pool(1, 0, /*multi_numa=*/false);

  std::thread::id tid1;
  std::thread::id tid2;

  auto f1 = pool.Schedule([&tid1]() { tid1 = std::this_thread::get_id(); });
  f1.wait();

  auto f2 = pool.Schedule([&tid2]() { tid2 = std::this_thread::get_id(); });
  f2.wait();

  EXPECT_EQ(tid1, tid2);
  EXPECT_NE(tid1, std::thread::id());
}

}  // namespace
}  // namespace tpu_raiden
