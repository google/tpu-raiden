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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xla/tsl/platform/logging.h"
#include "core/tpu_utils.h"

namespace tpu_raiden {

class NumaThreadPool {
 public:
  // If target_numa_node is set, this pool is restricted to that node.
  // If target_numa_node is nullopt and multi_numa is true, this pool
  // will internally spawn sub-pools for each NUMA node in the system.
  explicit NumaThreadPool(size_t threads,
                          std::optional<int> target_numa_node = std::nullopt,
                          bool multi_numa = true);
  ~NumaThreadPool();

  // Schedule a task, optionally targeting a specific NUMA node.
  template <class F, class... Args>
  auto Schedule(std::optional<int> numa_node, F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    if (is_multi_numa_) {
      int node = numa_node.value_or(0);
      if (node < 0 || node >= static_cast<int>(sub_pools_.size())) {
        node = 0;
      }
      return sub_pools_[node]->Schedule(std::nullopt, std::forward<F>(f),
                                        std::forward<Args>(args)...);
    }

    // Leaf pool (single NUMA node) implementation
    // No dynamic pinning here! The worker threads are already pinned at
    // startup.
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [f = std::forward<F>(f),
         args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
          return std::apply(f, std::move(args_tuple));
        });

    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) {
        throw std::runtime_error("Schedule on stopped NumaThreadPool");
      }
      tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  // Overload for scheduling without a specific NUMA node.
  template <class F, class... Args>
  auto Schedule(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    return Schedule(std::nullopt, std::forward<F>(f),
                    std::forward<Args>(args)...);
  }

  // Returns true if the current thread is a worker thread of ANY
  // NumaThreadPool.
  static bool IsCurrentThreadWorker();

 private:
  void WorkerLoop();

  std::optional<int> target_numa_node_;
  bool is_multi_numa_ = false;
  std::vector<std::unique_ptr<NumaThreadPool>> sub_pools_;

  // Members for the single-node pool (only used if !is_multi_numa_)
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_
