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

#include <functional>
#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <queue>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {

class NumaThreadPool {
 public:
  // Creates a thread pool with one worker thread per specified NUMA node.
  // Pins each worker thread to the CPU cores of its corresponding NUMA node.
  static absl::StatusOr<std::unique_ptr<NumaThreadPool>> Create(
      const std::vector<int>& numa_nodes);

  ~NumaThreadPool();

  // Enqueues a task to be executed on the worker thread of a specific NUMA
  // node. Returns a std::future that will contain the result of the task. If
  // the numa_node is not in the pool (e.g. -1 or invalid), it executes on a
  // fallback worker (or the first worker).
  template <typename F, typename... Args>
  auto Schedule(int numa_node, F&& f, Args&&... args)
      -> std::future<typename std::invoke_result<F, Args...>::type> {
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    EnqueueTask(numa_node, [task]() { (*task)(); });

    return res;
  }

 private:
  struct Worker {
    int numa_node;
    std::thread thread;  // NOLINT(build/c++11)
    std::queue<std::function<void()>> tasks ABSL_GUARDED_BY(queue_mutex);
    absl::Mutex queue_mutex;
    bool stop ABSL_GUARDED_BY(queue_mutex) = false;

    Worker(int node) : numa_node(node) {}
  };

  explicit NumaThreadPool(const std::vector<int>& numa_nodes);
  absl::Status Initialize();
  void EnqueueTask(int numa_node, std::function<void()> task);
  void WorkerLoop(Worker* worker);
  static bool WorkerCondition(Worker* worker) ABSL_NO_THREAD_SAFETY_ANALYSIS;

  std::vector<int> numa_nodes_;
  absl::flat_hash_map<int, std::unique_ptr<Worker>> workers_;
  std::unique_ptr<Worker> fallback_worker_;  // For invalid NUMA nodes (e.g. -1)
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_NUMA_THREAD_POOL_H_
