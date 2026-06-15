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

#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "xla/tsl/platform/logging.h"
#include "core/tpu_utils.h"

namespace tpu_raiden {

namespace {
// Thread-local flag to identify worker threads.
thread_local bool is_worker_thread = false;
}  // namespace

NumaThreadPool::NumaThreadPool(size_t threads,
                               std::optional<int> target_numa_node,
                               bool multi_numa)
    : target_numa_node_(target_numa_node), stop(false) {
  if (multi_numa && !target_numa_node.has_value()) {
    is_multi_numa_ = true;
    int num_nodes = tpu_raiden::GetNumaNodeCount();
    VLOG(1) << "NumaThreadPool (Composite) created with " << threads
            << " threads. Spawning " << num_nodes << " sub-pools.";
    size_t threads_per_node =
        threads > 0 ? std::max<size_t>(1, threads / num_nodes) : 0;
    for (int i = 0; i < num_nodes; ++i) {
      sub_pools_.push_back(std::make_unique<NumaThreadPool>(
          threads_per_node, /*target_numa_node=*/i, /*multi_numa=*/false));
    }
  } else {
    is_multi_numa_ = false;
    VLOG(1) << "NumaThreadPool (Leaf) created with " << threads
            << " threads on NUMA node "
            << (target_numa_node.has_value() ? std::to_string(*target_numa_node)
                                             : "none")
            << ". Pool instance: " << this;
    for (size_t i = 0; i < threads; ++i) {
      workers.emplace_back([this, i, target_numa_node] {
        is_worker_thread = true;  // Mark this thread as a worker
        if (target_numa_node.has_value() && *target_numa_node >= 0) {
          PinCurrentThreadToNumaNode(*target_numa_node);
        }
        VLOG(1) << "NumaThreadPool worker thread " << i
                << " started on NUMA node "
                << (target_numa_node.has_value()
                        ? std::to_string(*target_numa_node)
                        : "none")
                << ". Thread ID: " << std::this_thread::get_id()
                << ", Pool: " << this;
        this->WorkerLoop();
        VLOG(1) << "NumaThreadPool worker thread " << i
                << " exiting. Thread ID: " << std::this_thread::get_id();
      });
    }
  }
}

void NumaThreadPool::WorkerLoop() {
  for (;;) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(this->queue_mutex);
      this->condition.wait(
          lock, [this] { return this->stop || !this->tasks.empty(); });
      if (this->stop && this->tasks.empty()) {
        return;
      }
      task = std::move(this->tasks.front());
      this->tasks.pop();
    }
    VLOG(1) << "NumaThreadPool worker " << std::this_thread::get_id()
            << " executing task. Pool: " << this;
    task();
    VLOG(1) << "NumaThreadPool worker " << std::this_thread::get_id()
            << " finished task. Pool: " << this;
  }
}

NumaThreadPool::~NumaThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

bool NumaThreadPool::IsCurrentThreadWorker() { return is_worker_thread; }

}  // namespace tpu_raiden
