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

#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/tpu_utils.h"

namespace tpu_raiden {

absl::StatusOr<std::unique_ptr<NumaThreadPool>> NumaThreadPool::Create(
    const std::vector<int>& numa_nodes) {
  auto pool = std::unique_ptr<NumaThreadPool>(new NumaThreadPool(numa_nodes));
  auto status = pool->Initialize();
  if (!status.ok()) {
    return status;
  }
  return pool;
}

NumaThreadPool::NumaThreadPool(const std::vector<int>& numa_nodes)
    : numa_nodes_(numa_nodes) {}

absl::Status NumaThreadPool::Initialize() {
  // 1. Initialize pinned workers for each unique NUMA node
  for (int node : numa_nodes_) {
    if (node < 0) continue;
    if (workers_.contains(node)) continue;  // Skip duplicates

    auto worker = std::make_unique<Worker>(node);
    worker->thread = std::thread(&NumaThreadPool::WorkerLoop, this,
                                 worker.get());  // NOLINT(build/c++11)
    workers_[node] = std::move(worker);
  }

  // 2. Initialize fallback worker (unpinned)
  fallback_worker_ = std::make_unique<Worker>(-1);
  fallback_worker_->thread =
      std::thread(&NumaThreadPool::WorkerLoop, this,
                  fallback_worker_.get());  // NOLINT(build/c++11)

  return absl::OkStatus();
}

NumaThreadPool::~NumaThreadPool() {
  // Stop all pinned workers
  for (auto& [node, worker] : workers_) {
    {
      absl::MutexLock lock(worker->queue_mutex);
      worker->stop = true;
    }
    if (worker->thread.joinable()) {
      worker->thread.join();
    }
  }

  // Stop fallback worker
  if (fallback_worker_) {
    {
      absl::MutexLock lock(fallback_worker_->queue_mutex);
      fallback_worker_->stop = true;
    }
    if (fallback_worker_->thread.joinable()) {
      fallback_worker_->thread.join();
    }
  }
}

void NumaThreadPool::EnqueueTask(int numa_node, std::function<void()> task) {
  Worker* worker = nullptr;
  auto it = workers_.find(numa_node);
  if (it != workers_.end()) {
    worker = it->second.get();
  } else {
    worker = fallback_worker_.get();
  }

  {
    absl::MutexLock lock(worker->queue_mutex);
    if (worker->stop) {
      throw std::runtime_error("Enqueue on stopped NumaThreadPool");
    }
    worker->tasks.push(std::move(task));
  }
}

bool NumaThreadPool::WorkerCondition(Worker* worker) {
  return worker->stop || !worker->tasks.empty();
}

void NumaThreadPool::WorkerLoop(Worker* worker) {
  // Pin this thread to the NUMA node's CPU cores (if valid)
  if (worker->numa_node >= 0) {
    std::vector<int> cores = GetNumaNodeCpuCores(worker->numa_node);
    if (!cores.empty()) {
      int rc = PinCurrentThreadToCores(cores);
      if (rc == 0) {
        VLOG(1) << "[NUMA POOL] Successfully pinned worker thread to NUMA Node "
                << worker->numa_node << " (" << cores.size() << " cores)";
      } else {
        LOG(ERROR) << "[NUMA POOL] Failed to pin worker thread to NUMA Node "
                   << worker->numa_node;
      }
    } else {
      LOG(WARNING) << "[NUMA POOL] No CPU cores found for NUMA Node "
                   << worker->numa_node << ", running unpinned";
    }

    // Bind memory allocations of this thread to the NUMA node as well
    SetThreadMempolicy(2, worker->numa_node);  // MPOL_BIND
  } else {
    VLOG(1) << "[NUMA POOL] Initialized unpinned fallback worker thread";
  }

  // Task loop
  while (true) {
    std::function<void()> task;
    {
      absl::MutexLock lock(worker->queue_mutex);
      worker->queue_mutex.Await(absl::Condition(&NumaThreadPool::WorkerCondition, worker));
      if (worker->stop && worker->tasks.empty()) {
        break;
      }
      task = std::move(worker->tasks.front());
      worker->tasks.pop();
    }
    task();
  }

  // Restore memory policy on exit (just in case)
  if (worker->numa_node >= 0) {
    SetThreadMempolicy(0);  // MPOL_DEFAULT
  }
}

}  // namespace tpu_raiden
