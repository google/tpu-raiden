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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_ORCHESTRATOR_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_ORCHESTRATOR_H_

#include <atomic>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"

namespace tpu_raiden {
namespace kv_cache {

// Central orchestrator that acts as a directory service for Raiden Controllers.
// It maintains a mapping from RaidenId to Controller physical addresses.
class RaidenOrchestrator {
 public:
  explicit RaidenOrchestrator(int port, const std::string& bind_ip = "");
  ~RaidenOrchestrator();

  RaidenOrchestrator(const RaidenOrchestrator&) = delete;
  RaidenOrchestrator& operator=(const RaidenOrchestrator&) = delete;

  // Registers a controller address for a given RaidenId.
  absl::Status RegisterController(const RaidenId& raiden_id,
                                  const std::string& address);

  // Resolves the controller address for a given RaidenId.
  absl::StatusOr<std::string> ResolveController(const RaidenId& raiden_id);

  int port() const { return port_; }

 private:
  void ListenerLoop();
  void ConnectionWorker(int client_fd);

  int port_;
  int server_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;

  mutable absl::Mutex mutex_;
  // Maps RaidenId to Controller Address (e.g., "ip:port")
  absl::flat_hash_map<RaidenId, std::string, RaidenIdHash> registry_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_ORCHESTRATOR_H_
