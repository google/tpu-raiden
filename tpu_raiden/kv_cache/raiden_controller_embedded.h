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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_CONTROLLER_EMBEDDED_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_CONTROLLER_EMBEDDED_H_

#include <atomic>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_raiden/kv_cache/kv_cache_store.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

// Embedded C++ implementation of RaidenController running in the control plane.
// Manages asynchronous remote fetching via FetchWorkQueue and
// FetchCompletionQueue.
class RaidenControllerEmbedded {
 public:
  RaidenControllerEmbedded(
      KVCacheStore* store, int port, const std::string& orchestrator_address,
      int local_worker_port,
      const std::vector<std::string>& local_worker_data_addresses,
      size_t bytes_per_block, size_t num_shards, size_t num_listeners = 1);
  ~RaidenControllerEmbedded();

  RaidenControllerEmbedded(const RaidenControllerEmbedded&) = delete;
  RaidenControllerEmbedded& operator=(const RaidenControllerEmbedded&) = delete;

  // Starts the background threads (Listener and Queue Poller).
  absl::Status Start();

  // Stops the background threads.
  void Stop();

  int port() const { return port_; }

 private:
  // Thread function to poll FetchWorkQueue and process requests.
  void WorkQueuePollerLoop();

  // Thread function to poll LoadWorkQueue and process requests.
  void LoadWorkQueuePollerLoop();

  // Thread function to poll SaveWorkQueue and process requests.
  void SaveWorkQueuePollerLoop();

  // Thread function to listen for incoming control connections from other
  // controllers.
  void ListenerLoop();

  // Worker thread function to handle a specific control connection.
  void ConnectionWorker(int client_fd);

  // Helper to register this controller with the central orchestrator.
  absl::Status RegisterWithOrchestrator();

  // Helper to resolve a remote controller address via the orchestrator.
  absl::StatusOr<std::string> ResolveRemoteController(
      const RaidenId& remote_id);

  // Helper to negotiate fetch with a remote source controller.
  absl::StatusOr<tpu_raiden::rpc::ControlResponse> NegotiateFetch(
      const std::string& remote_addr, const FetchRequestItem& request,
      uint64_t uuid);

  // Helper to handle load completion from a local worker
  void HandleLoadCompletionFromWorker(uint64_t load_id, bool success,
                                      const std::string& error_message);

  // Helper to handle save completion from a local worker
  void HandleSaveCompletionFromWorker(
      uint64_t save_id, bool success, const std::string& error_message,
      const std::vector<int>& completed_block_ids = {});

  // Thread function to poll EvictWorkQueue and process requests.
  void EvictWorkQueuePollerLoop();

  // Helper to handle evict completion from a local worker
  void HandleEvictCompletionFromWorker(uint64_t evict_id, bool success,
                                       const std::string& error_message);

  KVCacheStore* store_;  // Not owned.
  int port_;
  std::string orchestrator_address_;
  std::string self_address_;

  int local_worker_port_;
  std::vector<std::string> local_worker_data_addresses_;
  size_t bytes_per_block_;
  size_t num_shards_;
  size_t num_listeners_;
  std::vector<size_t> listener_shard_counts_;

  int server_fd_ = -1;

  std::atomic<bool> stopping_{false};

  absl::Mutex pending_mu_;
  // Maps a fetch transfer uuid to its ORDERED block hashes, so a completion
  // (which carries worker-side block ids in the same order, possibly
  // auto-allocated) can be re-keyed to the block hash. Keyed by hash downstream.
  absl::flat_hash_map<uint64_t, std::vector<std::string>> pending_fetches_
      ABSL_GUARDED_BY(pending_mu_);
  // Maps block hash to number of listeners that have completed it.
  absl::flat_hash_map<std::string, size_t> block_completion_counts_
      ABSL_GUARDED_BY(pending_mu_);
  // Maps a fetch uuid to how many of its hashes are still outstanding, so the
  // pending_fetches_ entry can be dropped once every hash finished.
  absl::flat_hash_map<uint64_t, size_t> fetch_remaining_
      ABSL_GUARDED_BY(pending_mu_);

  struct ControllerLoadState {
    uint64_t load_id;
    size_t expected_workers = 0;
    size_t completed_workers = 0;
    std::vector<LoadCompletionItem> items;
  };
  absl::flat_hash_map<uint64_t, ControllerLoadState> active_loads_
      ABSL_GUARDED_BY(pending_mu_);

  struct ControllerSaveState {
    uint64_t save_id;
    size_t expected_workers = 0;
    size_t completed_workers = 0;
    std::vector<SaveCompletionItem> items;
  };
  absl::flat_hash_map<uint64_t, ControllerSaveState> active_saves_
      ABSL_GUARDED_BY(pending_mu_);

  struct ControllerEvictState {
    uint64_t evict_id;
    size_t expected_workers = 0;
    size_t completed_workers = 0;
    std::vector<EvictCompletionItem> items;
  };
  absl::flat_hash_map<uint64_t, ControllerEvictState> active_evicts_
      ABSL_GUARDED_BY(pending_mu_);

  std::thread poller_thread_;
  std::thread load_poller_thread_;
  std::thread save_poller_thread_;
  std::thread evict_poller_thread_;
  std::thread listener_thread_;
  std::vector<std::thread> worker_threads_;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_RAIDEN_CONTROLLER_EMBEDDED_H_
