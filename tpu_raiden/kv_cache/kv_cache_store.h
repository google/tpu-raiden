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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/numa_thread_pool.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {

namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {

namespace global_registry {
class GlobalRegistryClient;
}

enum class BlockStatus {
  INIT,
  REMOTE,
  HBM,
  HOST,
  HOST_AND_HBM,
};

struct RaidenBlockID {
  RaidenId raiden_id;
  int host_block_id = -1;
  int device_block_id = -1;
  BlockStatus status = BlockStatus::INIT;

  RaidenBlockID() = default;
  /* implicit */ RaidenBlockID(RaidenId id, int host_id = -1,
                               BlockStatus stat = BlockStatus::INIT)
      : raiden_id(std::move(id)), host_block_id(host_id), status(stat) {}

  RaidenBlockID(RaidenId id, int host_block_id, int device_block_id,
                BlockStatus stat = BlockStatus::INIT)
      : raiden_id(std::move(id)),
        host_block_id(host_block_id),
        device_block_id(device_block_id),
        status(stat) {}

  bool operator==(const RaidenBlockID& other) const {
    return raiden_id == other.raiden_id &&
           host_block_id == other.host_block_id &&
           device_block_id == other.device_block_id && status == other.status;
  }
};

using BlockSliceList = std::vector<std::pair<std::string, RaidenBlockID>>;

// KV Store that manages the indices and routing of prefix cache across serving
// nodes and microservice slices.
class KVCacheStore {
 public:
  explicit KVCacheStore(size_t capacity,
                        absl::string_view global_registry_address = "",
                        RaidenId raiden_id = {}, int num_shards = 0,
                        int64_t shard_size_bytes = 0,
                        int raiden_controller_port = 0,
                        absl::string_view raiden_orchestrator_address = "");

  // Test-only constructor for injecting mock controller
  explicit KVCacheStore(
      size_t capacity,
      std::unique_ptr<tpu_raiden::controller::RaidenController>
          raiden_controller,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {});

  ~KVCacheStore();

  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  // Authoritative KVCacheStore API implementations

  // Checks the LRU directory for cached block hashes. Returns a list of all
  // matched replica pairs (block hash and RaidenBlockID) encountered
  // in sequence prior to the first miss.
  // If enable_global is true, it will query the global registry for any
  // misses after the local lookup.
  absl::StatusOr<BlockSliceList> Lookup(
      const std::vector<std::string>& block_hashes, bool enable_global = false);

  // Caches sharded buffers into host-RAM/HBM backing store.
  // Returns:
  // - bool: whether all blocks were successfully inserted (i.e. none already
  // existed)
  // - BlockSliceList: list of entries evicted from the LRU cache during
  // insertion
  std::pair<bool, BlockSliceList> Insert(
      const std::vector<std::string>& block_hashes,
      const std::vector<RaidenBlockID>& slices, bool on_host);

  // Pins all existing block hashes, and inserts and pins new block hashes if
  // there is sufficient available space in the LRU cache.
  // Returns:
  // - bool: whether the entire InsertAndPin operation succeeded (i.e. all
  //         existing keys were pinned, all new keys inserted and pinned)
  // - BlockSliceList: list of entries evicted during insertion
  std::pair<bool, BlockSliceList> InsertAndPin(
      const std::vector<std::string>& block_hashes,
      const std::vector<RaidenBlockID>& slices, bool on_host);

  // Reverts an InsertAndPin operation by unpinning all block_hashes in the
  // LRU cache, deleting any block_hash in REMOTE status whose pin count is 0,
  // and putting back evicted entries in reverse order for each deleted remote
  // block.
  // Returns:
  // - size_t: number of remote blocks deleted
  // - BlockSliceList: remaining evicted entries that were not restored
  std::pair<size_t, BlockSliceList> ReleaseAndDelete(
      const std::vector<std::string>& block_hashes,
      BlockSliceList pending_evict_entries = {});

  // Deletes cached sharded buffers from host-RAM/HBM backing store entirely.
  void Delete(const std::vector<std::string>& block_hashes,
              const std::vector<RaidenBlockID>& slices);

  // Pins cached block hashes in memory, protecting them against LRU eviction
  // while in active use. Returns true if all keys exist and were successfully
  // pinned.
  bool Pin(const std::vector<std::string>& block_hashes);

  // Releases previously pinned block hashes, making them eligible for LRU
  // eviction when capacity is exceeded.
  void Release(const std::vector<std::string>& block_hashes);

  // Saves blocks from device (HBM) to host (DRAM).
  absl::Status Save(const std::vector<std::string>& block_hashes);

  // Loads blocks from host (DRAM) to device (HBM).
  absl::Status Load(const std::vector<std::string>& block_hashes,
                    const std::vector<int>& device_block_ids);

  int GetPinCount(const std::string& hash) const;

  size_t capacity() const;
  int raiden_controller_port() const;

  const RaidenId& raiden_id() const { return raiden_id_; }

  // Polls the status of all active/inflight Save operations.
  // Iterates over pending futures, updates cache metadata to HOST_AND_HBM
  // upon successful transfers, and deallocates host blocks on failure.
  //
  // Returns:
  //   A tuple of {done_block_hashes, failed_block_hashes, pending_block_hashes}
  //   representing the status of all block hashes tracked by active saves.
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollSaveStatus();

  // Polls the status of all active/inflight Load operations.
  // Iterates over pending futures, and updates cache metadata to HOST_AND_HBM
  // upon successful H2D transfers.
  //
  // Returns:
  //   A tuple of {done_block_hashes, failed_block_hashes, pending_block_hashes}
  //   representing the status of all block hashes tracked by active loads.
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollLoadStatus();

 private:
  struct SaveState {
    tsl::Future<> future;
    std::vector<std::string> block_hashes;
    std::vector<int> host_block_ids;
  };

  struct LoadState {
    tsl::Future<> future;
    std::vector<std::string> block_hashes;
    std::vector<int> device_block_ids;
  };

  mutable absl::Mutex mutex_;
  mutable LRUCache<std::string, RaidenBlockID> lru_cache_
      ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_;
  RaidenId raiden_id_;
  std::unique_ptr<tpu_raiden::controller::RaidenController> raiden_controller_;

  std::vector<SaveState> active_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<LoadState> active_loads_ ABSL_GUARDED_BY(mutex_);

  std::vector<std::string> done_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_loads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_loads_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_set<std::string> saving_hashes_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_set<std::string> loading_hashes_ ABSL_GUARDED_BY(mutex_);

  std::unique_ptr<std::thread> poller_thread_;
  std::unique_ptr<tpu_raiden::NumaThreadPool> write_through_pool_;
  std::atomic<bool> stop_poller_{false};

  void PollerLoop();
  void PollFuturesInternal();
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
