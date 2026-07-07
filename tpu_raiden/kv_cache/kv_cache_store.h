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

#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"

namespace tpu_raiden {
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
  int hbm_block_id = -1;
  BlockStatus status = BlockStatus::INIT;

  RaidenBlockID() = default;
  /* implicit */ RaidenBlockID(RaidenId id, int host_id = -1,
                               BlockStatus stat = BlockStatus::INIT)
      : raiden_id(std::move(id)), host_block_id(host_id), status(stat) {}

  RaidenBlockID(RaidenId id, int host_id, int hbm_id,
                BlockStatus stat = BlockStatus::INIT)
      : raiden_id(std::move(id)),
        host_block_id(host_id),
        hbm_block_id(hbm_id),
        status(stat) {}

  bool operator==(const RaidenBlockID& other) const {
    return raiden_id == other.raiden_id &&
           host_block_id == other.host_block_id &&
           hbm_block_id == other.hbm_block_id && status == other.status;
  }
};

using BlockSliceList = std::vector<std::pair<std::string, RaidenBlockID>>;

struct FetchRequestItem {
  RaidenId src_raiden_id;
  std::vector<std::string> block_hashes;
  std::vector<int> dst_block_ids;
};

using FetchRequest = std::vector<FetchRequestItem>;

struct FetchCompletionItem {
  std::string block_hash;
  int host_block_id;
  bool success;
  std::string error_message;
};

using FetchCompletion = std::vector<FetchCompletionItem>;

struct LoadRequest {
  uint64_t load_id;
  std::vector<std::string> block_hashes;
  std::vector<int> src_block_ids;
  std::vector<int> dst_block_ids;
};

struct LoadCompletionItem {
  std::string block_hash;
  int dst_hbm_block_id;
  bool success;
  std::string error_message;
};

struct LoadCompletion {
  uint64_t load_id;
  std::vector<LoadCompletionItem> items;
};

struct SaveRequest {
  uint64_t save_id;
  std::vector<std::string> block_hashes;
  std::vector<int> src_block_ids;  // HBM block IDs
};

struct SaveCompletionItem {
  std::string block_hash;
  int dst_host_block_id;
  bool success;
  std::string error_message;
};

struct SaveCompletion {
  uint64_t save_id;
  std::vector<SaveCompletionItem> items;
};

struct EvictRequest {
  uint64_t evict_id;
  std::vector<std::string> block_hashes;
  std::vector<int> host_block_ids;
};

struct EvictCompletionItem {
  std::string block_hash;
  bool success;
  std::string error_message;
};

struct EvictCompletion {
  uint64_t evict_id;
  std::vector<EvictCompletionItem> items;
};

template <typename T>
class ThreadSafeQueue;

using LoadWorkQueue = ThreadSafeQueue<LoadRequest>;
using LoadCompletionQueue = ThreadSafeQueue<LoadCompletion>;
using SaveWorkQueue = ThreadSafeQueue<SaveRequest>;
using SaveCompletionQueue = ThreadSafeQueue<SaveCompletion>;
using EvictWorkQueue = ThreadSafeQueue<EvictRequest>;
using EvictCompletionQueue = ThreadSafeQueue<EvictCompletion>;

template <typename T>
class ThreadSafeQueue {
 public:
  ThreadSafeQueue() = default;
  ~ThreadSafeQueue() = default;

  ThreadSafeQueue(const ThreadSafeQueue&) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

  void Push(T item) {
    absl::MutexLock lock(&mutex_);
    queue_.push(std::move(item));
    cond_.Signal();
  }

  bool Pop(T& item) {
    absl::MutexLock lock(&mutex_);
    while (queue_.empty() && !stopping_) {
      cond_.Wait(&mutex_);
    }
    if (stopping_ && queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool Empty() const {
    absl::MutexLock lock(&mutex_);
    return queue_.empty();
  }

  void Stop() {
    absl::MutexLock lock(&mutex_);
    stopping_ = true;
    cond_.SignalAll();
  }

 private:
  mutable absl::Mutex mutex_;
  std::queue<T> queue_ ABSL_GUARDED_BY(mutex_);
  absl::CondVar cond_;
  bool stopping_ = false;
};

using FetchWorkQueue = ThreadSafeQueue<FetchRequest>;
using FetchCompletionQueue = ThreadSafeQueue<FetchCompletion>;

class RaidenControllerEmbedded;

struct RemoteFetchConfig {
  std::string orchestrator_address;
  int controller_port = 0;
  int local_worker_port = 0;
  size_t bytes_per_block = 0;
  size_t num_shards = 0;
  size_t num_listeners = 1;
};

// KV Store that manages the indices and routing of prefix cache across serving
// nodes and microservice slices.
class KVCacheStore {
 public:
  struct FetchState {
    uint64_t fetch_id;
    std::string block_hash;
    bool failed = false;
    std::string error_message;
    absl::Notification notification;
  };

  class FetchFuture {
   public:
    FetchFuture() = default;
    explicit FetchFuture(std::shared_ptr<FetchState> state)
        : state_(std::move(state)) {}

    absl::Status Await();
    bool IsDone() const;

   private:
    std::shared_ptr<FetchState> state_;
  };

  struct LoadState {
    uint64_t load_id;
    std::string block_hash;
    absl::flat_hash_set<RaidenId, RaidenIdHash> pending_workers;
    bool failed = false;
    std::string error_message;
    absl::Notification notification;
  };

  class LoadFuture {
   public:
    LoadFuture() = default;
    explicit LoadFuture(std::shared_ptr<LoadState> state)
        : state_(std::move(state)) {}

    absl::Status Await();
    bool IsDone() const;

   private:
    std::shared_ptr<LoadState> state_;
  };

  struct SaveState {
    uint64_t save_id;
    std::string block_hash;
    absl::flat_hash_set<RaidenId, RaidenIdHash> pending_workers;
    bool failed = false;
    std::string error_message;
    absl::Notification notification;
  };

  class SaveFuture {
   public:
    SaveFuture() = default;
    explicit SaveFuture(std::shared_ptr<SaveState> state)
        : state_(std::move(state)) {}

    absl::Status Await();
    bool IsDone() const;

   private:
    std::shared_ptr<SaveState> state_;
  };

  struct EvictState {
    uint64_t evict_id;
    std::string block_hash;
    absl::flat_hash_set<RaidenId, RaidenIdHash> pending_workers;
    bool failed = false;
    std::string error_message;
    absl::Notification notification;
  };

  class EvictFuture {
   public:
    EvictFuture() = default;
    explicit EvictFuture(std::shared_ptr<EvictState> state)
        : state_(std::move(state)) {}

    absl::Status Await();
    bool IsDone() const;

   private:
    std::shared_ptr<EvictState> state_;
  };

  explicit KVCacheStore(
      size_t capacity, std::string global_registry_address = "",
      RaidenId raiden_id = {},
      std::optional<RemoteFetchConfig> remote_config = std::nullopt);

  ~KVCacheStore();

  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  // Dedicated Fetch function to trigger remote fetching.
  absl::flat_hash_map<std::string, FetchFuture> FetchRemote(
      const std::vector<std::string>& block_hashes);

  // Polls the status of all active fetches.
  // Returns {done_block_hashes, failed_block_hashes, pending_block_hashes}
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollFetchRemoteStatus();

  // Triggers local H2D load from host memory to device memory (HBM).
  absl::flat_hash_map<std::string, LoadFuture> Load(
      const std::vector<std::string>& block_hashes,
      const std::vector<int>& dst_hbm_block_ids);

  // Polls the status of all active loads.
  // Returns {done_block_hashes, failed_block_hashes, pending_block_hashes}
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollLoadStatus();

  // Triggers local D2H save from device memory (HBM) to host memory.
  absl::flat_hash_map<std::string, SaveFuture> Save(
      const std::vector<std::string>& block_hashes,
      const std::vector<int>& src_hbm_block_ids);

  // Polls the status of all active saves.
  // Returns {done_block_hashes, failed_block_hashes, pending_block_hashes}
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollSaveStatus();

  // Triggers host memory eviction (unlocks host blocks).
  absl::flat_hash_map<std::string, EvictFuture> Evict(
      const std::vector<std::string>& block_hashes);

  // Polls the status of all active evicts.
  // Returns {done_block_hashes, failed_block_hashes, pending_block_hashes}
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollEvictStatus();

  // Authoritative KVCacheStore API implementations

  // Checks the LRU directory for cached block hashes. Returns a list of all
  // matched replica pairs (block hash and vector of RaidenBlockIDs) encountered
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
  // LRU cache, deleting any non-local (not HOST or HOST_AND_HBM) block_hash
  // whose pin count is 0, and putting back evicted entries in reverse order
  // for each deleted non-local block.
  // Returns:
  // - size_t: number of non-local blocks deleted
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

  int GetPinCount(const std::string& hash) const;

  size_t capacity() const;

  const RaidenId& raiden_id() const { return raiden_id_; }

  // Fetch Queue Accessors
  bool PopFetchWork(FetchRequest& req) { return work_queue_.Pop(req); }
  void PushFetchCompletion(FetchCompletion completion) {
    completion_queue_.Push(std::move(completion));
  }
  void PushFetchWork(FetchRequest req) { work_queue_.Push(std::move(req)); }
  bool PopFetchCompletion(FetchCompletion& completion) {
    return completion_queue_.Pop(completion);
  }

  // Load Queue Accessors
  bool PopLoadWork(LoadRequest& req) { return load_work_queue_.Pop(req); }
  void PushLoadCompletion(LoadCompletion completion) {
    load_completion_queue_.Push(std::move(completion));
  }
  void PushLoadWork(LoadRequest req) { load_work_queue_.Push(std::move(req)); }
  bool PopLoadCompletion(LoadCompletion& completion) {
    return load_completion_queue_.Pop(completion);
  }

  // Save Queue Accessors
  bool PopSaveWork(SaveRequest& req) { return save_work_queue_.Pop(req); }
  void PushSaveCompletion(SaveCompletion completion) {
    save_completion_queue_.Push(std::move(completion));
  }
  void PushSaveWork(SaveRequest req) { save_work_queue_.Push(std::move(req)); }
  bool PopSaveCompletion(SaveCompletion& completion) {
    return save_completion_queue_.Pop(completion);
  }

  // Evict Queue Accessors
  bool PopEvictWork(EvictRequest& req) { return evict_work_queue_.Pop(req); }
  void PushEvictCompletion(EvictCompletion completion) {
    evict_completion_queue_.Push(std::move(completion));
  }
  void PushEvictWork(EvictRequest req) {
    evict_work_queue_.Push(std::move(req));
  }
  bool PopEvictCompletion(EvictCompletion& completion) {
    return evict_completion_queue_.Pop(completion);
  }

 private:
  mutable absl::Mutex mutex_;
  mutable LRUCache<std::string, RaidenBlockID> lru_cache_
      ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_;
  RaidenId raiden_id_;
  FetchWorkQueue work_queue_;
  FetchCompletionQueue completion_queue_;
  std::unique_ptr<RaidenControllerEmbedded> controller_;
  std::optional<RemoteFetchConfig> config_;
  void CompletionPollerLoop();
  std::thread completion_poller_thread_;
  void LoadCompletionPollerLoop();
  std::thread load_completion_poller_thread_;
  void SaveCompletionPollerLoop();
  std::thread save_completion_poller_thread_;
  void EvictCompletionPollerLoop();
  std::thread evict_completion_poller_thread_;

  mutable absl::Mutex fetch_mu_;
  uint64_t next_fetch_id_ ABSL_GUARDED_BY(fetch_mu_) = 1;
  absl::flat_hash_map<uint64_t, std::shared_ptr<FetchState>> active_fetches_
      ABSL_GUARDED_BY(fetch_mu_);
  // Maps block hash -> fetch_id. Keyed by hash (not dst block id) because the
  // dst host block may be auto-allocated by the receiving worker during fetch.
  absl::flat_hash_map<std::string, uint64_t> hash_to_fetch_
      ABSL_GUARDED_BY(fetch_mu_);

  std::vector<std::string> done_fetches_ ABSL_GUARDED_BY(fetch_mu_);
  std::vector<std::string> failed_fetches_ ABSL_GUARDED_BY(fetch_mu_);

  LoadWorkQueue load_work_queue_;
  LoadCompletionQueue load_completion_queue_;

  mutable absl::Mutex load_mu_;
  uint64_t next_load_id_ ABSL_GUARDED_BY(load_mu_) = 1;
  absl::flat_hash_map<
      uint64_t, absl::flat_hash_map<std::string, std::shared_ptr<LoadState>>>
      active_loads_ ABSL_GUARDED_BY(load_mu_);
  std::vector<std::string> done_loads_ ABSL_GUARDED_BY(load_mu_);
  std::vector<std::string> failed_loads_ ABSL_GUARDED_BY(load_mu_);

  SaveWorkQueue save_work_queue_;
  SaveCompletionQueue save_completion_queue_;

  mutable absl::Mutex save_mu_;
  uint64_t next_save_id_ ABSL_GUARDED_BY(save_mu_) = 1;
  absl::flat_hash_map<
      uint64_t, absl::flat_hash_map<std::string, std::shared_ptr<SaveState>>>
      active_saves_ ABSL_GUARDED_BY(save_mu_);
  std::vector<std::string> done_saves_ ABSL_GUARDED_BY(save_mu_);
  std::vector<std::string> failed_saves_ ABSL_GUARDED_BY(save_mu_);

  EvictWorkQueue evict_work_queue_;
  EvictCompletionQueue evict_completion_queue_;

  mutable absl::Mutex evict_mu_;
  uint64_t next_evict_id_ ABSL_GUARDED_BY(evict_mu_) = 1;
  absl::flat_hash_map<
      uint64_t, absl::flat_hash_map<std::string, std::shared_ptr<EvictState>>>
      active_evicts_ ABSL_GUARDED_BY(evict_mu_);
  std::vector<std::string> done_evicts_ ABSL_GUARDED_BY(evict_mu_);
  std::vector<std::string> failed_evicts_ ABSL_GUARDED_BY(evict_mu_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
