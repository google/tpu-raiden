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
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/numa_thread_pool.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_server.h"
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

// KV Store that manages the indices and routing of prefix cache across serving
// nodes and microservice slices.
class KVCacheStore {
 public:
  friend class KVCacheStoreTest;

  // NETWORK ADDRESSING (`store_server_ip`, `raiden_controller_port`)
  //
  // `store_server_ip` is the IP peers use to reach this node. It is
  // BIND-AND-ADVERTISE: both this store's KVCacheStoreService and its
  // RaidenController bind that interface, and it is the host published to the
  // global registry. It must therefore be an IP this process can bind -- not a
  // hostname, and not a NAT/service address. Ports are chosen by gRPC (the
  // store server always; the controller when `raiden_controller_port` is 0),
  // and the bound port is spliced into the advertised address. Note that the IP
  // address of the RaidenController reuses `store_server_ip`.
  //
  // `store_server_ip` is MANDATORY and must not be a wildcard: `Create()`
  // rejects an empty or wildcard value, and the raw constructors abort on it.
  // Whether this store is discoverable is decided by the registry, not by this
  // field -- with no `global_registry_address` no server is started at all.
  //
  // `expected_worker_count` > 0 makes construction block until that many
  // workers have registered with the in-process RaidenController, and fail
  // (Create() throws, the raw constructors abort the process via the
  // controller's exception) if they have not all registered within
  // RAIDEN_EXPECTED_WORKERS_TIMEOUT_S seconds (default 120). Callers whose
  // workers register only after the store exists must leave it 0.

  // Safe factory method to create KVCacheStore from a single BackendConfig.
  static absl::StatusOr<std::unique_ptr<KVCacheStore>> Create(
      const BackendConfig& config, size_t capacity = 0,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {},
      int num_shards = 0, int64_t shard_size_bytes = 0,
      absl::string_view raiden_orchestrator_address = "",
      absl::string_view store_server_ip = "", int raiden_controller_port = 0,
      std::optional<KVCacheMetadata> metadata = std::nullopt,
      int expected_worker_count = 0);

  // Safe factory method to create KVCacheStore from a list of BackendConfigs.
  static absl::StatusOr<std::unique_ptr<KVCacheStore>> Create(
      absl::Span<const BackendConfig> backend_configs, size_t capacity = 0,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {},
      int num_shards = 0, int64_t shard_size_bytes = 0,
      absl::string_view raiden_orchestrator_address = "",
      absl::string_view store_server_ip = "", int raiden_controller_port = 0,
      std::optional<KVCacheMetadata> metadata = std::nullopt,
      int expected_worker_count = 0);

  // Flexible constructor accepting a custom root backend
  explicit KVCacheStore(std::shared_ptr<KVCacheStoreBackend> backend,
                        RaidenId raiden_id = {}, int num_shards = 0,
                        int64_t shard_size_bytes = 0,
                        absl::string_view raiden_orchestrator_address = "",
                        absl::string_view store_server_ip = "",
                        int raiden_controller_port = 0,
                        absl::string_view global_registry_address = "",
                        int expected_worker_count = 0);

  // Constructor accepting an ordered list of backends.
  // `global_registry_address` is what this store publishes itself to. The
  // backends may each hold their own registry client; this one belongs to the
  // store, and without it the store cannot make itself discoverable.
  explicit KVCacheStore(
      std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
      RaidenId raiden_id = {}, int num_shards = 0, int64_t shard_size_bytes = 0,
      absl::string_view raiden_orchestrator_address = "",
      absl::string_view store_server_ip = "", int raiden_controller_port = 0,
      absl::string_view global_registry_address = "",
      int expected_worker_count = 0);

  // Links RaidenController to all backends in backends_. Returns an error iff
  // this store owns a store_server_ip and starting/publishing its server
  // failed -- store initialization requires the store to be discoverable
  // whenever the caller asked for one.
  absl::Status SetRaidenController(
      tpu_raiden::controller::RaidenController* controller);

  // If `metadata` is provided, every LRU cache entry whose data lives in
  // local host memory is mirrored into it, keeping a crash-persistent copy of
  // the LRU cache in the caller's region (see KVCacheMetadata).
  explicit KVCacheStore(size_t capacity,
                        absl::string_view global_registry_address = "",
                        RaidenId raiden_id = {}, int num_shards = 0,
                        int64_t shard_size_bytes = 0,
                        absl::string_view raiden_orchestrator_address = "",
                        absl::string_view store_server_ip = "",
                        int raiden_controller_port = 0,
                        std::optional<KVCacheMetadata> metadata = std::nullopt,
                        int expected_worker_count = 0);

  // Test-only constructor for injecting mock controller
  explicit KVCacheStore(
      size_t capacity,
      std::unique_ptr<tpu_raiden::controller::RaidenController>
          raiden_controller,
      absl::string_view global_registry_address = "", RaidenId raiden_id = {},
      std::optional<KVCacheMetadata> metadata = std::nullopt,
      absl::string_view store_server_ip = "");

  ~KVCacheStore();

  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  // Authoritative KVCacheStore API implementations

  // Checks the LRU cache for cached block hashes. Returns a list of all
  // matched replica pairs (block hash and RaidenBlockID) encountered
  // in sequence prior to the first miss.
  // If enable_global is true, it will query the global registry for any
  // misses after the local lookup.
  absl::StatusOr<BlockSliceList> Lookup(
      const std::vector<std::string>& block_hashes, bool enable_global = true);

  // Caches sharded buffers into host-RAM/HBM backing store.
  // Returns:
  // - bool: whether all blocks were successfully inserted (i.e. none already
  // existed)
  // - BlockSliceList: list of entries evicted from the LRU cache during
  // insertion
  std::pair<bool, BlockSliceList> Insert(
      const std::vector<std::string>& block_hashes,
      const std::vector<RaidenBlockID>& slices, bool on_host);

  // Pins all existing block hashes, and inserts and locks new block hashes if
  // there is sufficient available space in the LRU cache.
  // New items are inserted in reverse order to ensure tail blocks are evicted
  // first. Returns:
  // - bool: whether the entire InsertAndLock operation succeeded (i.e. all
  //         existing keys were locked, all new keys inserted and locked)
  bool InsertAndLock(const std::vector<std::string>& block_hashes,
                     const std::vector<RaidenBlockID>& slices, bool on_host);

  // Reverts an InsertAndLock operation by unlocking all block_hashes in the
  // LRU cache, deleting any block_hash NOT in HOST or HOST_AND_HBM status
  // whose pin count is 0, and restoring evicted entries from the LRU cache's
  // candidate list for each deleted block.
  // Returns:
  // - size_t: number of blocks deleted
  size_t ReleaseAndDelete(const std::vector<std::string>& block_hashes);

  // Deletes cached sharded buffers from host-RAM/HBM backing store entirely.
  void Delete(const std::vector<std::string>& block_hashes,
              const std::vector<RaidenBlockID>& slices);

  // Pins cached block hashes in memory, protecting them against LRU eviction
  // while in active use. Returns true if all keys exist and were successfully
  // pinned.
  bool Pin(const std::vector<std::string>& block_hashes);

  // Releases previously pinned block hashes (a.k.a. Unpin), making them
  // eligible for LRU eviction when capacity is exceeded.
  // The blocks are released in reversed order internally to ensure that the
  // last blocks (tails of sequences) are evicted first.
  void Release(const std::vector<std::string>& block_hashes);

  // Saves blocks from device (HBM) to host (DRAM) asynchronously.
  //
  // NOTE: The block_hashes must be pinned in the LRU cache (e.g., via
  // InsertAndLock) before calling Save. Once the operation is complete (as
  // reported by PollSaveStatus), the caller must manually release/unpin them
  // via Release so they become eligible for eviction.
  //
  // Recommended usage flow:
  //   0. [optional for save] lookup block hashes
  //   1. insert_and_lock(block_hashes)
  //   2. save(block_hashes)
  //   3. poll_save_status() -> wait for completion
  //   4. release(completed_block_hashes)
  absl::Status Save(const std::vector<std::string>& block_hashes);

  // Loads blocks from host (DRAM) to device (HBM) asynchronously.
  //
  // NOTE: The block_hashes must be pinned in the LRU cache (e.g., via Pin or
  // InsertAndLock) before calling Load. Once the operation is complete (as
  // reported by PollLoadStatus), the caller must manually release/unpin them
  // via Release so they become eligible for eviction.
  //
  // Recommended usage flow:
  //   1. insert_and_lock(block_hashes) (or Pin if they already exist in store)
  //   2. load(block_hashes, device_block_ids)
  //   3. poll_load_status() -> wait for completion
  //   4. release(completed_block_hashes)
  absl::Status Load(const std::vector<std::string>& block_hashes,
                    const std::vector<int>& device_block_ids);

  int GetPinCount(const std::string& hash) const;

  // Source-side ReadRemote (All-or-Nothing validate & pin block hashes at the
  // src controller): verifies ALL block_hashes exist in the LRU with status
  // HOST/HOST_AND_HBM and pins them (all-or-nothing: any miss rolls back and
  // aborts). On success returns the authoritative source host_block_ids
  // (re-derived from the LRU). NotFound => a hash is absent
  // (BLOCK_HASH_NOT_FOUND); FailedPrecondition => present but not
  // host-resident. Registered as a hook on the RaidenController so a peer's
  // ReadRemote RPC can reach this store's LRU. Public for testability.
  absl::StatusOr<std::vector<int32_t>> ValidateAndPinHostBlocks(
      absl::Span<const std::string> block_hashes);
  // Releases the pins taken by ValidateAndPinHostBlocks.
  void UnpinHostBlocks(absl::Span<const std::string> block_hashes);

  size_t capacity() const;
  std::string raiden_controller_address() const;

  // "host:port" of this store's KVCacheStoreService, as published to the
  // global registry. Empty when this store has no registry configured, since
  // that is exactly when no server is started -- not when `store_server_ip`
  // was omitted, which is now a construction error.
  std::string store_server_address() const { return store_server_address_; }

  // The store server this node serves peers from, or nullptr if it has none.
  // Owned either by this store or by whichever backend hosts it.
  KVCacheStoreServer* store_server() const { return store_server_; }

  const std::vector<std::shared_ptr<KVCacheStoreBackend>>& backends() const {
    return backends_;
  }

  const std::shared_ptr<KVCacheStoreBackend>& backend() const {
    return backends_[0];
  }

  const RaidenId& raiden_id() const { return raiden_id_; }

  tpu_raiden::controller::RaidenController* raiden_controller() const {
    return raiden_controller_.get();
  }

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

  // Launches an async receiver-initiated read of REMOTE blocks from their
  // owning peers. Returns as soon as the reads are issued; poll with
  // PollRemoteReadStatus().
  //
  // device_block_ids selects the destination:
  //   empty                    -> read to host. On success the entries become
  //                               HOST.
  //   size == block_hashes     -> read to HBM. The bytes land in the caller's
  //                               device blocks, with the host landing blocks
  //                               as the staging hop, so the entries become
  //                               HOST_AND_HBM and a later load() can reuse
  //                               the host copy.
  //   any other size           -> InvalidArgument.
  //
  // CALLER CONTRACT: every requested hash must already be pinned, and must
  // stay pinned until PollRemoteReadStatus() reports it terminal (done or
  // failed). Releasing early makes the entry eligible for deletion mid-read;
  // the read is then discarded and the WHOLE batch reported failed.
  //
  // In read-to-HBM mode the device blocks are written before the source's
  // verdict is known, so on failure their contents are UNDEFINED -- treat
  // supplied device blocks as scratch until the read reports success. Nothing
  // in the cache ever points at them unless the read commits.
  absl::Status ReadRemote(const std::vector<std::string>& block_hashes,
                          const std::vector<int32_t>& device_block_ids = {});

  // Polls status of active remote reads.
  // Returns {done_hashes, failed_hashes, pending_hashes}
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollRemoteReadStatus();

  // Offers `block_hashes` to a peer, which takes its own copy. The use case is
  // proactive eviction: a node under memory pressure hands cold blocks to a
  // peer and then frees them locally.
  //
  // Returns as soon as the destination has ACCEPTED (or refused); poll with
  // PollRemoteWriteStatus(). The blocks stay pinned internally until the
  // operation goes terminal or HOLD expires, whichever comes first -- the
  // caller's own pin is separate and the caller releases it itself.
  //
  // Requires a global registry at both ends: that is how the destination is
  // resolved, and how the blocks become reachable once they land.
  //
  // IMPORTANT: the destination allocates landing blocks from FREE blocks only
  // and never evicts to make room, so a peer with a warm cache refuses every
  // offer with RESOURCE_EXHAUSTED. Destination-side eviction is separate,
  // unimplemented work.
  absl::Status WriteRemote(const std::vector<std::string>& block_hashes,
                           const RaidenId& dst_raiden_id);

  // Polls all active/inflight WriteRemote operations.
  //
  // Returns {done, failed, pending, existing, unregistered}. The last two are
  // annotations on failures, not separate outcomes -- a hash in either also
  // appears in `failed` -- and they exist because this store never decides on
  // the caller's behalf:
  //
  //   existing:     the destination already held these, so it refused the
  //                 batch. Reissuing with the remainder is the caller's call.
  //   unregistered: the destination HAS these -- the transfer succeeded -- but
  //                 could not publish them, so no peer can find them. Reported
  //                 as failed because the safe default is for the caller to
  //                 keep its own copy. A caller that only needed the peer to
  //                 have the bytes may treat it as success; one that was about
  //                 to free its copy must not.
  //
  // Terminal results are drained on read, as with PollSaveStatus.
  std::tuple<std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>, std::vector<std::string>,
             std::vector<std::string>>
  PollRemoteWriteStatus();

  // Rebuilds this store's LRU cache after an engine restart from the
  // crash-persistent KVCacheMetadata table in local shared memory, without
  // consulting the global registry.
  //
  // - Requires a raiden controller, an attached metadata table, and an empty
  //   LRU cache.
  // - Allocates the recorded host block IDs in the controller, then
  //   repopulates the LRU cache with unpinned HOST entries in
  //   ascending-seq order (oldest first).
  // - Entries exceeding the LRU cache capacity overflow into eviction
  //   candidates, keeping their blocks and metadata entries.
  // - The seq counter resumes past the largest recovered stamp.
  // - If two blocks claim the same hash, the entry with the largest seq is
  //   the newest binding and wins; stale ones are not recovered and are
  //   cleared from the table.
  // - On any failure the store and the table are left untouched so the
  //   caller can fall back to a cold start.
  //
  // Returns the number of recovered blocks.
  absl::StatusOr<size_t> RecoverFromLocalManifest();

 private:
  // Tag selecting the constructor overload below that does NOT auto-wire the
  // controller (SetRaidenController) or FATAL on failure -- used exclusively
  // by Create(), which does that wiring itself afterward so a publish failure
  // can return a Status instead of aborting the process. The public
  // constructor with the same parameters delegates to this one and then
  // wires+FATALs, for direct (non-Create()) callers.
  struct CreateTag {};
  KVCacheStore(std::vector<std::shared_ptr<KVCacheStoreBackend>> backends,
               RaidenId raiden_id, int num_shards, int64_t shard_size_bytes,
               absl::string_view raiden_orchestrator_address,
               absl::string_view store_server_ip, int raiden_controller_port,
               absl::string_view global_registry_address,
               int expected_worker_count, CreateTag);

  // Registers ValidateAndPinHostBlocks/UnpinHostBlocks as ReadRemote step-6a
  // hooks on raiden_controller_ (no-op if there is no controller).
  void RegisterReadRemoteHooks();

  // Evicts host blocks by their logical block hashes.
  //
  // Checks if the blocks exist in the cache and are in HOST or HOST_AND_HBM
  // status and erases them from the cache.
  // Releases the host blocks via RaidenController and unregisters them from
  // GlobalRegistry.
  //
  // Input: `block_hashes` - list of block hashes to evict.
  // Output: Number of successfully evicted host blocks.
  size_t Evict(const std::vector<std::string>& block_hashes);

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

  struct RemoteReadState {
    std::vector<std::string> block_hashes;
    // The local landing blocks. These live HERE and nowhere else until the
    // poller commits -- stamping them into the LRU entry at issue time would
    // destroy the peer coordinate the entry needs for a retry.
    std::vector<int> host_block_ids;
    // Empty for a read to host; otherwise the caller's device blocks.
    std::vector<int32_t> device_block_ids;
  };

  struct FutureHash {
    size_t operator()(const tsl::Future<>& f) const {
      return reinterpret_cast<size_t>(f.async_value());
    }
  };

  struct FutureEqual {
    bool operator()(const tsl::Future<>& lhs, const tsl::Future<>& rhs) const {
      return lhs.async_value() == rhs.async_value();
    }
  };

  // Starts (if needed) the peer-facing store server, computes
  // store_server_address_, and publishes it to the global registry.
  // Idempotent: SetRaidenController may be called more than once, and Create
  // calls it again after construction. With no registry_client_, this is
  // a no-op -- no server is started anywhere. Returns an error iff a registry
  // is configured and starting or publishing the server failed.
  absl::Status EnsureStoreServerAndRegister(
      tpu_raiden::controller::RaidenController* controller);

  // Shuts down every server hosted by one of backends_, skipping
  // `already_shut` (the adopted server, when this store had one). Called from
  // the destructor body, where raiden_controller_ is still alive: a running
  // service holds it in a pointer it cannot re-seat.
  void ShutdownBackendStoreServers(KVCacheStoreServer* already_shut);

  mutable absl::Mutex mutex_;
  std::vector<std::shared_ptr<KVCacheStoreBackend>> backends_;
  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_;
  RaidenId raiden_id_;
  std::unique_ptr<tpu_raiden::controller::RaidenController> raiden_controller_;

  // The IP peers reach this node on. Never empty and never a wildcard --
  // construction rejects both.
  std::string store_server_ip_;
  // Advertised "host:port" of store_server_, empty until it is started.
  std::string store_server_address_;
  // Non-owning. Points either into owned_store_server_ or at a server owned by
  // one of backends_.
  KVCacheStoreServer* store_server_ = nullptr;
  // Set only when no backend hosts a store server and this store made its own.
  std::unique_ptr<KVCacheStoreServer> owned_store_server_;
  // True once this store published itself, so teardown knows to unpublish.
  bool registered_in_global_registry_ = false;

  struct RemoteWriteState {
    RaidenId dst_raiden_id;
    uint64_t operation_id = 0;
    std::vector<std::string> block_hashes;
    // When this source stops protecting the blocks and gives up, whatever the
    // destination is doing. Must outlive the deadline the destination granted,
    // or this store would unpin while the destination could still legitimately
    // commit bytes read out of blocks it has released.
    absl::Time hold_expiry;
  };

  std::vector<SaveState> active_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<LoadState> active_loads_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<tsl::Future<>, RemoteReadState, FutureHash, FutureEqual>
      active_remote_reads_ ABSL_GUARDED_BY(mutex_);

  std::vector<RemoteWriteState> active_remote_writes_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_remote_writes_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_remote_writes_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> existing_remote_writes_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> unregistered_remote_writes_ ABSL_GUARDED_BY(mutex_);

  std::vector<std::string> done_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_saves_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_loads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_loads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> done_remote_reads_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::string> failed_remote_reads_ ABSL_GUARDED_BY(mutex_);

  absl::flat_hash_set<std::string> saving_hashes_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_set<std::string> loading_hashes_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_set<std::string> reading_hashes_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_set<std::string> writing_hashes_ ABSL_GUARDED_BY(mutex_);

  std::unique_ptr<std::thread> poller_thread_;
  std::unique_ptr<tpu_raiden::NumaThreadPool> write_through_pool_;
  std::atomic<bool> stop_poller_{false};

  absl::StatusOr<std::vector<int>> AllocateBlockIds(int needed);
  void DeallocateBlockIds(absl::Span<const int> block_ids);

  // Asks the destination what became of each accepted offer, and gives up on
  // any whose HOLD has expired. Runs on the store's own poller, at the same
  // cadence as the save/load pollers, so remote writes do not introduce a
  // second one.
  void PollRemoteWritesInternal();

  // Releases this store's internal pin and clears writing_hashes_. Called once
  // per operation, whichever way it ends.
  void FinishRemoteWrite(const RemoteWriteState& state, bool succeeded,
                         std::vector<std::string> existing,
                         std::vector<std::string> unregistered = {});

  void PollerLoop();
  void PollSavesInternal(std::vector<SaveState> ready_saves);
  void PollLoadsInternal(std::vector<LoadState> ready_loads);
  void PollRemoteReadsInternal(
      std::vector<std::pair<tsl::Future<>, RemoteReadState>>
          ready_remote_reads);
  void PollFuturesInternal();
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_H_
