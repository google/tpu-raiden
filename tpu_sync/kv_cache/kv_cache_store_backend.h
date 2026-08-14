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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_sync/kv_cache/lru_cache.h"
#include "tpu_sync/kv_cache/raiden_id.h"

namespace tpu_raiden {

namespace kv_cache {

// Forward-declared rather than included: kv_cache_store_server.h includes this
// header, so including it back would be circular.
class KVCacheStoreServer;

enum class BlockStatus {
  INIT,
  REMOTE,
  HBM,
  HOST,
  HOST_AND_HBM,
};

struct RaidenBlockID {
  RaidenId raiden_id;
  // When status is REMOTE, it represents the remote host block ID.
  // When status is HOST or HOST_AND_HBM, it represents the local host block ID.
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
  bool operator!=(const RaidenBlockID& other) const {
    return !(*this == other);
  }
};

using BlockSliceList = std::vector<std::pair<std::string, RaidenBlockID>>;

// A pinned, stable reference to one entry in a backend's index.
//
// For callers that resolve a hash once and then need to reach its entry again
// later -- an async operation that must record its outcome when it completes.
// Looking the hash up a second time is what this avoids, which matters because
// that lookup can fall through to the global registry and become a blocking
// RPC.
//
// The pin the handle holds is what keeps it valid, so it must be released
// exactly once. See LRUCache::Handle.
using BlockHandle = LRUCache<std::string, RaidenBlockID>::Handle;

// Options controlling lookup behavior across storage backends.
struct LookupOptions {
  // Controls whether the lookup is allowed to query the global registry.
  // Default is true.
  bool enable_global = true;

  // Controls whether a block cached remotely may sit between two blocks cached
  // locally in the answer. When true, the local index and the global registry
  // are consulted for every hash and the answer runs to the first hash neither
  // can resolve. When false, the answer stops at the first LOCAL miss and only
  // the hashes after it are looked for remotely, so a locally cached block can
  // never follow a remote one.
  // Default is true.
  bool enable_interleaved_lookup = true;

  // Controls whether matched local blocks in store/backends are automatically
  // pinned in memory to protect against LRU eviction.
  // Default is false.
  bool pin_found = false;
};

// Abstract interface for KV cache index storage backends.
// Implementations must be thread-safe.
class KVCacheStoreBackend {
 public:
  virtual ~KVCacheStoreBackend() = default;

  // Name identifying the backend type (e.g., "LruCacheBackend",
  // "GlobalMemoryPoolingBackend").
  virtual std::string name() const = 0;

  // Resolves cached block hashes in sequence.
  // Returns a list of matched (block_hash, RaidenBlockID) pairs up to the first
  // miss.
  virtual absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) = 0;

  // Resolves one hash and pins it, returning a handle that reaches the entry
  // without probing again. The pin is the handle's own reference, independent
  // of any the caller holds, so the entry cannot be evicted or deleted while
  // the handle is alive -- an async operation can therefore rely on it even if
  // the caller releases early.
  //
  // Returns an empty handle if the hash is absent, or from a backend with no
  // index to hand one out of. Callers must treat that as failure: there is
  // nothing to hold.
  virtual BlockHandle AcquireBlockHandle(const std::string& block_hash) {
    return BlockHandle();
  }

  // Drops the pin AcquireBlockHandle took and empties the handle. No-op on an
  // empty handle, so unwinding a partly acquired batch needs no bookkeeping.
  virtual void ReleaseBlockHandle(BlockHandle& handle) {}

  // Executes a mutator on the block handle under the backend's internal lock.
  // This is required to safely mutate fields like status or raiden_id without
  // introducing data races with concurrent readers (e.g. peer gRPC lookups).
  virtual void MutateBlockHandle(BlockHandle& handle, absl::AnyInvocable<void(RaidenBlockID*)> mutator) {}

  // Asynchronously loads KV cache blocks to device (HBM), either from local
  // host DRAM or from the peer named by `remote_id`.
  //
  // `device_block_ids` is the destination and must name one device block per
  // hash.
  //
  // If `slices` is non-empty, the caller's pre-looked up RaidenBlockIDs are
  // used directly. Note that blocks in `slices` must be already pinned
  // externally (when Load from local host), and remote loads will re-resolve
  // hashes at the peer, ignoring `slices`.
  virtual tsl::Future<> Load(const RaidenId& remote_id,
                             absl::Span<const std::string> block_hashes,
                             absl::Span<const int32_t> device_block_ids,
                             absl::Span<const RaidenBlockID> slices = {}) = 0;

  // Inserts key-block mappings into the backend.
  // Returns:
  //   - bool: true if all hashes were newly inserted (none already existed)
  //   - BlockSliceList: entries evicted from this backend during insertion
  virtual std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockID> slices, bool on_host) = 0;

  // Pins existing hashes and inserts & locks new hashes if space permits.
  // Performs complete rollback on failure. Returns true on full success.
  virtual bool InsertAndLock(absl::Span<const std::string> block_hashes,
                             absl::Span<const RaidenBlockID> slices,
                             bool on_host) = 0;

  // Reverts an InsertAndLock operation: unpins hashes, erases non-host blocks
  // whose pin count drops to 0, and restores candidate evictions.
  // Returns the number of deleted blocks.
  virtual size_t ReleaseAndDelete(
      absl::Span<const std::string> block_hashes) = 0;

  // Explicitly deletes cached block entries from the backend.
  virtual void Delete(absl::Span<const std::string> block_hashes,
                      absl::Span<const RaidenBlockID> slices) = 0;

  // Pins block hashes to protect them from LRU eviction.
  // Returns true if all hashes exist and were successfully pinned.
  virtual bool Pin(absl::Span<const std::string> block_hashes) = 0;

  // Releases (unpins) previously pinned block hashes.
  virtual void Release(absl::Span<const std::string> block_hashes) = 0;

  // Returns current pin count for a single hash (0 if absent).
  virtual int GetPinCount(const std::string& hash) const = 0;

  // Maximum capacity of entries supported by this backend.
  virtual size_t GetCapacity() const = 0;

  // Current number of active entries stored in this backend.
  virtual size_t GetSize() const = 0;

  // Remaining evictable/unpinned space in this backend.
  virtual size_t GetAvailableSpace() const = 0;

  // Reconstructs index entries from a local persistent manifest/metadata table.
  // Default implementation returns 0 (no recovery).
  virtual absl::StatusOr<size_t> RecoverFromLocalManifest() { return 0; }

  // Validates host DRAM block IDs and pins them in local memory.
  // Default implementation returns false.
  virtual bool ValidateAndPinHostBlocks(absl::Span<const int> host_block_ids) {
    return false;
  }

  // Retrieves evictable keys from the backend.
  virtual std::vector<std::string> GetEvictableKeys(size_t count) { return {}; }

  // Evicts keys from the backend and returns deallocated host block IDs.
  virtual std::vector<int> Evict(const std::vector<std::string>& block_hashes) {
    Delete(block_hashes, {});
    return {};
  }

  // Retrieves eviction candidate keys from the backend (for
  // testing/diagnostics).
  virtual std::vector<std::string> GetEvictCandidateKeys() const { return {}; }

  // The peer-facing KVCacheStoreService server this backend hosts, if any.
  // Returning non-null tells the owning KVCacheStore to publish THIS server
  // --- Remote write (WriteRemote) ------------------------------------------
  //
  // These four exist because a WriteRemote handler holds only this interface,
  // and because each needs to happen under ONE acquisition of the backend's
  // lock. Composing them at the service out of Lookup/Insert/Delete would
  // check and then act with the lock dropped in between, which is exactly the
  // window a concurrent writer needs.
  //
  // The defaults are all refusals, so a backend that does not implement them
  // makes WriteRemote fail rather than half-succeed.

  // Which of `block_hashes` this backend already holds in host DRAM.
  //
  // Not Lookup(): Lookup stops at the first miss (it answers a prefix
  // question) so it cannot report a scattered subset, and it promotes LRU
  // order as a side effect. This has to see eviction CANDIDATES too -- a
  // candidate still has its host block, so treating it as absent would let a
  // duplicate through.
  virtual std::vector<std::string> AlreadyPresentHostResident(
      absl::Span<const std::string> block_hashes) const {
    return {};
  }

  // Inserts every hash or none, returning false if any part fails.
  //
  // Not Insert(): Insert rebinds a hash that is already present, which for a
  // remote write would orphan the old host block, and it reports partial
  // success, which would let the source free blocks the destination does not
  // have.
  virtual bool InsertAllOrNothing(absl::Span<const std::string> block_hashes,
                                  absl::Span<const RaidenBlockID> slices) {
    return false;
  }

  // Undoes an InsertAllOrNothing: erases the entries, clears their metadata,
  // AND returns the host blocks to the pool.
  //
  // Not Delete(): Delete silently skips pinned hashes and issues its own
  // Unregister. A rollback that skipped a pinned entry would leave an LRU
  // entry pointing at a block this then frees.
  virtual void RollbackInsert(absl::Span<const std::string> block_hashes,
                              absl::Span<const int32_t> host_block_ids) {}

  // Publishes `block_hashes` to the global registry and reports whether that
  // succeeded.
  //
  // Not Insert()'s inline Register: that one logs and swallows the failure. A
  // remote write may only report COMMITTED once the blocks are globally
  // reachable, or the source frees its copy of blocks no lookup can find.
  virtual absl::Status RegisterBlocksSync(
      absl::Span<const std::string> block_hashes,
      absl::Span<const int32_t> host_block_ids) {
    return absl::UnimplementedError(
        "Backend does not implement RegisterBlocksSync.");
  }

  // The peer-facing KVCacheStoreService server this backend hosts, if any.
  // Returning non-null tells the owning KVCacheStore to publish THIS server
  // rather than start a second one, so a node always serves peers from exactly
  // one port.
  virtual KVCacheStoreServer* store_server() const { return nullptr; }
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_BACKEND_H_
