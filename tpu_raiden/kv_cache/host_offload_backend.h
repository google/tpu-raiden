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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_metadata.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend.h"
#include "tpu_raiden/kv_cache/kv_cache_store_backend_factory.h"
#include "tpu_raiden/kv_cache/kv_cache_store_client.h"
#include "tpu_raiden/kv_cache/kv_cache_store_server.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/proto/kv_cache_store_service.pb.h"

namespace tpu_raiden {

namespace controller {
class RaidenController;
}  // namespace controller

namespace kv_cache {

class HostOffloadBackend : public KVCacheStoreBackend {
 public:
  static absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> Create(
      const BackendConfig& config);
  static absl::StatusOr<std::shared_ptr<KVCacheStoreBackend>> Create(
      const BackendConfig& config,
      controller::RaidenController* absl_nonnull controller);

  explicit HostOffloadBackend(
      size_t capacity, std::optional<KVCacheMetadata> metadata = std::nullopt,
      RaidenId raiden_id = {},
      controller::RaidenController* raiden_controller = nullptr,
      std::shared_ptr<global_registry::GlobalRegistryClient> registry_client =
          nullptr);

  ~HostOffloadBackend() override;

  std::string name() const override { return "HostOffloadBackend"; }

  absl::StatusOr<BlockSliceList> Lookup(
      absl::Span<const std::string> block_hashes,
      const LookupOptions& options = {}) override;

  std::pair<bool, BlockSliceList> Insert(
      absl::Span<const std::string> block_hashes,
      absl::Span<const RaidenBlockID> slices, bool on_host) override;

  bool InsertAndLock(absl::Span<const std::string> block_hashes,
                     absl::Span<const RaidenBlockID> slices,
                     bool on_host) override;

  size_t ReleaseAndDelete(absl::Span<const std::string> block_hashes) override;

  void Delete(absl::Span<const std::string> block_hashes,
              absl::Span<const RaidenBlockID> slices) override;

  bool Pin(absl::Span<const std::string> block_hashes) override;

  void Release(absl::Span<const std::string> block_hashes) override;

  int GetPinCount(const std::string& hash) const override;

  size_t GetCapacity() const override;

  size_t GetSize() const override;

  size_t GetAvailableSpace() const override;

  absl::StatusOr<size_t> RecoverFromLocalManifest() override;

  bool ValidateAndPinHostBlocks(absl::Span<const int> host_block_ids) override;

  std::vector<std::string> GetEvictableKeys(size_t count) override;

  std::vector<int> Evict(const std::vector<std::string>& block_hashes) override;

  std::vector<std::string> GetEvictCandidateKeys() const override;

  void SetRaidenController(controller::RaidenController* controller) override;

  void set_raiden_controller(controller::RaidenController* controller) {
    SetRaidenController(controller);
  }

  // --- Global Memory Pooling & RPC Methods ---
  absl::Status StartServer(absl::string_view server_address);

  tsl::Future<> Load(const RaidenId& remote_id,
                     absl::Span<const std::string> block_hashes,
                     absl::Span<const int32_t> device_block_ids = {});

 private:
  absl::StatusOr<std::shared_ptr<KVCacheStoreClient>> GetKVCacheStoreClient(
      const RaidenId& remote_id);

  void SetMetadataEntry(absl::string_view hash, const RaidenBlockID& block)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void ClearMetadataEntry(const RaidenBlockID& block)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Reclaims the state of a stale eviction candidate that is about to be
  // replaced by a fresh insert of the same hash: clears its metadata entry
  // and returns its host block (if it owned one) to the controller's block
  // allocator. Without this, Put() on a candidate hash overwrites the value
  // in place and the old host block leaks permanently.
  void ReclaimStaleCandidate(const std::string& hash)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  std::vector<std::string> GetSortedHashes(
      absl::Span<const std::string> hashes) const;

  mutable absl::Mutex mutex_;
  LRUCache<std::string, RaidenBlockID> lru_cache_ ABSL_GUARDED_BY(mutex_);
  std::optional<KVCacheMetadata> metadata_ ABSL_GUARDED_BY(mutex_);
  uint64_t next_metadata_seq_ ABSL_GUARDED_BY(mutex_) = 0;
  RaidenId raiden_id_ ABSL_GUARDED_BY(mutex_);
  controller::RaidenController* raiden_controller_ ABSL_GUARDED_BY(mutex_) =
      nullptr;
  absl::flat_hash_map<std::vector<std::string>, size_t> pending_eviction_counts_
      ABSL_GUARDED_BY(mutex_);

  std::shared_ptr<global_registry::GlobalRegistryClient> registry_client_
      ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<KVCacheStoreServer> server_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<RaidenId, std::shared_ptr<KVCacheStoreClient>,
                      RaidenIdHash>
      store_clients_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_RAIDEN_KV_CACHE_HOST_OFFLOAD_BACKEND_H_
