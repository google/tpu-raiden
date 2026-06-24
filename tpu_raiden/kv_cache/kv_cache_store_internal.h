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

#ifndef THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_INTERNAL_H_
#define THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_INTERNAL_H_

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {
namespace kv_cache {

class KVCacheStoreInternal
    : public tpu_raiden::transport::BlockTransportDelegate {
 public:
  KVCacheStoreInternal(int capacity,
                       std::string global_registry_address = "",
                       std::string local_address = "");
  virtual ~KVCacheStoreInternal();
  void Clear();

  int capacity() const { return capacity_; }

  absl::StatusOr<std::pair<std::vector<bool>, raiden::PjRtCopyFuture>>
  LookupAndFetch(const std::vector<uint64_t>& block_hashes,
                 KVCacheManagerBase& manager,
                 const std::vector<int>& dst_offsets_major_dim,
                 const std::vector<int>& copy_sizes_major_dim);

  absl::Status Insert(const std::vector<uint64_t>& block_hashes,
                      KVCacheManagerBase& manager,
                      const std::vector<int>& src_offsets_major_dim,
                      const std::vector<int>& copy_sizes_major_dim);

  // BlockTransportDelegate overrides
  uint8_t* GetHostPointer(size_t layer_idx, size_t shard_idx) override {
    return nullptr;
  }
  size_t GetHostSize(size_t layer_idx, size_t shard_idx) override { return 0; }
  uint8_t* GetBlockHostPointer(size_t layer_idx, size_t shard_idx,
                               int block_id) override;

  absl::StatusOr<std::vector<int>> AllocateBlocks(
      size_t num_blocks, uint64_t uuid = 0) override;
  int GetRemoteReadBlockId(int base_remote_id, int chunk_k) override {
    return base_remote_id + chunk_k;
  }
  absl::Status OnDataReceived() override { return absl::OkStatus(); }
  absl::Status OnSingleBlockReceived(int block_id, size_t size_bytes) override;

  size_t num_layers() const override { return num_layers_; }
  size_t num_shards() const override { return num_shards_; }
  size_t slice_byte_size() const override { return slice_byte_size_; }
  size_t shard_factor() const override { return shard_factor_; }

 private:
  struct CacheEntry {
    uint64_t block_hash;
    std::vector<int> internal_block_ids;
    std::shared_ptr<std::vector<std::vector<uint8_t>>> host_buffers;
    raiden::PjRtCopyFuture insert_future;
  };

  class BlockUnlocker;

  absl::Status LookupAndFetchRemote(
      const std::vector<uint64_t>& block_hashes, KVCacheManagerBase& manager,
      const std::vector<int>& dst_offsets_major_dim,
      const std::vector<int>& copy_sizes_major_dim, std::vector<bool>& hits,
      std::vector<raiden::PjRtCopyFuture>& futures_to_join);

  absl::Status RegisterBlocksInGlobalRegistry(
      const std::vector<uint64_t>& block_hashes,
      const std::vector<int>& copy_sizes_major_dim,
      const std::vector<int>& allocated_block_ids);

  void CleanupCompletedFuturesLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  absl::Mutex mutex_;

  int capacity_;

  std::list<CacheEntry> lru_list_;
  absl::flat_hash_map<uint64_t, std::list<CacheEntry>::iterator> cache_map_;

  std::unique_ptr<global_registry::GlobalRegistryClient> registry_client_;

  std::string local_address_;

  std::unique_ptr<LogicalBlockManager> host_block_manager_;
  absl::flat_hash_map<int, std::vector<uint8_t*>> block_to_ptrs_
      ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<tpu_raiden::transport::BlockTransport> server_;

  size_t num_layers_ = 0;
  size_t num_shards_ = 0;
  size_t slice_byte_size_ = 0;
  size_t shard_factor_ = 1;
};

}  // namespace kv_cache
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_KV_CACHE_KV_CACHE_STORE_INTERNAL_H_
