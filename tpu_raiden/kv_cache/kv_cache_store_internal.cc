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

#include "tpu_raiden/kv_cache/kv_cache_store_internal.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "grpcpp/security/credentials.h"
#include "tpu_raiden/core/raw_transfer_core.h"
#include "tpu_raiden/core/status_macros.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/kv_cache_manager_base.h"
#include "tpu_raiden/kv_cache/logical_block_manager.h"
#include "tpu_raiden/transport/block_transport.h"

namespace tpu_raiden {
namespace kv_cache {

namespace {
int ParsePort(absl::string_view addr) {
  size_t idx = addr.rfind(':');
  if (idx != absl::string_view::npos && idx + 1 < addr.size()) {
    int port = 0;
    if (absl::SimpleAtoi(addr.substr(idx + 1), &port)) {
      return port;
    }
  }
  return 0;
}
}  // namespace

KVCacheStoreInternal::KVCacheStoreInternal(int capacity,
                                           std::string global_registry_address,
                                           std::string local_address)
    : capacity_(capacity),
      local_address_(local_address) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(global_registry_address,
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_unique<global_registry::GlobalRegistryClient>(channel);
  }

  host_block_manager_ = std::make_unique<LogicalBlockManager>(capacity);

  if (!local_address.empty()) {
    int port = ParsePort(local_address);
    server_ =
        std::make_unique<tpu_raiden::transport::BlockTransport>(this, port);

    int actual_port = server_->local_port();
    size_t idx = local_address.rfind(':');
    if (idx != std::string::npos) {
      local_address_ =
          local_address.substr(0, idx + 1) + std::to_string(actual_port);
    } else {
      local_address_ = local_address;
    }
  }
}

KVCacheStoreInternal::~KVCacheStoreInternal() {
  Clear();
  server_.reset();
  host_block_manager_.reset();
}

void KVCacheStoreInternal::Clear() {
  absl::MutexLock lock(mutex_);
  lru_list_.clear();
  cache_map_.clear();
  block_to_ptrs_.clear();
  if (host_block_manager_) {
    host_block_manager_ = std::make_unique<LogicalBlockManager>(capacity_);
  }
}

uint8_t* KVCacheStoreInternal::GetBlockHostPointer(size_t layer_idx,
                                                   size_t shard_idx,
                                                   int block_id) {
  absl::MutexLock lock(mutex_);
  if (num_layers_ == 0) return nullptr;
  auto it = block_to_ptrs_.find(block_id);
  if (it != block_to_ptrs_.end()) {
    return it->second[layer_idx * num_shards_ + shard_idx];
  }
  return nullptr;
}

absl::StatusOr<std::vector<int>> KVCacheStoreInternal::AllocateBlocks(
    size_t num_blocks, uint64_t uuid) {
  absl::MutexLock lock(mutex_);
  if (!host_block_manager_) {
    return absl::FailedPreconditionError("Block manager is not initialized");
  }
  return host_block_manager_->Allocate(num_blocks);
}

absl::Status KVCacheStoreInternal::OnSingleBlockReceived(int block_id,
                                                         size_t size_bytes) {
  return absl::OkStatus();
}

absl::StatusOr<std::pair<std::vector<bool>, raiden::PjRtCopyFuture>>
KVCacheStoreInternal::LookupAndFetch(
    const std::vector<uint64_t>& block_hashes, KVCacheManagerBase& manager,
    const std::vector<int>& dst_offsets_major_dim,
    const std::vector<int>& copy_sizes_major_dim) {
  size_t num_chunks = block_hashes.size();
  if (dst_offsets_major_dim.size() != num_chunks ||
      copy_sizes_major_dim.size() != num_chunks) {
    return absl::InvalidArgumentError("Lengths of lists must match");
  }

  std::vector<bool> hits(num_chunks, false);
  std::vector<raiden::PjRtCopyFuture> futures_to_join;

  {
    absl::MutexLock lock(mutex_);
    CleanupCompletedFuturesLocked();
  }

  if (num_layers_ == 0) {
    num_layers_ = manager.num_layers();
    num_shards_ = manager.num_shards();
    slice_byte_size_ = manager.slice_byte_size();
    shard_factor_ = manager.shard_factor();
  }

  size_t miss_index = num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    uint64_t hash = block_hashes[i];
    std::shared_ptr<std::vector<std::vector<uint8_t>>> host_buffers;
    std::vector<int> host_block_ids;
    raiden::PjRtCopyFuture insert_future;

    {
      absl::MutexLock lock(mutex_);
      auto it = cache_map_.find(hash);
      if (it != cache_map_.end()) {
        hits[i] = true;
        host_buffers = it->second->host_buffers;
        host_block_ids = it->second->internal_block_ids;
        insert_future = it->second->insert_future;

        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
      }
    }

    if (hits[i]) {
      if (insert_future.IsValid()) {
        (void)insert_future.Await();
      }
      int needed = copy_sizes_major_dim[i];
      if (host_block_ids.size() < needed) {
        return absl::InternalError("Cached entry does not have enough blocks");
      }

      int dst_major_dim_offset = dst_offsets_major_dim[i];
      std::vector<int64_t> dummy_src_offsets;
      std::vector<int64_t> hit_dst_offsets;
      std::vector<int64_t> hit_sizes;
      dummy_src_offsets.reserve(needed);
      hit_dst_offsets.reserve(needed);
      hit_sizes.reserve(needed);

      for (int k = 0; k < needed; ++k) {
        dummy_src_offsets.push_back(k);
        hit_dst_offsets.push_back(dst_major_dim_offset + k);
        hit_sizes.push_back(1);
      }

      std::vector<const uint8_t*> host_ptrs;
      host_ptrs.reserve(host_buffers->size());
      for (const auto& buf : *host_buffers) {
        host_ptrs.push_back(buf.data());
      }
      std::vector<size_t> host_sizes;
      host_sizes.reserve(host_buffers->size());
      for (const auto& buf : *host_buffers) {
        host_sizes.push_back(buf.size());
      }
      manager.SetExternalHostPointers(host_ptrs, host_sizes);
      auto fut_or = manager.H2d(dummy_src_offsets, hit_dst_offsets, hit_sizes);
      if (!fut_or.ok()) {
        return fut_or.status();
      }
      futures_to_join.push_back(std::move(fut_or.value()));
    } else {
      miss_index = i;
      break;
    }
  }

  if (miss_index == 0 && num_chunks > 0 && registry_client_) {
    RETURN_IF_ERROR(
        LookupAndFetchRemote(block_hashes, manager, dst_offsets_major_dim,
                             copy_sizes_major_dim, hits, futures_to_join));
  }

  return std::pair<std::vector<bool>, raiden::PjRtCopyFuture>{
      std::move(hits),
      raiden::JoinPjRtCopyFutures(absl::MakeSpan(futures_to_join))};
}

absl::Status KVCacheStoreInternal::Insert(
    const std::vector<uint64_t>& block_hashes, KVCacheManagerBase& manager,
    const std::vector<int>& src_offsets_major_dim,
    const std::vector<int>& copy_sizes_major_dim) {
  {
    absl::MutexLock lock(mutex_);
    CleanupCompletedFuturesLocked();
  }

  size_t num_chunks = block_hashes.size();
  if (src_offsets_major_dim.size() != num_chunks ||
      copy_sizes_major_dim.size() != num_chunks) {
    return absl::InvalidArgumentError("Lengths of lists must match");
  }

  if (num_layers_ == 0) {
    num_layers_ = manager.num_layers();
    num_shards_ = manager.num_shards();
    slice_byte_size_ = manager.slice_byte_size();
    shard_factor_ = manager.shard_factor();
  }

  size_t bytes_per_block = slice_byte_size_;

  std::vector<int> store_block_ids;
  int total_needed_blocks = 0;
  std::vector<int> blocks_per_chunk;
  blocks_per_chunk.reserve(num_chunks);
  for (int copy_size : copy_sizes_major_dim) {
    int needed = copy_size;
    total_needed_blocks += needed;
    blocks_per_chunk.push_back(needed);
  }

  {
    absl::MutexLock lock(mutex_);
    ASSIGN_OR_RETURN(store_block_ids,
                     host_block_manager_->Allocate(total_needed_blocks));
  }

  size_t shard_alloc_size = total_needed_blocks * bytes_per_block;
  auto host_buffers = std::make_shared<std::vector<std::vector<uint8_t>>>(
      num_layers_ * num_shards_);
  std::vector<const uint8_t*> host_ptrs;
  std::vector<size_t> host_sizes;
  for (size_t j = 0; j < num_layers_ * num_shards_; ++j) {
    (*host_buffers)[j].resize(shard_alloc_size);
    host_ptrs.push_back((*host_buffers)[j].data());
    host_sizes.push_back(shard_alloc_size);
  }

  manager.SetExternalHostPointers(host_ptrs, host_sizes);
  std::vector<int64_t> src_offsets_64(src_offsets_major_dim.begin(),
                                      src_offsets_major_dim.end());
  std::vector<int64_t> copy_sizes_64(copy_sizes_major_dim.begin(),
                                     copy_sizes_major_dim.end());

  auto fut_or = manager.D2hAutoAllocate(src_offsets_64, copy_sizes_64);
  if (!fut_or.ok()) {
    absl::MutexLock lock(mutex_);
    (void)host_block_manager_->Unlock(store_block_ids);
    return fut_or.status();
  }

  auto [dummy_block_ids, insert_future] = std::move(fut_or).value();

  size_t block_idx = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    uint64_t hash = block_hashes[i];
    int needed = blocks_per_chunk[i];
    std::vector<int> chunk_block_ids(
        store_block_ids.begin() + block_idx,
        store_block_ids.begin() + block_idx + needed);

    {
      absl::MutexLock lock(mutex_);
      for (int k = 0; k < needed; ++k) {
        int store_block_id = chunk_block_ids[k];
        block_to_ptrs_[store_block_id].resize(num_layers_ * num_shards_);
        for (size_t l = 0; l < num_layers_; ++l) {
          for (size_t sh = 0; sh < num_shards_; ++sh) {
            uint8_t* ptr = (*host_buffers)[l * num_shards_ + sh].data() +
                           (block_idx + k) * bytes_per_block;
            block_to_ptrs_[store_block_id][l * num_shards_ + sh] = ptr;
          }
        }
      }

      auto map_it = cache_map_.find(hash);
      if (map_it != cache_map_.end()) {
        auto entry = *map_it->second;
        (void)host_block_manager_->Unlock(entry.internal_block_ids);
        for (int block_id : entry.internal_block_ids) {
          block_to_ptrs_.erase(block_id);
        }
        lru_list_.erase(map_it->second);
        cache_map_.erase(map_it);
      }

      while (lru_list_.size() >= capacity_) {
        auto back = lru_list_.back();
        (void)host_block_manager_->Unlock(back.internal_block_ids);
        for (int block_id : back.internal_block_ids) {
          block_to_ptrs_.erase(block_id);
        }
        cache_map_.erase(back.block_hash);
        lru_list_.pop_back();
      }

      lru_list_.push_front(
          {hash, chunk_block_ids, host_buffers, insert_future});
      cache_map_[hash] = lru_list_.begin();
    }
    block_idx += needed;
  }

  RETURN_IF_ERROR(RegisterBlocksInGlobalRegistry(
      block_hashes, copy_sizes_major_dim, store_block_ids));

  return absl::OkStatus();
}

absl::Status KVCacheStoreInternal::LookupAndFetchRemote(
    const std::vector<uint64_t>& block_hashes, KVCacheManagerBase& manager,
    const std::vector<int>& dst_offsets_major_dim,
    const std::vector<int>& copy_sizes_major_dim, std::vector<bool>& hits,
    std::vector<raiden::PjRtCopyFuture>& futures_to_join) {
  size_t num_chunks = block_hashes.size();
  std::vector<std::string> remaining_hashes;
  remaining_hashes.reserve(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    remaining_hashes.push_back(std::to_string(block_hashes[i]));
  }

  ASSIGN_OR_RETURN(const auto& lookup_results,
                   registry_client_->Lookup(remaining_hashes),
                   absl::OkStatus());
  if (lookup_results.empty()) {
    return absl::OkStatus();
  }

  struct FetchTask {
    std::string peer;
    std::vector<size_t> chunk_indices;
    std::vector<int> remote_block_ids;
    std::vector<int> blocks_needed;
  };
  absl::flat_hash_map<std::string, FetchTask> fetch_tasks;

  for (size_t idx = 0; idx < lookup_results.size(); ++idx) {
    const auto& meta = lookup_results[idx];
    std::string peer = meta.host_address();
    int start_remote_id = meta.block_id();
    int needed = copy_sizes_major_dim[idx];

    auto& task = fetch_tasks[peer];
    task.peer = peer;
    task.chunk_indices.push_back(idx);
    task.blocks_needed.push_back(needed);
    for (int k = 0; k < needed; ++k) {
      task.remote_block_ids.push_back(start_remote_id + k);
    }
  }

  struct ActiveFetch {
    FetchTask task;
    std::shared_ptr<std::vector<std::vector<uint8_t>>> staging_buffers;
  };
  std::vector<ActiveFetch> active_fetches;
  active_fetches.reserve(fetch_tasks.size());

  size_t bytes_per_block = slice_byte_size_;

  for (auto& [peer, task] : fetch_tasks) {
    size_t total_blocks = 0;
    for (int needed : task.blocks_needed) {
      total_blocks += needed;
    }

    auto staging_buffers = std::make_shared<std::vector<std::vector<uint8_t>>>(
        num_layers_ * num_shards_,
        std::vector<uint8_t>(total_blocks * bytes_per_block));

    std::vector<uint8_t*> staging_ptrs;
    staging_ptrs.reserve(num_layers_ * num_shards_);
    for (auto& buf : *staging_buffers) {
      staging_ptrs.push_back(buf.data());
    }

    std::vector<int> dummy_local_ids(total_blocks);
    std::iota(dummy_local_ids.begin(), dummy_local_ids.end(), 0);

    ASSIGN_OR_RETURN(auto fut,
                     manager.H2hReadExplicit(peer, task.remote_block_ids,
                                             dummy_local_ids, staging_ptrs));
    active_fetches.push_back({std::move(task), std::move(staging_buffers)});
  }

  for (const auto& fetch : active_fetches) {
    const auto& task = fetch.task;
    const auto& staging_buffers = *fetch.staging_buffers;

    size_t chunk_block_offset_in_task = 0;
    for (size_t i = 0; i < task.chunk_indices.size(); ++i) {
      size_t chunk_idx = task.chunk_indices[i];
      int needed = task.blocks_needed[i];

      auto host_buffers = std::make_shared<std::vector<std::vector<uint8_t>>>(
          num_layers_ * num_shards_,
          std::vector<uint8_t>(needed * bytes_per_block));

      for (size_t l = 0; l < num_layers_; ++l) {
        for (size_t sh = 0; sh < num_shards_; ++sh) {
          const uint8_t* src_ptr =
              staging_buffers[l * num_shards_ + sh].data() +
              chunk_block_offset_in_task * bytes_per_block;
          std::memcpy((*host_buffers)[l * num_shards_ + sh].data(), src_ptr,
                      needed * bytes_per_block);
        }
      }

      KVCacheCopySpec copy_spec;
      copy_spec.src_offsets = {0};
      copy_spec.dst_offsets = {
          static_cast<int64_t>(dst_offsets_major_dim[chunk_idx])};
      copy_spec.sizes = {static_cast<int64_t>(needed)};

      std::vector<const uint8_t*> host_ptrs;
      host_ptrs.reserve(num_layers_ * num_shards_);
      std::vector<size_t> host_sizes;
      host_sizes.reserve(num_layers_ * num_shards_);
      for (size_t l = 0; l < num_layers_; ++l) {
        for (size_t sh = 0; sh < num_shards_; ++sh) {
          host_ptrs.push_back((*host_buffers)[l * num_shards_ + sh].data());
          host_sizes.push_back(needed * bytes_per_block);
        }
      }
      manager.SetExternalHostPointers(host_ptrs, host_sizes);

      ASSIGN_OR_RETURN(auto chunk_insert_future,
                       manager.H2d(copy_spec.src_offsets, copy_spec.dst_offsets,
                                   copy_spec.sizes));
      chunk_insert_future.AddKeepAlive(host_buffers);

      futures_to_join.push_back(chunk_insert_future);
      hits[chunk_idx] = true;
      chunk_block_offset_in_task += needed;
    }
  }

  return absl::OkStatus();
}

absl::Status KVCacheStoreInternal::RegisterBlocksInGlobalRegistry(
    const std::vector<uint64_t>& block_hashes,
    const std::vector<int>& copy_sizes_major_dim,
    const std::vector<int>& allocated_block_ids) {
  if (!registry_client_ || local_address_.empty()) {
    return absl::OkStatus();
  }

  std::vector<global_registry::Registration> metadata_list;
  metadata_list.reserve(block_hashes.size());

  size_t block_idx = 0;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    global_registry::Registration meta;
    meta.prefix_hash = std::to_string(block_hashes[i]);
    meta.host_address = local_address_;
    meta.block_id = allocated_block_ids[block_idx];
    metadata_list.push_back(std::move(meta));

    int needed = copy_sizes_major_dim[i];
    block_idx += needed;
  }

  return registry_client_->Register(metadata_list);
}

void KVCacheStoreInternal::CleanupCompletedFuturesLocked() {
  for (auto it = lru_list_.begin(); it != lru_list_.end();) {
    if (it->insert_future.IsValid() && it->insert_future.IsReady()) {
      it->insert_future = raiden::PjRtCopyFuture();
    }
    ++it;
  }
}

}  // namespace kv_cache
}  // namespace tpu_raiden
