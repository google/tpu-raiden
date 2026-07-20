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

#include "tpu_raiden/kv_cache/kv_cache_store.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
#include "tpu_raiden/core/buffer.h"
#include "tpu_raiden/core/controller/raiden_controller.h"
#include "tpu_raiden/core/numa_thread_pool.h"
#include "tpu_raiden/kv_cache/global_registry/global_registry_client.h"
#include "tpu_raiden/kv_cache/lru_cache.h"
#include "tpu_raiden/kv_cache/raiden_id.h"
#include "tpu_raiden/rpc/raiden_service.pb.h"

namespace tpu_raiden {
namespace kv_cache {

KVCacheStore::KVCacheStore(size_t capacity,
                           absl::string_view global_registry_address,
                           RaidenId raiden_id, int num_shards,
                           int64_t shard_size_bytes,
                           absl::string_view raiden_orchestrator_address,
                           absl::string_view raiden_controller_address)
    : lru_cache_(capacity),
      raiden_id_(std::move(raiden_id)),
      write_through_pool_(std::make_unique<NumaThreadPool>(4)) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
  if (num_shards > 0) {
    raiden_controller_ =
        std::make_unique<tpu_raiden::controller::RaidenController>(
            raiden_id_.ToProto(), capacity, num_shards, shard_size_bytes,
            raiden_orchestrator_address, raiden_controller_address);
  }
  if (raiden_controller_) {
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::KVCacheStore(
    size_t capacity,
    std::unique_ptr<tpu_raiden::controller::RaidenController> raiden_controller,
    absl::string_view global_registry_address, RaidenId raiden_id)
    : lru_cache_(capacity),
      raiden_id_(std::move(raiden_id)),
      raiden_controller_(std::move(raiden_controller)),
      write_through_pool_(std::make_unique<NumaThreadPool>(4)) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
  if (raiden_controller_) {
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::~KVCacheStore() {
  if (poller_thread_) {
    stop_poller_.store(true);
    if (poller_thread_->joinable()) {
      poller_thread_->join();
    }
  }
  std::vector<tsl::Future<>> futures_to_await;
  {
    absl::MutexLock lock(mutex_);
    for (auto& state : active_saves_) {
      futures_to_await.push_back(state.future);
    }
    for (auto& state : active_loads_) {
      futures_to_await.push_back(state.future);
    }
  }
  for (auto& fut : futures_to_await) {
    (void)fut.Await();
  }
}

absl::StatusOr<BlockSliceList> KVCacheStore::Lookup(
    const std::vector<std::string>& block_hashes, bool enable_global) {
  BlockSliceList results;

  size_t local_hits = 0;
  size_t limit = 0;
  {
    absl::MutexLock lock(mutex_);
    limit = std::min(block_hashes.size(), lru_cache_.available_space());
    results.reserve(limit);
    for (size_t i = 0; i < limit; ++i) {
      const std::string& hash = block_hashes[i];
      RaidenBlockID* existing = lru_cache_.Peek(hash);
      if (!existing) {
        break;
      }
      results.push_back(std::make_pair(hash, *existing));
      local_hits++;
    }
  }

  if (enable_global && local_hits < limit && registry_client_) {
    std::vector<std::string> remaining_hashes(block_hashes.begin() + local_hits,
                                              block_hashes.begin() + limit);
    auto global_results_or = registry_client_->Lookup(remaining_hashes);
    if (global_results_or.ok()) {
      const auto& global_results = global_results_or.value();
      size_t num_results =
          std::min(global_results.size(), remaining_hashes.size());
      for (size_t i = 0; i < num_results; ++i) {
        const auto& metadata = global_results[i];
        const auto& proto_id = metadata.raiden_id();
        RaidenId remote_id = RaidenId::FromProto(proto_id);
        results.push_back(std::make_pair(
            remaining_hashes[i], RaidenBlockID(remote_id, metadata.block_id(),
                                               BlockStatus::REMOTE)));
      }
    } else {
      LOG(WARNING) << "Global registry lookup failed: "
                   << global_results_or.status().message();
    }
  }

  return results;
}

std::pair<bool, BlockSliceList> KVCacheStore::Insert(
    const std::vector<std::string>& block_hashes,
    const std::vector<RaidenBlockID>& slices, bool /*on_host*/) {
  absl::MutexLock lock(mutex_);
  BlockSliceList evicted_entries;
  bool all_inserted = true;

  for (size_t i = 0; i < block_hashes.size(); ++i) {
    const std::string& hash = block_hashes[i];
    if (lru_cache_.Contains(hash)) {
      // NOTE(jcgu): This is technically true, as the key already exists in the
      // cache. However, for the purpose of this insert, we treat it as if it
      // was inserted.
      all_inserted = false;
      continue;
    }
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, RaidenBlockID());
    }
    if (evicted.has_value()) {
      evicted_entries.push_back(std::move(*evicted));
    }
  }

  return std::make_pair(all_inserted, std::move(evicted_entries));
}

// Categorizes block_hashes into existing (local) and new (remote) hashes.
// 1. Pins all existing block hashes.
// 2. Inserts new block hashes into the LRU cache if space permits.
// 3. Pins the newly inserted block hashes, with full rollback on failure.
bool KVCacheStore::InsertAndLock(const std::vector<std::string>& block_hashes,
                                 const std::vector<RaidenBlockID>& slices,
                                 bool /*on_host*/) {
  absl::MutexLock lock(mutex_);

  std::vector<size_t> existing_indices;
  std::vector<size_t> new_indices;
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (lru_cache_.Contains(block_hashes[i])) {
      existing_indices.push_back(i);
    } else {
      new_indices.push_back(i);
    }
  }

  // 1. Pin all existing block_hashes
  for (size_t idx = 0; idx < existing_indices.size(); ++idx) {
    size_t i = existing_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[existing_indices[j]]);
      }
      return false;
    }
  }

  // 2. Check if free space in lru_cache can hold all new block_hashes
  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return false;
  }

  size_t eviction_count = 0;
  // Insert all new block hashes into the lru cache list in reverse order
  for (auto it = new_indices.rbegin(); it != new_indices.rend(); ++it) {
    size_t i = *it;
    const std::string& hash = block_hashes[i];
    std::optional<std::pair<std::string, RaidenBlockID>> evicted;
    if (i < slices.size()) {
      evicted = lru_cache_.Put(hash, slices[i]);
    } else {
      evicted = lru_cache_.Put(hash, RaidenBlockID());
    }
    if (evicted.has_value()) {
      eviction_count++;
    }
  }

  // 3. Pin the newly inserted block_hashes in the lru cache list
  for (size_t idx = 0; idx < new_indices.size(); ++idx) {
    size_t i = new_indices[idx];
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < idx; ++j) {
        lru_cache_.Unpin(block_hashes[new_indices[j]]);
      }
      for (size_t j : existing_indices) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      for (size_t j : new_indices) {
        lru_cache_.Erase(block_hashes[j]);
      }
      for (size_t j = 0; j < eviction_count; ++j) {
        lru_cache_.RestoreLastCandidate();
      }
      return false;
    }
  }

  if (eviction_count > 0) {
    pending_eviction_counts_[GetSortedHashes(block_hashes)] = eviction_count;
  }
  return true;
}

// Reverts an InsertAndLock operation.
// 1. Unpins all block hashes in the LRU cache in reverse order.
// 2. Erases block hashes (not HOST and not HOST_AND_HBM) whose pin count
// reaches 0.
// 3. Restores evicted entries to the back of the LRU cache.
// Returns the number of deleted blocks.
size_t KVCacheStore::ReleaseAndDelete(
    const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  size_t deleted_blocks = 0;

  // 1. Unpin in reverse order
  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }

  // 2. Erase blocks that are not HOST and not HOST_AND_HBM when pin count is 0
  for (const std::string& hash : block_hashes) {
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
        val->status != BlockStatus::HOST &&
        val->status != BlockStatus::HOST_AND_HBM) {
      lru_cache_.Erase(hash);
      deleted_blocks++;
    }
  }

  size_t restoration_count = 0;
  auto it = pending_eviction_counts_.find(GetSortedHashes(block_hashes));
  if (it != pending_eviction_counts_.end()) {
    restoration_count = it->second;
    pending_eviction_counts_.erase(it);
  }

  // 3. Restore candidates
  size_t to_restore = std::min(deleted_blocks, restoration_count);
  for (size_t i = 0; i < to_restore; ++i) {
    lru_cache_.RestoreLastCandidate();
  }

  return deleted_blocks;
}

void KVCacheStore::Delete(const std::vector<std::string>& block_hashes,
                          const std::vector<RaidenBlockID>& slices) {
  absl::MutexLock lock(mutex_);
  for (const std::string& hash : block_hashes) {
    lru_cache_.Erase(hash);
  }
}

bool KVCacheStore::Pin(const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (size_t i = 0; i < block_hashes.size(); ++i) {
    if (!lru_cache_.Pin(block_hashes[i])) {
      for (size_t j = 0; j < i; ++j) {
        lru_cache_.Unpin(block_hashes[j]);
      }
      return false;
    }
  }
  return true;
}

void KVCacheStore::Release(const std::vector<std::string>& block_hashes) {
  absl::MutexLock lock(mutex_);
  for (auto it = block_hashes.rbegin(); it != block_hashes.rend(); ++it) {
    lru_cache_.Unpin(*it);
  }
  pending_eviction_counts_.erase(GetSortedHashes(block_hashes));
}

absl::Status KVCacheStore::Save(const std::vector<std::string>& block_hashes) {
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<int64_t> src_device_block_ids;
  src_device_block_ids.reserve(block_hashes.size());

  {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      RaidenBlockID* existing = lru_cache_.Get(hash);
      if (!existing) {
        return absl::NotFoundError(
            absl::StrCat("Block hash not found: ", hash));
      }
      if (existing->status != BlockStatus::HBM) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not in HBM status: ", hash));
      }
      if (existing->device_block_id == -1) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block device_block_id is -1: ", hash));
      }
      if (lru_cache_.GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (saving_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already saving: ", hash));
      }
      src_device_block_ids.push_back(existing->device_block_id);
    }
    for (const auto& hash : block_hashes) {
      saving_hashes_.insert(hash);
    }
  }

  // Allocate host blocks on controller
  auto host_blocks_or = AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      saving_hashes_.erase(hash);
    }
    return host_blocks_or.status();
  }
  const auto& host_block_ids = host_blocks_or.value();

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_device_block_ids.size());
  for (int64_t id : src_device_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_HBM);
  }
  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(host_block_ids.size());
  for (int id : host_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_DRAM);
  }

  // Trigger transfer
  tsl::Future<> future =
      raiden_controller_->TransferBuffers(src_buffers, dst_buffers);

  {
    absl::MutexLock lock(mutex_);
    active_saves_.push_back(SaveState{
        .future = std::move(future),
        .block_hashes = block_hashes,
        .host_block_ids = host_block_ids,
    });
  }

  return absl::OkStatus();
}

absl::Status KVCacheStore::Load(const std::vector<std::string>& block_hashes,
                                const std::vector<int>& device_block_ids) {
  if (block_hashes.size() != device_block_ids.size()) {
    return absl::InvalidArgumentError(
        "block_hashes and device_block_ids size mismatch");
  }
  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<int64_t> src_host_block_ids;
  src_host_block_ids.reserve(block_hashes.size());

  {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      RaidenBlockID* existing = lru_cache_.Get(hash);
      if (!existing) {
        return absl::NotFoundError(
            absl::StrCat("Block hash not found: ", hash));
      }
      if (existing->status != BlockStatus::HOST &&
          existing->status != BlockStatus::HOST_AND_HBM) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not on host: ", hash));
      }
      if (existing->host_block_id == -1) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block host_block_id is -1: ", hash));
      }
      if (lru_cache_.GetPinCount(hash) <= 0) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      if (loading_hashes_.contains(hash)) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already loading: ", hash));
      }
      src_host_block_ids.push_back(existing->host_block_id);
    }
    for (const auto& hash : block_hashes) {
      loading_hashes_.insert(hash);
    }
  }

  std::vector<Buffer> src_buffers;
  src_buffers.reserve(src_host_block_ids.size());
  for (int64_t id : src_host_block_ids) {
    src_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_DRAM);
  }
  std::vector<Buffer> dst_buffers;
  dst_buffers.reserve(device_block_ids.size());
  for (int id : device_block_ids) {
    dst_buffers.emplace_back(id, std::vector<BufferShard>{}, std::nullopt,
                             rpc::MEMORY_TYPE_HBM);
  }

  // Trigger transfer
  tsl::Future<> future =
      raiden_controller_->TransferBuffers(src_buffers, dst_buffers);

  {
    absl::MutexLock lock(mutex_);
    active_loads_.push_back(LoadState{
        .future = std::move(future),
        .block_hashes = block_hashes,
        .device_block_ids = device_block_ids,
    });
  }

  return absl::OkStatus();
}

absl::Status KVCacheStore::ReadRemote(
    const std::vector<std::string>& block_hashes) {
  std::vector<std::string> successfully_marked_as_reading;
  successfully_marked_as_reading.reserve(block_hashes.size());
  auto cleanup = absl::MakeCleanup([this, &successfully_marked_as_reading]() {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : successfully_marked_as_reading) {
      reading_hashes_.erase(hash);
    }
  });

  if (!raiden_controller_) {
    return absl::FailedPreconditionError("RaidenController is not initialized");
  }

  std::vector<std::pair<RaidenId, int32_t>> src_info;
  src_info.reserve(block_hashes.size());

  {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      RaidenBlockID* existing = lru_cache_.PeekMutable(hash);
      if (!existing) {
        return absl::InvalidArgumentError(
            absl::StrCat("Block hash not found: ", hash));
      }
      if (existing->status != BlockStatus::REMOTE) {
        return absl::InvalidArgumentError(
            absl::StrCat("Block status is not REMOTE: ", hash));
      }
      if (lru_cache_.GetPinCount(hash) <= 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Block is not pinned: ", hash));
      }
      auto [it, inserted] = reading_hashes_.insert(hash);
      if (!inserted) {
        return absl::FailedPreconditionError(
            absl::StrCat("Block is already reading: ", hash));
      }
      successfully_marked_as_reading.push_back(hash);
      src_info.push_back({existing->raiden_id, existing->host_block_id});
    }
  }

  // Allocate host blocks on controller
  auto host_blocks_or = AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    LOG(WARNING) << "Failed to allocate local block IDs for ReadRemote: "
                 << host_blocks_or.status().message();
    return host_blocks_or.status();
  }
  const auto& dest_host_block_ids = host_blocks_or.value();

  {
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < block_hashes.size(); ++i) {
      RaidenBlockID* existing = lru_cache_.PeekMutable(block_hashes[i]);
      if (existing) {
        existing->host_block_id = dest_host_block_ids[i];
      }
    }
  }

  // Group by source RaidenId
  absl::flat_hash_map<RaidenId, std::vector<size_t>, RaidenIdHash> groups;
  for (size_t i = 0; i < src_info.size(); ++i) {
    groups[src_info[i].first].push_back(i);
  }

  std::vector<tsl::Future<>> futures;
  futures.reserve(groups.size());
  for (const auto& [src_raiden_id, indices] : groups) {
    std::vector<int32_t> group_src_ids;
    std::vector<int32_t> group_dst_ids;
    group_src_ids.reserve(indices.size());
    group_dst_ids.reserve(indices.size());
    for (size_t idx : indices) {
      group_src_ids.push_back(src_info[idx].second);
      group_dst_ids.push_back(dest_host_block_ids[idx]);
    }
    futures.push_back(raiden_controller_->ReadRemote(
        src_raiden_id, group_src_ids, group_dst_ids));
  }

  tsl::Future<> combined_future = tsl::JoinFutures(futures);

  {
    absl::MutexLock lock(mutex_);
    active_remote_reads_.emplace(std::move(combined_future),
                                 RemoteReadState{
                                     .block_hashes = block_hashes,
                                     .host_block_ids = dest_host_block_ids,
                                 });
  }

  std::move(cleanup).Cancel();
  return absl::OkStatus();
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollRemoteReadStatus() {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> done = std::move(done_remote_reads_);
  done_remote_reads_.clear();
  std::vector<std::string> failed = std::move(failed_remote_reads_);
  failed_remote_reads_.clear();

  std::vector<std::string> pending;
  size_t total_pending_blocks = 0;
  for (const auto& [future, state] : active_remote_reads_) {
    total_pending_blocks += state.block_hashes.size();
  }
  pending.reserve(total_pending_blocks);
  for (const auto& [future, state] : active_remote_reads_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }

  return {std::move(done), std::move(failed), std::move(pending)};
}

int KVCacheStore::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t KVCacheStore::capacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
}

std::string KVCacheStore::raiden_controller_address() const {
  if (raiden_controller_) {
    return raiden_controller_->controller_address();
  }
  return "";
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollSaveStatus() {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> done = std::move(done_saves_);
  done_saves_.clear();
  std::vector<std::string> failed = std::move(failed_saves_);
  failed_saves_.clear();

  std::vector<std::string> pending;
  size_t total_pending_blocks = 0;
  for (const auto& state : active_saves_) {
    total_pending_blocks += state.block_hashes.size();
  }
  pending.reserve(total_pending_blocks);
  for (const auto& state : active_saves_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }

  return {std::move(done), std::move(failed), std::move(pending)};
}

std::tuple<std::vector<std::string>, std::vector<std::string>,
           std::vector<std::string>>
KVCacheStore::PollLoadStatus() {
  absl::MutexLock lock(mutex_);
  std::vector<std::string> done = std::move(done_loads_);
  done_loads_.clear();
  std::vector<std::string> failed = std::move(failed_loads_);
  failed_loads_.clear();

  std::vector<std::string> pending;
  size_t total_pending_blocks = 0;
  for (const auto& state : active_loads_) {
    total_pending_blocks += state.block_hashes.size();
  }
  pending.reserve(total_pending_blocks);
  for (const auto& state : active_loads_) {
    for (const auto& hash : state.block_hashes) {
      pending.push_back(hash);
    }
  }

  return {std::move(done), std::move(failed), std::move(pending)};
}

absl::StatusOr<size_t> KVCacheStore::RecoverFromRegistry() {
  if (!registry_client_) {
    return absl::FailedPreconditionError(
        "RecoverFromRegistry requires a global registry connection");
  }
  if (!raiden_controller_) {
    return absl::FailedPreconditionError(
        "RecoverFromRegistry requires a raiden controller");
  }

  auto pulled_or = registry_client_->PullOwned(raiden_id_);
  if (!pulled_or.ok()) {
    return pulled_or.status();
  }
  std::vector<global_registry::GlobalRegistryClient::PulledEntry> pulled =
      *std::move(pulled_or);
  if (pulled.empty()) {
    return 0;
  }

  // Sort so that entries closest to expiry come first: they are inserted
  // first and therefore evicted first. 0 means "never expires" and sorts as
  // the most recent.
  auto effective_ttl =
      [](const global_registry::GlobalRegistryClient::PulledEntry& entry) {
        return entry.remaining_ttl_seconds == 0
                   ? std::numeric_limits<int64_t>::max()
                   : entry.remaining_ttl_seconds;
      };
  std::sort(pulled.begin(), pulled.end(),
            [&effective_ttl](const auto& a, const auto& b) {
              return effective_ttl(a) < effective_ttl(b);
            });

  absl::MutexLock lock(mutex_);

  // Skip hashes already present, then keep only the most recently registered
  // entries that fit in the remaining directory capacity. The skip is
  // defensive: the directory is normally empty here, but it keeps a re-pull
  // after a partial recovery (or a call on a live store) from double-inserting.
  std::vector<const global_registry::GlobalRegistryClient::PulledEntry*>
      eligible;
  eligible.reserve(pulled.size());
  for (const auto& entry : pulled) {
    if (lru_cache_.PeekIncludingCandidates(entry.prefix_hash) == nullptr) {
      eligible.push_back(&entry);
    }
  }
  size_t available = lru_cache_.capacity() > lru_cache_.active_size()
                         ? lru_cache_.capacity() - lru_cache_.active_size()
                         : 0;
  if (eligible.size() > available) {
    LOG(WARNING) << "RecoverFromRegistry: " << eligible.size()
                 << " recoverable entries exceed remaining capacity "
                 << available << "; skipping the oldest ones";
    eligible.erase(eligible.begin(), eligible.end() - available);
  }
  if (eligible.empty()) {
    return 0;
  }

  std::vector<int> block_ids;
  block_ids.reserve(eligible.size());
  for (const auto* entry : eligible) {
    block_ids.push_back(entry->block_id);
  }
  auto restore_status =
      raiden_controller_->RestoreAllocatedBlockIds(block_ids);
  if (!restore_status.ok()) {
    return restore_status;
  }

  for (const auto* entry : eligible) {
    auto evicted = lru_cache_.Put(
        entry->prefix_hash,
        RaidenBlockID(raiden_id_, entry->block_id, BlockStatus::HOST));
    if (evicted.has_value()) {
      // Not expected: inserts are capped to the remaining capacity above.
      LOG(WARNING) << "RecoverFromRegistry: unexpected eviction of "
                   << evicted->first;
      DeallocateBlockIds({evicted->second.host_block_id});
    }
  }
  return eligible.size();
}

absl::StatusOr<std::vector<int>> KVCacheStore::AllocateBlockIds(int needed) {
  std::vector<std::string> hashes_to_deallocate;
  {
    absl::MutexLock l(mutex_);
    int free_count = raiden_controller_->block_manager()->num_free_blocks();
    int to_free = needed - free_count;
    if (to_free > 0) {
      // Get evictable keys. This automatically scans candidates first, then
      // active LRU.
      hashes_to_deallocate = lru_cache_.GetEvictableKeys(to_free);
      if (hashes_to_deallocate.size() < to_free) {
        return absl::ResourceExhaustedError(
            absl::StrCat("Insufficient free blocks and not enough evictable "
                         "blocks. Needed: ",
                         needed, ", Free: ", free_count,
                         ", Evictable: ", hashes_to_deallocate.size()));
      }
    }
  }

  // Perform eviction outside lock
  if (!hashes_to_deallocate.empty()) {
    Evict(hashes_to_deallocate);
  }

  // Now allocate from the controller
  return raiden_controller_->AllocateBlockIds(needed);
}

void KVCacheStore::DeallocateBlockIds(absl::Span<const int> block_ids) {
  if (raiden_controller_) {
    auto status = raiden_controller_->DeallocateBlockIds(block_ids);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to deallocate host block IDs: "
                   << status.message();
    }
  }
}

void KVCacheStore::PollerLoop() {
  while (!stop_poller_.load()) {
    PollFuturesInternal();
    absl::SleepFor(absl::Milliseconds(10));
  }
}

void KVCacheStore::PollRemoteReadsInternal(
    std::vector<std::pair<tsl::Future<>, RemoteReadState>> ready_remote_reads) {
  for (auto& [future, state] : ready_remote_reads) {
    absl::Status status = future.Await();
    absl::MutexLock lock(mutex_);
    if (status.ok()) {
      std::vector<global_registry::Registration> write_through_regs;
      write_through_regs.reserve(state.block_hashes.size());
      for (size_t i = 0; i < state.block_hashes.size(); ++i) {
        const auto& hash = state.block_hashes[i];
        RaidenBlockID* existing = lru_cache_.Get(hash);
        if (existing) {
          existing->host_block_id = state.host_block_ids[i];
          existing->status = BlockStatus::HOST;
          if (registry_client_) {
            write_through_regs.push_back({
                .prefix_hash = hash,
                .raiden_id = raiden_id_,
                .block_id = state.host_block_ids[i],
            });
          }
        } else {
          DeallocateBlockIds({state.host_block_ids[i]});
        }
        done_remote_reads_.push_back(hash);
      }
      if (!write_through_regs.empty() && registry_client_ &&
          write_through_pool_) {
        write_through_pool_->Schedule([client = registry_client_,
                                       regs = std::move(write_through_regs)]() {
          auto status = client->Register(regs);
          if (!status.ok()) {
            LOG(WARNING) << "Async write-through failed after ReadRemote: "
                         << status.message();
          } else {
            LOG(INFO) << "Async write-through succeeded after ReadRemote for "
                      << regs.size() << " blocks";
          }
        });
      }
    } else {
      LOG(WARNING) << "Async ReadRemote failed: " << status.ToString();
      DeallocateBlockIds(state.host_block_ids);
      for (const auto& hash : state.block_hashes) {
        failed_remote_reads_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      reading_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollSavesInternal(std::vector<SaveState> ready_saves) {
  for (auto& state : ready_saves) {
    absl::Status status = state.future.Await();
    absl::MutexLock lock(mutex_);
    if (status.ok()) {
      std::vector<global_registry::Registration> write_through_regs;
      write_through_regs.reserve(state.block_hashes.size());
      for (size_t i = 0; i < state.block_hashes.size(); ++i) {
        const auto& hash = state.block_hashes[i];
        RaidenBlockID* existing = lru_cache_.Get(hash);
        if (existing) {
          existing->host_block_id = state.host_block_ids[i];
          existing->status = BlockStatus::HOST_AND_HBM;
          if (registry_client_) {
            write_through_regs.push_back({
                .prefix_hash = hash,
                .raiden_id = raiden_id_,
                .block_id = state.host_block_ids[i],
            });
          }
        } else {
          DeallocateBlockIds({state.host_block_ids[i]});
        }
        done_saves_.push_back(hash);
      }
      if (!write_through_regs.empty() && registry_client_ &&
          write_through_pool_) {
        write_through_pool_->Schedule([client = registry_client_,
                                       regs = std::move(write_through_regs)]() {
          auto status = client->Register(regs);
          if (!status.ok()) {
            LOG(WARNING) << "Async write-through failed after Save: "
                         << status.message();
          } else {
            LOG(INFO) << "Async write-through succeeded after Save for "
                      << regs.size() << " blocks";
          }
        });
      }
    } else {
      LOG(ERROR) << "Async Save failed: " << status.ToString();
      DeallocateBlockIds(state.host_block_ids);
      for (const auto& hash : state.block_hashes) {
        failed_saves_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      saving_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollLoadsInternal(std::vector<LoadState> ready_loads) {
  for (auto& state : ready_loads) {
    absl::Status status = state.future.Await();
    absl::MutexLock lock(mutex_);
    if (status.ok()) {
      for (size_t i = 0; i < state.block_hashes.size(); ++i) {
        const auto& hash = state.block_hashes[i];
        RaidenBlockID* existing = lru_cache_.Get(hash);
        if (existing) {
          existing->device_block_id = state.device_block_ids[i];
          existing->status = BlockStatus::HOST_AND_HBM;
        }
        done_loads_.push_back(hash);
      }
    } else {
      LOG(ERROR) << "Async Load failed: " << status.ToString();
      for (const auto& hash : state.block_hashes) {
        failed_loads_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      loading_hashes_.erase(hash);
    }
  }
}

void KVCacheStore::PollFuturesInternal() {
  std::vector<SaveState> ready_saves;
  std::vector<LoadState> ready_loads;
  std::vector<std::pair<tsl::Future<>, RemoteReadState>> ready_remote_reads;

  {
    absl::MutexLock lock(mutex_);
    auto it = active_saves_.begin();
    while (it != active_saves_.end()) {
      if (it->future.IsReady()) {
        ready_saves.push_back(std::move(*it));
        it = active_saves_.erase(it);
      } else {
        ++it;
      }
    }

    auto jt = active_loads_.begin();
    while (jt != active_loads_.end()) {
      if (jt->future.IsReady()) {
        ready_loads.push_back(std::move(*jt));
        jt = active_loads_.erase(jt);
      } else {
        ++jt;
      }
    }

    auto kt = active_remote_reads_.begin();
    while (kt != active_remote_reads_.end()) {
      if (kt->first.IsReady()) {
        ready_remote_reads.push_back({kt->first, std::move(kt->second)});
        active_remote_reads_.erase(kt++);
      } else {
        ++kt;
      }
    }
  }

  PollSavesInternal(std::move(ready_saves));
  PollLoadsInternal(std::move(ready_loads));
  PollRemoteReadsInternal(std::move(ready_remote_reads));
}

size_t KVCacheStore::Evict(const std::vector<std::string>& block_hashes) {
  if (block_hashes.empty()) {
    return 0;
  }
  std::vector<std::string> erased_hashes;
  std::vector<int> host_ids_to_deallocate;
  erased_hashes.reserve(block_hashes.size());
  host_ids_to_deallocate.reserve(block_hashes.size());
  {
    absl::MutexLock l(mutex_);
    for (const auto& hash : block_hashes) {
      RaidenBlockID* existing = lru_cache_.PeekMutableIncludingCandidates(hash);
      if (existing != nullptr && lru_cache_.GetPinCount(hash) == 0 &&
          (existing->status == BlockStatus::HOST ||
           existing->status == BlockStatus::HOST_AND_HBM)) {
        host_ids_to_deallocate.push_back(existing->host_block_id);
        erased_hashes.push_back(hash);
        lru_cache_.Erase(hash);
      }
    }
  }

  if (host_ids_to_deallocate.empty()) {
    return 0;
  }

  // 1. Unregister from global registry (batched)
  if (registry_client_) {
    auto status = registry_client_->Unregister(erased_hashes, raiden_id_);
    if (!status.ok()) {
      LOG(WARNING)
          << "Failed to unregister proactively evicted blocks: "
          << status.message();
    }
  }

  // 2. Deallocate host block IDs (batched)
  DeallocateBlockIds(host_ids_to_deallocate);

  return host_ids_to_deallocate.size();
}

std::vector<std::string> KVCacheStore::GetSortedHashes(
    const std::vector<std::string>& hashes) const {
  // Dummy comment to trigger presubmit retry.
  std::vector<std::string> sorted = hashes;
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
