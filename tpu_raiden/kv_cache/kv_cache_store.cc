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
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "grpcpp/create_channel.h"
#include "grpcpp/grpcpp.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/security/credentials.h"
#include "xla/tsl/concurrency/future.h"
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
                           int64_t shard_size_bytes, int raiden_controller_port,
                           absl::string_view raiden_orchestrator_address)
    : lru_cache_(capacity),
      raiden_id_(std::move(raiden_id)),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
  if (!global_registry_address.empty()) {
    auto channel = grpc::CreateChannel(std::string(global_registry_address),
                                       grpc::InsecureChannelCredentials());
    registry_client_ =
        std::make_shared<global_registry::GlobalRegistryClient>(channel);
  }
  if (num_shards > 0) {
    ::tpu_raiden::rpc::RaidenIdProto unit_proto;
    unit_proto.set_job_name(raiden_id_.job_name);
    unit_proto.set_job_replica_id(raiden_id_.job_replica_id);
    unit_proto.set_data_name(raiden_id_.data_name);
    unit_proto.set_data_replica_idx(raiden_id_.data_replica_idx);

    raiden_controller_ =
        std::make_unique<::tpu_raiden::controller::RaidenController>(
            unit_proto, capacity, num_shards, shard_size_bytes,
            raiden_controller_port, raiden_orchestrator_address);
  }
  if (raiden_controller_) {
    poller_thread_ =
        std::make_unique<std::thread>(&KVCacheStore::PollerLoop, this);
  }
}

KVCacheStore::KVCacheStore(
    size_t capacity,
    std::unique_ptr<::tpu_raiden::controller::RaidenController>
        raiden_controller,
    absl::string_view global_registry_address, RaidenId raiden_id)
    : lru_cache_(capacity),
      raiden_id_(std::move(raiden_id)),
      raiden_controller_(std::move(raiden_controller)),
      write_through_pool_(std::make_unique<::tpu_raiden::NumaThreadPool>(4)) {
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
      RaidenBlockID* existing = lru_cache_.Get(hash);
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
      for (size_t i = 0; i < global_results.size(); ++i) {
        const auto& metadata = global_results[i];
        const auto& proto_id = metadata.raiden_id();
        RaidenId remote_id{
            .job_name = proto_id.job_name(),
            .job_replica_id = proto_id.job_replica_id(),
            .data_name = proto_id.data_name(),
            .data_replica_idx = proto_id.data_replica_idx(),
        };
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
std::pair<bool, BlockSliceList> KVCacheStore::InsertAndPin(
    const std::vector<std::string>& block_hashes,
    const std::vector<RaidenBlockID>& slices, bool /*on_host*/) {
  absl::MutexLock lock(mutex_);
  BlockSliceList evicted_entries;

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
      return std::make_pair(false, std::move(evicted_entries));
    }
  }

  // 2. Check if free space in lru_cache can hold all new block_hashes
  if (lru_cache_.available_space() < new_indices.size()) {
    for (size_t i : existing_indices) {
      lru_cache_.Unpin(block_hashes[i]);
    }
    return std::make_pair(false, std::move(evicted_entries));
  }

  // Insert all new block hashes into the lru cache list
  for (size_t i : new_indices) {
    const std::string& hash = block_hashes[i];
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
      for (auto it = evicted_entries.rbegin(); it != evicted_entries.rend();
           ++it) {
        lru_cache_.PutBack(it->first, std::move(it->second));
      }
      return std::make_pair(false, BlockSliceList{});
    }
  }

  return std::make_pair(true, std::move(evicted_entries));
}

// Reverts an InsertAndPin operation.
// 1. Unpins all block hashes in the LRU cache.
// 2. Erases remote block hashes whose pin count reaches 0.
// 3. Restores evicted entries to the back of the LRU cache in reverse order for
//    each deleted remote block. Returns the number of deleted remote blocks and
//    the remaining unrestored evicted entries.
std::pair<size_t, BlockSliceList> KVCacheStore::ReleaseAndDelete(
    const std::vector<std::string>& block_hashes,
    BlockSliceList pending_evict_entries) {
  absl::MutexLock lock(mutex_);
  size_t deleted_remote_blocks = 0;
  for (const std::string& hash : block_hashes) {
    lru_cache_.Unpin(hash);
    auto* val = lru_cache_.Peek(hash);
    if (val != nullptr && val->status == BlockStatus::REMOTE &&
        lru_cache_.GetPinCount(hash) == 0) {
      lru_cache_.Erase(hash);
      deleted_remote_blocks++;
    }
  }

  if (deleted_remote_blocks > pending_evict_entries.size()) {
    LOG(WARNING) << "Number of deleted remote blocks (" << deleted_remote_blocks
                 << ") exceeds number of pending evict entries ("
                 << pending_evict_entries.size() << ").";
  }

  size_t to_restore =
      std::min(deleted_remote_blocks, pending_evict_entries.size());
  for (size_t i = 0; i < to_restore; ++i) {
    auto& entry = pending_evict_entries.back();
    lru_cache_.PutBack(entry.first, std::move(entry.second));
    pending_evict_entries.pop_back();
  }

  return std::make_pair(deleted_remote_blocks,
                        std::move(pending_evict_entries));
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
  for (const std::string& hash : block_hashes) {
    lru_cache_.Unpin(hash);
  }
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
  auto host_blocks_or =
      raiden_controller_->AllocateBlockIds(block_hashes.size());
  if (!host_blocks_or.ok()) {
    absl::MutexLock lock(mutex_);
    for (const auto& hash : block_hashes) {
      saving_hashes_.erase(hash);
    }
    return host_blocks_or.status();
  }
  const auto& host_block_ids = host_blocks_or.value();
  std::vector<int64_t> host_block_ids_64(host_block_ids.begin(),
                                         host_block_ids.end());

  // Trigger transfer
  tsl::Future<> future = raiden_controller_->TransferBuffers(
      rpc::MEMORY_TYPE_HBM, rpc::MEMORY_TYPE_DRAM, src_device_block_ids,
      host_block_ids_64);

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

  std::vector<int64_t> dst_device_block_ids(device_block_ids.begin(),
                                            device_block_ids.end());

  // Trigger transfer
  tsl::Future<> future = raiden_controller_->TransferBuffers(
      rpc::MEMORY_TYPE_DRAM, rpc::MEMORY_TYPE_HBM, src_host_block_ids,
      dst_device_block_ids);

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

int KVCacheStore::GetPinCount(const std::string& hash) const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.GetPinCount(hash);
}

size_t KVCacheStore::capacity() const {
  absl::MutexLock lock(mutex_);
  return lru_cache_.capacity();
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

void KVCacheStore::PollerLoop() {
  while (!stop_poller_.load()) {
    PollFuturesInternal();
    absl::SleepFor(absl::Milliseconds(10));
  }
}

void KVCacheStore::PollFuturesInternal() {
  std::vector<SaveState> ready_saves;
  std::vector<LoadState> ready_loads;

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
  }

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
          if (raiden_controller_) {
            (void)raiden_controller_->DeallocateBlockIds(
                {state.host_block_ids[i]});
          }
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
      if (raiden_controller_) {
        (void)raiden_controller_->DeallocateBlockIds(state.host_block_ids);
      }
      for (const auto& hash : state.block_hashes) {
        failed_saves_.push_back(hash);
      }
    }
    for (const auto& hash : state.block_hashes) {
      saving_hashes_.erase(hash);
    }
  }

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

int KVCacheStore::raiden_controller_port() const {
  if (raiden_controller_) {
    return raiden_controller_->raiden_controller_port();
  }
  return 0;
}

}  // namespace kv_cache
}  // namespace tpu_raiden
